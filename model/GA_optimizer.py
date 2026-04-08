import numpy as np
import pandas as pd
from collections import Counter

# No single item should exceed this % of total budget
MAX_ITEM_BUDGET_PCT = 0.15

# Category constraints: min/max item count and max budget share
CATEGORY_TARGETS = {
    'meat': {'min': 3, 'max': 6, 'max_budget_pct': 0.30},
    'dairy': {'min': 3, 'max': 6, 'max_budget_pct': 0.20},
    'vegetables': {'min': 2, 'max': 8, 'max_budget_pct': 0.20},
    'staples': {'min': 2, 'max': 5, 'max_budget_pct': 0.20},
    'snacks': {'min': 1, 'max': 5, 'max_budget_pct': 0.15},
    'seafood': {'min': 1, 'max': 3, 'max_budget_pct': 0.20},
    'soup': {'min': 1, 'max': 2, 'max_budget_pct': 0.10},
}


# Preselect candidates
def preselect_candidates(data, targets, n_per_category=20):
    data = data.copy()
    data['cal_eff'] = data['calories_kcal'] / data['price']
    data['prot_eff'] = data['protein_g'] / data['price']
    data['combined'] = (
        data['protein_g'] * 0.35 + data['fat_g'] * 0.15 +
        data['carbs_g'] * 0.20 + data['calories_kcal'] / 100 * 0.30
    ) / data['price']
    data['low_cal'] = data['price'] / (data['calories_kcal'] + 1)

    n_each = max(n_per_category // 4, 3)
    candidates = []
    for _, group in data.groupby('category'):
        merged = pd.concat([
            group.nlargest(n_each, 'cal_eff'),
            group.nlargest(n_each, 'prot_eff'),
            group.nlargest(n_each, 'combined'),
            group.nlargest(n_each, 'low_cal'),
        ]).drop_duplicates(subset='id')
        candidates.append(merged.head(n_per_category))

    result = pd.concat(candidates, ignore_index=True)
    result = result.drop(columns=['cal_eff', 'prot_eff', 'combined', 'low_cal'])
    return result


class GroceryGA:
    def __init__(self, data, targets, budget, max_qty=2,
                 pop_size=200, generations=300, crossover_rate=0.85,
                 mutation_rate=0.1, tournament_size=5, elitism_count=10,
                 n_per_category=20, preferences=None):

        # Step 0: Build candidate pool
        self.candidates = preselect_candidates(data, targets, n_per_category)

        # Add preference-matched items to candidate pool
        if preferences:
            existing_ids = set(self.candidates['id'])
            names = data['name'].str.lower()
            cats = data['category'].str.lower()
            for kw in preferences:
                matched = data[names.str.contains(kw, na=False) | cats.str.contains(kw, na=False)]
                new_items = matched[~matched['id'].isin(existing_ids)]
                if len(new_items) > 0:
                    # Add top 5 matched items by protein efficiency
                    new_items = new_items.copy()
                    new_items['_eff'] = new_items['protein_g'] / new_items['price']
                    top = new_items.nlargest(5, '_eff').drop(columns='_eff')
                    self.candidates = pd.concat([self.candidates, top], ignore_index=True)
                    existing_ids.update(top['id'])

        self.n_items = len(self.candidates)
        self.targets = targets
        self.budget = budget
        self.max_qty = max_qty
        self.preferences = preferences or []

        # Filter out items exceeding single-item price cap
        max_item_price = budget * MAX_ITEM_BUDGET_PCT
        keep = self.candidates['price'] <= max_item_price
        self.candidates = self.candidates[keep].reset_index(drop=True)
        self.n_items = len(self.candidates)

        # Genetic algorithm parameters
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count

        # Extract arrays for vectorized computation
        self.prices = self.candidates['price'].values.astype(np.float64)
        self.protein = self.candidates['protein_g'].values.astype(np.float64)
        self.fat = self.candidates['fat_g'].values.astype(np.float64)
        self.carbs = self.candidates['carbs_g'].values.astype(np.float64)
        self.calories = self.candidates['calories_kcal'].values.astype(np.float64)
        self.categories = self.candidates['category'].values

        # Category encoding
        self.cat_labels = pd.Categorical(self.categories)
        self.cat_ids = self.cat_labels.codes
        self.cat_names = list(self.cat_labels.categories)
        self.name_to_code = {name: code for code, name in enumerate(self.cat_names)}

        # Map category targets to codes
        self.cat_targets = {}
        for name, t in CATEGORY_TARGETS.items():
            if name in self.name_to_code:
                self.cat_targets[self.name_to_code[name]] = t

        # Nutrition targets and weights
        self.target_arr = np.array([
            targets['calories_kcal'], targets['protein_g'],
            targets['fat_g'], targets['carbs_g']
        ])
        self.nutrition_weights = np.array([0.30, 0.35, 0.15, 0.20])

        # Preference scoring
        self.pref_scores = np.zeros(self.n_items)
        if self.preferences:
            names = self.candidates['name'].str.lower().values
            cats = self.candidates['category'].str.lower().values
            for i in range(self.n_items):
                for kw in self.preferences:
                    if kw in names[i]:
                        self.pref_scores[i] += 2.0
                    if kw in cats[i]:
                        self.pref_scores[i] += 1.0

        # Tracking
        self.fitness_history = []
        self.avg_fitness_history = []

        print(f"Candidate pool: {self.n_items} items from {len(self.cat_names)} categories")
        if self.preferences:
            matched = np.sum(self.pref_scores > 0)
            print(f"Preference keywords: {', '.join(self.preferences)} ({matched} matched)")

    # Step 1: Generate initial population
    def _init_population(self):
        pop = np.zeros((self.pop_size, self.n_items), dtype=np.int8)

        for i in range(self.pop_size):
            remaining_budget = self.budget
            remaining_cal = self.targets['calories_kcal']
            cat_count = {}

            # Step 1a: Fill each category to its minimum target
            for cat_code, t in self.cat_targets.items():
                cat_items = np.where(self.cat_ids == cat_code)[0]
                if len(cat_items) == 0:
                    continue
                picks = np.random.choice(cat_items, size=min(t['min'], len(cat_items)), replace=False)
                for pick in picks:
                    qty = np.random.randint(1, self.max_qty + 1)
                    cost = self.prices[pick] * qty
                    if cost <= remaining_budget:
                        pop[i, pick] = qty
                        remaining_budget -= cost
                        remaining_cal -= self.calories[pick] * qty
                        cat_count[cat_code] = cat_count.get(cat_code, 0) + qty

            # Step 1b: Add preferred items
            if len(self.preferences) > 0:
                pref_items = np.where(self.pref_scores > 0)[0]
                if len(pref_items) > 0:
                    n_pref = min(3, len(pref_items))
                    for pick in np.random.choice(pref_items, size=n_pref, replace=False):
                        if pop[i, pick] > 0:
                            continue
                        cat = self.cat_ids[pick]
                        cat_max = self.cat_targets.get(cat, {}).get('max', 4)
                        if cat_count.get(cat, 0) >= cat_max:
                            continue
                        qty = np.random.randint(1, self.max_qty + 1)
                        cost = self.prices[pick] * qty
                        if cost <= remaining_budget:
                            pop[i, pick] = qty
                            remaining_budget -= cost
                            remaining_cal -= self.calories[pick] * qty
                            cat_count[cat] = cat_count.get(cat, 0) + qty

            # Step 1c: Fill until calorie target met
            if len(self.preferences) > 0:
                order = np.argsort(-self.pref_scores + np.random.random(self.n_items) * 0.5)
            else:
                order = np.random.permutation(self.n_items)
            for idx in order:
                if remaining_cal <= 0 or remaining_budget <= 0:
                    break
                if pop[i, idx] > 0:
                    continue
                cat = self.cat_ids[idx]
                cat_max = self.cat_targets.get(cat, {}).get('max', 4)
                if cat_count.get(cat, 0) >= cat_max:
                    continue
                qty = np.random.randint(1, self.max_qty + 1)
                cost = self.prices[idx] * qty
                if cost <= remaining_budget:
                    pop[i, idx] = qty
                    remaining_budget -= cost
                    remaining_cal -= self.calories[idx] * qty
                    cat_count[cat] = cat_count.get(cat, 0) + qty

            # Step 1d: Spend remaining budget on variety
            if remaining_budget > 5:
                for idx in np.random.permutation(self.n_items):
                    if remaining_budget <= 5:
                        break
                    if pop[i, idx] > 0:
                        continue
                    cat = self.cat_ids[idx]
                    cat_max = self.cat_targets.get(cat, {}).get('max', 4)
                    if cat_count.get(cat, 0) >= cat_max:
                        continue
                    cost = self.prices[idx]
                    if cost <= remaining_budget:
                        pop[i, idx] = 1
                        remaining_budget -= cost
                        cat_count[cat] = cat_count.get(cat, 0) + 1

        return pop

    # Step 2: Evaluate fitness for entire population
    def _evaluate_population(self, pop):
        pop_float = pop.astype(np.float64)
        total_cost = pop_float @ self.prices
        total_cal = pop_float @ self.calories
        total_prot = pop_float @ self.protein
        total_fat = pop_float @ self.fat
        total_carbs = pop_float @ self.carbs

        # Step 2a: Nutrition score (0-40 pts, no penalty in 80-120% range)
        actuals = np.column_stack([total_cal, total_prot, total_fat, total_carbs])
        ratios = actuals / self.target_arr
        under = np.maximum(0, 0.8 - ratios)
        over = np.maximum(0, ratios - 1.2)
        weighted_dev = (under + over) @ self.nutrition_weights
        nutrition_score = np.maximum(0, 40 * (1 - weighted_dev * 2))

        # Step 2b: Budget penalty 
        budget_penalty = np.maximum(0, total_cost - self.budget) / self.budget * 200

        # Step 2c: Budget utilization bonus (0-25 pts)
        util = np.where(total_cost <= self.budget, total_cost / self.budget, 0.0)
        cost_bonus = util * 25

        # Step 2d: Category balance score
        cat_score = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            selected = pop[i] > 0
            if not np.any(selected):
                continue
            sel_cats = self.cat_ids[selected]
            counts = Counter(sel_cats)

            for cat_code, t in self.cat_targets.items():
                count = counts.get(cat_code, 0)
                if count >= t['min']:
                    cat_score[i] += 2.5
                elif count > 0:
                    cat_score[i] += 1.0
                else:
                    cat_score[i] -= 3.0
                if count > t['max']:
                    cat_score[i] -= (count - t['max']) * 3

                # Penalize category budget overflow
                cat_indices = np.where((pop[i] > 0) & (self.cat_ids == cat_code))[0]
                if len(cat_indices) > 0:
                    cat_spend = np.sum(self.prices[cat_indices] * pop[i][cat_indices])
                    max_spend = self.budget * t.get('max_budget_pct', 0.25)
                    if cat_spend > max_spend:
                        cat_score[i] -= (cat_spend - max_spend) / self.budget * 20

        # Step 2e: Preference bonus (0-20 pts)
        pref_bonus = np.zeros(self.pop_size)
        if len(self.preferences) > 0:
            for i in range(self.pop_size):
                selected = pop[i] > 0
                if np.any(selected):
                    n_pref = np.sum(self.pref_scores[selected] > 0)
                    if n_pref > 0:
                        pref_bonus[i] = min(n_pref * 5, 20)
                    else:
                        pref_bonus[i] = -10

        # Step 2f: Final fitness
        return nutrition_score - budget_penalty + cost_bonus + cat_score + pref_bonus

    # Step 3: Tournament selection
    def _tournament_select(self, pop, fitnesses):
        indices = np.random.choice(self.pop_size, size=self.tournament_size, replace=False)
        return pop[indices[np.argmax(fitnesses[indices])]].copy()

    # Step 4: Uniform crossover
    def _crossover(self, p1, p2):
        if np.random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        mask = np.random.randint(0, 2, size=self.n_items).astype(bool)
        c1 = np.where(mask, p1, p2).astype(np.int8)
        c2 = np.where(mask, p2, p1).astype(np.int8)
        return c1, c2

    # Step 5: Mutation (5 strategies)
    def _mutate(self, chrom):
        m = chrom.copy()

        # Step 5a: Random reset (10% per gene)
        mask = np.random.random(self.n_items) < self.mutation_rate
        if np.any(mask):
            m[mask] = np.random.randint(0, self.max_qty + 1, size=mask.sum()).astype(np.int8)

        # Step 5b: Swap two items (15%)
        if np.random.random() < 0.15:
            i, j = np.random.choice(self.n_items, size=2, replace=False)
            m[i], m[j] = m[j], m[i]

        # Step 5c: Smart nutrient adjustment (25%)
        if np.random.random() < 0.25:
            ratios = np.array([
                np.dot(m, self.calories) / self.targets['calories_kcal'],
                np.dot(m, self.protein) / self.targets['protein_g'],
                np.dot(m, self.fat) / self.targets['fat_g'],
                np.dot(m, self.carbs) / self.targets['carbs_g']
            ])
            nutrient_arrs = [self.calories, self.protein, self.fat, self.carbs]

            if np.any(ratios < 0.7):
                w = np.argmin(ratios)
                top = np.argsort(nutrient_arrs[w] / self.prices)[-20:]
                pick = np.random.choice(top)
                if m[pick] < self.max_qty:
                    m[pick] += 1
            elif np.any(ratios > 1.5):
                s = np.argmax(ratios)
                sel = np.where(m > 0)[0]
                if len(sel) > 0:
                    worst = sel[np.argmax(nutrient_arrs[s][sel] * m[sel])]
                    m[worst] = max(0, m[worst] - 1)

        # Step 5d: Quality upgrade (25%)
        if np.random.random() < 0.25:
            total_cost = np.dot(m, self.prices)
            remaining = self.budget - total_cost
            if remaining > 0:
                sel = np.where(m > 0)[0]
                if len(sel) > 0:
                    cal_density = self.calories[sel] / self.prices[sel]
                    to_replace = sel[np.argmax(cal_density)]
                    old_cat = self.cat_ids[to_replace]
                    same_cat = np.where(self.cat_ids == old_cat)[0]
                    upgrades = same_cat[
                        (self.prices[same_cat] > self.prices[to_replace]) &
                        (self.prices[same_cat] <= self.prices[to_replace] + remaining) &
                        (self.calories[same_cat] / self.prices[same_cat] <
                         self.calories[to_replace] / self.prices[to_replace])
                    ]
                    if len(upgrades) > 0:
                        pick = np.random.choice(upgrades)
                        m[to_replace] = 0
                        m[pick] = 1

        # Step 5e: Budget fill — add items from under-filled categories (30%)
        if np.random.random() < 0.3:
            total_cost = np.dot(m, self.prices)
            total_cal = np.dot(m, self.calories)
            remaining = self.budget - total_cost
            if remaining > self.budget * 0.2:
                unsel = np.where(m == 0)[0]
                affordable = unsel[self.prices[unsel] <= remaining]
                if len(affordable) > 0:
                    sel_cats = Counter(self.cat_ids[m > 0])
                    under_cats = []
                    for cat_code, t in self.cat_targets.items():
                        if sel_cats.get(cat_code, 0) < t['min']:
                            under_cats.append(cat_code)
                    if under_cats:
                        pool = [i for i in affordable if self.cat_ids[i] in under_cats]
                    else:
                        pool = list(affordable)
                    if not pool:
                        pool = list(affordable)
                    pool = np.array(pool)
                    if total_cal > self.targets['calories_kcal'] * 0.9:
                        cpd = self.calories[pool] / self.prices[pool]
                        n_low = max(len(pool) // 4, 1)
                        pool = pool[np.argsort(cpd)[:n_low]]
                    m[np.random.choice(pool)] = 1

        return m

    # Step 6: Repair invalid chromosomes
    def _repair(self, chrom):
        r = chrom.copy()

        # Step 6a: Fix category count and budget overflow
        for cat_code, t in self.cat_targets.items():
            cat_indices = np.where((r > 0) & (self.cat_ids == cat_code))[0]
            if len(cat_indices) == 0:
                continue
            cat_total = sum(r[idx] for idx in cat_indices)
            if cat_total > t['max']:
                sorted_idx = sorted(cat_indices, key=lambda i: self.prices[i])
                for idx in sorted_idx:
                    while r[idx] > 0 and cat_total > t['max']:
                        r[idx] -= 1
                        cat_total -= 1
            max_spend = self.budget * t.get('max_budget_pct', 0.25)
            cat_spend = np.sum(self.prices[cat_indices] * r[cat_indices])
            if cat_spend > max_spend:
                sorted_idx = sorted(cat_indices, key=lambda i: self.prices[i], reverse=True)
                for idx in sorted_idx:
                    while r[idx] > 0 and cat_spend > max_spend:
                        r[idx] -= 1
                        cat_spend -= self.prices[idx]

        # Step 6b: Fix calorie overshoot (>150% target)
        total_cal = np.dot(r, self.calories)
        if total_cal > self.targets['calories_kcal'] * 1.5:
            sel = np.where(r > 0)[0]
            if len(sel) > 0:
                eff = self.protein[sel] / (self.calories[sel] + 1)
                for idx in sel[np.argsort(eff)]:
                    while r[idx] > 0 and total_cal > self.targets['calories_kcal'] * 1.2:
                        r[idx] -= 1
                        total_cal -= self.calories[idx]
                    if total_cal <= self.targets['calories_kcal'] * 1.2:
                        break

        # Step 6c: Fix over-budget
        total_cost = np.dot(r, self.prices)
        if total_cost > self.budget:
            sel = np.where(r > 0)[0]
            if len(sel) > 0:
                score = (self.protein[sel] * 0.35 + self.calories[sel] / 100 * 0.30 +
                         self.carbs[sel] * 0.20 + self.fat[sel] * 0.15)
                for idx in sel[np.argsort(score / self.prices[sel])]:
                    while r[idx] > 0 and total_cost > self.budget:
                        r[idx] -= 1
                        total_cost -= self.prices[idx]
                    if total_cost <= self.budget:
                        break

        return r

    # Step 7: Main evolution loop
    def optimize(self, verbose=True):
        # Step 7a: Initialize population
        pop = self._init_population()
        fitnesses = self._evaluate_population(pop)
        best_fitness = -np.inf
        best_chrom = None
        stagnation = 0

        for gen in range(self.generations):
            best_idx = np.argmax(fitnesses)
            gen_best = fitnesses[best_idx]
            gen_avg = np.mean(fitnesses)

            if gen_best > best_fitness + 0.01:
                best_fitness = gen_best
                best_chrom = pop[best_idx].copy()
                stagnation = 0
            else:
                stagnation += 1

            self.fitness_history.append(gen_best)
            self.avg_fitness_history.append(gen_avg)

            if verbose and gen % 50 == 0:
                cost = np.dot(pop[best_idx], self.prices)
                cal = np.dot(pop[best_idx], self.calories)
                print(f"Gen {gen:>4d} | Best: {gen_best:.2f} | "
                      f"Avg: {gen_avg:.2f} | Cost: ${cost:.2f} | "
                      f"Cal: {cal:.0f}/{self.targets['calories_kcal']:.0f}")

            # Step 7b: Inject fresh individuals if stagnant
            if stagnation >= 40:
                n_replace = self.pop_size // 2
                worst = np.argsort(fitnesses)[:n_replace]
                fresh = self._init_population()[:n_replace]
                for k, wi in enumerate(worst):
                    pop[wi] = fresh[k]
                fitnesses = self._evaluate_population(pop)
                stagnation = 0
                continue

            # Step 7c: Elitism — keep top individuals
            elite = np.argsort(fitnesses)[-self.elitism_count:]
            new_pop = [pop[i].copy() for i in elite]

            # Step 7d: Breed new generation
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._repair(self._mutate(c1)))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._repair(self._mutate(c2)))

            pop = np.array(new_pop, dtype=np.int8)
            fitnesses = self._evaluate_population(pop)

        # Final check
        final_best = np.argmax(fitnesses)
        if fitnesses[final_best] > best_fitness:
            best_fitness = fitnesses[final_best]
            best_chrom = pop[final_best].copy()

        if verbose:
            print(f"\nDone. Best fitness: {best_fitness:.2f}")

        return best_chrom, best_fitness

    # Step 8: Format chromosome into readable grocery list
    def format_result(self, chrom):
        sel = chrom > 0
        items = self.candidates[sel].copy()
        items['quantity'] = chrom[sel]
        items['subtotal'] = items['price'] * items['quantity']
        items['total_protein'] = items['protein_g'] * items['quantity']
        items['total_fat'] = items['fat_g'] * items['quantity']
        items['total_carbs'] = items['carbs_g'] * items['quantity']
        items['total_cal'] = items['calories_kcal'] * items['quantity']

        summary = {
            'total_cost': items['subtotal'].sum(),
            'total_protein_g': items['total_protein'].sum(),
            'total_fat_g': items['total_fat'].sum(),
            'total_carbs_g': items['total_carbs'].sum(),
            'total_calories_kcal': items['total_cal'].sum(),
            'num_items': len(items),
            'num_categories': items['category'].nunique(),
            'budget': self.budget,
            'targets': self.targets,
            'category_breakdown': items.groupby('category')['quantity'].sum().to_dict()
        }

        display = items[['name', 'store', 'category', 'price', 'quantity', 'subtotal']].copy()
        display = display.sort_values(['category', 'subtotal'], ascending=[True, False]).reset_index(drop=True)
        return display, summary

# Print the summary
def print_summary(display, summary):
    t = summary['targets']
    pct = lambda a, b: a / b * 100

    print("\n" + "=" * 70)
    print("OPTIMIZED WEEKLY GROCERY LIST")
    print("=" * 70)

    current_cat = None
    for _, row in display.iterrows():
        if row['category'] != current_cat:
            current_cat = row['category']
            print(f"\n  [{current_cat.upper()}]")
        print(f"    x{int(row['quantity'])}  ${row['price']:>6.2f}  [{row['store']}]  {row['name'][:48]}")

    print("\n" + "-" * 70)
    print(f"  Items: {summary['num_items']} ({summary['num_categories']} categories)")
    print(f"  Cost:  ${summary['total_cost']:.2f} / ${summary['budget']:.2f}")
    print(f"  Cal:   {summary['total_calories_kcal']:>8.0f} / {t['calories_kcal']:.0f} ({pct(summary['total_calories_kcal'], t['calories_kcal']):.0f}%)")
    print(f"  Prot:  {summary['total_protein_g']:>8.1f} / {t['protein_g']:.1f} ({pct(summary['total_protein_g'], t['protein_g']):.0f}%)")
    print(f"  Fat:   {summary['total_fat_g']:>8.1f} / {t['fat_g']:.1f} ({pct(summary['total_fat_g'], t['fat_g']):.0f}%)")
    print(f"  Carbs: {summary['total_carbs_g']:>8.1f} / {t['carbs_g']:.1f} ({pct(summary['total_carbs_g'], t['carbs_g']):.0f}%)")

    print(f"\n  Category breakdown:")
    for cat, count in sorted(summary['category_breakdown'].items()):
        t_info = CATEGORY_TARGETS.get(cat, {})
        target_str = f"(target: {t_info.get('min', '?')}-{t_info.get('max', '?')})" if t_info else ""
        print(f"    {cat:<12} {count} items {target_str}")

    print("=" * 70)