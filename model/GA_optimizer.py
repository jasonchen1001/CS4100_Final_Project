import numpy as np
import pandas as pd
from collections import Counter


# Select diverse candidate items per category to reduce search space
def preselect_candidates(data, targets, n_per_category=20):
    data = data.copy()
    data['cal_eff'] = data['calories_kcal'] / data['price']
    data['prot_eff'] = data['protein_g'] / data['price']
    data['combined'] = (
        data['protein_g'] * 0.35 + data['fat_g'] * 0.15 +
        data['carbs_g'] * 0.20 + data['calories_kcal'] / 100 * 0.30
    ) / data['price']
    # High price per calorie = premium items that spend budget without overshooting
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
                 n_per_category=20):

        self.candidates = preselect_candidates(data, targets, n_per_category)
        self.n_items = len(self.candidates)
        self.targets = targets
        self.budget = budget
        self.max_qty = max_qty

        # GA parameters
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
        self.max_per_category = 4

        # Group related categories to prevent redundancy
        self.super_categories = {
            'seafood': {'seafood', 'seafood-wf'},
            'meat': {'meat-packaged', 'meat-poultry-wf', 'sausage-bacon', 'jerky'},
            'drinks': {'drink-shakes-other', 'drink-coffee', 'drink-juice',
                       'drink-juice-wf', 'drink-soft-energy-mixes', 'drink-tea'},
            'dairy_drinks': {'dairy-yogurt-drink', 'milk-milk-substitute'},
            'grains': {'rice-grains-packaged', 'rice-grains-wf'},
            'bread_all': {'bread', 'rolls-buns-wraps', 'muffins-bagels'},
            'sweets': {'pastry-chocolate-candy', 'cookies-biscuit',
                       'ice-cream-dessert', 'cakes', 'pudding-jello'},
            'snacks_all': {'snacks-bars', 'snacks-chips', 'snacks-mixes-crackers',
                           'snacks-popcorn', 'snacks-nuts-seeds', 'snacks-dips-salsa'},
        }
        self.max_per_super = 4

        # Map each item index to its super-category
        cat_names = list(self.cat_labels.categories)
        self.item_super_cat = {}
        for i in range(self.n_items):
            cat_name = cat_names[self.cat_ids[i]]
            for sname, members in self.super_categories.items():
                if cat_name in members:
                    self.item_super_cat[i] = sname
                    break

        # Required categories
        required_names = {
            'meat-packaged', 'meat-poultry-wf',
            'produce-packaged', 'produce-beans-wf',
            'dairy-yogurt-drink', 'milk-milk-substitute',
            'rice-grains-packaged', 'pasta-noodles', 'bread',
            'seafood', 'seafood-wf', 'eggs-wf', 'cheese',
        }
        name_to_code = {name: code for code, name in enumerate(self.cat_labels.categories)}
        self.required_cat_codes = {name_to_code[n] for n in required_names if n in name_to_code}

        # Nutrition target array and weights
        self.target_arr = np.array([
            targets['calories_kcal'], targets['protein_g'],
            targets['fat_g'], targets['carbs_g']
        ])
        self.nutrition_weights = np.array([0.30, 0.35, 0.15, 0.20])

        # Tracking
        self.fitness_history = []
        self.avg_fitness_history = []

        print(f"Candidate pool: {self.n_items} items from {self.candidates['category'].nunique()} categories")

    # --- Initialization ---
    def _init_population(self):
        pop = np.zeros((self.pop_size, self.n_items), dtype=np.int8)

        for i in range(self.pop_size):
            remaining_budget = self.budget
            remaining_cal = self.targets['calories_kcal']
            cat_count = {}
            super_count = {}

            # Phase 1: one item from each required category
            for cat_code in self.required_cat_codes:
                cat_items = np.where(self.cat_ids == cat_code)[0]
                if len(cat_items) == 0:
                    continue
                pick = np.random.choice(cat_items)
                qty = np.random.randint(1, self.max_qty + 1)
                cost = self.prices[pick] * qty
                if cost <= remaining_budget:
                    pop[i, pick] = qty
                    remaining_budget -= cost
                    remaining_cal -= self.calories[pick] * qty
                    cat_count[cat_code] = cat_count.get(cat_code, 0) + qty
                    sc = self.item_super_cat.get(pick)
                    if sc:
                        super_count[sc] = super_count.get(sc, 0) + qty

            # Phase 2: fill until calorie target met
            for idx in np.random.permutation(self.n_items):
                if remaining_cal <= 0 or remaining_budget <= 0:
                    break
                if pop[i, idx] > 0:
                    continue
                cat = self.cat_ids[idx]
                if cat_count.get(cat, 0) >= self.max_per_category:
                    continue
                sc = self.item_super_cat.get(idx)
                if sc and super_count.get(sc, 0) >= self.max_per_super:
                    continue
                qty = np.random.randint(1, self.max_qty + 1)
                cost = self.prices[idx] * qty
                if cost <= remaining_budget:
                    pop[i, idx] = qty
                    remaining_budget -= cost
                    remaining_cal -= self.calories[idx] * qty
                    cat_count[cat] = cat_count.get(cat, 0) + qty
                    if sc:
                        super_count[sc] = super_count.get(sc, 0) + qty

            # Phase 3: spend remaining budget on variety
            if remaining_budget > 5:
                for idx in np.random.permutation(self.n_items):
                    if remaining_budget <= 5:
                        break
                    if pop[i, idx] > 0:
                        continue
                    cat = self.cat_ids[idx]
                    if cat_count.get(cat, 0) >= self.max_per_category:
                        continue
                    sc = self.item_super_cat.get(idx)
                    if sc and super_count.get(sc, 0) >= self.max_per_super:
                        continue
                    cost = self.prices[idx]
                    if cost <= remaining_budget:
                        pop[i, idx] = 1
                        remaining_budget -= cost
                        cat_count[cat] = cat_count.get(cat, 0) + 1
                        if sc:
                            super_count[sc] = super_count.get(sc, 0) + 1

        return pop

    # --- Fitness ---
    def _evaluate_population(self, pop):
        pop_float = pop.astype(np.float64)
        total_cost = pop_float @ self.prices
        total_cal = pop_float @ self.calories
        total_prot = pop_float @ self.protein
        total_fat = pop_float @ self.fat
        total_carbs = pop_float @ self.carbs

        # Nutrition: penalize only outside 80-120% range
        actuals = np.column_stack([total_cal, total_prot, total_fat, total_carbs])
        ratios = actuals / self.target_arr
        under = np.maximum(0, 0.8 - ratios)
        over = np.maximum(0, ratios - 1.2)
        weighted_dev = (under + over) @ self.nutrition_weights
        nutrition_score = np.maximum(0, 40 * (1 - weighted_dev * 2))

        # Budget: hard penalty if over
        budget_penalty = np.maximum(0, total_cost - self.budget) / self.budget * 200

        # Budget utilization: reward spending wisely
        util = np.where(total_cost <= self.budget, total_cost / self.budget, 0.0)
        cost_bonus = util * 25

        # Diversity + required categories
        diversity_score = np.zeros(self.pop_size)
        n_unique = np.zeros(self.pop_size)
        req_bonus = np.zeros(self.pop_size)

        for i in range(self.pop_size):
            selected = pop[i] > 0
            n_unique[i] = np.sum(selected)
            if not np.any(selected):
                continue

            sel_cats = self.cat_ids[selected]
            unique_cats = set(sel_cats)
            diversity_score[i] = len(unique_cats)

            # Reward for required categories present
            req_bonus[i] = len(unique_cats & self.required_cat_codes) * 1.5

            # Penalize category overflow
            for count in Counter(sel_cats).values():
                if count > self.max_per_category:
                    req_bonus[i] -= (count - self.max_per_category) * 2

            # Penalize super-category overflow
            sc_counts = Counter()
            for idx in np.where(pop[i] > 0)[0]:
                sc = self.item_super_cat.get(idx)
                if sc:
                    sc_counts[sc] += pop[i, idx]
            for count in sc_counts.values():
                if count > self.max_per_super:
                    req_bonus[i] -= (count - self.max_per_super) * 3

        cat_b = np.minimum(diversity_score / 12, 1.0)
        item_b = np.minimum(n_unique / 25, 1.0)
        diversity_total = (cat_b * 0.5 + item_b * 0.5) * 20

        return nutrition_score - budget_penalty + cost_bonus + diversity_total + req_bonus

    # --- Selection ---
    def _tournament_select(self, pop, fitnesses):
        indices = np.random.choice(self.pop_size, size=self.tournament_size, replace=False)
        return pop[indices[np.argmax(fitnesses[indices])]].copy()

    # --- Crossover ---
    def _crossover(self, p1, p2):
        if np.random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        mask = np.random.randint(0, 2, size=self.n_items).astype(bool)
        c1 = np.where(mask, p1, p2).astype(np.int8)
        c2 = np.where(mask, p2, p1).astype(np.int8)
        return c1, c2

    # --- Mutation ---
    def _mutate(self, chrom):
        m = chrom.copy()

        # Random reset
        mask = np.random.random(self.n_items) < self.mutation_rate
        if np.any(mask):
            m[mask] = np.random.randint(0, self.max_qty + 1, size=mask.sum()).astype(np.int8)

        # Swap
        if np.random.random() < 0.15:
            i, j = np.random.choice(self.n_items, size=2, replace=False)
            m[i], m[j] = m[j], m[i]

        # Smart: adjust toward weakest/strongest nutrient
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

        # Quality upgrade: replace high-cal-density item with pricier low-cal one
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

        # Budget fill: add low-calorie items when underspending
        if np.random.random() < 0.3:
            total_cost = np.dot(m, self.prices)
            total_cal = np.dot(m, self.calories)
            remaining = self.budget - total_cost
            if remaining > self.budget * 0.2:
                unsel = np.where(m == 0)[0]
                affordable = unsel[self.prices[unsel] <= remaining]
                if len(affordable) > 0:
                    sel_cats = set(self.cat_ids[m > 0])
                    new_cat = [i for i in affordable if self.cat_ids[i] not in sel_cats]
                    pool = np.array(new_cat if new_cat else list(affordable))
                    if total_cal > self.targets['calories_kcal'] * 0.9:
                        cpd = self.calories[pool] / self.prices[pool]
                        n_low = max(len(pool) // 4, 1)
                        pool = pool[np.argsort(cpd)[:n_low]]
                    m[np.random.choice(pool)] = 1

        return m

    # --- Repair ---
    def _repair(self, chrom):
        r = chrom.copy()

        # Fix calorie overshoot (>150% target)
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

        # Fix over-budget
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

    # --- Main optimization loop ---
    def optimize(self, verbose=True):
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

            # Inject fresh individuals if stuck
            if stagnation >= 40:
                n_replace = self.pop_size // 2
                worst = np.argsort(fitnesses)[:n_replace]
                fresh = self._init_population()[:n_replace]
                for k, wi in enumerate(worst):
                    pop[wi] = fresh[k]
                fitnesses = self._evaluate_population(pop)
                stagnation = 0
                continue

            # Elitism + breeding
            elite = np.argsort(fitnesses)[-self.elitism_count:]
            new_pop = [pop[i].copy() for i in elite]

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

    # --- Format results ---
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
            'targets': self.targets
        }

        display = items[['name', 'store', 'category', 'price', 'quantity', 'subtotal']].copy()
        display = display.sort_values('subtotal', ascending=False).reset_index(drop=True)
        return display, summary


def print_summary(display, summary):
    t = summary['targets']
    pct = lambda a, b: a / b * 100

    print("\n" + "=" * 70)
    print("OPTIMIZED WEEKLY GROCERY LIST")
    print("=" * 70)
    for _, row in display.iterrows():
        print(f"  x{int(row['quantity'])}  ${row['price']:>6.2f}  [{row['store']}]  {row['name'][:50]}")
    print("-" * 70)
    print(f"  Items: {summary['num_items']} ({summary['num_categories']} categories)")
    print(f"  Cost:  ${summary['total_cost']:.2f} / ${summary['budget']:.2f}")
    print(f"  Cal:   {summary['total_calories_kcal']:>8.0f} / {t['calories_kcal']:.0f} ({pct(summary['total_calories_kcal'], t['calories_kcal']):.0f}%)")
    print(f"  Prot:  {summary['total_protein_g']:>8.1f} / {t['protein_g']:.1f} ({pct(summary['total_protein_g'], t['protein_g']):.0f}%)")
    print(f"  Fat:   {summary['total_fat_g']:>8.1f} / {t['fat_g']:.1f} ({pct(summary['total_fat_g'], t['fat_g']):.0f}%)")
    print(f"  Carbs: {summary['total_carbs_g']:>8.1f} / {t['carbs_g']:.1f} ({pct(summary['total_carbs_g'], t['carbs_g']):.0f}%)")
    print("=" * 70)