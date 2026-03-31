const DATASET_URL = "data/cleaned_grocery.csv";

const ACTIVITY_MULTIPLIER = {
  sedentary: 1.2,
  light: 1.375,
  moderate: 1.55,
  high: 1.725,
};

const GOAL_CALORIE_ADJUSTMENT = {
  fat_loss: -450,
  maintenance: 0,
  muscle_gain: 300,
};

const GOAL_PROTEIN_FACTOR = {
  fat_loss: 2.0,
  maintenance: 1.7,
  muscle_gain: 2.1,
};

const GA_CONFIG = {
  weekly: {
    populationSize: 100,
    generations: 130,
    maxGenes: 90,
    maxPerCategory: 6,
    mutationRate: 0.05,
    crossoverRate: 0.9,
    elitismCount: 6,
  },
  monthly: {
    populationSize: 130,
    generations: 160,
    maxGenes: 130,
    maxPerCategory: 8,
    mutationRate: 0.045,
    crossoverRate: 0.9,
    elitismCount: 8,
  },
};

const state = {
  items: [],
  datasetLoaded: false,
};

const elements = {
  age: document.getElementById("ageInput"),
  sex: document.getElementById("sexInput"),
  height: document.getElementById("heightInput"),
  weight: document.getElementById("weightInput"),
  activity: document.getElementById("activityInput"),
  goal: document.getElementById("goalInput"),
  budget: document.getElementById("budgetInput"),
  period: document.getElementById("periodInput"),
  vegetarian: document.getElementById("vegetarianInput"),
  nuts: document.getElementById("nutsInput"),
  dairy: document.getElementById("dairyInput"),
  gluten: document.getElementById("glutenInput"),
  egg: document.getElementById("eggInput"),
  categoryPrefs: document.getElementById("categoryPrefsInput"),
  keywordPrefs: document.getElementById("keywordPrefsInput"),
  history: document.getElementById("historyInput"),
  prefWeight: document.getElementById("prefWeightInput"),
  datasetMetaText: document.getElementById("datasetMetaText"),
  runButton: document.getElementById("runOptimizerBtn"),
  clearButton: document.getElementById("clearResultsBtn"),
  status: document.getElementById("statusText"),
  summaryBlock: document.getElementById("summaryBlock"),
  resultBody: document.getElementById("resultBody"),
  emptyState: document.getElementById("emptyState"),
};

function setStatus(message, kind = "neutral") {
  elements.status.textContent = message;
  elements.status.dataset.kind = kind;
}

function toNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function tokenizeCsvLine(line) {
  const fields = [];
  let token = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      token += '"';
      i += 1;
      continue;
    }
    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (char === "," && !inQuotes) {
      fields.push(token.trim());
      token = "";
      continue;
    }
    token += char;
  }

  fields.push(token.trim());
  return fields;
}

function parseGroceryCsv(csvText) {
  const lines = csvText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    throw new Error("Bundled CSV must include a header and at least one row.");
  }

  const headers = tokenizeCsvLine(lines[0]);
  const indexByHeader = Object.fromEntries(headers.map((header, index) => [header, index]));
  const requiredHeaders = [
    "id",
    "name",
    "store",
    "category",
    "price",
    "package_weight_g",
    "protein_g",
    "fat_g",
    "carbs_g",
    "calories_kcal",
    "contains_nuts",
    "is_vegetarian",
    "has_dairy",
    "has_gluten",
    "has_egg",
  ];

  const missingHeaders = requiredHeaders.filter((header) => indexByHeader[header] === undefined);
  if (missingHeaders.length > 0) {
    throw new Error(`Dataset missing required columns: ${missingHeaders.join(", ")}`);
  }

  const parsedItems = [];
  for (let lineIndex = 1; lineIndex < lines.length; lineIndex += 1) {
    const row = tokenizeCsvLine(lines[lineIndex]);
    const item = {
      id: row[indexByHeader.id],
      name: row[indexByHeader.name],
      store: row[indexByHeader.store],
      category: row[indexByHeader.category],
      price: toNumber(row[indexByHeader.price]),
      package_weight_g: toNumber(row[indexByHeader.package_weight_g]),
      protein_g: toNumber(row[indexByHeader.protein_g]),
      fat_g: toNumber(row[indexByHeader.fat_g]),
      carbs_g: toNumber(row[indexByHeader.carbs_g]),
      calories_kcal: toNumber(row[indexByHeader.calories_kcal]),
      contains_nuts: toNumber(row[indexByHeader.contains_nuts]),
      is_vegetarian: toNumber(row[indexByHeader.is_vegetarian]),
      has_dairy: toNumber(row[indexByHeader.has_dairy]),
      has_gluten: toNumber(row[indexByHeader.has_gluten]),
      has_egg: toNumber(row[indexByHeader.has_egg]),
    };

    if (!item.id || !item.name || item.price <= 0 || item.calories_kcal <= 0) {
      continue;
    }
    parsedItems.push(item);
  }

  if (parsedItems.length === 0) {
    throw new Error("Bundled CSV has no valid grocery rows.");
  }
  return parsedItems;
}

async function ensureDatasetLoaded() {
  if (state.datasetLoaded) return;
  setStatus("Loading bundled grocery dataset...", "neutral");

  let response;
  try {
    response = await fetch(DATASET_URL, { cache: "no-store" });
  } catch {
    throw new Error(
      "Failed to fetch bundled dataset. Serve this project over HTTP (not file://) so fetch can load data/cleaned_grocery.csv."
    );
  }

  if (!response.ok) {
    throw new Error(`Could not load bundled dataset (${response.status} ${response.statusText}).`);
  }

  const csvText = await response.text();
  state.items = parseGroceryCsv(csvText);
  state.datasetLoaded = true;
  elements.datasetMetaText.textContent = `Dataset: ${state.items.length} items loaded from data/cleaned_grocery.csv`;
  setStatus("Bundled dataset ready.", "success");
}

function splitTokens(value) {
  return value
    .toLowerCase()
    .split(",")
    .map((token) => token.trim())
    .filter(Boolean);
}

function getUserInputs() {
  const age = toNumber(elements.age.value, 22);
  const heightCm = toNumber(elements.height.value, 175);
  const weightKg = toNumber(elements.weight.value, 70);
  const sex = elements.sex.value;
  const activity = elements.activity.value;
  const goal = elements.goal.value;
  const budget = toNumber(elements.budget.value, 120);
  const period = elements.period.value;
  const preferenceWeight = toNumber(elements.prefWeight.value, 45) / 100;

  return {
    age,
    heightCm,
    weightKg,
    sex,
    activity,
    goal,
    budget,
    period,
    preferenceWeight,
    categoryPrefs: splitTokens(elements.categoryPrefs.value),
    keywordPrefs: splitTokens(elements.keywordPrefs.value),
    historyTokens: splitTokens(elements.history.value),
    vegetarianOnly: elements.vegetarian.checked,
    excludeNuts: elements.nuts.checked,
    excludeDairy: elements.dairy.checked,
    excludeGluten: elements.gluten.checked,
    excludeEgg: elements.egg.checked,
  };
}

function calculateNutritionTargets(inputs) {
  const { sex, weightKg, heightCm, age, activity, goal, period } = inputs;
  let bmr;
  if (sex === "male") {
    bmr = 10 * weightKg + 6.25 * heightCm - 5 * age + 5;
  } else if (sex === "female") {
    bmr = 10 * weightKg + 6.25 * heightCm - 5 * age - 161;
  } else {
    bmr = 10 * weightKg + 6.25 * heightCm - 5 * age - 78;
  }

  const tdee = bmr * (ACTIVITY_MULTIPLIER[activity] ?? ACTIVITY_MULTIPLIER.light);
  const dailyCalories = Math.max(1200, tdee + (GOAL_CALORIE_ADJUSTMENT[goal] ?? 0));
  const proteinPerDay = Math.max(50, weightKg * (GOAL_PROTEIN_FACTOR[goal] ?? 1.7));
  const fatPerDay = Math.max(35, weightKg * 0.8);
  const carbPerDay = Math.max(80, (dailyCalories - proteinPerDay * 4 - fatPerDay * 9) / 4);
  const days = period === "monthly" ? 30.4 : 7;

  return {
    calories_kcal: dailyCalories * days,
    protein_g: proteinPerDay * days,
    fat_g: fatPerDay * days,
    carbs_g: carbPerDay * days,
    days,
  };
}

function applyDietaryFilters(items, inputs) {
  return items.filter((item) => {
    if (inputs.vegetarianOnly && item.is_vegetarian !== 1) return false;
    if (inputs.excludeNuts && item.contains_nuts === 1) return false;
    if (inputs.excludeDairy && item.has_dairy === 1) return false;
    if (inputs.excludeGluten && item.has_gluten === 1) return false;
    if (inputs.excludeEgg && item.has_egg === 1) return false;
    return true;
  });
}

function preferenceMatchScore(item, inputs) {
  const name = item.name.toLowerCase();
  const category = item.category.toLowerCase();
  let score = 0;

  for (const token of inputs.categoryPrefs) {
    if (category.includes(token)) score += 1.1;
  }
  for (const token of inputs.keywordPrefs) {
    if (name.includes(token)) score += 1.3;
  }
  for (const token of inputs.historyTokens) {
    if (name.includes(token) || category.includes(token)) score += 2.1;
  }
  return score;
}

function quantityCap(item, period) {
  const periodFactor = period === "monthly" ? 2.8 : 1;
  let cap;
  if (item.price < 2) cap = 8;
  else if (item.price < 5) cap = 6;
  else if (item.price < 9) cap = 4;
  else cap = 3;

  if (item.package_weight_g > 1400) cap -= 1;
  return Math.max(1, Math.min(30, Math.round(cap * periodFactor)));
}

function buildCandidatePool(items, targets, inputs) {
  const gaProfile = GA_CONFIG[inputs.period] ?? GA_CONFIG.weekly;
  const weighted = items
    .map((item) => {
      const prefScore = preferenceMatchScore(item, inputs);
      const safePrice = Math.max(item.price, 0.01);
      const proteinDensity = item.protein_g / safePrice / targets.protein_g;
      const calorieDensity = item.calories_kcal / safePrice / targets.calories_kcal;
      const carbDensity = item.carbs_g / safePrice / targets.carbs_g;
      const fatDensity = item.fat_g / safePrice / targets.fat_g;
      const nutritionScore = proteinDensity * 0.4 + calorieDensity * 0.25 + carbDensity * 0.2 + fatDensity * 0.15;

      return {
        item,
        prefScore,
        rankScore: nutritionScore + prefScore * (0.2 + inputs.preferenceWeight * 0.6),
      };
    })
    .sort((a, b) => b.rankScore - a.rankScore);

  const perCategoryLimit = gaProfile.maxPerCategory;
  const selected = [];
  const categoryCounts = new Map();
  for (const entry of weighted) {
    const currentCount = categoryCounts.get(entry.item.category) ?? 0;
    if (currentCount >= perCategoryLimit) continue;
    selected.push(entry);
    categoryCounts.set(entry.item.category, currentCount + 1);
    if (selected.length >= gaProfile.maxGenes) break;
  }

  if (selected.length < gaProfile.maxGenes) {
    const selectedIds = new Set(selected.map((entry) => entry.item.id));
    for (const entry of weighted) {
      if (selected.length >= gaProfile.maxGenes) break;
      if (selectedIds.has(entry.item.id)) continue;
      selected.push(entry);
      selectedIds.add(entry.item.id);
    }
  }

  return selected.map((entry) => ({
    ...entry,
    maxQty: quantityCap(entry.item, inputs.period),
  }));
}

function createEmptyTotals() {
  return {
    cost: 0,
    calories_kcal: 0,
    protein_g: 0,
    fat_g: 0,
    carbs_g: 0,
    preferenceScore: 0,
    uniqueItems: 0,
    categoryCount: 0,
  };
}

function evaluateChromosome(chromosome, genes, targets, inputs) {
  const totals = createEmptyTotals();
  const categories = new Set();
  let quantitySum = 0;

  for (let i = 0; i < genes.length; i += 1) {
    const qty = chromosome[i];
    if (qty <= 0) continue;
    quantitySum += qty;
    const gene = genes[i];
    const { item } = gene;
    totals.uniqueItems += 1;
    categories.add(item.category);
    totals.cost += item.price * qty;
    totals.calories_kcal += item.calories_kcal * qty;
    totals.protein_g += item.protein_g * qty;
    totals.fat_g += item.fat_g * qty;
    totals.carbs_g += item.carbs_g * qty;
    totals.preferenceScore += gene.prefScore * qty;
  }

  totals.categoryCount = categories.size;

  const deficits =
    Math.max(0, targets.calories_kcal - totals.calories_kcal) / targets.calories_kcal +
    Math.max(0, targets.protein_g - totals.protein_g) / targets.protein_g +
    Math.max(0, targets.fat_g - totals.fat_g) / targets.fat_g +
    Math.max(0, targets.carbs_g - totals.carbs_g) / targets.carbs_g;

  const overshoot =
    Math.max(0, totals.calories_kcal - targets.calories_kcal * 1.28) / targets.calories_kcal +
    Math.max(0, totals.protein_g - targets.protein_g * 1.45) / targets.protein_g +
    Math.max(0, totals.fat_g - targets.fat_g * 1.45) / targets.fat_g +
    Math.max(0, totals.carbs_g - targets.carbs_g * 1.45) / targets.carbs_g;

  const budgetOverflow = Math.max(0, totals.cost - inputs.budget) / inputs.budget;
  const budgetPenalty = budgetOverflow > 0 ? budgetOverflow * 16 + budgetOverflow * budgetOverflow * 30 : 0;

  const veryLowCoveragePenalty = totals.calories_kcal < targets.calories_kcal * 0.55 ? 9 : 0;
  const sparsityPenalty = Math.max(0, totals.uniqueItems - 20) * 0.04;
  const preferenceReward = totals.preferenceScore * (0.01 + 0.035 * inputs.preferenceWeight);
  const diversityReward = totals.categoryCount * 0.04;

  const fitness =
    deficits * 10 +
    overshoot * 2.4 +
    budgetPenalty +
    veryLowCoveragePenalty +
    sparsityPenalty -
    preferenceReward -
    diversityReward;

  return {
    fitness,
    totals,
    quantitySum,
  };
}

function randomInt(minInclusive, maxInclusive) {
  return Math.floor(Math.random() * (maxInclusive - minInclusive + 1)) + minInclusive;
}

function createRandomChromosome(genes) {
  const chromosome = new Array(genes.length).fill(0);
  for (let i = 0; i < genes.length; i += 1) {
    const maxQty = genes[i].maxQty;
    const pick = Math.random();
    if (pick < 0.78) {
      chromosome[i] = 0;
    } else if (pick < 0.93) {
      chromosome[i] = randomInt(1, Math.max(1, Math.floor(maxQty / 2)));
    } else {
      chromosome[i] = randomInt(1, maxQty);
    }
  }
  return chromosome;
}

function createGreedyChromosome(genes, targets, inputs) {
  const chromosome = new Array(genes.length).fill(0);
  let runningCost = 0;
  for (let i = 0; i < genes.length; i += 1) {
    const gene = genes[i];
    if (runningCost + gene.item.price > inputs.budget * 0.98) continue;
    const qty = Math.max(1, Math.min(gene.maxQty, Math.round((targets.calories_kcal * 0.03) / gene.item.calories_kcal)));
    chromosome[i] = qty;
    runningCost += qty * gene.item.price;
    if (runningCost > inputs.budget * 0.9) break;
  }
  return chromosome;
}

function repairChromosomeBudget(chromosome, genes, budget) {
  let totalCost = 0;
  for (let i = 0; i < genes.length; i += 1) {
    totalCost += genes[i].item.price * chromosome[i];
  }

  const budgetCeiling = budget * 1.22;
  let guard = 0;
  while (totalCost > budgetCeiling && guard < 800) {
    guard += 1;
    const activeIndexes = [];
    for (let i = 0; i < chromosome.length; i += 1) {
      if (chromosome[i] > 0) activeIndexes.push(i);
    }
    if (activeIndexes.length === 0) break;
    activeIndexes.sort((a, b) => genes[b].item.price - genes[a].item.price);
    const chosen = activeIndexes[Math.floor(Math.random() * Math.min(6, activeIndexes.length))];
    chromosome[chosen] -= 1;
    totalCost -= genes[chosen].item.price;
  }
}

function tournamentSelect(population, evaluations, tournamentSize) {
  let bestIndex = randomInt(0, population.length - 1);
  for (let i = 1; i < tournamentSize; i += 1) {
    const challenger = randomInt(0, population.length - 1);
    if (evaluations[challenger].fitness < evaluations[bestIndex].fitness) {
      bestIndex = challenger;
    }
  }
  return population[bestIndex];
}

function crossover(parentA, parentB, crossoverRate) {
  const length = parentA.length;
  if (Math.random() > crossoverRate || length < 2) {
    return [parentA.slice(), parentB.slice()];
  }

  const cutPoint = randomInt(1, length - 1);
  const childA = new Array(length);
  const childB = new Array(length);
  for (let i = 0; i < length; i += 1) {
    if (i < cutPoint) {
      childA[i] = parentA[i];
      childB[i] = parentB[i];
    } else {
      childA[i] = parentB[i];
      childB[i] = parentA[i];
    }
  }
  return [childA, childB];
}

function mutate(chromosome, genes, mutationRate) {
  for (let i = 0; i < chromosome.length; i += 1) {
    if (Math.random() >= mutationRate) continue;
    const direction = Math.random();
    if (direction < 0.34) {
      chromosome[i] = Math.max(0, chromosome[i] - 1);
    } else if (direction < 0.68) {
      chromosome[i] = Math.min(genes[i].maxQty, chromosome[i] + 1);
    } else {
      chromosome[i] = randomInt(0, genes[i].maxQty);
    }
  }
}

function chromosomeToSelectedRows(chromosome, genes) {
  const rows = [];
  for (let i = 0; i < chromosome.length; i += 1) {
    const qty = chromosome[i];
    if (qty <= 0) continue;
    rows.push({
      item: genes[i].item,
      qty,
      prefScore: genes[i].prefScore,
    });
  }
  rows.sort((a, b) => b.item.price * b.qty - a.item.price * a.qty);
  return rows;
}

async function runGeneticOptimization(allItems, inputs, targets) {
  const filtered = applyDietaryFilters(allItems, inputs);
  if (filtered.length === 0) {
    throw new Error("No grocery items remain after applying dietary restrictions.");
  }

  const genes = buildCandidatePool(filtered, targets, inputs);
  if (genes.length === 0) {
    throw new Error("No candidate items available for optimization.");
  }

  const profile = GA_CONFIG[inputs.period] ?? GA_CONFIG.weekly;
  const population = [];
  population.push(createGreedyChromosome(genes, targets, inputs));
  for (let i = 1; i < profile.populationSize; i += 1) {
    population.push(createRandomChromosome(genes));
  }
  for (const chromosome of population) {
    repairChromosomeBudget(chromosome, genes, inputs.budget);
  }

  let evaluations = population.map((chromosome) => evaluateChromosome(chromosome, genes, targets, inputs));
  let bestIndex = 0;
  for (let i = 1; i < evaluations.length; i += 1) {
    if (evaluations[i].fitness < evaluations[bestIndex].fitness) bestIndex = i;
  }

  let bestChromosome = population[bestIndex].slice();
  let bestEvaluation = evaluations[bestIndex];
  const tournamentSize = 4;

  for (let generation = 0; generation < profile.generations; generation += 1) {
    if (generation % 20 === 0) {
      setStatus(`Running genetic optimizer... generation ${generation + 1}/${profile.generations}`, "neutral");
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    const ranked = evaluations
      .map((evaluation, index) => ({ index, fitness: evaluation.fitness }))
      .sort((a, b) => a.fitness - b.fitness);

    const nextPopulation = [];
    for (let i = 0; i < profile.elitismCount; i += 1) {
      nextPopulation.push(population[ranked[i].index].slice());
    }

    while (nextPopulation.length < profile.populationSize) {
      const parentA = tournamentSelect(population, evaluations, tournamentSize);
      const parentB = tournamentSelect(population, evaluations, tournamentSize);
      const [childA, childB] = crossover(parentA, parentB, profile.crossoverRate);
      mutate(childA, genes, profile.mutationRate);
      mutate(childB, genes, profile.mutationRate);
      repairChromosomeBudget(childA, genes, inputs.budget);
      repairChromosomeBudget(childB, genes, inputs.budget);
      nextPopulation.push(childA);
      if (nextPopulation.length < profile.populationSize) {
        nextPopulation.push(childB);
      }
    }

    for (let i = 0; i < population.length; i += 1) {
      population[i] = nextPopulation[i];
    }
    evaluations = population.map((chromosome) => evaluateChromosome(chromosome, genes, targets, inputs));

    let generationBest = 0;
    for (let i = 1; i < evaluations.length; i += 1) {
      if (evaluations[i].fitness < evaluations[generationBest].fitness) generationBest = i;
    }
    if (evaluations[generationBest].fitness < bestEvaluation.fitness) {
      bestChromosome = population[generationBest].slice();
      bestEvaluation = evaluations[generationBest];
    }
  }

  return {
    selected: chromosomeToSelectedRows(bestChromosome, genes),
    totals: bestEvaluation.totals,
    targets,
    filteredCount: filtered.length,
    candidatePoolCount: genes.length,
    fitness: bestEvaluation.fitness,
    gaDiagnostics: {
      populationSize: profile.populationSize,
      generations: profile.generations,
      geneCount: genes.length,
    },
  };
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value);
}

function renderSummary(result, inputs) {
  const budgetUtilization = (result.totals.cost / inputs.budget) * 100;
  const macro = (actual, target) => `${Math.round(actual)} / ${Math.round(target)} (${((actual / target) * 100).toFixed(0)}%)`;

  elements.summaryBlock.innerHTML = `
    <div class="summary-grid">
      <div><strong>Budget:</strong> ${formatCurrency(result.totals.cost)} / ${formatCurrency(inputs.budget)} (${budgetUtilization.toFixed(1)}%)</div>
      <div><strong>Calories:</strong> ${macro(result.totals.calories_kcal, result.targets.calories_kcal)}</div>
      <div><strong>Protein:</strong> ${macro(result.totals.protein_g, result.targets.protein_g)} g</div>
      <div><strong>Fat:</strong> ${macro(result.totals.fat_g, result.targets.fat_g)} g</div>
      <div><strong>Carbs:</strong> ${macro(result.totals.carbs_g, result.targets.carbs_g)} g</div>
      <div><strong>Filtered items:</strong> ${result.filteredCount} (${result.candidatePoolCount} genes)</div>
      <div><strong>GA run:</strong> pop ${result.gaDiagnostics.populationSize}, gen ${result.gaDiagnostics.generations}</div>
      <div><strong>Best fitness:</strong> ${result.fitness.toFixed(4)}</div>
    </div>
  `;
}

function renderResultTable(selectedRows) {
  elements.resultBody.innerHTML = "";
  for (const row of selectedRows) {
    const subtotal = row.item.price * row.qty;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.item.name}</td>
      <td>${row.item.store}</td>
      <td>${row.item.category}</td>
      <td>${row.qty}</td>
      <td>${formatCurrency(row.item.price)}</td>
      <td>${formatCurrency(subtotal)}</td>
      <td>${Math.round(row.item.calories_kcal * row.qty)}</td>
      <td>${Math.round(row.item.protein_g * row.qty)}</td>
      <td>${Math.round(row.item.fat_g * row.qty)}</td>
      <td>${Math.round(row.item.carbs_g * row.qty)}</td>
    `;
    elements.resultBody.appendChild(tr);
  }
}

function renderEmptyState(isEmpty) {
  elements.emptyState.hidden = !isEmpty;
}

function clearResults(options = {}) {
  const { keepStatus = false } = options;
  elements.summaryBlock.innerHTML = "";
  elements.resultBody.innerHTML = "";
  renderEmptyState(true);
  if (!keepStatus) {
    setStatus("Ready.", "neutral");
  }
}

async function runOptimizer() {
  try {
    await ensureDatasetLoaded();
    const inputs = getUserInputs();
    if (inputs.budget <= 0) {
      throw new Error("Budget must be greater than zero.");
    }
    const targets = calculateNutritionTargets(inputs);
    const result = await runGeneticOptimization(state.items, inputs, targets);

    if (result.selected.length === 0) {
      throw new Error("Genetic algorithm did not find a feasible grocery list with current constraints.");
    }

    renderSummary(result, inputs);
    renderResultTable(result.selected);
    renderEmptyState(false);
    setStatus(`Optimization complete. ${result.selected.length} items recommended.`, "success");
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected optimization error.";
    clearResults({ keepStatus: true });
    setStatus(message, "error");
  }
}

function attachEvents() {
  elements.runButton.addEventListener("click", runOptimizer);
  elements.clearButton.addEventListener("click", clearResults);
}

async function init() {
  attachEvents();
  clearResults();
  try {
    await ensureDatasetLoaded();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not initialize dataset.";
    setStatus(message, "error");
    elements.datasetMetaText.textContent = "Dataset: load failed";
  }
}

init();
