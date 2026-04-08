# Grocery List Optimizer

**CS4100 Final Project** · Kashish Sethi, Yanzhen Chen, Rongxuan Zhang

A genetic-algorithm tool that builds a weekly grocery list under budget and dietary limits, aiming to match personalized nutrition targets and food preferences.

## Features

- **Nutrition targets**: Mifflin–St Jeor for BMR/TDEE; weekly calories and macros (protein, fat, carbs) from activity level and goal (lose weight / maintain / gain muscle).
- **Constraints**: Weekly budget cap; optional vegetarian, no nuts, no dairy, gluten-free; optional store filter (Target, Walmart, Whole Foods).
- **Optimization**: `GroceryGA` runs on a candidate item pool, balancing nutrition fit, cost, category diversity, and keyword preferences (e.g., chicken, rice).
- **Data**: `data/cleaned_grocery.csv` is a curated static price and nutrition table; no live APIs.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit web UI |
| `model/main.py` | CLI with the same pipeline |
| `model/data_processor.py` | CSV load, categories, dietary filters |
| `model/nutrition_calculator.py` | Weekly nutrition targets |
| `model/GA_optimizer.py` | Genetic algorithm and result formatting |
| `data/cleaned_grocery.csv` | Item dataset |

## How to run

**Dependencies** (Python 3): `streamlit`, `pandas`, `numpy`, `plotly`

```bash
pip install streamlit pandas numpy plotly
```

**Web app** (recommended), from the repo root:

```bash
streamlit run app.py
```

**CLI**:

```bash
cd model && python main.py
```

## Note

Nutrition and prices are static demo data for coursework. For real shopping, use in-store prices and professional medical or dietary advice.
