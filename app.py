import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

import streamlit as st
import pandas as pd

from data_processor import load_cleaned_data, filter_by_dietary
from nutrition_calculator import calc_weekly_targets
from GA_optimizer import GroceryGA

# Page config
st.set_page_config(page_title="Grocery Optimizer", layout="wide")

# BMI
def calculate_bmi(weight_kg, height_cm):
    h = height_cm / 100
    return weight_kg / (h ** 2) if h > 0 else 0

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"

# Title
st.title("🛒 Grocery List Optimizer")
st.markdown("Build a budget-aware grocery list with nutrition balance.")

# Layout
col1, col2 = st.columns(2)

# rofile
with col1:
    st.subheader("1. Profile & Goals")

    weight = st.number_input("Weight (kg)", value=70.0)
    height = st.number_input("Height (cm)", value=175.0)
    age = st.number_input("Age", value=22)

    sex = st.selectbox("Sex", ["male", "female"])
    activity = st.selectbox("Activity", ["sedentary", "light", "moderate", "active", "very_active"])
    goal = st.selectbox("Goal", ["lose_weight", "maintain", "gain_muscle"])

    bmi = calculate_bmi(weight, height)
    st.markdown(f"**BMI:** {bmi:.2f} ({bmi_category(bmi)})")

# Budget
with col2:
    st.subheader("2. Budget & Constraints")

    budget = st.number_input("Weekly Budget ($)", value=100.0)

    vegetarian = st.checkbox("Vegetarian")
    no_nuts = st.checkbox("No Nuts")
    no_dairy = st.checkbox("No Dairy")
    no_gluten = st.checkbox("No Gluten")

# Preferences
st.subheader("3. Preferences")

pref_input = st.text_input("Food preferences (comma separated)")
preferences = [p.strip().lower() for p in pref_input.split(",") if p.strip()]

# Run Button
run = st.button("🚀 Generate Optimized Grocery List")

# Run Logic
if run:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "cleaned_grocery.csv")

    data = load_cleaned_data(data_path)

    data = filter_by_dietary(
        data,
        vegetarian=vegetarian,
        no_nuts=no_nuts,
        no_dairy=no_dairy,
        no_gluten=no_gluten
    )

    st.success(f"Dataset loaded: {len(data)} items")

    # Targets
    targets = calc_weekly_targets(weight, height, age, sex, activity, goal)

    st.subheader("📊 Weekly Targets")

    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    tcol1.metric("Calories", f"{targets['calories_kcal']:.0f}")
    tcol2.metric("Protein (g)", f"{targets['protein_g']:.1f}")
    tcol3.metric("Fat (g)", f"{targets['fat_g']:.1f}")
    tcol4.metric("Carbs (g)", f"{targets['carbs_g']:.1f}")

    # Run GA
    with st.spinner("Running optimization..."):
        ga = GroceryGA(
            data,
            targets,
            budget,
            preferences=preferences
        )

        chrom, fitness = ga.optimize(verbose=False)
        display, summary = ga.format_result(chrom)

    # Result Table
    st.subheader("🛍️ Optimized Grocery List")

    display = display.copy()
    display['price'] = display['price'].round(2)
    display['subtotal'] = display['subtotal'].round(2)

    st.dataframe(display, use_container_width=True)

    # Summary
    st.subheader("📈 Summary")

    scol1, scol2, scol3, scol4 = st.columns(4)

    scol1.metric("Total Cost", f"${summary['total_cost']:.2f}")
    scol2.metric("Items", summary['num_items'])
    scol3.metric("Calories", f"{summary['total_calories_kcal']:.0f}")
    scol4.metric("Protein", f"{summary['total_protein_g']:.1f}")

    # Nutrition Comparison
    st.markdown("### Nutrition vs Target")

    comp_df = pd.DataFrame({
        "Metric": ["Calories", "Protein", "Fat", "Carbs"],
        "Actual": [
            summary['total_calories_kcal'],
            summary['total_protein_g'],
            summary['total_fat_g'],
            summary['total_carbs_g']
        ],
        "Target": [
            targets['calories_kcal'],
            targets['protein_g'],
            targets['fat_g'],
            targets['carbs_g']
        ]
    })

    comp_df["% Achieved"] = (comp_df["Actual"] / comp_df["Target"] * 100).round(1)

    st.dataframe(comp_df, use_container_width=True)

    # Category Breakdown
    st.markdown("### Category Breakdown")

    cat_df = pd.DataFrame.from_dict(
        summary['category_breakdown'],
        orient='index',
        columns=['Count']
    ).reset_index().rename(columns={'index': 'Category'})

    st.bar_chart(cat_df.set_index("Category"))