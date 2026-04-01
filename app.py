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

# Profile
with col1:
    st.subheader("1. Profile & Goals")

    weight = st.number_input("Weight (kg)", value=70, step=1)
    height = st.number_input("Height (cm)", value=175, step=1)
    age = st.number_input("Age", value=22)

    sex = st.selectbox("Sex", ["male", "female"])
    activity_options = {
        "sedentary (little exercise)": "sedentary",
        "light (1-3 days/week)": "light",
        "moderate (3-5 days/week)": "moderate",
        "active (6-7 days/week)": "active",
        "very active (physical job)": "very_active"
    }

    activity_label = st.selectbox("Activity Level", list(activity_options.keys()))
    activity = activity_options[activity_label]
    goal_options = {
        "Lose weight": "lose_weight",
        "Maintain": "maintain",
        "Gain muscle": "gain_muscle"
    }

    goal_label = st.selectbox("Goal", list(goal_options.keys()))
    goal = goal_options[goal_label]

# Budget
with col2:
    st.subheader("2. Budget & Constraints")

    budget = st.number_input("Weekly Budget ($)", value=100.0)

    if "diet" not in st.session_state:
        st.session_state.diet = {
            "vegetarian": False,
            "no_nuts": False,
            "no_dairy": False,
            "no_gluten": False
        }

    st.markdown("**Dietary Restrictions**")

    bcol1, bcol2 = st.columns(2)

    def toggle(key):
        st.session_state.diet[key] = not st.session_state.diet[key]

    with bcol1:
        st.button(
            "🥦 Vegetarian",
            use_container_width=True,
            on_click=toggle,
            args=("vegetarian",),
            type="primary" if st.session_state.diet["vegetarian"] else "secondary"
        )

        st.button(
            "🥜 No Nuts",
            use_container_width=True,
            on_click=toggle,
            args=("no_nuts",),
            type="primary" if st.session_state.diet["no_nuts"] else "secondary"
        )

    with bcol2:
        st.button(
            "🥛 No Dairy",
            use_container_width=True,
            on_click=toggle,
            args=("no_dairy",),
            type="primary" if st.session_state.diet["no_dairy"] else "secondary"
        )

        st.button(
            "🌾 No Gluten",
            use_container_width=True,
            on_click=toggle,
            args=("no_gluten",),
            type="primary" if st.session_state.diet["no_gluten"] else "secondary"
        )

    vegetarian = st.session_state.diet["vegetarian"]
    no_nuts = st.session_state.diet["no_nuts"]
    no_dairy = st.session_state.diet["no_dairy"]
    no_gluten = st.session_state.diet["no_gluten"]

    # BMI
    bmi = calculate_bmi(weight, height)
    bmi_text = bmi_category(bmi)

    st.markdown(f"""
    <div style="
        width: 92px;
        height: 92px;
        background: #f9fafb;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 10px;
        margin-bottom: 10px;
    ">
        <div style="font-size: 24px; font-weight: 800; color: #111827; line-height: 1.1;">
            BMI
        </div>
        <div style="font-size: 20px; font-weight: 600; color: #111827; margin-top: 8px; line-height: 1.1;">
            {bmi:.2f}
        </div>
        <div style="font-size: 13px; color: #10b981; margin-top: 8px;">
            {bmi_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # Food preferences
    st.markdown("### 3. Preferences")

    pref_input = st.text_input(
        "Food Preferences (e.g. chicken, rice, yogurt, salmon)"
    )

    preferences = [p.strip().lower() for p in pref_input.split(",") if p.strip()]

# Store preferences
st.subheader("Store Preferences")

store_options = {
    "All Stores": None,
    "Target": "Target",
    "Walmart": "Walmart",
    "WholeFoods": "WholeFoods"
}

store_label = st.selectbox(
    "Store Preference",
    list(store_options.keys()),
    label_visibility="collapsed"
)
store = store_options[store_label]

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

    if store:
        data = data[data['store'] == store]

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

    st.plotly_chart(
        {
            "data": [{
                "labels": cat_df["Category"],
                "values": cat_df["Count"],
                "type": "pie"
            }],
            "layout": {"margin": {"t": 0, "b": 0}}
        },
        use_container_width=True
    )

    # Result Table
    st.subheader("🛍️ Optimized Grocery List")

    display = display.copy()
    display['price'] = display['price'].round(2)
    display['subtotal'] = display['subtotal'].round(2)

    st.dataframe(display, use_container_width=True)