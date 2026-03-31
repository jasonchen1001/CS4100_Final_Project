import os
import pandas as pd
import numpy as np
from data_processor import load_cleaned_data, filter_by_dietary
from nutrition_calculator import calc_weekly_targets
from GA_optimizer import GroceryGA, print_summary


def get_user_profile():
    print("\n=== Grocery List Optimizer ===\n")

    weight = float(input("Weight (kg): "))
    height = float(input("Height (cm): "))
    age = int(input("Age: "))

    print("\nSex:")
    print("  1. Male")
    print("  2. Female")
    sex_map = {'1': 'male', '2': 'female'}
    sex = sex_map[input("Choose (1-2): ").strip()]

    print("\nActivity Level:")
    print("  1. Sedentary (little exercise)")
    print("  2. Light (1-3 days/week)")
    print("  3. Moderate (3-5 days/week)")
    print("  4. Active (6-7 days/week)")
    print("  5. Very Active (physical job)")
    activity_map = {'1': 'sedentary', '2': 'light', '3': 'moderate', '4': 'active', '5': 'very_active'}
    activity = activity_map[input("Choose (1-5): ").strip()]

    print("\nGoal:")
    print("  1. Lose Weight")
    print("  2. Maintain Weight")
    print("  3. Gain Muscle")
    goal_map = {'1': 'lose_weight', '2': 'maintain', '3': 'gain_muscle'}
    goal = goal_map[input("Choose (1-3): ").strip()]

    budget = float(input("\nWeekly Budget ($): "))

    print("\nStore Preference:")
    print("  1. All Stores")
    print("  2. Target")
    print("  3. Walmart")
    print("  4. WholeFoods")
    store_map = {'1': None, '2': 'Target', '3': 'Walmart', '4': 'WholeFoods'}
    store = store_map[input("Choose (1-4): ").strip()]

    print("\nDietary Restrictions (y/n):")
    vegetarian = input("  Vegetarian? ").strip().lower() == 'y'
    no_nuts = input("  Nut allergy? ").strip().lower() == 'y'
    no_dairy = input("  No dairy? ").strip().lower() == 'y'
    no_gluten = input("  Gluten-free? ").strip().lower() == 'y'

    return {
        'weight_kg': weight, 'height_cm': height, 'age': age,
        'sex': sex, 'activity': activity, 'goal': goal,
        'budget': budget, 'store': store,
        'vegetarian': vegetarian, 'no_nuts': no_nuts,
        'no_dairy': no_dairy, 'no_gluten': no_gluten
    }


def run_optimization(profile, data_path=None):
    if data_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '..', 'data', 'cleaned_grocery.csv')

    # Load and filter
    data = load_cleaned_data(data_path)
    data = filter_by_dietary(
        data, vegetarian=profile['vegetarian'], no_nuts=profile['no_nuts'],
        no_dairy=profile['no_dairy'], no_gluten=profile['no_gluten']
    )
    if profile.get('store'):
        data = data[data['store'] == profile['store']]

    print(f"\nAvailable items: {len(data)}")

    # Targets
    targets = calc_weekly_targets(
        profile['weight_kg'], profile['height_cm'],
        profile['age'], profile['sex'],
        profile['activity'], profile['goal']
    )
    print(f"\nWeekly targets:")
    print(f"  Calories: {targets['calories_kcal']:.0f} kcal")
    print(f"  Protein:  {targets['protein_g']:.1f} g")
    print(f"  Fat:      {targets['fat_g']:.1f} g")
    print(f"  Carbs:    {targets['carbs_g']:.1f} g")
    print(f"  Budget:   ${profile['budget']:.2f}")

    # Run GA
    ga = GroceryGA(
        data, targets, profile['budget'],
        max_qty=2, pop_size=200, generations=300,
        crossover_rate=0.85, mutation_rate=0.1,
        tournament_size=5, elitism_count=10
    )
    chrom, fitness = ga.optimize(verbose=True)
    display, summary = ga.format_result(chrom)
    print_summary(display, summary)

    return ga, chrom, summary


if __name__ == '__main__':
    profile = get_user_profile()
    run_optimization(profile)
