# Calculate Basal Metabolic Rate using Mifflin-St Jeor equation
def calc_bmr(weight_kg, height_cm, age, sex):
    if sex == 'male':
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


# Activity level multipliers for TDEE
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,        # little or no exercise
    'light': 1.375,          # light exercise 1-3 days/week
    'moderate': 1.55,        # moderate exercise 3-5 days/week
    'active': 1.725,         # hard exercise 6-7 days/week
    'very_active': 1.9       # very hard exercise, physical job
}

# Goal adjustments (daily calorie offset)
GOAL_OFFSETS = {
    'lose_weight': -500,     # ~0.45 kg/week deficit
    'maintain': 0,
    'gain_muscle': 300        # slight surplus for muscle gain
}

# Macronutrient ratio by goal (protein%, fat%, carbs%)
MACRO_RATIOS = {
    'lose_weight': (0.30, 0.25, 0.45),
    'maintain': (0.25, 0.30, 0.45),
    'gain_muscle': (0.30, 0.25, 0.45)
}

# Calculate weekly nutritional targets
def calc_weekly_targets(weight_kg, height_cm, age, sex, activity, goal):
    bmr = calc_bmr(weight_kg, height_cm, age, sex)
    tdee = bmr * ACTIVITY_MULTIPLIERS[activity]
    daily_cal = tdee + GOAL_OFFSETS[goal]

    # Ensure minimum daily calories
    daily_cal = max(daily_cal, 1200)

    prot_ratio, fat_ratio, carb_ratio = MACRO_RATIOS[goal]

    # Daily macros in grams (protein=4kcal/g, fat=9kcal/g, carbs=4kcal/g)
    daily_protein = (daily_cal * prot_ratio) / 4
    daily_fat = (daily_cal * fat_ratio) / 9
    daily_carbs = (daily_cal * carb_ratio) / 4

    # Scale to weekly
    return {
        'calories_kcal': round(daily_cal * 7, 2),
        'protein_g': round(daily_protein * 7, 2),
        'fat_g': round(daily_fat * 7, 2),
        'carbs_g': round(daily_carbs * 7, 2)
    }


if __name__ == '__main__':
    targets = calc_weekly_targets(70, 175, 25, 'male', 'moderate', 'maintain')
    print("Weekly nutritional targets:")
    for k, v in targets.items():
        print(f"  {k}: {v}")