import pandas as pd
import re
import html
import os

# Clean ID
def clean_id(item_id):
    if pd.isna(item_id):
        return ""
    item_id = str(item_id)

    # For Whole Foods IDs
    if item_id.startswith('wf_'):
        match = re.search(r'([a-z0-9]+)$', item_id, flags=re.IGNORECASE)
        if match:
            return f"wf-{match.group()}"
    return item_id

# Preprocess grocery dataset
def preprocess_grocery_data(input_path, output_path):
    df = pd.read_csv(input_path)
    original_count = len(df)

    # Select columns
    required_cols = ['original_ID', 'name', 'store', 'harmonized single category', 'price',
                     'Protein', 'Total Fat', 'Carbohydrate', 'has10_nuts']
    df = df[required_cols].copy()

    # Drop missing values
    df = df.dropna()

    # Clean ID
    df['original_ID'] = df['original_ID'].apply(clean_id)

    # Rename columns
    df.columns = ['id', 'name', 'store', 'category', 'price', 'protein_g', 'fat_g', 'carbs_g', 'contains_nuts']

    # Convert types
    df['price'] = df['price'].astype(float)
    df['protein_g'] = df['protein_g'].astype(float)
    df['fat_g'] = df['fat_g'].astype(float)
    df['carbs_g'] = df['carbs_g'].astype(float)
    df['contains_nuts'] = df['contains_nuts'].astype(int)

    # Round to 2 decimal places
    df['price'] = df['price'].round(2)
    df['protein_g'] = df['protein_g'].round(2)
    df['fat_g'] = df['fat_g'].round(2)
    df['carbs_g'] = df['carbs_g'].round(2)

    # Filter zero values
    df = df[(df['protein_g'] > 0) & (df['fat_g'] > 0) & (df['carbs_g'] > 0)]

    # Filter out of range (per 100g)
    df = df[(df['protein_g'] <= 100) & (df['fat_g'] <= 100) & (df['carbs_g'] <= 100)]

    # Calculate calories
    df['calories_kcal'] = df['protein_g'] * 4 + df['carbs_g'] * 4 + df['fat_g'] * 9

    # Round calories to 2 decimal
    df['calories_kcal'] = df['calories_kcal'].round(2)

    # Filter calories max
    df = df[df['calories_kcal'] <= 900]

    # Select final columns (id first)
    df = df[['id', 'name', 'store', 'category', 'price', 'protein_g', 'fat_g', 'carbs_g', 'calories_kcal', 'contains_nuts']]

    # Save output
    df.to_csv(output_path, index=False)

    # Print stats
    print(f"Raw dataset: {original_count} rows")
    print(f"After cleaning: {len(df)} rows")

    return len(df)


def load_cleaned_data(filepath):
    # Load cleaned data
    return pd.read_csv(filepath)


def filter_by_store(data, store_name):
    # Filter by store
    return data[data['store'] == store_name]


def filter_by_category(data, category):
    # Filter by category
    return data[data['category'] == category]


def filter_by_dietary(data, no_nuts=False):
    # Filter by dietary restrictions
    if no_nuts:
        return data[data['contains_nuts'] == 0]
    return data


def get_nutrition_summary(data):
    # Get nutrition summary
    return {
        'count': len(data),
        'protein': {'avg': data['protein_g'].mean(), 'min': data['protein_g'].min(), 'max': data['protein_g'].max()},
        'fat': {'avg': data['fat_g'].mean(), 'min': data['fat_g'].min(), 'max': data['fat_g'].max()},
        'carbs': {'avg': data['carbs_g'].mean(), 'min': data['carbs_g'].min(), 'max': data['carbs_g'].max()},
        'calories': {'avg': data['calories_kcal'].mean(), 'min': data['calories_kcal'].min(), 'max': data['calories_kcal'].max()},
        'price': {'avg': data['price'].mean(), 'min': data['price'].min(), 'max': data['price'].max()}
    }


if __name__ == '__main__':
    # Run preprocessing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, '../data/GroceryDB_foods.csv')
    output_path = os.path.join(script_dir, '../data/cleaned_grocery.csv')
    preprocess_grocery_data(input_path, output_path)
