import pandas as pd
import re
import html
import os

# Dietary category mapping
DIETARY_TAGS = {
    'non_vegetarian': {
        'meat-packaged', 'meat-poultry-wf', 'sausage-bacon',
        'jerky', 'seafood', 'seafood-wf'
    },
    'contains_dairy': {
        'dairy-yogurt-drink', 'cheese', 'milk-milk-substitute',
        'ice-cream-dessert', 'pudding-jello', 'mac-cheese'
    },
    'likely_gluten': {
        'bread', 'rolls-buns-wraps', 'muffins-bagels',
        'cookies-biscuit', 'cakes', 'pastry-chocolate-candy',
        'pasta-noodles', 'pizza', 'cereal', 'baking'
    },
    'contains_egg': {
        'eggs-wf'
    }
}


def clean_id(item_id):
    if pd.isna(item_id):
        return ""
    item_id = str(item_id)
    if item_id.startswith('wf_'):
        match = re.search(r'([a-z0-9]+)$', item_id, flags=re.IGNORECASE)
        if match:
            return f"wf-{match.group()}"
    return item_id


def preprocess_grocery_data(input_path, output_path):
    df = pd.read_csv(input_path)
    original_count = len(df)

    # Select columns
    required_cols = ['original_ID', 'name', 'store', 'harmonized single category', 'price',
                     'Protein', 'Total Fat', 'Carbohydrate', 'has10_nuts', 'package_weight']
    df = df[required_cols].copy()

    # Drop missing values
    df = df.dropna()

    # Clean ID
    df['original_ID'] = df['original_ID'].apply(clean_id)

    # Clean HTML entities in name
    df['name'] = df['name'].apply(lambda x: html.unescape(str(x)))

    # Rename columns
    df.columns = ['id', 'name', 'store', 'category', 'price',
                   'protein_g', 'fat_g', 'carbs_g', 'contains_nuts', 'package_weight_g']

    # Convert types
    df['price'] = df['price'].astype(float).round(2)
    df['protein_g'] = df['protein_g'].astype(float).round(2)
    df['fat_g'] = df['fat_g'].astype(float).round(2)
    df['carbs_g'] = df['carbs_g'].astype(float).round(2)
    df['contains_nuts'] = df['contains_nuts'].astype(int)
    df['package_weight_g'] = df['package_weight_g'].astype(float).round(2)

    # Filter: nutrition per 100g must be > 0 and <= 100
    df = df[(df['protein_g'] > 0) & (df['fat_g'] > 0) & (df['carbs_g'] > 0)]
    df = df[(df['protein_g'] <= 100) & (df['fat_g'] <= 100) & (df['carbs_g'] <= 100)]

    # Calculate calories per 100g
    df['calories_kcal'] = (df['protein_g'] * 4 + df['carbs_g'] * 4 + df['fat_g'] * 9).round(2)
    df = df[df['calories_kcal'] <= 900]

    # drop categories that are not suitable for adult optimization
    excluded_categories = {'baby-food', 'spices-seasoning', 'culinary-ingredients'}
    df = df[~df['category'].isin(excluded_categories)]

    # calculate the actual nutrition per package
    weight_ratio = df['package_weight_g'] / 100
    df['total_protein_g'] = (df['protein_g'] * weight_ratio).round(2)
    df['total_fat_g'] = (df['fat_g'] * weight_ratio).round(2)
    df['total_carbs_g'] = (df['carbs_g'] * weight_ratio).round(2)
    df['total_calories_kcal'] = (df['calories_kcal'] * weight_ratio).round(2)

    # add dietary tags
    df['is_vegetarian'] = (~df['category'].isin(DIETARY_TAGS['non_vegetarian'])).astype(int)
    df['has_dairy'] = df['category'].isin(DIETARY_TAGS['contains_dairy']).astype(int)
    df['has_gluten'] = df['category'].isin(DIETARY_TAGS['likely_gluten']).astype(int)
    df['has_egg'] = df['category'].isin(DIETARY_TAGS['contains_egg']).astype(int)

    # Drop per-100g columns, then rename total columns
    df = df.drop(columns=['protein_g', 'fat_g', 'carbs_g', 'calories_kcal'])
    df = df.rename(columns={
        'total_protein_g': 'protein_g',
        'total_fat_g': 'fat_g',
        'total_carbs_g': 'carbs_g',
        'total_calories_kcal': 'calories_kcal'
    })

    # Select final columns — all nutrition values are per package
    df = df[['id', 'name', 'store', 'category', 'price', 'package_weight_g',
             'protein_g', 'fat_g', 'carbs_g', 'calories_kcal',
             'contains_nuts', 'is_vegetarian', 'has_dairy', 'has_gluten', 'has_egg']]

    # Save
    df.to_csv(output_path, index=False)

    print(f"Raw dataset: {original_count} rows")
    print(f"After cleaning: {len(df)} rows")
    return len(df)


def load_cleaned_data(filepath):
    return pd.read_csv(filepath)


def filter_by_store(data, store_name):
    return data[data['store'] == store_name]


def filter_by_category(data, category):
    return data[data['category'] == category]


def filter_by_dietary(data, no_nuts=False, vegetarian=False, no_dairy=False, no_gluten=False, no_egg=False):
    """filter by dietary restrictions"""
    if no_nuts:
        data = data[data['contains_nuts'] == 0]
    if vegetarian:
        data = data[data['is_vegetarian'] == 1]
    if no_dairy:
        data = data[data['has_dairy'] == 0]
    if no_gluten:
        data = data[data['has_gluten'] == 0]
    if no_egg:
        data = data[data['has_egg'] == 0]
    return data


def get_nutrition_summary(data):
    return {
        'count': len(data),
        'protein': {'avg': data['protein_g'].mean(), 'min': data['protein_g'].min(), 'max': data['protein_g'].max()},
        'fat': {'avg': data['fat_g'].mean(), 'min': data['fat_g'].min(), 'max': data['fat_g'].max()},
        'carbs': {'avg': data['carbs_g'].mean(), 'min': data['carbs_g'].min(), 'max': data['carbs_g'].max()},
        'calories': {'avg': data['calories_kcal'].mean(), 'min': data['calories_kcal'].min(), 'max': data['calories_kcal'].max()},
        'price': {'avg': data['price'].mean(), 'min': data['price'].min(), 'max': data['price'].max()}
    }


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, '../data/GroceryDB_foods.csv')
    output_path = os.path.join(script_dir, '../data/cleaned_grocery.csv')
    preprocess_grocery_data(input_path, output_path)