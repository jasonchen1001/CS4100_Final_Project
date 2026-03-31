import pandas as pd
import re
import html
import os

# Dietary tags based on original category names (before remapping)
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

# Categories to remove (low nutritional value, not core grocery items)
EXCLUDED_CATEGORIES = {
    'baby-food', 'spices-seasoning', 'culinary-ingredients',
    'pastry-chocolate-candy', 'cookies-biscuit', 'cakes',
    'ice-cream-dessert', 'pudding-jello',
    'sauce-all', 'dressings', 'spread-squeeze',
    'drink-shakes-other', 'drink-coffee', 'drink-juice',
    'drink-soft-energy-mixes', 'drink-tea', 'drink-juice-wf',
    'prepared-meals-dishes', 'pizza', 'salad',
    'snacks-chips', 'snacks-mixes-crackers', 'snacks-popcorn', 'snacks-dips-salsa',
    'canned-goods',
}

# Remap original categories into 7 groups
CATEGORY_MAP = {
    'meat-packaged': 'meat', 'meat-poultry-wf': 'meat',
    'sausage-bacon': 'meat', 'jerky': 'meat',
    'seafood': 'seafood', 'seafood-wf': 'seafood',
    'produce-packaged': 'vegetables', 'produce-beans-wf': 'vegetables',
    'nuts-seeds-wf': 'vegetables',
    'dairy-yogurt-drink': 'dairy', 'milk-milk-substitute': 'dairy',
    'cheese': 'dairy', 'eggs-wf': 'dairy',
    'bread': 'staples', 'rolls-buns-wraps': 'staples',
    'muffins-bagels': 'staples', 'cereal': 'staples',
    'rice-grains-packaged': 'staples', 'rice-grains-wf': 'staples',
    'pasta-noodles': 'staples', 'mac-cheese': 'staples',
    'baking': 'staples', 'breakfast': 'staples',
    'soup-stew': 'soup',
    'snacks-bars': 'snacks', 'snacks-nuts-seeds': 'snacks',
}


# Normalize Whole Foods IDs to consistent format
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

    # Keep only required columns
    required_cols = ['original_ID', 'name', 'store', 'harmonized single category', 'price',
                     'Protein', 'Total Fat', 'Carbohydrate', 'has10_nuts', 'package_weight']
    df = df[required_cols].copy()
    df = df.dropna()

    # Clean ID and decode HTML entities in names
    df['original_ID'] = df['original_ID'].apply(clean_id)
    df['name'] = df['name'].apply(lambda x: html.unescape(str(x)))

    # Standardize column names
    df.columns = ['id', 'name', 'store', 'category', 'price',
                   'protein_g', 'fat_g', 'carbs_g', 'contains_nuts', 'package_weight_g']

    # Convert to numeric types
    df['price'] = df['price'].astype(float).round(2)
    df['protein_g'] = df['protein_g'].astype(float).round(2)
    df['fat_g'] = df['fat_g'].astype(float).round(2)
    df['carbs_g'] = df['carbs_g'].astype(float).round(2)
    df['contains_nuts'] = df['contains_nuts'].astype(int)
    df['package_weight_g'] = df['package_weight_g'].astype(float).round(2)

    # Remove items with zero or unrealistic per-100g nutrition values
    df = df[(df['protein_g'] > 0) & (df['fat_g'] > 0) & (df['carbs_g'] > 0)]
    df = df[(df['protein_g'] <= 100) & (df['fat_g'] <= 100) & (df['carbs_g'] <= 100)]

    # Calculate calories per 100g (4-4-9 formula) and cap at 900
    df['calories_kcal'] = (df['protein_g'] * 4 + df['carbs_g'] * 4 + df['fat_g'] * 9).round(2)
    df = df[df['calories_kcal'] <= 900]

    # Tag dietary labels using ORIGINAL category names (before remapping)
    df['is_vegetarian'] = (~df['category'].isin(DIETARY_TAGS['non_vegetarian'])).astype(int)
    df['has_dairy'] = df['category'].isin(DIETARY_TAGS['contains_dairy']).astype(int)
    df['has_gluten'] = df['category'].isin(DIETARY_TAGS['likely_gluten']).astype(int)
    df['has_egg'] = df['category'].isin(DIETARY_TAGS['contains_egg']).astype(int)

    # Remove unwanted categories
    df = df[~df['category'].isin(EXCLUDED_CATEGORIES)]

    # Remap to 7 groups (AFTER dietary tagging, order matters)
    df['category'] = df['category'].map(CATEGORY_MAP)
    df = df.dropna(subset=['category'])

    # Remove herbs/spices that slipped through in produce categories
    herb_keywords = ['thyme', 'rosemary', 'basil', 'oregano', 'cilantro',
                     'parsley', 'dill', 'mint', 'sage', 'chives']
    herb_mask = df['name'].str.lower().str.contains('|'.join(herb_keywords), na=False)
    df = df[~herb_mask]

    # Convert per-100g nutrition to per-package (actual amount per item)
    weight_ratio = df['package_weight_g'] / 100
    df['total_protein_g'] = (df['protein_g'] * weight_ratio).round(2)
    df['total_fat_g'] = (df['fat_g'] * weight_ratio).round(2)
    df['total_carbs_g'] = (df['carbs_g'] * weight_ratio).round(2)
    df['total_calories_kcal'] = (df['calories_kcal'] * weight_ratio).round(2)

    # Replace per-100g columns with per-package values
    df = df.drop(columns=['protein_g', 'fat_g', 'carbs_g', 'calories_kcal'])
    df = df.rename(columns={
        'total_protein_g': 'protein_g',
        'total_fat_g': 'fat_g',
        'total_carbs_g': 'carbs_g',
        'total_calories_kcal': 'calories_kcal'
    })

    # Select final columns
    df = df[['id', 'name', 'store', 'category', 'price', 'package_weight_g',
             'protein_g', 'fat_g', 'carbs_g', 'calories_kcal',
             'contains_nuts', 'is_vegetarian', 'has_dairy', 'has_gluten', 'has_egg']]

    df.to_csv(output_path, index=False)

    # Print summary
    print(f"Raw: {original_count} rows")
    print(f"Cleaned: {len(df)} rows")
    cats = sorted(df['category'].unique())
    print(f"Categories: {df['category'].nunique()} ({', '.join(cats)})")
    print(f"\nBreakdown:")
    print(df['category'].value_counts().to_string())

    return len(df)


def load_cleaned_data(filepath):
    return pd.read_csv(filepath)


def filter_by_store(data, store_name):
    return data[data['store'] == store_name]


def filter_by_category(data, category):
    return data[data['category'] == category]


# Apply dietary restrictions to filter out items
def filter_by_dietary(data, no_nuts=False, vegetarian=False, no_dairy=False, no_gluten=False, no_egg=False):
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


# Get statistical summary for a dataset
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
    input_path = os.path.join(script_dir, '..', 'data', 'GroceryDB_foods.csv')
    output_path = os.path.join(script_dir, '..', 'data', 'cleaned_grocery.csv')
    preprocess_grocery_data(input_path, output_path)