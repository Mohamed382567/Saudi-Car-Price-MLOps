import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import re

def master_extension_cleaner(text):
    """ Standardize car trims based on keywords using whole-word matching. """
    text = str(text).lower().replace('\t', ' ').strip()

    # heplper function for the full words only
    def contains_word(word_list, target_text):
        for word in word_list:
            if re.search(r'\b' + re.escape(word) + r'\b', target_text):
                return True
        return False

    # 1. Performance & Sport
    if contains_word(['sport', 'rs', 'gt', 'performance', 'amg', '63', 'f-sport', 'f sport', 'n line', 'srt', 'skyactive g', 'skyactiv'], text):
        return 'Sport/Performance'

    # 2. Offroad & 4x4
    if contains_word(['4x4', '4wd', 'double', 'sahara', 'unlimited', 'willys', 'jk', 'z71', 'at4', 'adventure', 'txl', 'gxr', 'vxr', 'gx', 'fj cruiser', 'جي كي'], text):
        return 'Offroad/4x4'

    # 3. Hybrid & Electric 
    if contains_word(['hybrid', 'hev', 'phev', 'h ', 'electric', 'ev'], text):
        return 'Hybrid'

    # 4. Full Option & Luxury
    full_keywords = ['premium', 'full', 'elite', 'limited', 'platinum', 'signature', 'royal', 'vip',
                     'calligraphy', 'titanium', 'prestige', 'vx', 'sl', 'le', 'grande', 'diamond',
                     'black ed', 'grand touring', 'high line', 'ليمتد', 'فل']
    if contains_word(full_keywords, text):
        return 'Full Option'

    # 5. Mid Option & Comfort
    mid_keywords = ['smart', 'mid', 'comfort', 'classic', 'se', 'sv', 'sr', 'ex', 'gle', 'gls',
                    'sel', 'trend', 'exclusive', 'design', 'hl', 'semi']
    if contains_word(mid_keywords, text):
        return 'Mid Option'

    # 6. Standard & Economy
    std_keywords = ['basic', 'fleet', 'gl', 'bse', 's', 'l', 'core', 'lx', 'ls', 'dx', 'xli', 'y',
                    'base', 'std', 'xe', 'xl', 'dlx', 'cargo', 'van', 'بضاعة', 'ثلاجة', 'ركاب']
    if contains_word(std_keywords, text):
        return 'Standard'

    # 7. Specific Luxury/Engine Codes (Regex for codes like V6, G500, etc.)
    if re.search(r'\b\w?\d{3,}\w?\b', text) and len(text) < 8:
        return text.upper()

    return 'Other'

def clean_numeric_string(value):
    """ Extracts numbers from strings (e.g., '100,000 KM' -> 100000.0). """
    if value is None or value == '': return 0.0
    cleaned = re.sub(r'[^\d.]', '', str(value))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def clean_dataframe(df):
    """
    Applies cleaning, handles missing data precisely, and removes outliers.
    """
    df_clean = df.copy()
    
    # --- NEW ADDITION: MISSING DATA HANDLING ---
    
    # 1. Level 1: "Ghost Cars" (Absolute minimum required)
    # If we don't know the brand, model, or price, it's useless. Drop it.
    essential_cols = [col for col in ['brand', 'model_name', 'price'] if col in df_clean.columns]
    if essential_cols:
        df_clean = df_clean.dropna(subset=essential_cols)

    # 2. Level 2: "Major Nulls" (Too many missing technical details)
    # These are cars that exist but have mostly empty cells, which ruins training.
    tech_columns = ['year', 'fuel', 'gear', 'condition', 'engine_size', 'mileage', 'cylinders', 'drivetrain']
    available_tech_cols = [col for col in tech_columns if col in df_clean.columns]
    
    if available_tech_cols:
        # Count how many of these core technical features are missing (null) per row
        missing_tech_count = df_clean[available_tech_cols].isnull().sum(axis=1)
        
        # Rule: If a car is missing MORE THAN 0 of these key features, drop it.
        # Level 3 (Minor Nulls): If missing <= 0, keep it and let SimpleImputer handle it later.
        MAX_ALLOWED_MISSING = 0
        df_clean = df_clean[missing_tech_count <= MAX_ALLOWED_MISSING]
        # This is the best desicion to not allow any null value so it don't affect the data badly and then affects accuracy of the model.
    # ----------------------------------------------------------------------

    # 1. Textual Cleaning (Extension) - Using your original logic and name
    if 'extension_clean' in df_clean.columns:
        df_clean['extension_clean'] = df_clean['extension_clean'].apply(master_extension_cleaner)
    
    # 2. Convert to numeric for outlier filtering
    if 'mileage' in df_clean.columns:
        df_clean['mileage'] = df_clean['mileage'].apply(clean_numeric_string)
    
    for col in ['year', 'seats']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

    # ---  CYLINDERS CLEANING ---
    if 'cylinders' in df_clean.columns:
        # Step 1: Extract only digits using Regex (Notebook logic)
        df_clean['cylinders'] = df_clean['cylinders'].astype(str).str.extract(r'(\d+)')
        # Step 2: Convert to numeric and fill NaN
        df_clean['cylinders'] = pd.to_numeric(df_clean['cylinders'], errors='coerce').fillna(0)
        
    # --- 3. OUTLIER HANDLING (Notebook Logic) ---
    
    # Handle Mileage outliers: Keep cars below 250k KM
    if 'mileage' in df_clean.columns:
        df_clean = df_clean[df_clean['mileage'] < 250000]
        
    # Handle Year outliers: Keep cars from 2016 onwards
    if 'year' in df_clean.columns:
        df_clean = df_clean[df_clean['year'] >= 2016]
        
    # Price/Luxury: Keeping high values as per your decision to preserve luxury data
    
    return df_clean

def create_preprocessor():
    """ Creates the transformation pipeline. """
    categorical_features = [
        'brand', 'model_name', 'exterior_color', 'interior_color', 
        'origin', 'fuel', 'gear',  'condition', 
        'drivetrain', 'extension_clean'
    ]
    numerical_features = ['year', 'mileage', 'seats','cylinders','engine_size']
    
    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    num_pipe = Pipeline(steps=[
        ('scaler', RobustScaler())
    ])
    
    return ColumnTransformer(transformers=[
        ('num', num_pipe, numerical_features),
        ('cat', cat_pipe, categorical_features)
    ])