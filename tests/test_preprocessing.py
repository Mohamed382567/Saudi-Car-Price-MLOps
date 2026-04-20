# test_preprocessing.py
import pandas as pd
from src.preprocessing import master_extension_cleaner, clean_dataframe

def test_extension_cleaning_logic():
    """Verify that keywords match your specific categories in preprocessing.py."""
    assert master_extension_cleaner("Camry Sport SE") == "Sport/Performance"
    assert master_extension_cleaner("Land Cruiser VXR") == "Offroad/4x4"
    assert master_extension_cleaner("Tesla Electric") == "Hybrid" # As per your hybrid keywords

def test_filtering_outliers():
    """Verify rows are dropped based on your exact constraints (2016 and 250k)."""
    df = pd.DataFrame({
        'year': [2010, 2020], 
        'mileage': [10000, 300000],
        'price': [30000, 40000]
    })
    cleaned = clean_dataframe(df)
    # Both rows should be removed (one for year < 2016, one for mileage > 250k)
    assert len(cleaned) == 0