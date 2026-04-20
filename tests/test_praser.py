# test_parser.py
from scraper.parser import extract_price_from_text

def test_price_extraction_variants():
    """Testing the Regex logic in parser.py for different currency formats."""
    assert extract_price_from_text("Price is 150,000 SAR") == 150000.0
    assert extract_price_from_text("السعر 95000 ريال") == 95000.0
    assert extract_price_from_text("ر.س 45,000") == 45000.0

def test_large_number_validation():
    """Verify it ignores small numbers like years or cylinders and picks the price."""
    # Based on your logic in parser.py: min(valid_prices) if > 5000
    text = "Model 2022, 6 cylinders, price 85000 SAR"
    assert extract_price_from_text(text) == 85000.0