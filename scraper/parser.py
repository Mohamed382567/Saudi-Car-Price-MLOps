import re
import json
from bs4 import BeautifulSoup
from src.preprocessing import master_extension_cleaner 

def extract_price_from_text(full_text):
    """ Extracts price precisely by looking for currency keywords or formatting. """
    if not full_text: return None
    
    # First attempt: Look for a number associated with currency keywords
    match = re.search(r'([\d,]+)\s*(?:ريال|ر\.س|SAR)', full_text)
    if match:
        return float(match.group(1).replace(',', ''))
    
    # Second attempt: Look for large numbers (car prices are usually above 5000)
    # This helps avoid mistakenly extracting the manufacturing year (e.g., 2023) or cylinders count
    numbers = re.findall(r'\d{1,3}(?:,\d{3})+|\d{4,}', full_text)
    if numbers:
        valid_prices = [float(n.replace(',', '')) for n in numbers if float(n.replace(',', '')) > 5000]
        if valid_prices:
            # The smallest valid large number is often the cash price 
            # (Used to avoid picking the total installment price which is usually higher)
            return min(valid_prices) 
            
    return None

def parse_car_details(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        script = soup.find("script", string=re.compile("window.FULL_PAGE_DATA"))
        if not script: return None

        json_text = re.search(r'window\.FULL_PAGE_DATA\s*=\s*(\{.*\});', script.string).group(1)
        data = json.loads(json_text)
        
        post_details = data.get('postDetails', {}).get('details', {})
        card = post_details.get('details_card', {})
        
        # --- 1. Condition Fix (Looking for 'is_new') ---
        # The JSON shows condition is now inside 'is_new'
        condition = card.get('is_new', {}).get('name') 
        if not condition:
            condition = post_details.get('condition') # Old fallback
            
        # --- 2. Mileage Fix (The 'Milage' spelling error) ---
        # Notice the spelling in JSON: 'milage' (without 'e')
        mileage = card.get('milage', {}).get('name')
        if not mileage:
            mileage = post_details.get('mileage') # Old fallback
            
        return {
            "brand": card.get('make', {}).get('name'),
            "model_name": card.get('model', {}).get('name'),
            "year": card.get('years', {}).get('name'),
            "exterior_color": card.get('exterior_color', {}).get('name'),
            "interior_color": card.get('interior_color', {}).get('name'),
            "origin": card.get('car_origin', {}).get('name'),
            "fuel": card.get('fuel_types', {}).get('name'),
            "gear": card.get('transmission_type', {}).get('name'),
            "cylinders": card.get('cylinders', {}).get('name'),
            "condition": str(condition).lower() if condition else None,
            "engine_size": card.get('engine_size', {}).get('name'),
            "mileage": mileage,
            "drivetrain": card.get('drivetrain_type', {}).get('name'),
            "seats": card.get('seats', {}).get('name'),
            "extension_clean": master_extension_cleaner(card.get('extension', {}).get('name'))
        }
    except Exception as e:
        print(f"⚠️ Parser Error: {e}")
        return None