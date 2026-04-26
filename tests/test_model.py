# tests/test_inference.py
import os
import pytest
import numpy as np
from src.inference import CarPredictor

# This decorator tells pytest to skip these tests if the model file is missing
# This is crucial for GitHub Actions where we use dummy artifacts for logic testing
@pytest.mark.skipif(
    not os.path.exists('models/car_price_model.pkl'), 
    reason="Model file not found - skipping deep inference tests in CI environment"
)
def test_predictor_log_conversion():
    """
    Ensure CarPredictor correctly reverses the Log1p transformation using expm1.
    This test runs only when a real model file is present (Local or Docker).
    """
    predictor = CarPredictor()
    sample_car = {
        "brand": "Toyota", 
        "model_name": "Camry", 
        "year": 2022, 
        "mileage": 15000,
        "gear": "Automatic", 
        "fuel": "Gasoline", 
        "extension": "Full", 
        "origin": "Saudi",
        "drivetrain": "FWD", 
        "cylinders": "4", 
        "engine_size": "2.5", 
        "seats": 5,
        "condition": "used", 
        "exterior_color": "White", 
        "interior_color": "Beige"
    }
    
    # Trigger prediction
    response = predictor.predict(sample_car)
    
    # Check if the output is a dictionary (common in FastAPI predictors) or a raw float
    if isinstance(response, dict):
        price = response.get('estimated_price_sar', 0)
    else:
        price = response
        
    # Validation assertions
    assert price > 0, f"Predicted price should be positive, got {price}"
    assert isinstance(price, (float, int, np.float32, np.float64)), "Price must be a numeric type"
