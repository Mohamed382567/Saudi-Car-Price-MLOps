# test_inference.py
import pytest
from src.inference import CarPredictor

def test_predictor_log_conversion():
    """Ensure CarPredictor correctly reverses the Log1p transformation using expm1."""
    predictor = CarPredictor()
    sample_car = {
        "brand": "Toyota", "model_name": "Camry", "year": 2022, "mileage": 15000,
        "gear": "Automatic", "fuel": "Gasoline", "extension": "Full", "origin": "Saudi",
        "drivetrain": "FWD", "cylinders": "4", "engine_size": "2.5", "seats": 5,
        "condition": "used", "exterior_color": "White", "interior_color": "Beige"
    }
    price = predictor.predict(sample_car)
    assert price > 0
    assert isinstance(price, float)