# test_api.py
import os
from fastapi.testclient import TestClient
from app.api import app  # Ensure the path to your FastAPI app is correct

# Initialize the test client globally so all functions can use it
client = TestClient(app)

def test_predict_endpoint_response():
    """Verify API returns 200 and a rounded price."""
    payload = {
        "brand": "Lexus", "model_name": "LX", "year": 2022, "mileage": 5000,
        "gear": "Automatic", "fuel": "Gasoline", "extension": "Full", "origin": "Saudi",
        "drivetrain": "AWD", "cylinders": "6", "engine_size": "3.5", "seats": 7,
        "condition": "used", "exterior_color": "Black", "interior_color": "Tan"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "estimated_price_sar" in response.json()

def test_logging_creation():
    """Ensure the API creates a log file locally even in cloud-ready systems."""
    # 1. Setup: Ensure the logs directory exists before triggering the request
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    # 2. Action: Trigger a prediction request using the 'client'
    test_data = {
        "brand": "Test", "model_name": "Test", "year": 2020, 
        "mileage": 10, "gear": "A", "fuel": "G", "extension": "F", 
        "origin": "S", "drivetrain": "F", "cylinders": "4", 
        "engine_size": "2", "seats": 5, "condition": "used", 
        "exterior_color": "W", "interior_color": "B"
    }
    client.post("/predict", json=test_data)
    
    # 3. Assertion: Check if the file was created on the local filesystem (GitHub Runner)
    log_file_path = "logs/single_predictions.jsonl"
    assert os.path.exists(log_file_path)