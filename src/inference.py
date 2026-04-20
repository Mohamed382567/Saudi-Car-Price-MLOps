import joblib
import pandas as pd
import numpy as np
import os
# Import cleaning logic from our preprocessing module
from src.preprocessing import master_extension_cleaner, clean_numeric_string

class CarPredictor:
    """
    Unified class to handle car price predictions.
    It bridges the gap between raw UI input and the trained XGBoost model.
    """
    def __init__(self, model_path='models/car_price_model.pkl', preprocessor_path='models/preprocessor.pkl'):
        # Load the artifacts saved during the training pipeline
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Model or Preprocessor not found. Run trainer.py first.")
            
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, raw_input_dict):
        """
        Full inference cycle: Raw Input -> Cleaning -> Transformation -> Prediction -> Inverse Log.
        """
        # 1. Convert the input dictionary to a pandas DataFrame
        df = pd.DataFrame([raw_input_dict])

        # 2. Map 'extension' to 'extension_clean' using the master logic
        # This ensures inputs like "فل كامل" are categorized exactly as in training
        if 'extension' in df.columns:
            df['extension_clean'] = df['extension'].apply(master_extension_cleaner)
            df.drop(columns=['extension'], inplace=True)

        # 3. Manual numeric cleaning (Crucial for API/Gradio inputs)
        # Preprocessor expects floats, so we must remove commas/units (KM, Liter, etc.)
        if 'mileage' in df.columns:
            df['mileage'] = df['mileage'].apply(clean_numeric_string)
        
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0)
            
        if 'seats' in df.columns:
            df['seats'] = pd.to_numeric(df['seats'], errors='coerce').fillna(0)

        # 4. Transform the cleaned data using the fitted preprocessor
        # We use .transform() to apply the exact same scaling/encoding as training
        try:
            processed_features = self.preprocessor.transform(df)
        except Exception as e:
            return {"error": f"Preprocessing failed: {str(e)}"}

        # 5. Get prediction on the Log scale (as used in the notebook)
        log_prediction = self.model.predict(processed_features)

        # 6. Apply Inverse Log (expm1) to get the actual price in SAR
        final_price = np.expm1(log_prediction)[0]

        return round(float(final_price), 2)

# Self-test block (Optional)
if __name__ == "__main__":
    # Example input similar to what Gradio or FastAPI would send
    sample_car = {
        "brand": "Toyota",
        "model_name": "Camry",
        "year": 2023,
        "mileage": "15,000 KM", # String with units
        "gear": "Automatic",
        "fuel": "Gasoline",
        "extension": "Full Option", # Raw extension
        "origin": "Saudi",
        "drivetrain": "FWD",
        "cylinders": "4",
        "engine_size": "2.5",
        "seats": 5,
        "condition": "used",
        "exterior_color": "White",
        "interior_color": "Beige"
    }
    
    predictor = CarPredictor()
    result = predictor.predict(sample_car)
    print(f"💰 Predicted Price: {result} SAR")