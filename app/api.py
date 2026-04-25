# app/api.py
from fastapi import FastAPI, UploadFile, File
from typing import List
import joblib
import pandas as pd
import numpy as np
import io
import os
import json
from datetime import datetime
from pydantic import BaseModel
from src.preprocessing import master_extension_cleaner, clean_dataframe
from pymongo import MongoClient
from dotenv import load_dotenv

# This command reads the .env file and makes variables available to os.getenv
load_dotenv() 

APP_ENV = os.getenv("APP_ENV", "local")
MONGO_URI = os.getenv("MONGO_URI")

# Add this line temporarily to verify the mode in the terminal
print(f"DEBUG: System identified environment as: [{APP_ENV}]")

def log_single_request(data: dict, prediction: float):
    """Unified logger for single predictions (Cloud & Local)"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input_data": data,
        "prediction": round(float(prediction), 2)
    }
    
    # 1. Cloud Logging (MongoDB)
    if APP_ENV == "cloud" and MONGO_URI:
        try:
            client = MongoClient(MONGO_URI)
            client.saudi_cars_db.inference_logs.insert_one(log_entry)
        except Exception as e:
            print(f"Cloud Logging Error: {e}")

    # 2. Local Logging (JSONL) - Backup
    log_path = os.path.join(LOG_DIR, "single_predictions.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

app = FastAPI(title="Saudi Car Price API", version="1.3")

# --- LOGGING SETUP ---
LOG_DIR = "logs"
UPLOAD_LOG_DIR = os.path.join(LOG_DIR, "uploaded_files")

# Ensure log directories exist without manual intervention
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(UPLOAD_LOG_DIR, exist_ok=True)

# Load ML artifacts during startup
model = joblib.load('models/car_price_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

class CarInput(BaseModel):
    brand: str
    model_name: str
    year: int
    mileage: float
    gear: str
    fuel: str
    extension: str
    origin: str
    drivetrain: str
    cylinders: str
    engine_size: str
    seats: int
    condition: str
    exterior_color: str
    interior_color: str

@app.post("/predict")
def predict(data: CarInput):
    """Predicts the price for a single car input and logs it."""
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])
    
    # Pre-process extension
    input_df['extension_clean'] = input_df['extension'].apply(master_extension_cleaner)
    input_df.drop(columns=['extension'], inplace=True)
    
    # Transform and Predict
    processed_X = preprocessor.transform(input_df)
    log_pred = model.predict(processed_X)
    final_price = np.expm1(log_pred)[0]
    
    # LOGGING: Record the individual request (Uses the hybrid function above)
    try:
        log_single_request(input_dict, final_price)
    except Exception as e:
        print(f"Logging error: {e}") 
    
    return {"estimated_price_sar": round(float(final_price), 2)}

@app.post("/predict_batch")
def predict_batch(data_list: List[CarInput]):
    """Predicts prices for a list of JSON payloads."""
    input_df = pd.DataFrame([item.dict() for item in data_list])
    
    input_df['extension_clean'] = input_df['extension'].apply(master_extension_cleaner)
    input_df.drop(columns=['extension'], inplace=True)
    
    processed_X = preprocessor.transform(input_df)
    log_preds = model.predict(processed_X)
    final_prices = np.expm1(log_preds)
    
    return {"predictions": [round(float(p), 2) for p in final_prices]}

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Predicts prices from CSV and archives detailed statistical metadata to MongoDB.
    """
    try:
        # Read the file content
        contents = await file.read()
        
        # 1. Local Archiving: Save the raw file with a timestamp for traceability
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{timestamp_str}_{file.filename}"
        with open(os.path.join(UPLOAD_LOG_DIR, archive_name), "wb") as f:
            f.write(contents)

        # 2. Data Loading & Initial Formatting
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df.columns = [col.lower() for col in df.columns]
        if 'extension' in df.columns:
            df.rename(columns={'extension': 'extension_clean'}, inplace=True)
            
        # Clean the dataframe using our custom source logic
        cleaned_df = clean_dataframe(df)
        
        # --- [STATISTICAL METADATA CALCULATION] ---
        
        # Calculate Brand Distribution (e.g., {"Toyota": 50, "Ford": 20})
        brand_dist = cleaned_df['brand'].value_counts().to_dict() if 'brand' in cleaned_df.columns else {}
        
        # Calculate Year Distribution (e.g., {"2025": 10, "2024": 100})
        # Note: MongoDB requires keys to be strings, so we convert numeric years to str
        year_dist = {str(year): int(count) for year, count in cleaned_df['year'].value_counts().items()} if 'year' in cleaned_df.columns else {}
        
        # Calculate Model Distribution (Top 15 most requested models)
        model_dist = cleaned_df['model_name'].value_counts().head(15).to_dict() if 'model_name' in cleaned_df.columns else {}

        # 3. Construct the Expanded Metadata Object
        file_metadata = {
            "filename": archive_name,
            "timestamp": datetime.now().isoformat(),
            "source": "gradio_upload",
            "status": "processed",
            "total_rows": len(cleaned_df),
            "unique_brands": int(cleaned_df['brand'].nunique()) if 'brand' in cleaned_df.columns else 0,
            "unique_models": int(cleaned_df['model_name'].nunique()) if 'model_name' in cleaned_df.columns else 0,
            "unique_years": int(cleaned_df['year'].nunique()) if 'year' in cleaned_df.columns else 0,
            "brand_distribution": brand_dist,
            "year_distribution": year_dist,
            "model_distribution": model_dist
        }

        # 4. Cloud Logging: Send metadata to MongoDB Atlas if in Cloud Mode
        if APP_ENV == "cloud" and MONGO_URI:
            try:
                client = MongoClient(MONGO_URI)
                # Store analytics in the 'uploaded_files_metadata' collection
                client.saudi_cars_db.uploaded_files_metadata.insert_one(file_metadata)
                print(f"✅ Detailed analytics for {file.filename} synced to Cloud.")
            except Exception as e:
                print(f"❌ Cloud Metadata Error: {e}")
        else:
            print(f"🏠 Local Mode: Metadata calculated locally for {file.filename}.")

        # 5. ML Inference: Transform and Predict
        processed_X = preprocessor.transform(cleaned_df)
        log_preds = model.predict(processed_X)
        final_prices = np.expm1(log_preds)
        
        cleaned_df['predicted_price_sar'] = [round(float(p), 2) for p in final_prices]
        
        # Update local batch history log
        with open(os.path.join(LOG_DIR, "batch_history.log"), "a") as f:
            f.write(f"{datetime.now().isoformat()} - Processed file: {file.filename} - Rows: {len(cleaned_df)}\n")

        return {"predictions_list": cleaned_df.to_dict(orient="records")}
        
    except Exception as e:
        return {"error": str(e)}