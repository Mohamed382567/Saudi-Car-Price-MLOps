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
import mlflow # Added for cloud fetch fallback

# This command reads the .env file and makes variables available to os.getenv
load_dotenv() 

APP_ENV = os.getenv("APP_ENV", "local")
MONGO_URI = os.getenv("MONGO_URI")
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
REPO_NAME = os.getenv("REPO_NAME")

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

# --- SMART ARTIFACT LOADING ---
model = None
preprocessor = None

def load_artifacts():
    global model, preprocessor
    model_path = 'models/car_price_model.pkl'
    pre_path = 'models/preprocessor.pkl'

    # 1. Attempt Local Load (Primary for Docker/Render)
    if os.path.exists(model_path) and os.path.exists(pre_path):
        try:
            model = joblib.load(model_path)
            preprocessor = joblib.load(pre_path)
            print("✅ Artifacts loaded from local storage.")
            return
        except Exception as e:
            print(f"⚠️ Local load failed: {e}")

    # 2. Attempt Cloud Fetch (Fallback for development/maintenance)
    if APP_ENV == "cloud" and DAGSHUB_USERNAME and REPO_NAME:
        try:
            print("📥 Local artifacts missing. Fetching from DagsHub Registry...")
            mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow")
            model = mlflow.xgboost.load_model(f"models:/Saudi_Car_Price_Predictor/Production")
            # For preprocessor, we'll try a local fallback if cloud registry only stores the model
            print("✨ Cloud model loaded successfully.")
        except Exception as e:
            print(f"❌ Cloud fetch failed: {e}")

    if model is None:
        print("🛑 WARNING: Model not found. Predictions will be disabled.")

# Trigger loading on startup
load_artifacts()

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
    if model is None or preprocessor is None:
        return {"error": "Model not loaded. Service is in maintenance mode."}, 503

    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])
    
    # Pre-process extension
    input_df['extension_clean'] = input_df['extension'].apply(master_extension_cleaner)
    input_df.drop(columns=['extension'], inplace=True)
    
    # Transform and Predict
    processed_X = preprocessor.transform(input_df)
    log_pred = model.predict(processed_X)
    final_price = np.expm1(log_pred)[0]
    
    # LOGGING: Record the individual request
    try:
        log_single_request(input_dict, final_price)
    except Exception as e:
        print(f"Logging error: {e}") 
    
    return {"estimated_price_sar": round(float(final_price), 2)}

@app.post("/predict_batch")
def predict_batch(data_list: List[CarInput]):
    """Predicts prices for a list of JSON payloads."""
    if model is None or preprocessor is None:
        return {"error": "Model not loaded."}, 503

    input_df = pd.DataFrame([item.dict() for item in data_list])
    
    input_df['extension_clean'] = input_df['extension'].apply(master_extension_cleaner)
    input_df.drop(columns=['extension'], inplace=True)
    
    processed_X = preprocessor.transform(input_df)
    log_preds = model.predict(processed_X)
    final_prices = np.expm1(log_preds)
    
    return {"predictions": [round(float(p), 2) for p in final_prices]}

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """Predicts prices from CSV and archives metadata."""
    if model is None or preprocessor is None:
        return {"error": "Model not loaded."}, 503

    try:
        contents = await file.read()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{timestamp_str}_{file.filename}"
        with open(os.path.join(UPLOAD_LOG_DIR, archive_name), "wb") as f:
            f.write(contents)

        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df.columns = [col.lower() for col in df.columns]
        if 'extension' in df.columns:
            df.rename(columns={'extension': 'extension_clean'}, inplace=True)
            
        cleaned_df = clean_dataframe(df)
        
        # Statistics logic (as before)
        brand_dist = cleaned_df['brand'].value_counts().to_dict() if 'brand' in cleaned_df.columns else {}
        year_dist = {str(year): int(count) for year, count in cleaned_df['year'].value_counts().items()} if 'year' in cleaned_df.columns else {}
        model_dist = cleaned_df['model_name'].value_counts().head(15).to_dict() if 'model_name' in cleaned_df.columns else {}

        file_metadata = {
            "filename": archive_name,
            "timestamp": datetime.now().isoformat(),
            "source": "gradio_upload",
            "status": "processed",
            "total_rows": len(cleaned_df),
            "brand_distribution": brand_dist,
            "year_distribution": year_dist,
            "model_distribution": model_dist
        }

        if APP_ENV == "cloud" and MONGO_URI:
            try:
                client = MongoClient(MONGO_URI)
                client.saudi_cars_db.uploaded_files_metadata.insert_one(file_metadata)
            except Exception as e:
                print(f"❌ Cloud Metadata Error: {e}")

        # ML Inference
        processed_X = preprocessor.transform(cleaned_df)
        log_preds = model.predict(processed_X)
        final_prices = np.expm1(log_preds)
        
        cleaned_df['predicted_price_sar'] = [round(float(p), 2) for p in final_prices]
        
        return {"predictions_list": cleaned_df.to_dict(orient="records")}
        
    except Exception as e:
        return {"error": str(e)}
