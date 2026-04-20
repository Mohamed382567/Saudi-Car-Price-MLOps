import sqlite3
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import json
import os
import mlflow
import mlflow.xgboost
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.preprocessing import create_preprocessor, clean_dataframe
from pymongo import MongoClient

# Environment Setup
APP_ENV = os.getenv("APP_ENV", "local")
MONGO_URI = os.getenv("MONGO_URI")
DB_PATH = 'database/cars_warehouse.db'

def load_data_from_db():
    """
    Fetches data from either MongoDB Atlas (Cloud) or SQLite (Local).
    """
    if APP_ENV == "cloud":
        print("🌐 Connecting to MongoDB Atlas for training data...")
        client = MongoClient(MONGO_URI)
        db = client.saudi_cars_db
        # Filter for cars that have a price (target variable)
        df = pd.DataFrame(list(db.cars.find({"price": {"$ne": None}})))
        # Remove MongoDB internal ID for processing
        if not df.empty and '_id' in df.columns:
            df = df.drop(columns=['_id'])
    else:
        print("🏠 Connecting to local SQLite for training data...")
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM raw_cars WHERE price IS NOT NULL ORDER BY id ASC", conn)
        conn.close()

    if df.empty:
        return df

    # 1. General Formatting (Title Case & Strip Spaces)
    title_cols = ['brand', 'fuel', 'gear', 'origin', 'exterior_color', 'model_name']
    for col in title_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    
    # 2. Condition: MUST be lowercase (used, new)
    if 'condition' in df.columns:
        df['condition'] = df['condition'].astype(str).str.strip().str.lower()

    # 3. Model Name: Fix specific uppercase models (LX, IS, RAV4, etc.)
    if 'model_name' in df.columns:
        upper_models = {'Lx': 'LX', 'Is': 'IS', 'Nx': 'NX', 'Es': 'ES', 'Rx': 'RX', 'Ux': 'UX', 'Rav4': 'RAV4'}
        df['model_name'] = df['model_name'].replace(upper_models)

    # 4. Acronyms Mapping (Brands & Gear)
    acronyms = {
        'Bmw': 'BMW', 'Byd': 'BYD', 'Mg': 'MG', 'Gmc': 'GMC', 
        'Cvt': 'CVT', 'Fwd': 'FWD', 'Awd': 'AWD', 'Rwd': 'RWD'
    }
    df['brand'] = df['brand'].replace(acronyms)
    df['gear'] = df['gear'].replace(acronyms)

    print("🎯 Data formatted with surgical precision to match Notebook patterns.")
    return df

def get_all_metrics(y_train_real, y_train_pred, y_test_real, y_test_pred):
    return {
        'Train_R2_Score': r2_score(y_train_real, y_train_pred) * 100,
        'Test_R2_Score': r2_score(y_test_real, y_test_pred) * 100,
        'Train_MAE_SAR': mean_absolute_error(y_train_real, y_train_pred),
        'Test_MAE_SAR': mean_absolute_error(y_test_real, y_test_pred),
        'Train_RMSE_SAR': np.sqrt(mean_squared_error(y_train_real, y_train_pred)),
        'Test_RMSE_SAR': np.sqrt(mean_squared_error(y_test_real, y_test_pred))
    }

def run_trainer():
    print("🚀 Training Final Model (Notebook Architecture)...")
    
    # Initialize DagsHub/MLflow tracking
    from src.tuning import setup_dagshub
    setup_dagshub()
    mlflow.set_experiment("Final_Model_Production")

    with mlflow.start_run(run_name="XGB_Production_Pipeline"):
        # Step 1: Load and Clean
        raw_df = load_data_from_db()
        df = clean_dataframe(raw_df)
        
        # --- THE GOLDEN NOTEBOOK PARAMETERS (Baseline) ---
        params = {
            'n_estimators': 2000,
            'learning_rate': 0.05,
            'max_depth': 4,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 10,
            'random_state': 42,
            'tree_method': 'hist',
            'early_stopping_rounds': 50 
        }

        # --- THE OPTUNA CONDITION ---
        # Update parameters only if Optuna tuning was previously executed
        params_path = 'models/best_params.json'
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                optuna_params = json.load(f)
                params.update(optuna_params)
                print(f"📝 Optimized parameters loaded from {params_path}")
        else:
            print("⚠️ Optuna params not found. Using default notebook parameters.")

        # Feature/Target Split
        X = df.drop(columns=['id', 'url', 'price', 'full_text', 'scraped_at', 'is_trained'], errors='ignore')
        y = np.log1p(df['price'])

        from sklearn.model_selection import train_test_split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Preprocessing
        preprocessor = create_preprocessor()
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        # Initialize and Fit
        final_model = xgb.XGBRegressor(**params)
        final_model.fit(
            X_train, y_train, 
            eval_set=[(X_test, y_test)], 
            verbose=False
        )

        # Evaluation (Inverse Log to Real SAR)
        y_test_pred = np.expm1(final_model.predict(X_test))
        y_test_real = np.expm1(y_test)
        y_train_pred = np.expm1(final_model.predict(X_train))
        y_train_real = np.expm1(y_train)
        
        metrics = get_all_metrics(y_train_real, y_train_pred, y_test_real, y_test_pred)
        
        # --- [ARTIFACTS & REGISTRATION] ---
        # Log Metrics and Parameters
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        # Log and Register the model in DagsHub Model Registry
        mlflow.xgboost.log_model(
            xgb_model=final_model, 
            artifact_path="car_price_model_prod",
            registered_model_name="Saudi_Car_Price_Predictor"
        )
        
        # Save local copies as backup
        if not os.path.exists('models'): os.makedirs('models')
        joblib.dump(final_model, 'models/car_price_model.pkl')
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        
        # --- [ENVIRONMENT-AWARE DB UPDATE] ---
        # Mark records as trained in the appropriate database
        try:
            if APP_ENV == "cloud" and MONGO_URI:
                client = MongoClient(MONGO_URI)
                db = client.saudi_cars_db
                update_result = db.cars.update_many({"is_trained": 0}, {"$set": {"is_trained": 1}})
                print(f"🌐 Cloud: Marked {update_result.modified_count} cars as trained in MongoDB.")
            else:
                conn = sqlite3.connect(DB_PATH)
                conn.execute("UPDATE raw_cars SET is_trained = 1")
                conn.commit()
                conn.close()
                print("🏠 Local: All cars marked as trained in SQLite.")
        except Exception as e:
            print(f"⚠️ Database Update Warning: {e}")

        print("\n📊 Final Performance Report:")
        print(f"✅ R2 Test Score: {metrics['Test_R2_Score']:.2f}%")
        print(f"📉 RMSE (SAR): {metrics['Test_RMSE_SAR']:,.2f}")

if __name__ == "__main__":
    run_trainer()