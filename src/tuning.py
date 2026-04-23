import optuna
import xgboost as xgb
import json
import os
import mlflow
import dagshub
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.preprocessing import create_preprocessor, clean_dataframe
from src.trainer import load_data_from_db

# DagsHub Configuration
REPO_OWNER = os.getenv("DAGSHUB_USERNAME") 
REPO_NAME = os.getenv("REPO_NAME")

def setup_dagshub():
    """ Connects the local script to the DagsHub MLflow remote server. """
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")
    mlflow.set_experiment("Hyperparameter_Tuning_RMSE")

def objective(trial, X_train, y_train, w_train, X_val, y_val):
    """ 
    Optuna objective function tightly constrained around the Notebook's best parameters 
    to prevent Overfitting while adapting to new scraped data.
    """
    with mlflow.start_run(nested=True):
        params = {
            'n_estimators': 2000,
            
            # --- CONSTRAINED SEARCH SPACE (Notebook-Informed) ---
            
            # Notebook value: 4. Range: 3 to 5 (Keeps trees shallow to prevent overfitting)
            'max_depth': trial.suggest_int('max_depth', 3, 5), 
            
            # Notebook value: 5. Range: 4 to 7
            'min_child_weight': trial.suggest_int('min_child_weight', 4, 7),
            
            # Notebook value: 0.05. Range: 0.03 to 0.08
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.08),
            
            # Notebook value: 0.8. Range: 0.7 to 0.9
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            
            # Notebook value: 0.8. Range: 0.7 to 0.9
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            
            # Notebook value: 10. Range: 8.0 to 15.0 (High regularization)
            'reg_lambda': trial.suggest_float('reg_lambda', 8.0, 15.0), 
            
            'early_stopping_rounds': 50,
            'tree_method': 'hist',
            'random_state': 42
        }
        
        mlflow.log_params(params)
        model = xgb.XGBRegressor(**params)
        
        # Train with custom weights (if you want to keep punishing luxury/new car errors)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight=w_train, verbose=False)
        
        preds_log = model.predict(X_val)
        
        # Revert log values back to real Saudi Riyals (SAR) for evaluation
        preds_real = np.expm1(preds_log)
        y_val_real = np.expm1(y_val)
        
        rmse_real_money = np.sqrt(mean_squared_error(y_val_real, preds_real))
        mlflow.log_metric("rmse_sar", rmse_real_money)
        return rmse_real_money 

def run_tuner():
    setup_dagshub()
    print("🔍 Starting Hyperparameter Optimization (Notebook-Constrained Bounds)...")
    
    df = clean_dataframe(load_data_from_db())
    
    # Define Sample Weights (Keeps your logic for Lexus and Camry 2023)
    weights = np.ones(len(df))
    weights[df['year'] >= 2023] = 3.0 
    weights[df['price'] > 200000] = 5.0 
    
    X = df.drop(columns=['id', 'url', 'price', 'full_text', 'scraped_at', 'is_trained'], errors='ignore')
    y = np.log1p(df['price'])
    
    X_train_raw, X_val_raw, y_train, y_val, w_train, w_val = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    
    preprocessor = create_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    with mlflow.start_run(run_name="Optuna_Notebook_Constrained"):
        study = optuna.create_study(direction='minimize')
        
        study.optimize(lambda trial: objective(trial, X_train, y_train, w_train, X_val, y_val), n_trials=20)

        if not os.path.exists('models'): os.makedirs('models')
        with open('models/best_params.json', 'w') as f:
            json.dump(study.best_params, f)
            
        mlflow.log_params(study.best_params)
        print(f"✨ Tuning complete. Parameters safely optimized around baseline.")

if __name__ == "__main__":
    run_tuner()
