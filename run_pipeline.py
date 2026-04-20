import asyncio
import os
import json
from pymongo import MongoClient # Added for cloud metadata persistence
# Import the helper functions we just added to db_manager
from database.db_manager import (
    init_db, 
    migrate_csv_to_sql, 
    is_db_empty, 
    get_untrained_count, 
    get_total_count, 
    mark_as_trained
)
from scraper.engine import run_phase1, run_phase2
from src.tuning import run_tuner
from src.trainer import run_trainer

# --- Configuration & Environment Setup ---
APP_ENV = os.getenv("APP_ENV", "local")
MONGO_URI = os.getenv("MONGO_URI")
# Minimum number of new cars required to trigger a re-train
TRAINING_THRESHOLD = 500
# Path to store the state of the last tuning session (Local only)
METADATA_PATH = 'models/pipeline_metadata.json'

def get_last_tuned_count_metadata():
    """ 
    Fetches the last_tuned_count based on the environment.
    Cloud: MongoDB Atlas | Local: pipeline_metadata.json 
    """
    if APP_ENV == "cloud" and MONGO_URI:
        try:
            client = MongoClient(MONGO_URI)
            db = client.saudi_cars_db
            config = db.pipeline_config.find_one({"type": "tuning_state"})
            return config.get("last_tuned_count", 0) if config else 0
        except Exception as e:
            print(f"⚠️ Cloud Metadata Fetch Error: {e}")
            return 0
    else:
        # Local Logic
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
                return metadata.get("last_tuned_count", 0)
        return 0

def update_last_tuned_count_metadata(total_count):
    """ 
    Updates the last_tuned_count based on the environment.
    Cloud: MongoDB Atlas | Local: pipeline_metadata.json 
    """
    if APP_ENV == "cloud" and MONGO_URI:
        try:
            client = MongoClient(MONGO_URI)
            db = client.saudi_cars_db
            db.pipeline_config.update_one(
                {"type": "tuning_state"},
                {"$set": {"last_tuned_count": total_count}},
                upsert=True
            )
            print(f"📡 Cloud Metadata Updated: New base count is {total_count}")
        except Exception as e:
            print(f"⚠️ Cloud Metadata Update Error: {e}")
    else:
        # Local Logic
        if not os.path.exists('models'): os.makedirs('models')
        with open(METADATA_PATH, 'w') as f:
            json.dump({"last_tuned_count": total_count}, f)
        print(f"🏠 Local Metadata Updated: New base count is {total_count}")


async def start_mlops_pipeline():
    """ 
    Smart MLOps Pipeline: 
    1. Initializes DB and migrates legacy CSV data if necessary.
    2. Performs incremental scraping to fetch the latest car listings.
    3. Executes a 'Smart Tuning' decision: runs Optuna only if data has grown by 50%.
    4. Re-trains the model to incorporate new data.
    """
    
    # --- Step 1: Environment Setup ---
    print("🛠️  Step 1: Initializing database...")
    init_db()
    
    # --- Step 2: Zero-Redundancy Migration ---
    csv_path = 'data/raw/final_cleaned_scraped_cars.csv'
    if is_db_empty() and os.path.exists(csv_path):
        print("📂 DB is empty. Migrating legacy CSV data...")
        migrate_csv_to_sql(csv_path)
    else:
        print("✅ Skipping migration: Database already populated.")

    # --- Step 3: Incremental Scraping ---
    print("🕷️ Step 3: Checking Syarah.com for new listings...")
    await run_phase1(pages=2) 
    
    print("🔍 Step 4: Fetching technical details for new entries...")
    run_phase2()

    # --- Step 5: Smart Execution Logic ---
    new_data_count = get_untrained_count()
    total_count = get_total_count()
    print(f"📊 New untrained records found: {new_data_count}")

    if new_data_count >= TRAINING_THRESHOLD:
        print(f"🧪 Training threshold reached ({TRAINING_THRESHOLD}). Evaluating tuning trigger...")
        
        # --- Environment-Aware 50% Growth Trigger Logic ---
        last_tuned_count = get_last_tuned_count_metadata()
        
        # Calculate growth ratio since the last Optuna run
        increase_ratio = (total_count - last_tuned_count) / last_tuned_count if last_tuned_count > 0 else 1.0
        
        if increase_ratio >= 0.5:
            print(f"📈 Growth of {increase_ratio:.1%} detected. Triggering Optuna Hyperparameter Tuner...")
            run_tuner()
            
            # Update metadata memory after successful tuning (Cloud or Local)
            update_last_tuned_count_metadata(total_count)
        else:
            print(f"ℹ️ Growth ({increase_ratio:.1%}) is below 50%. Skipping Tuner, using stable baseline parameters.")

        # --- Step 6: Final Model Training ---
        # Re-train the model with the latest data using the best known parameters
        run_trainer()
        print("🚀 Model successfully updated and logged to DagsHub/MLflow.")
    else:
        print(f"😴 Insufficient new data ({new_data_count}/{TRAINING_THRESHOLD}). Skipping training cycle.")

    print("🏁 Pipeline execution complete!")

if __name__ == "__main__":
    asyncio.run(start_mlops_pipeline())