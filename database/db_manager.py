import sqlite3
import pandas as pd
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# This is the "bridge" that connects your .env file to the script
load_dotenv() 
APP_ENV = os.getenv("APP_ENV", "local")
print(f"DEBUG: The system is currently running in [{APP_ENV}] mode.")
MONGO_URI = os.getenv("MONGO_URI")

# --- Configuration & Environment Setup ---
# APP_ENV should be set to 'cloud' on Render and 'local' on your PC
APP_ENV = os.getenv("APP_ENV", "local")
MONGO_URI = os.getenv("MONGO_URI") # Set this in your .env or Render settings

# Local Path
DB_PATH = os.path.join('database', 'cars_warehouse.db')

def get_mongo_db():
    """Returns the MongoDB database object if in cloud mode."""
    if not MONGO_URI:
        raise ValueError("MONGO_URI environment variable is not set!")
    client = MongoClient(MONGO_URI)
    return client.saudi_cars_db

def init_db():
    """Initializes the SQLite database. (Used locally only)"""
    if APP_ENV == "cloud":
        print("ℹ️ Skipping SQLite init: Using MongoDB Atlas in the cloud.")
        return

    os.makedirs('database', exist_ok=True)
    schema_path = os.path.join('database', 'schema.sql')
    
    if not os.path.exists(schema_path):
        print(f"❌ Error: {schema_path} not found!")
        return

    with sqlite3.connect(DB_PATH) as conn:
        with open(schema_path, 'r') as f:
            conn.executescript(f.read())
    print("✅ Local SQLite database initialized successfully.")

def insert_stage1_data(all_data):
    """Inserts Phase 1 links. Deduplicates by URL."""
    if APP_ENV == "cloud":
        db = get_mongo_db()
        for item in all_data:
            # Use upsert with $setOnInsert to mimic "INSERT OR IGNORE"
            db.cars.update_one(
                {"url": item['Link']},
                {"$setOnInsert": {
                    "price": item['Price'], 
                    "full_text": item['Full_Text'],
                    "scraped_at": pd.Timestamp.now()
                }},
                upsert=True
            )
        print(f"✅ Cloud Phase 1: {len(all_data)} links processed via MongoDB.")
    else:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for item in all_data:
            cursor.execute('''
                INSERT OR IGNORE INTO raw_cars (url, price, full_text)
                VALUES (?, ?, ?)
            ''', (item['Link'], item['Price'], item['Full_Text']))
        conn.commit()
        conn.close()
        print(f"✅ Local Phase 1: {len(all_data)} links processed via SQLite.")

def get_links_to_scrape():
    """Returns URLs where brand info is missing."""
    if APP_ENV == "cloud":
        db = get_mongo_db()
        # Find documents where brand does not exist or is null
        cursor = db.cars.find({"brand": {"$exists": False}}, {"url": 1})
        return [doc['url'] for doc in cursor]
    else:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT url FROM raw_cars WHERE brand IS NULL")
        links = [row[0] for row in cursor.fetchall()]
        conn.close()
        return links

def update_car_details(url, details):
    """Updates car record with full technical features."""
    if APP_ENV == "cloud":
        db = get_mongo_db()
        db.cars.update_one({"url": url}, {"$set": details})
    else:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query = '''
            UPDATE raw_cars SET 
            brand = ?, model_name = ?, year = ?, exterior_color = ?, 
            interior_color = ?, origin = ?, fuel = ?, gear = ?, 
            cylinders = ?, condition = ?, engine_size = ?, mileage = ?, 
            drivetrain = ?, seats = ?, extension_clean = ?
            WHERE url = ?
        '''
        cursor.execute(query, (
            details.get('brand'), details.get('model_name'), details.get('year'),
            details.get('exterior_color'), details.get('interior_color'), 
            details.get('origin'), details.get('fuel'), details.get('gear'),
            details.get('cylinders'), details.get('condition'), details.get('engine_size'),
            details.get('mileage'), details.get('drivetrain'), details.get('seats'),
            details.get('extension_clean'), url
        ))
        conn.commit()
        conn.close()

def migrate_csv_to_sql(csv_path):
    """Migrates legacy CSV data into the active database (SQLite or MongoDB)."""
    if not os.path.exists(csv_path):
        print(f"⚠️ Warning: {csv_path} not found. Skipping migration.")
        return
        
    df = pd.read_csv(csv_path)
    mapping = {
        'URL': 'url', 'Brand': 'brand', 'Model_Name': 'model_name', 'Year': 'year',
        'Exterior_Color': 'exterior_color', 'Interior_Color': 'interior_color',
        'Origin': 'origin', 'Fuel': 'fuel', 'Gear': 'gear', 'Cylinders': 'cylinders',
        'Condition': 'condition', 'Engine_Size': 'engine_size', 'Mileage': 'mileage',
        'Drivetrain': 'drivetrain', 'Seats': 'seats', 'Cash_Price': 'price', 
        'Extension': 'extension_clean'
    }
    
    available_cols = [col for col in mapping.keys() if col in df.columns]
    df_refined = df[available_cols].rename(columns=mapping)
    df_refined['is_trained'] = 1
    
    if APP_ENV == "cloud":
        db = get_mongo_db()
        # Convert DataFrame to list of dictionaries for MongoDB
        records = df_refined.to_dict(orient='records')
        if records:
            db.cars.insert_many(records)
            print(f"✅ Migrated {len(records)} records to MongoDB Atlas.")
    else:
        with sqlite3.connect(DB_PATH) as conn:
            df_refined.to_sql('raw_cars', conn, if_exists='append', index=False)
        print(f"✅ Migrated {len(df_refined)} records to Local SQLite.")

def get_untrained_count():
    """Returns the number of records where is_trained = 0 (Hybrid)."""
    if APP_ENV == "cloud":
        db = get_mongo_db()
        # MongoDB: Count documents where is_trained is either 0 or does not exist
        return db.cars.count_documents({"is_trained": {"$in": [0, False, None]}})
    else:
        if not os.path.exists(DB_PATH): return 0
        with sqlite3.connect(DB_PATH) as conn:
            # We use a context manager (with) to ensure connection closes
            cursor = conn.cursor()
            return cursor.execute("SELECT COUNT(*) FROM raw_cars WHERE is_trained = 0").fetchone()[0]

def get_total_count():
    """Returns the total number of car records (Hybrid)."""
    if APP_ENV == "cloud":
        db = get_mongo_db()
        return db.cars.count_documents({})
    else:
        if not os.path.exists(DB_PATH): return 0
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            return cursor.execute("SELECT COUNT(*) FROM raw_cars").fetchone()[0]

def is_db_empty():
    """Checks if the active database has no data (Hybrid)."""
    return get_total_count() == 0

def mark_as_trained():
    """Updates all records to is_trained = 1 after a successful run (Hybrid)."""
    if APP_ENV == "cloud":
        db = get_mongo_db()
        db.cars.update_many({}, {"$set": {"is_trained": 1}})
    else:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("UPDATE raw_cars SET is_trained = 1")
    print("✅ All records marked as trained in the database.")        