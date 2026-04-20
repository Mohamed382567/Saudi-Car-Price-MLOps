-- Cleaned Schema: Focusing only on the most available features
CREATE TABLE IF NOT EXISTS raw_cars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE,               -- Primary key for deduplication
    full_text TEXT,                -- Used for Phase 1 price extraction
    brand TEXT,
    model_name TEXT,
    year INTEGER,
    exterior_color TEXT,
    interior_color TEXT,
    origin TEXT,
    fuel TEXT,
    gear TEXT,
    cylinders TEXT,
    condition TEXT,
    engine_size TEXT,
    mileage TEXT,
    drivetrain TEXT,
    seats REAL,
    extension_clean TEXT,          -- The cleaned version of car trim
    price REAL,                    -- The target variable (Cash_Price)
    is_trained INTEGER DEFAULT 0,  -- MLOps Tracking
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);