import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import plotly.express as px
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv() 

APP_ENV = os.getenv("APP_ENV", "local")
MONGO_URI = os.getenv("MONGO_URI")

# Debugging: Verify current environment mode
print(f"DEBUG: System identified environment as: [{APP_ENV}]")

# Local Path Configuration
DB_PATH = "database/cars_warehouse.db"
LOG_SINGLE = "logs/single_predictions.jsonl"
BATCH_LOG_FILE = "logs/batch_history.log"
BATCH_DIR = "logs/uploaded_files"

st.set_page_config(page_title="Saudia Cars - MLOps Monitoring", layout="wide")

# --- Cloud-Aware Helper Functions ---

def get_db_connection():
    """Establishes connection to MongoDB Atlas if in cloud mode."""
    if APP_ENV == "cloud":
        client = MongoClient(MONGO_URI)
        return client.saudi_cars_db
    return None

def count_uploaded_files():
    """Counts processed files from MongoDB metadata (Cloud) or local directory (Local)."""
    if APP_ENV == "cloud":
        try:
            db = get_db_connection()
            # Count documents in the unified metadata collection
            return db.uploaded_files_metadata.count_documents({})
        except Exception as e:
            st.sidebar.error(f"Cloud Count Error: {e}")
            return 0
    else:
        if not os.path.exists(BATCH_DIR):
            return 0
        files = [f for f in os.listdir(BATCH_DIR) 
                 if os.path.isfile(os.path.join(BATCH_DIR, f)) and not f.endswith('.log')]
        return len(files)

def load_batch_history():
    """
    Unified Event History:
    Cloud: Formats records from uploaded_files_metadata.
    Local: Reads from the local batch_history.log file.
    """
    if APP_ENV == "cloud":
        try:
            db = get_db_connection()
            # Fetch latest 20 uploads to act as the activity log
            logs = list(db.uploaded_files_metadata.find().sort("timestamp", -1).limit(20))
            if logs:
                return "\n".join([
                    f"{l.get('timestamp')[:19]} - SUCCESS: {l.get('filename')} "
                    f"({l.get('total_rows', 0)} rows processed)" 
                    for l in logs
                ])
            return "No cloud upload history found."
        except Exception as e:
            return f"Cloud History Error: {e}"
    else:
        # Legacy Local Logic
        if os.path.exists(BATCH_LOG_FILE):
            with open(BATCH_LOG_FILE, "r") as f:
                return f.read()
        return "batch_history.log not found."

def load_upload_metadata():
    """Fetches detailed statistical metadata for Tab 4 analysis."""
    if APP_ENV == "cloud":
        db = get_db_connection()
        return list(db.uploaded_files_metadata.find().sort("timestamp", -1).limit(10))
    return []

def load_training_stats():
    """Loads baseline car data from MongoDB (Cloud) or SQLite (Local)."""
    if APP_ENV == "cloud":
        db = get_db_connection()
        data = list(db.cars.find({}, {"brand": 1, "year": 1, "price": 1, "scraped_at": 1}))
        df = pd.DataFrame(data)
        if not df.empty:
            df['scraped_at'] = pd.to_datetime(df['scraped_at'])
        return df
    else:
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT brand, year, price, scraped_at FROM raw_cars", conn)
            conn.close()
            df['scraped_at'] = pd.to_datetime(df['scraped_at'])
            return df
    return pd.DataFrame()

def load_inference_logs():
    """Loads single prediction logs for drift and activity tracking."""
    if APP_ENV == "cloud":
        db = get_db_connection()
        data = list(db.inference_logs.find().sort("timestamp", -1))
        flat_data = []
        for record in data:
            row = record.get('input_data', {}).copy()
            row['log_timestamp'] = record.get('timestamp')
            row['prediction_result'] = record.get('prediction')
            flat_data.append(row)
        df = pd.DataFrame(flat_data)
    else:
        flat_data = []
        if os.path.exists(LOG_SINGLE):
            with open(LOG_SINGLE, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        row = record.get('input_data', {}).copy()
                        row['log_timestamp'] = record.get('timestamp')
                        row['prediction_result'] = record.get('prediction')
                        flat_data.append(row)
                    except: continue
        df = pd.DataFrame(flat_data)
    
    if not df.empty:
        df['log_timestamp'] = pd.to_datetime(df['log_timestamp'])
    return df

# --- UI Layout ---
st.title("🚗 Saudi Cars MLOps Monitoring Dashboard")
st.markdown("Real-time observability for car price predictions.")

# Initialize Data Sources
train_df = load_training_stats()
logs_df = load_inference_logs()
num_uploaded = count_uploaded_files()
upload_meta = load_upload_metadata()

# --- Top Metrics Row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Scraped Cars", f"{len(train_df):,}")
col2.metric("Live Predictions", len(logs_df))
col3.metric("Batch Files", num_uploaded)
col4.metric("System Health", "Active ✅")

# --- Tabs Configuration ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Activity Logs", "📈 Data Drift Detection", "📊 Training Database", "📂 Upload Analytics"])

# --- Tab 1: Activity Logs (Hybrid Cloud/Local) ---
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Flattened Prediction Logs")
        if not logs_df.empty:
            st.dataframe(logs_df, use_container_width=True)
        else:
            st.info("No prediction logs found yet.")
    with c2:
        st.subheader("Batch Event History")
        # Pull history from cloud metadata or local log file
        history_text = load_batch_history()
        st.text_area("Latest Operations Log", history_text, height=400)

# --- Tab 2: Feature Drift (Distribution Shift Analysis) ---
with tab2:
    st.subheader("Feature Drift: Year Analysis")
    if not logs_df.empty and not train_df.empty and 'year' in logs_df.columns:
        train_dist = train_df[['year']].copy()
        train_dist['Source'] = 'Baseline (Training Data)'
        
        live_dist = logs_df[['year']].copy()
        live_dist['Source'] = 'Live Traffic (JSON Logs)'
        
        combined = pd.concat([train_dist, live_dist])
        fig = px.histogram(combined, x="year", color="Source", barmode="overlay", 
                           title="Distribution Shift: Car Year", marginal="box")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Awaiting more diversified JSON logs to visualize Drift.")

# --- Tab 3: Database Insights ---
with tab3:
    if not train_df.empty:
        st.subheader("Database Overview (Market Content)")
        brand_fig = px.bar(train_df['brand'].value_counts().head(15), 
                           title="Top 15 Brands in System")
        st.plotly_chart(brand_fig, use_container_width=True)

# --- Tab 4: Detailed Upload Analytics (Cloud Metadata Only) ---
with tab4:
    st.subheader("Statistical Analytics for User-Uploaded CSVs")
    if APP_ENV == "cloud" and upload_meta:
        # File selector for specific batch analysis
        file_options = {f"{m.get('filename', 'Unknown')} ({m.get('timestamp', 'N/A')[:16]})": m for m in upload_meta}
        selected_file_label = st.selectbox("Select an uploaded file to analyze:", list(file_options.keys()))
        
        selected_data = file_options[selected_file_label]
        
        # Display Batch Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows Processed", selected_data.get('total_rows', 0))
        m2.metric("Unique Brands", selected_data.get('unique_brands', 0))
        m3.metric("Unique Models", selected_data.get('unique_models', 0))
        m4.metric("Unique Years", selected_data.get('unique_years', 0))
        
        st.divider()
        
        # Visualizing Batch Distributions (Pie and Bar Charts)
        c1, c2 = st.columns(2)
        
        with c1:
            # Brand Share Analysis
            brand_dist = selected_data.get('brand_distribution', {})
            if brand_dist:
                fig_brand = px.pie(names=list(brand_dist.keys()), values=list(brand_dist.values()), 
                                   title="Brand Share in Selected File", hole=0.3)
                st.plotly_chart(fig_brand, use_container_width=True)
        
        with c2:
            # Year/Demand Analysis (Visualizing 2025+ models)
            year_dist = selected_data.get('year_distribution', {})
            if year_dist:
                sorted_years = dict(sorted(year_dist.items()))
                fig_year = px.bar(x=list(sorted_years.keys()), y=list(sorted_years.values()), 
                                  labels={'x': 'Manufacturing Year', 'y': 'Count'},
                                  title="Year Distribution (Identify 2025 Trends)")
                st.plotly_chart(fig_year, use_container_width=True)
                
        # Popular Models Visualization
        model_dist = selected_data.get('model_distribution', {})
        if model_dist:
            fig_model = px.bar(x=list(model_dist.keys()), y=list(model_dist.values()), 
                               title="Top 15 Most Frequent Models in Batch",
                               labels={'x': 'Model Name', 'y': 'Frequency'})
            st.plotly_chart(fig_model, use_container_width=True)
            
    elif APP_ENV == "local":
        st.warning("Advanced Upload Analytics are only available in Cloud Mode (MongoDB Atlas).")
    else:
        st.info("No CSV upload metadata found in the cloud yet.")

# --- Sidebar Controls ---
st.sidebar.header("System Settings")
if st.sidebar.button("Manual Refresh"):
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(f"Last DB Update: {train_df['scraped_at'].max() if not train_df.empty else 'N/A'}")