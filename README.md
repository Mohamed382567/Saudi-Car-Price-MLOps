# 🚗 Saudi Car Price Predictor: End-to-End MLOps Lifecycle

A production-grade MLOps project designed to scrape, train, and deploy a machine learning model for predicting car prices in the Saudi Arabian market. This project demonstrates a complete lifecycle—from automated data ingestion and strategy discovery to dynamic model tuning and cloud-based deployment.

---

## 🌟 Key Features
* **Automated Data Ingestion:** Asynchronous **Playwright & BeautifulSoup scraper** fetching real-time listings every 3 days.
* **Environment-Aware Architecture:** Seamlessly switches between Local development (SQLite) and Cloud production (MongoDB Atlas) based on the environment configuration.
* **Smart Training Gate:** The pipeline only triggers retraining when a threshold of **500 new unique records** is reached to ensure data quality and resource efficiency.
* **Dynamic Hyperparameter Tuning:** Automatically triggers `tuning.py` (Optuna) for re-optimization if the new data volume increases the dataset size by more than 50%.
* **Hybrid Artifact Strategy:** A robust CI/CD design that manages the transition from local preprocessing to cloud-coupled model registry artifacts.
* **Operational Monitoring:** A dedicated **Streamlit** dashboard to track market trends, data distribution, and inference logs.

---

## 🏗 System Architecture & Strategy

### 1. Research & Strategy Lab (The Notebook)
The file `MLCarsProjectNotebook.ipynb` serves as the primary **Research & Strategy Discovery** document. It is the foundation of the project and contains:
* **Exploratory Data Analysis (EDA):** Deep dives into the Saudi car market to identify key price drivers.
* **Strategy Justification:** Detailed explanations of the data cleaning steps, handling of Arabic car terminology, and the selection of the XGBoost architecture.
* **Baseline Validation:** The initial "Golden Parameters" were tested and validated here before being moved to the production pipeline.

### 2. The Hybrid Artifact Evolution ($v1$ vs. $v2$)
This project implements a sophisticated "Forward-Compatible" strategy to handle the evolution of model artifacts:

* **The $v1$ Strategy (Legacy Fallback):** Due to an initial development oversight where the preprocessor was not registered in the cloud, we adopted a **Hybrid Fallback**. The system stores a `preprocessor.pkl` file locally in the GitHub repository to ensure the first version of the model remains functional.
* **The $v2$ & Beyond Strategy (Cloud Coupling):** The training pipeline was subsequently upgraded. For every new version ($v2, v3, \dots$), the **Model** and its matching **Preprocessor** are coupled and uploaded as a single package to DagsHub.
* **The CI/CD Logic:** Our GitHub Actions workflow is programmed to prioritize cloud artifacts. It attempts to pull the coupled pair from DagsHub ($v2+$ logic); if it fails, it seamlessly falls back to the local repository preprocessor ($v1$ logic), ensuring 0% downtime during transitions.

### 3. Automated Workflow Logic
1.  **Ingestion:** The scraper collects listings and evaluates the "500-record" training gate.
2.  **Optimization:** If the data volume has significantly shifted (>50%), `tuning.py` is executed to find new optimal hyperparameters.
3.  **Registration:** New models are versioned and stored in the **DagsHub Model Registry**.
4.  **Deployment:** Docker images are rebuilt and pushed to Render only when a new model is successfully verified through automated tests.


### 4. Data Orchestration & Persistence (MongoDB Atlas)
MongoDB Atlas serves as the centralized backbone for the entire data lifecycle and pipeline control:
* **Inbound Ingestion:** All newly scraped vehicle data is automatically stored in MongoDB, serving as our primary "Data Warehouse."
* **Single Source of Truth:** Both training and testing pipelines pull their datasets directly from MongoDB collections to ensure the model is trained on the most up-to-date cloud data.
* **Pipeline Configuration & Control:** The logic that determines when **Optuna** (`tuning.py`) is triggered is governed by a `pipeline_config` collection in MongoDB. This collection stores critical metadata and state flags, allowing the system to dynamically decide whether to perform a full hyperparameter re-optimization based on data volume shifts.
* **Operational Logging:** Every individual prediction request and batch upload metadata is logged in real-time for continuous monitoring and audit trails.

---

## 🛠 Tech Stack
* **ML Core:** XGBoost, Scikit-learn, Optuna (Tuning).
* **Application:** FastAPI (Backend), Gradio (User Interface), Streamlit (Monitoring).
* **Data Scraping:** Playwright, BeautifulSoup4, Requests.
* **Data Storage**: MongoDB Atlas (Cloud Logs/Production) & SQLite (Local Data/Development).
* **DevOps:** Docker, GitHub Actions, MLflow, DagsHub, Render.

---

## 🔍 Technical Observations & Ethics

* **Market Specialization & Data Bias:** The model is designed to predict prices for both **new and used** vehicles. However, it currently demonstrates higher precision for **used cars**. This is a direct result of the data distribution from our source—a leading automotive marketplace in Saudi Arabia—where used car listings are significantly more prevalent. As the system continues to scrape more diverse data, we aim to further improve the model's generalization for brand-new car models.
* **Data Integrity:** To optimize storage and performance, the system logs only **statistical metadata** (averages, distributions, and counts) for batch uploads. In contrast, individual prediction requests are logged in full to provide a detailed audit trail for real-time monitoring and debugging.
* **Ethical Scraping:** We strictly follow a "Polite Scraping" protocol. This includes using asynchronous requests and randomized delays to ensure we respect the source server's stability and maintain a low bandwidth footprint.
* **Project Purpose:** This project is developed strictly for **educational and research purposes**, demonstrating a complete end-to-end MLOps architectural cycle.
* **Note about the shown data in the dashboard:** Individual sample inputs are manually curated for testing purposes, while the batch data showcased in Streamlit is identical to the training baseline to ensure consistency during experimental validation.


---
## 🔗 Project Links
* **Live Prediction UI (Gradio):** https://saudi-cars-prices-predictor-web.onrender.com
* **Monitoring Dashboard (Streamlit):** https://saudi-cars-project-dashboard.onrender.com
* **Model Registry & MLflow Tracking**: https://dagshub.com/Mohamed382567/Car-Price-Prediction-MLOps

---
## 📂 Directory Structure
```bash
├── .github/workflows/  # CI/CD Automation (run_pipeline.yml)
├── app/                # Web Application (FastAPI + Gradio UI)
│   ├── api.py          # API Gateway
│   └── ui.py           # Gradio Web Interface
├── data/raw/           #  the sample data used in the notebook and the baseline data for the created models from the pipeline       
├── database/           # DB connection and schema management
├── models/             
│   └── preprocessor.pkl # Legacy v1 Preprocessor (GitHub Fallback)
├── monitoring/         # for integrating the monitoring system using streamlit
├── scraper/            # Scraper engine (engine.py) and HTML parsers
├── src/                # Core ML & Engineering logic 
│   ├── trainer.py      # Main training script (Uploads coupled artifacts)
│   ├── preprocessing.py # Feature engineering & terminology cleaning
│   ├── tuning.py       # Automated Optuna optimization logic
│   └── inference.py    # v1/v2 compatible prediction engine
├── tests/              # Unit and integration tests (Pytest)
├── MLCarsProjectNotebook.ipynb # The Research & Strategy Lab
├── Dockerfile          # Multi-stage production build definition
├── requirements.txt    # Project dependencies
└── run_pipeline.py     # Main orchestration script
