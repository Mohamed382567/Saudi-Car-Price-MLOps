# --- Stage 1: Base Image (Common for all services) ---
# We use python:3.9-slim to keep the image size small but functional
FROM python:3.9-slim as base

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for Playwright and general operations
# We need wget and gnupg to handle package signing and downloads
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# [CRITICAL] Install Playwright Chromium browser and its system dependencies
# This allows the scraper to run inside the container on Render/Cloud
RUN playwright install --with-deps chromium

# Copy the rest of the application code
COPY . .

# Set Python to run in unbuffered mode to see logs in real-time
ENV PYTHONUNBUFFERED=1


# --- Stage 2: API & Gradio UI (Production Web Service) ---
# This stage builds the image for your FastAPI and Gradio interface
FROM base as api_stage

# Define environment to trigger cloud-specific logic (e.g., MongoDB connectivity)
ENV APP_ENV=cloud

# Expose the port FastAPI/Uvicorn will run on
EXPOSE 8000

# Start the API server
# Assuming your FastAPI instance is named 'app' inside 'app/api.py'
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]


# --- Stage 3: Streamlit Dashboard (Monitoring Service) ---
# This stage builds the image for your Streamlit analytics dashboard
FROM base as dashboard_stage

# Define environment to trigger cloud-specific logic
ENV APP_ENV=cloud

# Expose the port Streamlit will run on
EXPOSE 8501

# Start the Streamlit application
CMD ["streamlit", "run", "monitoring/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]