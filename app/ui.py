# app/ui.py
import gradio as gr
import requests
import pandas as pd
import os

# Set API endpoint (Use 'api' as hostname for Docker, or 127.0.0.1 for local)
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def single_predict(brand, model, year, mileage, gear, fuel, ext, origin, drive, cyl, eng, seats, cond, ext_c, int_c):
    """
    Calls the FastAPI /predict endpoint with all 15 required fields.
    """
    payload = {
        "brand": brand, 
        "model_name": model, 
        "year": year, 
        "mileage": mileage,
        "gear": gear, 
        "fuel": fuel, 
        "extension": ext, 
        "origin": origin,
        "drivetrain": drive, 
        "cylinders": cyl, 
        "engine_size": eng,
        "seats": int(seats), 
        "condition": cond, 
        "exterior_color": ext_c, 
        "interior_color": int_c
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        if response.status_code == 200:
            price = response.json()['estimated_price_sar']
            return f"{price:,} SAR"
        else:
            return f"API Error: {response.text}"
    except Exception as e:
        return f"Connection Error: {str(e)}"

def batch_predict(file):
    """
    Sends the uploaded CSV file directly to the new API endpoint.
    The API handles all the cleaning and prediction logic.
    """
    try:
        # Open the file in binary mode and send it as a multipart/form-data request
        with open(file.name, "rb") as f:
            files = {"file": (os.path.basename(file.name), f, "text/csv")}
            response = requests.post(f"{API_BASE_URL}/predict_csv", files=files)
            
        # Check if the API request was successful and contains the predictions
        if response.status_code == 200 and "predictions_list" in response.json():
            data = response.json()["predictions_list"]
            df_result = pd.DataFrame(data)
            
            # Save the result to a new CSV file for the user to download
            output_file = "batch_results.csv"
            df_result.to_csv(output_file, index=False)
            
            # Return the first few rows for the preview, and the file for download
            return df_result.head(), output_file
        else:
            error_msg = response.json().get('error', response.text)
            return f"API Error: {error_msg}", None
            
    except Exception as e:
        return f"Connection Error: {str(e)}", None

# Building the Interface with 3 Columns for better UX
with gr.Blocks() as demo:
    gr.Markdown("# 🚗 Saudi Car Price Prediction System")
    
    with gr.Tab("Single Car Appraisal"):
        with gr.Row():
            # Column 1: Basic Info
            with gr.Column():
                gr.Markdown("### 📋 Basic Details")
                brand = gr.Textbox(label="Brand", placeholder="e.g. Toyota")
                model = gr.Textbox(label="Model Name", placeholder="e.g. Camry")
                year = gr.Number(label="Manufacturing Year", value=2022)
                mileage = gr.Number(label="Mileage (KM)", value=50000)
                condition = gr.Radio(["used", "new"], label="Condition", value="used")

            # Column 2: Mechanical Specs
            with gr.Column():
                gr.Markdown("### ⚙️ Mechanical Specs")
                engine = gr.Textbox(label="Engine Size", placeholder="e.g. 2.5")
                cylinders = gr.Textbox(label="Cylinders", placeholder="e.g. 4 Cylinder")
                gear = gr.Radio(["Automatic", "Manual", "CVT"], label="Transmission", value="Automatic")
                fuel = gr.Radio(["Gasoline", "Diesel", "Hybrid"], label="Fuel Type", value="Gasoline")
                drive = gr.Radio(["FWD", "RWD", "AWD", "4WD", "Double (4x4)"], label="Drivetrain", value="FWD")

            # Column 3: Aesthetics & Origin
            with gr.Column():
                gr.Markdown("### 🎨 Aesthetics & Origin")
                origin = gr.Radio(["Saudi", "Gulf", "Other"], label="Origin", value="Saudi")
                ext_color = gr.Textbox(label="Exterior Color", placeholder="e.g. White")
                int_color = gr.Textbox(label="Interior Color", placeholder="e.g. Beige")
                seats = gr.Slider(2, 9, step=1, label="Number of Seats", value=5)
                extension = gr.Textbox(label="Extension/Grade", placeholder="e.g. Full Option")

        btn = gr.Button("Evaluate Price 🎯", variant="primary")
        result = gr.Label(label="Estimated Value (SAR)")
        
        # Mapping all 15 inputs to the function
        btn.click(
            single_predict, 
            inputs=[brand, model, year, mileage, gear, fuel, extension, origin, drive, cylinders, engine, seats, condition, ext_color, int_color], 
            outputs=result
        )

    with gr.Tab("Bulk Prediction (CSV)"):
        gr.Markdown("### 📊 Bulk Processing")
        gr.Markdown("Upload a CSV file to get mass valuations. The system will automatically clean and process the data.")
        csv_input = gr.File(label="Upload CSV")
        btn_bulk = gr.Button("Process CSV 🚀")
        with gr.Row():
            preview = gr.DataFrame(label="Results Preview")
            csv_output = gr.File(label="Download Full Results")
        btn_bulk.click(batch_predict, inputs=csv_input, outputs=[preview, csv_output])

# Launch the app
if __name__ == "__main__":
    print("Starting Gradio Interface...")
    demo.launch(
        server_name="127.0.0.1", # For local testing
        server_port=None,        # Let Gradio choose an open port
        theme=gr.themes.Soft(),  # Soft theme to match the appraisal vibe
        quiet=False              # Show details in case of an error
    )