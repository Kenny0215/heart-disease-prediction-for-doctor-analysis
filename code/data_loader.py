import pandas as pd
import kagglehub
import os

def load_data():
    print("--- Step 1: Downloading Data ---")
    # Download dataset
    path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
    print(f"Path to dataset files: {path}")

    # Find the CSV file inside the folder
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError("No CSV file found.")

    # Load into Pandas
    full_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(full_path)
    
    print("Data loaded successfully.")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    return df