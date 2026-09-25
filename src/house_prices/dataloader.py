import os
import pandas as pd
import kagglehub

def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the dataset from the given path into a pandas DataFrame.
    Assumes the path contains a CSV file.
    """
    # Find all csv files in the directory
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the directory: {path}")
    
    # Sort to ensure deterministic behavior if multiple files exist
    csv_files.sort()
    first_csv = os.path.join(path, csv_files[0])
    
    print(f"Loading dataset from: {first_csv}")
    df = pd.read_csv(first_csv)
    return df

if __name__ == "__main__":
    # Download latest version
    path = kagglehub.dataset_download("debayank2024/house-price-prediction")
    print("Path to dataset files:", path)

    # Load the dataset
    try:
        df = load_dataset(path)
        print("Dataset loaded successfully.")
        print(df.head())
    except Exception as e:
        print(f"Error loading dataset: {e}")

