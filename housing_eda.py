import marimo as mo
import pandas as pd
import kagglehub
from dataloader import load_dataset

# Load the dataset
path = kagglehub.dataset_download("debayank2024/house-price-prediction")
df = load_dataset(path)

# Display the first few rows
mo.display(df.head())

# Show summary statistics
mo.md(f"### Summary Statistics\n{df.describe()}")

# Show info
df.info()
