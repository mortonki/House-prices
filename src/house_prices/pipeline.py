import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from .dataloader import load_dataset
from .features import engineer_house_features
from .preprocessing import target_encode, one_hot_encode, log_transform
from .models import ModelTrainer
from sklearn.model_selection import train_test_split

class TrainingPipeline:
    """
    Orchestrates the full data science pipeline from raw data to model evaluation.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trainer = ModelTrainer(random_state=self.random_state)

    def run(self, dataset_path: str, config: Dict[str, Any]) -> None:
        """Executes the full pipeline."""
        print("--- Step 1: Data Loading ---")
        df = load_dataset(dataset_path)
        
        print("--- Step 2: Feature Engineering ---")
        df = engineer_house_features(df)
        
        print("--- Step 3: Preprocessing ---")
        # Log transform target
        df = log_transform(df, 'price')
        
        # Target encode cityzip
        df, _ = target_encode(df, 'cityzip', 'price')
        df.drop(columns=['cityzip'], inplace=True)
        
        # One-hot encode street_type
        df, _, dropped_col = one_hot_encode(df, 'street_type', drop_first=True)
        df.drop(columns=['street_type'], inplace=True)
        
        # Log transform skewed features
        df = log_transform(df, 'sqft_living')
        df = log_transform(df, 'sqft_lot')
        
        # Drop redundant columns
        df.drop(columns=['price_per_sqft'], inplace=True, errors='ignore')
        
        print("--- Step 4: Data Splitting ---")
        X = df.drop(columns=['price'])
        y = df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.get("test_size", 0.2), random_state=self.random_state
        )
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        print("--- Step 5: Model Training & Evaluation ---")
        self.trainer.train_ridge(X_train, y_train, params=config.get("ridge_params"))
        self.trainer.train_xgboost(X_train, y_train, params=config.get("xgb_params"))
        
        print("\nEvaluation Results:")
        self.trainer.evaluate(X_test, y_test)
        
        print("--- Step 6: Saving Models ---")
        self.trainer.save_models()
        print("Pipeline complete.")

if __name__ == "__main__":
    # Example usage (will be called from main.py)
    pass
