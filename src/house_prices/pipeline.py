import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from .dataloader import load_dataset
from .features import engineer_house_features as original_engineer_house_features
from .preprocessing import target_encode, one_hot_encode, log_transform, inverse_log_transform, standardize
from .models import ModelTrainer
from sklearn.model_selection import train_test_split

class TrainingPipeline:
    """
    Orchestrates the full data science pipeline from raw data to model evaluation. Follows best practices for preventing data leakage by splitting the dataset before any preprocessing that depends on target information or global statistics.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trainer = ModelTrainer(random_state=self.random_state)

    def run(self, dataset_path: str, config: Dict[str, Any]) -> None:
        """Executes the full pipeline."""
        print("--- Step 1: Data Loading ---")
        df = load_dataset(dataset_path)

        print("--- Step 1.5: Outlier Removal ---")

        # Drop rows where bedrooms or bathrooms are 0.0
        df = df[(df['bedrooms'] > 0.0) & (df['bathrooms'] > 0.0)]

        # Remove extreme price outliers (configurable quantile, default 0.99)
        outlier_quantile = config.get("outlier_quantile", 0.99)
        q = df['price'].quantile(outlier_quantile)
        df = df[df['price'] <= q].copy()
        print(f"Removed {len(df) - len(df[df['price'] <= q])} outliers based on quantile {outlier_quantile}.")

        print("--- Step 2: Data Splitting ---")
        X = df.drop(columns=['price'])
        y = df['price']

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=config.get("test_size", 0.2), random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state
        )

        # Ensure date columns are datetime objects before calculating stats
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()
        X_train['date'] = pd.to_datetime(X_train['date'])
        X_val['date'] = pd.to_datetime(X_val['date'])
        X_test['date'] = pd.to_datetime(X_test['date'])

        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

        print("--- Step 3: Feature Engineering (Training Data Only) ---")
        # Calculate statistics on training data only
        min_date = X_train['date'].min()

        # First pass: Create base features (including street_type) for training data
        X_train = original_engineer_house_features(X_train, min_date=min_date, rare_types=None)

        # Calculate rare types from the engineered training data
        rare_types = X_train['street_type'].value_counts(normalize=True).index[X_train['street_type'].value_counts(normalize=True) < 0.01].tolist()

        # Second pass: Create features for val and test data using training statistics
        X_val = original_engineer_house_features(X_val, min_date=min_date, rare_types=rare_types)
        X_test = original_engineer_house_features(X_test, min_date=min_date, rare_types=rare_types)

        print("--- Step 4: Preprocessing ---")
        # Log transform target
        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)
        y_test = np.log1p(y_test)

        # Target encode cityzip using training data statistics
        X_train, encoder_target = target_encode(X_train, 'cityzip', y_train)
        X_val, _ = target_encode(X_val, 'cityzip', y_train, encoder=encoder_target)
        X_test, _ = target_encode(X_test, 'cityzip', y_train, encoder=encoder_target)
        X_train.drop(columns=['cityzip'], inplace=True)
        X_val.drop(columns=['cityzip'], inplace=True)
        X_test.drop(columns=['cityzip'], inplace=True)

        # One-hot encode street_type
        X_train, encoder_oh, dropped_col = one_hot_encode(X_train, 'street_type', drop_first=True)
        X_val, _, _ = one_hot_encode(X_val, 'street_type', drop_first=True, encoder=encoder_oh)
        X_test, _, _ = one_hot_encode(X_test, 'street_type', drop_first=True, encoder=encoder_oh)
        X_train.drop(columns=['street_type'], inplace=True)
        X_val.drop(columns=['street_type'], inplace=True)
        X_test.drop(columns=['street_type'], inplace=True)

        # Log transform skewed features
        X_train = log_transform(X_train, 'sqft_living')
        X_train = log_transform(X_train, 'sqft_lot')
        X_train = log_transform(X_train, 'price_per_sqft')
        X_val = log_transform(X_val, 'sqft_living')
        X_val = log_transform(X_val, 'sqft_lot')
        X_val = log_transform(X_val, 'price_per_sqft')
        X_test = log_transform(X_test, 'sqft_living')
        X_test = log_transform(X_test, 'sqft_lot')
        X_test = log_transform(X_test, 'price_per_sqft')

        # Drop redundant columns
        #X_train.drop(columns=['price_per_sqft'], inplace=True, errors='ignore')
        #X_val.drop(columns=['price_per_sqft'], inplace=True, errors='ignore')
        #X_test.drop(columns=['price_per_sqft'], inplace=True, errors='ignore')

        # Standardize numerical features
        num_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view', 'condition', 'age_built', 'years_since_renovation', 'weekday', 'days_since_first', 'cityzip_encoded', 'price_per_sqft', 'sqft_living_per_age']
        X_train, scaler = standardize(X_train, num_cols)
        X_val, _ = standardize(X_val, num_cols, scaler=scaler)
        X_test, _ = standardize(X_test, num_cols, scaler=scaler)


        print("--- Step 5: Model Training & Evaluation ---")
        self.trainer.train_ridge(X_train, y_train, X_val=X_val, y_val=y_val, params=config.get("ridge_params"))
        self.trainer.train_xgboost(X_train, y_train, X_val=X_val, y_val=y_val, params=config.get("xgb_params"))

        print("\nValidation Results:")
        self.trainer.evaluate(X_val, y_val, inverse_transform_func=inverse_log_transform, target_column='price')

        print("\nTest Results:")
        self.trainer.evaluate(X_test, y_test, inverse_transform_func=inverse_log_transform, target_column='price')

        print("--- Step 6: Saving Models ---")
        self.trainer.save_models()
        print("Pipeline complete.")

    if __name__ == "__main__":
        # Example usage (will be called from main.py)
        pass
