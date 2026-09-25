import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from typing import Dict, Any, Tuple, Optional
import joblib

class ModelTrainer:
    """
    Handles training and evaluation of house price prediction models.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}

    def train_ridge(self, X: pd.DataFrame, y: pd.Series, params: Optional[Dict[str, Any]] = None) -> None:
        """Trains a Ridge regression model."""
        if params is None:
            params = {"alpha": 1.0}
        
        model = Ridge(**params, random_state=self.random_state)
        model.fit(X, y)
        self.models["ridge"] = model
        print("Ridge model trained.")

    def train_xgboost(self, X: pd.DataFrame, y: pd.Series, params: Optional[Dict[str, Any]] = None) -> None:
        """Trains an XGBoost regressor."""
        if params is None:
            params = {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": 6,
                "n_jobs": -1
            }
        
        model = XGBRegressor(**params, random_state=self.random_state)
        model.fit(X, y)
        self.models["xgboost"] = model
        print("XGBoost model trained.")

    def evaluate(self, X: pd.DataFrame, y: pd.Series, inverse_transform_func=None, target_column: Optional[str] = None) -> None:
        """Evaluates all trained models."""
        for name, model in self.models.items():
            predictions = model.predict(X)
            
            # Default metrics in log space
            rmse = np.sqrt(mean_squared_error(y, predictions))
            r2 = r2_score(y, predictions)
            
            # If inverse transform is provided, calculate metrics in original space
            if inverse_transform_func and target_column:
                # Create temporary DataFrames to use the existing inverse_log_transform
                # Ensure pred_df has the same index as y to avoid alignment issues
                pred_df = pd.DataFrame({'pred': predictions}, index=y.index)
                y_df = pd.DataFrame({target_column: y})
                
                # Apply inverse transform
                pred_df = inverse_transform_func(pred_df, 'pred')
                y_df = inverse_transform_func(y_df, target_column)
                
                # Recalculate metrics in original space
                rmse = np.sqrt(mean_squared_error(y_df[target_column], pred_df['pred']))
                r2 = r2_score(y_df[target_column], pred_df['pred'])
            
            self.results[name] = {"RMSE": rmse, "R2": r2}
            print(f"Model: {name} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    def save_models(self, directory: str = "models"):
        """Saves trained models to disk."""
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(directory, f"{name}.joblib"))
            print(f"Saved {name} model to {directory}")