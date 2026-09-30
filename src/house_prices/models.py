import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from typing import Dict, Any, Tuple, Optional
import joblib
import mlflow
import mlflow.metrics

class ModelTrainer:
    """
    Handles training and evaluation of house price prediction models.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}

    def train_ridge(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, params: Optional[Dict[str, Any]] = None) -> None:
        """Trains a Ridge regression model and logs to MLflow."""
        if params is None:
            params = {"alpha": 1.0}

        with mlflow.start_run(run_name="ridge"):
            mlflow.log_params(params)
            
            model = Ridge(**params, random_state=self.random_state)
            model.fit(X, y)
            self.models["ridge"] = model
            
            if X_val is not None and y_val is not None:
                predictions = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, predictions))
                r2 = r2_score(y_val, predictions)
                mae = mean_absolute_error(y_val, predictions)
                
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("R2", r2)
                mlflow.log_metric("MAE", mae)
                print(f"Ridge model trained and logged. RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
            else:
                print("Ridge model trained (no validation data provided for logging).")

    def train_xgboost(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, params: Optional[Dict[str, Any]] = None) -> None:
        """Trains an XGBoost regressor and logs to MLflow."""
        if params is None:
            params = {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": 6,
                "n_jobs": -1
            }

        # Convert inputs to numpy arrays to avoid index alignment issues with XGBoost eval_set
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else X
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else y

        eval_set = None

        if X_val is not None and y_val is not None:
            if X_val.empty:
                raise ValueError("Validation dataset is empty. Please check your data splitting logic.")

            X_val_arr = X_val.to_numpy() if hasattr(X_val, "to_numpy") else X_val
            y_val_arr = y_val.to_numpy() if hasattr(y_val, "to_numpy") else y_val

            # If user hasn't provided eval_set in params, we provide it from X_val/y_val
            if "eval_set" not in params:
                eval_set = [(X_val_arr, y_val_arr)]
                # If they didn't provide early_stopping_rounds either, set default
                if "early_stopping_rounds" not in params:
                    params["early_stopping_rounds"] = 50
            else:
                # User provided eval_set in params.
                # Since XGBRegressor constructor doesn't take eval_set, we pop it from params
                # and use it for the .fit() call.
                eval_set = params.pop("eval_set")

        # Remove n_jobs if early stopping is active, as it can cause issues in some versions
        if "early_stopping_rounds" in params or eval_set is not None:
            params.pop("n_jobs", None)

        with mlflow.start_run(run_name="xgboost"):
            mlflow.log_params(params)
            
            print(f"Training XGBoost with params: {params}")
            model = XGBRegressor(**params, random_state=self.random_state)

            # Pass eval_set to fit()
            model.fit(X_arr, y_arr, eval_set=eval_set, verbose=False)

            self.models["xgboost"] = model
            
            if X_val is not None and y_val is not None:
                predictions = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, predictions))
                r2 = r2_score(y_val, predictions)
                mae = mean_absolute_error(y_val, predictions)
                
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("R2", r2)
                mlflow.log_metric("MAE", mae)
                print(f"XGBoost model trained and logged. RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
            else:
                print("XGBoost model trained (no validation data provided for logging).")
            
            print("XGBoost model trained.")

    def evaluate(self, X: pd.DataFrame, y: pd.Series, inverse_transform_func=None, target_column: Optional[str] = None) -> None:
        """Evaluates all trained models."""
        for name, model in self.models.items():
            predictions = model.predict(X)

            # Default metrics in log space
            rmse = np.sqrt(mean_squared_error(y, predictions))
            r2 = r2_score(y, predictions)
            mae = mean_absolute_error(y, predictions)

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
                mae = mean_absolute_error(y_df[target_column], pred_df['pred'])

            self.results[name] = {"RMSE": rmse, "R2": r2, "MAE": mae}
            print(f"Model: {name} | RMSE: {rmse:.4f} | R2: {r2:.4f} | MAE: {mae:.4f}")

    def save_models(self, directory: str = "models"):
        """Saves trained models to disk."""
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(directory, f"{name}.joblib"))
            print(f"Saved {name} model to {directory}")
