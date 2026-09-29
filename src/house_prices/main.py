import sys
import argparse
import json
import kagglehub
from .pipeline import TrainingPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Run the house price prediction pipeline.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size (default: 0.2)")
    parser.add_argument("--ridge_params", type=str, default='{"alpha": 1.0}', help="Ridge parameters as JSON string")
    parser.add_argument("--xgb_params", type=str, default='{"n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6, "n_jobs": -1}', help="XGBoost parameters as JSON string")
    parser.add_argument("--outlier_quantile", type=float, default=0.99, help="Quantile for removing extreme price outliers (default: 0.99)")
    return parser.parse_args()

def main():
    """
    Main entry point for the house prices prediction project.
    Runs the full training pipeline.
    """
    args = parse_args()
    
    print("Starting House Price Prediction Pipeline...")
    
    # Download dataset path
    dataset_path = kagglehub.dataset_download("debayank2024/house-price-prediction")
    
    # Prepare configuration
    try:
        config = {
            "test_size": args.test_size,
            "ridge_params": json.loads(args.ridge_params),
            "xgb_params": json.loads(args.xgb_params),
            "outlier_quantile": args.outlier_quantile
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON parameters: {e}")
        sys.exit(1)
    
    # Initialize and run pipeline
    pipeline = TrainingPipeline(random_state=42)
    pipeline.run(dataset_path, config)
    
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
