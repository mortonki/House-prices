import argparse
import sys
import kagglehub
from house_prices.pipeline import TrainingPipeline

def main():
    parser = argparse.ArgumentParser(description="Train House Price Prediction Models")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size (default: 0.2)")
    parser.add_argument("--ridge_alpha", type=float, default=1.0, help="Alpha parameter for Ridge regression")
    parser.add_argument("--xgb_lr", type=float, default=0.05, help="Learning rate for XGBoost")
    parser.add_argument("--xgb_depth", type=int, default=6, help="Max depth for XGBoost")
    parser.add_argument("--xgb_estimators", type=int, default=1000, help="Number of estimators for XGBoost")

    args = parser.parse_args()

    # Construct configuration dictionary
    config = {
        "test_size": args.test_size,
        "ridge_params": {
            "alpha": args.ridge_alpha
        },
        "xgb_params": {
            "n_estimators": args.xgb_estimators,
            "learning_rate": args.xgb_lr,
            "max_depth": args.xgb_depth,
            "n_jobs": -1
        }
    }

    # Hardcoded dataset path
    dataset_path = kagglehub.dataset_download("debayank2024/house-price-prediction")

    pipeline = TrainingPipeline()
    pipeline.run(dataset_path, config)

if __name__ == "__main__":
    main()
