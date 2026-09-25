import sys
import kagglehub
from .pipeline import TrainingPipeline

def main():
    """
    Main entry point for the house prices prediction project.
    Runs the full training pipeline.
    """
    print("Starting House Price Prediction Pipeline...")
    
    # Download dataset path
    dataset_path = kagglehub.dataset_download("debayank2024/house-price-prediction")
    
    # Initialize and run pipeline
    pipeline = TrainingPipeline(random_state=42)
    pipeline.run(dataset_path)
    
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
