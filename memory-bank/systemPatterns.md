# System Patterns

## Architecture
- The project follows a modular design where data loading, feature engineering, preprocessing, model training, and evaluation are separated into distinct modules.
- A `TrainingPipeline` class orchestrates the flow between these modules.

## Key Technical Decisions
- **Configuration Management:** Parameters are passed as a configuration dictionary to the pipeline. This decouples the execution logic from the input source (CLI, config files, etc.).
- **Model Training:** The `ModelTrainer` class handles both Ridge and XGBoost models, providing a consistent interface for training and evaluation.
- **Reproducibility:** A fixed random state is used across all stochastic operations.

## Component Relationships
- `dataloader.py`: Fetches raw data.
- `features.py`: Transforms raw data into features.
- `preprocessing.py`: Handles encoding and scaling.
- `models.py`: Contains model definitions and training logic.
- `pipeline.py`: Coordinates the entire process.
- `main.py`: Entry point for running the pipeline with CLI arguments.
