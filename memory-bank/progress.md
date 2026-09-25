# Progress

## What Works
- Project structure established.
- Data loading script (`src/dataloader.py`) functional.
- Basic street information extraction logic.
- Core preprocessing functions (encoding, log transform).
- Initial EDA notebook completed.
- Feature engineering and multicollinearity analysis finished.
- Validation strategies defined.
- Modular training pipeline implemented.
- Command-line interface for hyperparameter configuration.

## What's Left to Build
- Hyperparameter optimization (e.g., GridSearch or Optuna).
- Final model evaluation and reporting.
- Deployment of the trained models.

## Current Status
- Pipeline is parameterized and can be executed with custom hyperparameters via CLI.

## Known Issues
- None currently known.

## Evolution of Project Decisions
- Switched from hardcoded parameters to a configuration dictionary passed through the pipeline.
- Decided to use `argparse` in `main.py` for user-facing configuration.
- Decided to use `kagglehub` for easier dataset management.
- Decided to implement custom street extraction due to inconsistent formatting in the source data.
- Decided to use a flat `src/` layout for core logic.
- Decided to use log transformations for skewed features.

