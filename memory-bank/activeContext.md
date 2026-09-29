# Active Context

## Current Work Focus
- Verifying the pipeline with different sets of hyperparameters via the command line.
- Planning and implementing hyperparameter optimization (GridSearch/Optuna).

## Recent Changes
- Modified `src/house_prices/models.py` to accept optional parameter dictionaries for Ridge and XGBoost models.
- Updated `src/house_prices/pipeline.py` to accept a configuration dictionary in its `run` method.
- Created `main.py` as an entry point that uses `argparse` to collect hyperparameters and pass them to the pipeline.
- Integrated `kagglehub` for automated dataset downloading.

## Next Steps
- Verify the pipeline with different sets of hyperparameters via the command line.
- Implement hyperparameter optimization.
- Finalize model evaluation and reporting.

## Important Decisions & Considerations
- Configuration is passed as a dictionary to the pipeline to decouple logic from input source (CLI, config files, etc.).
- Default parameters are handled within the `ModelTrainer` methods if not provided in the config.
- Using a fixed random state (42) for reproducibility.
- Log transformation applied to `price`, `sqft_living`, and `sqft_lot`.

## Learnings
- Target encoding for `cityzip` effectively captures neighborhood-specific price trends.
- Multicollinearity analysis confirmed the safety of dropping redundant features like `sqft_above` and `sqft_basement`.
- Using a configuration dictionary makes it much easier to add new parameters without changing multiple function signatures.

