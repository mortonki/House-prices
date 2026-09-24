# Active Context

## Current Work Focus
- Transitioning from exploratory data analysis and preprocessing to the model training phase.

## Recent Changes
- Completed EDA, feature engineering, and multicollinearity analysis.
- Appended "Validation Strategies" section to `housing_eda.ipynb`.
- Defined train/test split (80/20) and cross-validation plan.

## Next Steps
- Begin training phase using Ridge Regression and XGBoost models.
- Evaluate and compare model performances.
- Perform hyperparameter tuning.

## Important Decisions & Considerations
- Using a fixed random state (42) for reproducibility.
- Using K-Fold Cross-Validation (K=5) for training.
- Log transformation applied to `price`, `sqft_living`, and `sqft_lot`.

## Learnings
- Target encoding for `cityzip` effectively captures neighborhood-specific price trends.
- Multicollinearity analysis confirmed the safety of dropping redundant features like `sqft_above` and `sqft_basement`.
