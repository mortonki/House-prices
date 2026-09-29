# Tech Context

## Technologies Used
- Python 3.10+
- Pandas: Data manipulation
- NumPy: Numerical operations
- Scikit-learn: Machine learning models
- XGBoost: Gradient boosting
- Joblib: Model serialization
- Argparse: Command-line argument parsing
- Kagglehub: Dataset acquisition

## Development Setup
- Environment managed by `uv`
- Run using `uv run house_prices`
- Dataset acquisition via `uv run python src/dataloader.py`
- Dataset Source: `debayank2024/house-price-prediction`

## Technical Constraints
- Memory limits when handling large datasets
- Reproducibility requirements (fixed random states)

## Tool Usage Patterns
- Use `argparse` for all external configurations.
- Use `joblib` for saving/loading models.
- Use `log_transform` for skewed target variables.
