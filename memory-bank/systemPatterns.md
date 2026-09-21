# System Patterns

## Architecture
- **Modular Design**: The project is split into `src/` for core logic and `house_prices` for shared utilities.
- **Data Pipeline**: `dataloader.py` -> `preprocessing.py` -> Model Training (to be implemented/refined).

## Key Technical Decisions
- **Preprocessing Functions**: Pure functions used for transformations (e.g., `target_encode`, `one_hot_encode`) to ensure testability.
- **Street Extraction**: Custom regex-based extraction to handle diverse street formats.
- **Log Transformations**: Used to normalize skewed price distributions.

## Component Relationships
- `src.dataloader`: Fetches and loads the raw CSV.
- `src.preprocessing`: Contains the heavy lifting for feature engineering.
- `house_prices.utils`: Shared helper functions like street info extraction.
- `housing_eda.ipynb`: Entry point for exploratory analysis.

## Critical Implementation Paths
- Ensuring consistent encoding across train/test splits.
- Handling missing values consistently during preprocessing.
