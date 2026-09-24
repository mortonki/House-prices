# Progress

## What Works
- [x] Project structure established.
- [x] Data loading script (`src/dataloader.py`) functional.
- [x] Basic street information extraction logic.
- [x] Core preprocessing functions (encoding, log transform).
- [x] Initial EDA notebook completed.
- [x] Feature engineering and multicollinearity analysis finished.
- [x] Validation strategies defined.

## What's Left to Build
- [ ] Fix `pyproject.toml` script definition.
- [ ] Model training pipeline implementation.
- [ ] Hyperparameter tuning.
- [ ] Evaluation metrics and reporting.
- [ ] Production-ready inference script.

## Current Status
- EDA and preprocessing phase complete.
- Ready to begin model training phase.

## Known Issues
- Discrepancy in `pyproject.toml` script path.

## Evolution of Project Decisions
- Decided to use `kagglehub` for easier dataset management.
- Decided to implement custom street extraction due to inconsistent formatting in the source data.
- Decided to use a flat `src/` layout for core logic.
- Decided to use log transformations for skewed features.
