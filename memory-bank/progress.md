# Progress

## What Works
- [x] Project structure established.
- [x] Data loading script (`src/dataloader.py`) functional.
- [x] Basic street information extraction logic.
- [x] Core preprocessing functions (encoding, log transform).
- [x] Initial EDA notebook completed.

## What's Left to Build
- [ ] Fix `pyproject.toml` script definition.
- [ ] Finalize model selection and training pipeline.
- [ ] Hyperparameter tuning.
- [ ] Evaluation metrics and reporting.
- [ ] Production-ready inference script.

## Current Status
- Initial setup and memory bank initialization complete.
- Identified configuration discrepancy in `pyproject.toml`.
- Currently verifying preprocessing logic.

## Known Issues
- Discrepancy in `pyproject.toml` script path.

## Evolution of Project Decisions
- Decided to use `kagglehub` for easier dataset management.
- Decided to implement custom street extraction due to inconsistent formatting in the source data.
- Decided to use a flat `src/` layout for core logic.
