# Active Context

## Current Work Focus
- Verifying the project structure and aligning the Memory Bank with the actual codebase.
- Identifying discrepancies in configuration files (e.g., `pyproject.toml`).

## Recent Changes
- Verified that all core logic resides in the `src/` directory.
- Confirmed that `src/dataloader.py`, `src/preprocessing.py`, and `src/utils.py` contain the primary functionality.
- Identified a discrepancy in `pyproject.toml` where the `house-prices` script refers to a non-existent `house_prices` package instead of the `src/` directory contents.
- Initialized and populated the Memory Bank files.

## Next Steps
- Fix the `pyproject.toml` script definition to correctly point to the source code.
- Verify the `src/preprocessing.py` logic for consistency.
- Ensure all utility functions in `src/utils.py` are well-documented.
- Complete the initial EDA if any parts are missing.

## Important Decisions & Considerations
- Using `uv` for environment management as specified in `AGENTS.md`.
- Maintaining a clear separation between data loading (`src/dataloader.py`) and processing (`src/preprocessing.py`).
- Using `kagglehub` for dataset acquisition.

## Learnings
- The dataset contains specific street naming conventions that require custom extraction logic.
- The project uses a flat `src/` layout without a nested `house_prices` package, which conflicts with the current `pyproject.toml` entry.
