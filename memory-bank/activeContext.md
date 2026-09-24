# Active Context

## Current Work Focus
- Updating the Memory Bank to align with the actual project structure.
- Analyzing the existing codebase for data loading and preprocessing patterns.

## Recent Changes
- Corrected paths in Memory Bank (changed `house_prices/utils.py` to `src/utils.py`).
- Updated `systemPatterns.md` to reflect that `src/` contains all core logic and utilities.
- Created the `memory-bank` directory and core documentation files.

## Next Steps
- Verify the `src/preprocessing.py` logic.
- Ensure all utility functions in `src/utils.py` are well-documented.
- Complete the initial EDA if any parts are missing.
- Resolve discrepancy between `pyproject.toml` (referencing `house_prices` package) and actual file structure (all logic in `src/`).

## Important Decisions & Considerations
- Using `uv` for environment management as specified in `AGENTS.md`.
- Maintaining a clear separation between data loading (`src/dataloader.py`) and processing (`src/preprocessing.py`).

## Learnings
- The dataset contains specific street naming conventions that require custom extraction logic.
- There is a mismatch between the project's declared package structure (`house_prices`) and its physical layout (`src/`).
