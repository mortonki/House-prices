# Tech Context

## Technologies Used
- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Environment Management**: `uv`
- **Dataset Source**: Kaggle (via `kagglehub`)

## Development Setup
- Run project: `uv run house-prices`
- Download data: `uv run python src/dataloader.py`
- IDE: VS Code

## Technical Constraints
- Must handle large CSV files efficiently.
- Preprocessing must be idempotent and reproducible.

## Tool Usage Patterns
- Use `uv` for all dependency management.
- Follow the project structure defined in `AGENTS.md`.
