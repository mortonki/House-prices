# House Price Prediction

## Overview
This project aims to build a machine learning model to accurately predict house prices based on historical data and property characteristics. It provides a structured way to process raw Kaggle datasets into model-ready features, handling complex categorical data like street addresses and city locations.

## Goals
- Build a machine learning model to predict house prices accurately.
- Implement a robust data preprocessing pipeline.
- Perform comprehensive Exploratory Data Analysis (EDA).

## Scope
- Data acquisition using `kagglehub` and custom dataloaders.
- Feature engineering including street information extraction and geographical encoding.
- Preprocessing techniques like target encoding, one-hot encoding, and log transformations.
- Model development and evaluation.

## Technology Stack
- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Environment Management**: `uv`
- **Dataset Source**: Kaggle (via `kagglehub`)

## Getting Started

### Prerequisites
- Python installed
- `uv` installed

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   uv sync
   ```

### Running the Project
- Run the project:
  ```bash
   uv run house-prices
  ```
- Download the dataset:
  ```bash
  uv run python src/dataloader.py
  ```

## Project Structure
- `src/`: Core logic for data loading and preprocessing.
- `house_prices/`: Shared utilities and helper functions.
- `housing_eda.ipynb`: Entry point for exploratory analysis.

## Progress
- [x] Project structure established.
- [x] Data loading script (`src/dataloader.py`) functional.
- [x] Basic street information extraction logic.
- [x] Core preprocessing functions (encoding, log transform).
- [x] Initial EDA notebook completed.
- [ ] Finalize model selection and training pipeline.
- [ ] Hyperparameter tuning.
- [ ] Evaluation metrics and reporting.
- [ ] Production-ready inference script.

## Contact
For more information, please refer to the `memory-bank` directory for detailed project context and technical decisions.

