# Project Memory Bank

## Dataset Overview
- **Source**: Washington State housing market.
- **Records**: 4,600 complete records.
- **Features**: 18 distinct features.
- **Geographic Coverage**: 44 cities and 77 zip codes.
- **Data Quality**: Entirely free of missing or null values; ready for immediate EDA and ML.

## Key Variables
- **Target Variable**: `price` (property valuation).
- **Structural Dimensions**: `sqft_living`, `bedrooms`, `floors`.
- **Condition & History**: Condition ratings, `yr_built`, `yr_renovated`.
- **Geography**: Granular geographic identifiers (redundant data removed).

## Engineering & Pre-processing
- **Redundancy Removal**: Redundant geographical data has been removed.
- **Engineered Features**: `price_per_sqft` (calculated to streamline pricing evaluations and modeling).

## Current Status
- Initialized memory bank based on project description.
