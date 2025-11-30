# Land Value Prediction Project

This project aims to predict land and property values in the Île-de-France region using a variety of data sources and machine learning techniques. The workflow is organized around several Jupyter notebooks, each handling a specific step in the data science pipeline.

## Notebooks Overview

### 00_create_dataset.ipynb
Imports and merges multiple data tables to create a comprehensive dataset with as many features as possible to help predict the price per square meter of properties. Includes data cleaning and merging operations.

### 01_train_test_split.ipynb
Splits the dataset into training and test sets, with preliminary analysis to avoid data leakage. Ensures proper separation for model validation.

### 02_analyze_process_clean_data.ipynb
Performs exploratory data analysis and cleaning on the dataset. Visualizes distributions, checks for missing values, and processes features to prepare for modeling.

### 03_selection_features.ipynb
Selects relevant features for modeling using statistical and machine learning techniques (e.g., SHAP values, correlation analysis). Reduces dimensionality and improves model performance.

### 04_modelisation_RF_XG_theo.ipynb
Trains and evaluates Random Forest and XGBoost models on the selected features. Includes cross-validation and performance metrics.

### 04_modelisation_XGB_RF_Stacking_LigthGBM.ipynb
Implements advanced modeling techniques, including stacking and LightGBM, alongside XGBoost and Random Forest. Compares ensemble methods for improved prediction accuracy.

### 04_multiple_models.ipynb
Explores and compares multiple machine learning models for land value prediction. Includes model training, evaluation, and comparison of results.

### 04_ridge_regression.ipynb
Applies Ridge Regression to the dataset, including hyperparameter tuning and evaluation. Compares linear model performance to ensemble methods.

## Data Structure
- **data/raw/**: Contains original data files from various sources.
- **data/processed/**: Contains cleaned and processed datasets ready for analysis and modeling.
- **src/**: Source code for data processing, feature engineering, and modeling.
- **notebooks/**: Jupyter notebooks for each step of the workflow.

## Getting Started
1. Clone the repository.
2. Install required dependencies (see `pyproject.toml`).
3. Run the notebooks in order for a complete workflow from raw data to model evaluation.

## Documentation
Additional documentation is available in the `doc/` folder, describing variable dictionaries and data sources.
