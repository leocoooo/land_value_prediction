# Copilot Instructions for land_value_prediction

## Project Overview
- Predicts real estate value (€/m²) in Île-de-France using machine learning.
- Data flows from raw sources (`data/raw/`) through cleaning, feature engineering, and modeling, with results and analysis in Jupyter notebooks (`notebooks/`).
- Main code is in `src/land_value_prediction/` (feature engineering, model training, selection, and pipelines).

## Key Workflows
- **Run the full pipeline by executing notebooks in order:**
  1. `00_create_dataset.ipynb` (data fusion)
  2. `01_train_test_split.ipynb` (split, avoid leakage)
  3. `02_analyze_process_clean_data.ipynb` (clean, impute, engineer features)
  4. `03_selection_features.ipynb` (feature selection via SHAP/LightGBM)
  5. `04_*` notebooks (modeling, evaluation, stacking, zone-specific models)
- **Dependencies:** Managed in `pyproject.toml` (Python ≥3.12, heavy use of pandas, scikit-learn, xgboost, lightgbm, optuna, geopandas, etc.).
- **Data conventions:**
  - Use Parquet for processed datasets.
  - Paris vs. non-Paris split is a recurring pattern for model specialization.
  - Feature engineering and cleaning functions are in `pipeline_traitement.py`.
- **Model selection:**
  - Hyperparameter search via Optuna (see `multiple_models.py`).
  - Feature selection via SHAP values (see `feature_selection.py`).
  - Stacking and zone-specific models are common (see `04_multiple_models.ipynb`).

## Project-Specific Patterns
- **Notebooks drive the workflow**; scripts in `src/` are helpers, not entry points.
- **Zone-based modeling:** Many models are trained separately for Paris (code_departement == '75') and the rest.
- **Feature engineering:** Use `do_feature_engineering` and related functions from `pipeline_traitement.py`.
- **Model optimization:** Use `optimize_and_train` from `multiple_models.py` for both RandomForest and XGBoost.
- **Feature selection:** Use SHAP/LightGBM pipeline in `feature_selection.py`.

## Examples
- To train a model for Paris only:
  ```python
  X_train_paris = X_train[X_train["code_departement"] == "75"].drop(columns=["code_departement", "distance_paris"])
  y_train_paris = y_train[X_train["code_departement"] == "75"]
  model_paris_rf, params, score = optimize_and_train(X_train_paris, y_train_paris, ...)
  ```
- To engineer features:
  ```python
  from src.land_value_prediction.pipeline_traitement import do_feature_engineering
  df = do_feature_engineering(df)
  ```

## Testing & Debugging
- No explicit test suite; validate by running notebooks end-to-end.
- Data and model outputs are inspected via notebook cells and plots.

## Documentation
- See `README.md` for workflow and notebook descriptions.
- See `doc/` for variable dictionaries and data source explanations.

---
For new agents: follow notebook order, use helper functions from `src/`, and respect the Paris/hors Paris modeling split where relevant.