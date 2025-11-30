import pandas as pd
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

def objective(trial, X, y, sample_size=None, initial_params=None, model_cls=RandomForestRegressor):
    """Fonction objectif pour Optuna, compatible RF et XGBoost"""
    # Définir les plages de recherche autour des paramètres initiaux
    if model_cls.__name__ == "RandomForestRegressor":
        if initial_params is not None:
            params = {
                'n_estimators': trial.suggest_int(
                    'n_estimators', 
                    max(50, int(initial_params['n_estimators'] * 0.8)), 
                    min(300, int(initial_params['n_estimators'] * 1.2))
                ),
                'max_depth': trial.suggest_int(
                    'max_depth', 
                    max(5, initial_params['max_depth'] - 5), 
                    min(30, initial_params['max_depth'] + 5)
                ),
                'min_samples_split': trial.suggest_int(
                    'min_samples_split', 
                    max(2, initial_params['min_samples_split'] - 3), 
                    min(20, initial_params['min_samples_split'] + 3)
                ),
                'min_samples_leaf': trial.suggest_int(
                    'min_samples_leaf', 
                    max(1, initial_params['min_samples_leaf'] - 2), 
                    min(10, initial_params['min_samples_leaf'] + 2)
                ),
                'max_features': trial.suggest_float(
                    'max_features', 
                    max(0.1, initial_params['max_features'] - 0.2), 
                    min(1.0, initial_params['max_features'] + 0.2)
                ),
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'random_state': 42,
                'n_jobs': -1
            }
    else:  # XGBoost
        if initial_params is not None:
            params = {
                'n_estimators': trial.suggest_int(
                    'n_estimators', 
                    max(100, int(initial_params['n_estimators'] * 0.8)), 
                    min(1500, int(initial_params['n_estimators'] * 1.2))
                ),
                'max_depth': trial.suggest_int(
                    'max_depth', 
                    max(3, initial_params['max_depth'] - 3), 
                    min(20, initial_params['max_depth'] + 3)
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 
                    max(0.001, initial_params['learning_rate'] * 0.8), 
                    min(0.3, initial_params['learning_rate'] * 1.2)
                ),
                'subsample': trial.suggest_float(
                    'subsample', 
                    max(0.5, initial_params['subsample'] - 0.2), 
                    min(1.0, initial_params['subsample'] + 0.2)
                ),
                'colsample_bytree': trial.suggest_float(
                    'colsample_bytree', 
                    max(0.3, initial_params['colsample_bytree'] - 0.2), 
                    min(1.0, initial_params['colsample_bytree'] + 0.2)
                ),
                'min_child_weight': trial.suggest_int(
                    'min_child_weight', 
                    max(1, initial_params['min_child_weight'] - 2), 
                    min(10, initial_params['min_child_weight'] + 2)
                ),
                'gamma': trial.suggest_float(
                    'gamma', 
                    max(0, initial_params['gamma'] - 2), 
                    initial_params['gamma'] + 2
                ),
                'reg_alpha': trial.suggest_float(
                    'reg_alpha', 
                    max(0, initial_params['reg_alpha'] * 0.8), 
                    initial_params['reg_alpha'] * 1.2
                ),
                'reg_lambda': trial.suggest_float(
                    'reg_lambda', 
                    max(0, initial_params['reg_lambda'] * 0.8), 
                    initial_params['reg_lambda'] * 1.2
                ),
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'random_state': 42,
                'n_jobs': -1
            }
    # Échantillonnage si demandé
    if sample_size is not None:
        n_samples = min(sample_size, len(X))
        np.random.seed(42) 
        indices = np.random.choice(len(X), size=n_samples, replace=False)
        X_sample = X.iloc[indices]
        y_sample = y.iloc[indices]
    else:
        X_sample = X
        y_sample = y

    model = model_cls(**params)
    scores = cross_val_score(model, X_sample, y_sample, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()


def optimize_and_train(X_train, y_train, zone_name, sample_size=None, n_trials=20, initial_params=None, model_cls=RandomForestRegressor):
    """Optimise les hyperparamètres et entraîne le modèle final (RF ou XGBoost)"""
    print(f"\n{'='*60}")
    print(f"Optimisation pour {zone_name}")
    if sample_size is not None:
        actual_size = min(sample_size, len(X_train))
        print(f"Utilisation de {actual_size} observations pour l'optimisation (sur {len(X_train)} disponibles)")
    else:
        print(f"Utilisation de toutes les données pour l'optimisation ({len(X_train)} observations)")
    if initial_params is not None:
        print("Recherche LOCALE autour des hyperparamètres initiaux")
        print(f"Paramètres de départ: {initial_params}")
    else:
        print("Recherche GLOBALE des hyperparamètres")
    print(f"{'='*60}")

    study = optuna.create_study(direction='minimize', study_name=f"{model_cls.__name__}_{zone_name}")
    if initial_params is not None:
        print("Évaluation des hyperparamètres de départ...")
        study.enqueue_trial(initial_params)
    print(f"Recherche des meilleurs hyperparamètres ({n_trials} trials)...")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, sample_size=sample_size, initial_params=initial_params, model_cls=model_cls), 
        n_trials=n_trials, 
        show_progress_bar=True
    )
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    print("\nMeilleurs hyperparamètres trouvés :")
    for param, value in best_params.items():
        if param not in ['random_state', 'n_jobs']:
            initial_val = initial_params.get(param, 'N/A') if initial_params else 'N/A'
            print(f"  {param}: {value} (initial: {initial_val})")
    print(f"\nMeilleur MAE en CV : {study.best_value:.4f}")

    print(f"\nEntraînement du modèle final sur {zone_name} (100% des données - {len(X_train)} observations)...")
    final_model = model_cls(**best_params)
    final_model.fit(X_train, y_train)
    print(f"Modèle sur {zone_name} entraîné.")

    return final_model, best_params, study.best_value


def predict_xgb_by_zone(df, model_paris, model_hors_paris):
    preds = pd.Series(index=df.index, dtype="float32")
    paris_mask = df["code_departement"] == "75"
    hors_paris_mask = df["code_departement"] != "75"

    # Paris
    if paris_mask.any():
        X_paris = df[paris_mask].drop(columns=["code_departement", "distance_paris"], errors="ignore")
        preds.loc[paris_mask] = model_paris.predict(X_paris)
    # Hors Paris
    if hors_paris_mask.any():
        X_hors_paris = df[hors_paris_mask].drop(columns=["code_departement"], errors="ignore")
        preds.loc[hors_paris_mask] = model_hors_paris.predict(X_hors_paris)
    return preds