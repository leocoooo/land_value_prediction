# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("resultat_forecast")
RESULTS_DIR.mkdir(exist_ok=True)

# Configuration des données
years_train = ['2020-S2', '2021', '2022', '2023', '2024']
years_test = ['2025-S1']

df_train = pd.concat([
    pd.read_csv(f"data/ValeursFoncieres-{year}.txt", sep="|", low_memory=False)
    for year in years_train
], ignore_index=True)

df_test = pd.concat([
    pd.read_csv(f"data/ValeursFoncieres-{year}.txt", sep="|", low_memory=False)
    for year in years_test
], ignore_index=True)


def prepare_data(df):
    """Prépare les données avec les mêmes filtres que SARIMA"""
    # IDF
    departements_idf = ['75', '77', '78', '91', '92', '93', '94', '95']
    df_idf = df[df['Code departement'].isin(departements_idf)].copy()

    # Convertir date
    df_idf['Date mutation'] = pd.to_datetime(df_idf['Date mutation'], format='%d/%m/%Y', errors='coerce')
    df_idf['annee'] = df_idf['Date mutation'].dt.year
    df_idf['mois'] = df_idf['Date mutation'].dt.month
    df_idf['trimestre'] = df_idf['Date mutation'].dt.quarter
    df_idf['annee_mois'] = df_idf['Date mutation'].dt.to_period('M')

    # Conversions numériques
    numeric_cols = ['Valeur fonciere', 'Surface reelle bati', 'Surface terrain',
                    'Nombre pieces principales', 'Nombre de lots']
    for col in numeric_cols:
        if df_idf[col].dtype == 'object':
            df_idf[col] = df_idf[col].str.replace(',', '.').astype(float)
        else:
            df_idf[col] = pd.to_numeric(df_idf[col], errors='coerce')

    # Filtres qualité
    df_idf = df_idf[
        (df_idf['Valeur fonciere'] > 1000) &
        (df_idf['Surface reelle bati'] > 9) &
        (df_idf['Surface reelle bati'] < 1000) &
        (df_idf['Nombre pieces principales'] >= 1) &
        (df_idf['Nombre pieces principales'] <= 15)
    ].copy()

    # Prix au m²
    df_idf['prix_m2'] = df_idf['Valeur fonciere'] / df_idf['Surface reelle bati']

    # Filtrer prix au m² aberrants
    df_idf = df_idf[
        (df_idf['prix_m2'] > 500) &
        (df_idf['prix_m2'] < 20000)
    ].copy()

    # Zone = département uniquement
    df_idf['zone'] = 'Dept ' + df_idf['Code departement'].astype(str)

    return df_idf


def create_features_for_zone(ts, forecast_months=6):
    """Crée les features temporelles et de lag pour XGBoost"""
    df = pd.DataFrame({'prix_median_m2': ts})
    df['mois'] = df.index.month
    df['annee'] = df.index.year
    df['mois_depuis_debut'] = np.arange(len(df))

    # Features de lag
    for lag in [1, 2, 3, 6, 12]:
        if len(df) > lag:
            df[f'lag_{lag}'] = df['prix_median_m2'].shift(lag)

    # Moyennes mobiles
    for window in [3, 6, 12]:
        if len(df) > window:
            df[f'ma_{window}'] = df['prix_median_m2'].rolling(window=window).mean()

    # Statistiques sur les dernières périodes
    if len(df) > 3:
        df['std_3'] = df['prix_median_m2'].rolling(window=3).std()
    if len(df) > 6:
        df['std_6'] = df['prix_median_m2'].rolling(window=6).std()

    # Tendance
    if len(df) > 1:
        df['tendance'] = df['prix_median_m2'].diff()

    return df


# Préparer les données
print("\nPréparation des données...")
df_train_clean = prepare_data(df_train)
df_test_clean = prepare_data(df_test)

print(f"Données train nettoyées : {df_train_clean.shape[0]} lignes")
print(f"Données test nettoyées : {df_test_clean.shape[0]} lignes")

# Agrégation mensuelle par zone
print("\nAgrégation mensuelle par zone...")
train_monthly_by_zone = df_train_clean.groupby(['zone', 'annee_mois']).agg({
    'prix_m2': ['median', 'mean', 'count'],
    'Valeur fonciere': 'median'
}).reset_index()
train_monthly_by_zone.columns = ['zone', 'annee_mois', 'prix_median_m2', 'prix_moyen_m2', 'nb_transactions', 'valeur_mediane']

test_monthly_by_zone = df_test_clean.groupby(['zone', 'annee_mois']).agg({
    'prix_m2': ['median', 'mean', 'count'],
    'Valeur fonciere': 'median'
}).reset_index()
test_monthly_by_zone.columns = ['zone', 'annee_mois', 'prix_median_m2', 'prix_moyen_m2', 'nb_transactions', 'valeur_mediane']

# Convertir en index temporel
train_monthly_by_zone['date'] = train_monthly_by_zone['annee_mois'].dt.to_timestamp()
test_monthly_by_zone['date'] = test_monthly_by_zone['annee_mois'].dt.to_timestamp()

# Créer séries temporelles par zone
zones = train_monthly_by_zone['zone'].unique()

train_series_by_zone = {}
test_series_by_zone = {}

for zone in zones:
    # Train
    zone_train = train_monthly_by_zone[train_monthly_by_zone['zone'] == zone].copy()
    zone_train.set_index('date', inplace=True)
    zone_train.sort_index(inplace=True)
    train_series_by_zone[zone] = zone_train['prix_median_m2']

    # Test
    zone_test = test_monthly_by_zone[test_monthly_by_zone['zone'] == zone].copy()
    if len(zone_test) > 0:
        zone_test.set_index('date', inplace=True)
        zone_test.sort_index(inplace=True)
        test_series_by_zone[zone] = zone_test['prix_median_m2']
    else:
        test_series_by_zone[zone] = pd.Series(dtype=float)

print(f"\nPériode train : {train_monthly_by_zone['date'].min()} à {train_monthly_by_zone['date'].max()}")
print(f"Période test : {test_monthly_by_zone['date'].min()} à {test_monthly_by_zone['date'].max()}")

# Entraîner un modèle XGBoost pour chaque zone
print("\n" + "="*50)
print("Entraînement des modèles XGBoost par zone (PRIX AU M²)...")
print("="*50)

models_by_zone = {}
forecasts_by_zone = {}
metrics_by_zone = {}

for zone in sorted(zones):
    print(f"\n{'='*50}")
    print(f"Zone : {zone}")
    print(f"{'='*50}")

    ts_train = train_series_by_zone[zone]
    ts_test = test_series_by_zone[zone]

    # Vérifier qu'on a assez de données
    if len(ts_train) < 24:  # Au moins 2 ans de données
        print(f"⚠  Pas assez de données pour {zone} (seulement {len(ts_train)} mois)")
        continue

    if len(ts_test) == 0:
        print(f"⚠  Pas de données test pour {zone}")
        continue

    print(f"Données train : {len(ts_train)} mois")
    print(f"Données test : {len(ts_test)} mois")
    print(f"Prix médian m² train : min={ts_train.min():.0f}€, max={ts_train.max():.0f}€, moyen={ts_train.mean():.0f}€")

    try:
        # Créer les features
        df_features = create_features_for_zone(ts_train)

        # Supprimer les lignes avec des NaN (dues aux lags et moyennes mobiles)
        df_features = df_features.dropna()

        if len(df_features) < 12:
            print(f"⚠  Pas assez de données après création des features pour {zone}")
            continue

        # Séparer X et y
        feature_cols = [col for col in df_features.columns if col != 'prix_median_m2']
        X_train = df_features[feature_cols]
        y_train = df_features['prix_median_m2']

        print(f"Features utilisées : {len(feature_cols)}")

        # Entraîner le modèle XGBoost
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            early_stopping_rounds=20,
            eval_metric='rmse'
        )

        # Split pour early stopping
        split_idx = int(len(X_train) * 0.85)
        X_train_fit = X_train.iloc[:split_idx]
        y_train_fit = y_train.iloc[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:]

        model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        models_by_zone[zone] = model
        print(f"✓ Modèle XGBoost entraîné")

        # Prédictions récursives pour la période test
        forecast_list = []
        last_known_data = ts_train.copy()

        for i in range(len(ts_test)):
            # Créer features pour la prochaine prédiction
            df_pred = create_features_for_zone(last_known_data)
            df_pred = df_pred.dropna()

            if len(df_pred) == 0:
                # Fallback si pas de données
                forecast_list.append(last_known_data.mean())
                continue

            # Prédire le prochain mois
            X_pred = df_pred[feature_cols].iloc[[-1]]
            next_pred = model.predict(X_pred)[0]

            # Appliquer des contraintes raisonnables
            max_reasonable = ts_train.max() * 1.3
            min_reasonable = ts_train.min() * 0.7
            next_pred = np.clip(next_pred, min_reasonable, max_reasonable)

            forecast_list.append(next_pred)

            # Ajouter la prédiction aux données pour le prochain cycle
            next_date = ts_test.index[i]
            last_known_data[next_date] = next_pred

        forecasts_by_zone[zone] = pd.Series(forecast_list, index=ts_test.index)

        # Métriques
        mae = mean_absolute_error(ts_test.values, forecast_list)
        rmse = np.sqrt(mean_squared_error(ts_test.values, forecast_list))
        mape = np.mean(np.abs((ts_test.values - np.array(forecast_list)) / ts_test.values)) * 100
        r2 = r2_score(ts_test.values, forecast_list)

        metrics_by_zone[zone] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'n_features': len(feature_cols)
        }

        print(f"MAE: {mae:.0f}€/m² | RMSE: {rmse:.0f}€/m² | MAPE: {mape:.2f}% | R²: {r2:.4f}")

        # Feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        print(f"\nTop 5 features importantes:")
        print(feature_importance.head().to_string(index=False))

    except Exception as e:
        print(f"❌ Impossible d'entraîner un modèle XGBoost pour {zone}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Résumé global
print("\n" + "="*50)
print("RÉSUMÉ DES PERFORMANCES PAR ZONE (PRIX AU M² - XGBOOST)")
print("="*50)
metrics_df = pd.DataFrame(metrics_by_zone).T
metrics_df = metrics_df.sort_values('MAPE')
print(metrics_df.to_string())
print("\n")

# Générer le graphique d'ensemble
print("Génération du graphique d'ensemble...")

departements = sorted(forecasts_by_zone.keys())

# Vue d'ensemble de toutes les zones
fig, axes = plt.subplots(4, 2, figsize=(16, 16))
axes = axes.flatten()

for idx, dept in enumerate(departements):
    if idx >= len(axes):
        break

    ts_train = train_series_by_zone[dept]
    ts_test = test_series_by_zone[dept]
    forecast = forecasts_by_zone[dept]
    metrics = metrics_by_zone[dept]

    axes[idx].plot(ts_train.index, ts_train.values, label='Historique', color='#2E86AB', linewidth=1.5)
    axes[idx].plot(ts_test.index, ts_test.values, label='Réel', color='#06A77D', linewidth=2, marker='o')
    axes[idx].plot(ts_test.index, forecast.values, label='Prédiction XGBoost', color='#D62828', linewidth=2, marker='^', linestyle='--')

    axes[idx].set_title(f'{dept}\nMAPE: {metrics["MAPE"]:.1f}% | R²: {metrics["R2"]:.2f}', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Prix médian (€/m²)', fontsize=9)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(fontsize=8, loc='best')
    axes[idx].tick_params(axis='x', rotation=45)

plt.suptitle('Vue d\'ensemble des Prévisions de Prix - XGBoost - Tous Départements', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'forecast_prix_xgboost_overview_all.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*50)
print("ANALYSE TERMINÉE - PRIX AU M² (XGBOOST)")
print("="*50)
print(f"\nGraphique généré dans le dossier : {RESULTS_DIR}/")
print(f"  - forecast_prix_xgboost_overview_all.png")

# Sauvegarder les résultats détaillés
print("\nSauvegarde des résultats...")
metrics_df.to_csv(RESULTS_DIR / 'metrics_xgboost_prix.csv')
print("✓ Résultats sauvegardés")

# Comparaison avec SARIMA si disponible
try:
    sarima_metrics = pd.read_csv(RESULTS_DIR / 'metrics_sarima_prix.csv', index_col=0)
    print("\n" + "="*50)
    print("COMPARAISON XGBOOST vs SARIMA")
    print("="*50)

    comparison = pd.DataFrame({
        'XGBoost_MAPE': metrics_df['MAPE'],
        'SARIMA_MAPE': sarima_metrics['MAPE'],
        'Différence (%)': metrics_df['MAPE'] - sarima_metrics['MAPE']
    })
    comparison = comparison.sort_values('Différence (%)')
    print(comparison.to_string())
    print("\nNote: Une différence négative signifie que XGBoost est meilleur")

except FileNotFoundError:
    print("\nPas de fichier SARIMA trouvé pour comparaison")
