# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("resultat_forecast")
RESULTS_DIR.mkdir(exist_ok=True)


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
    # IDF
    departements_idf = ['75', '77', '78', '91', '92', '93', '94', '95']
    df_idf = df[df['Code departement'].isin(departements_idf)].copy()

    # Convertir date
    df_idf['Date mutation'] = pd.to_datetime(df_idf['Date mutation'], format='%d/%m/%Y', errors='coerce')
    df_idf['annee'] = df_idf['Date mutation'].dt.year
    df_idf['mois'] = df_idf['Date mutation'].dt.month
    df_idf['trimestre'] = df_idf['Date mutation'].dt.quarter
    df_idf['annee_mois'] = df_idf['Date mutation'].dt.to_period('M')

    # Conversions numeriques
    numeric_cols = ['Valeur fonciere', 'Surface reelle bati', 'Surface terrain',
                    'Nombre pieces principales', 'Nombre de lots']
    for col in numeric_cols:
        if df_idf[col].dtype == 'object':
            df_idf[col] = df_idf[col].str.replace(',', '.').astype(float)
        else:
            df_idf[col] = pd.to_numeric(df_idf[col], errors='coerce')

    # Filtres qualite
    df_idf = df_idf[
        (df_idf['Valeur fonciere'] > 1000) &
        (df_idf['Surface reelle bati'] > 9) &
        (df_idf['Surface reelle bati'] < 1000) &
        (df_idf['Nombre pieces principales'] >= 1) &
        (df_idf['Nombre pieces principales'] <= 15)
    ].copy()

    # Prix au m2
    df_idf['prix_m2'] = df_idf['Valeur fonciere'] / df_idf['Surface reelle bati']

    # Filtrer prix au m2 aberrants
    df_idf = df_idf[
        (df_idf['prix_m2'] > 500) &
        (df_idf['prix_m2'] < 20000)
    ].copy()

    # Zone = departement uniquement
    df_idf['zone'] = 'Dept ' + df_idf['Code departement'].astype(str)

    return df_idf

# Preparer les donnees
print("\nPreparation des donnees...")
df_train_clean = prepare_data(df_train)
df_test_clean = prepare_data(df_test)

print(f"Donnees train nettoyees : {df_train_clean.shape[0]} lignes")
print(f"Donnees test nettoyees : {df_test_clean.shape[0]} lignes")

# Agreger par mois et par zone (departement ou arrondissement)
print("\nAgregation mensuelle par zone...")
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

# Creer un dictionnaire de series temporelles par zone
zones = train_monthly_by_zone['zone'].unique()

train_series_by_zone = {}
test_series_by_zone = {}

for zone in zones:
    # Train
    zone_train = train_monthly_by_zone[train_monthly_by_zone['zone'] == zone].copy()
    zone_train.set_index('date', inplace=True)
    zone_train.sort_index(inplace=True)
    train_series_by_zone[zone] = zone_train['prix_median_m2']  # CHANGEMENT ICI : prix_median_m2 au lieu de nb_transactions

    # Test
    zone_test = test_monthly_by_zone[test_monthly_by_zone['zone'] == zone].copy()
    if len(zone_test) > 0:
        zone_test.set_index('date', inplace=True)
        zone_test.sort_index(inplace=True)
        test_series_by_zone[zone] = zone_test['prix_median_m2']  # CHANGEMENT ICI : prix_median_m2 au lieu de nb_transactions
    else:
        test_series_by_zone[zone] = pd.Series(dtype=float)

print(f"\nPeriode train : {train_monthly_by_zone['date'].min()} e {train_monthly_by_zone['date'].max()}")
print(f"Periode test : {test_monthly_by_zone['date'].min()} e {test_monthly_by_zone['date'].max()}")

# Fonction pour trouver les meilleurs parametres SARIMA
def find_best_sarima(ts, seasonal_period=12):
    """Trouve les meilleurs parametres SARIMA avec une recherche simplifiee"""
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None

    # Parametres e tester (reduit pour la vitesse)
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]

    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            try:
                                model = SARIMAX(ts,
                                              order=(p, d, q),
                                              seasonal_order=(P, D, Q, seasonal_period),
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)
                                results = model.fit(disp=False, maxiter=50)
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (p, d, q)
                                    best_seasonal_order = (P, D, Q, seasonal_period)
                            except:
                                continue

    return best_order, best_seasonal_order, best_aic

# Entrainer un modele SARIMA pour chaque zone
print("\n" + "="*50)
print("Entrainement des modeles SARIMA par zone (PRIX AU Me)...")
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

    # Verifier qu'on a assez de donnees
    if len(ts_train) < 24:  # Au moins 2 ans de donnees
        print(f"e  Pas assez de donnees pour {zone} (seulement {len(ts_train)} mois)")
        continue

    if len(ts_test) == 0:
        print(f"e  Pas de donnees test pour {zone}")
        continue

    print(f"Donnees train : {len(ts_train)} mois")
    print(f"Donnees test : {len(ts_test)} mois")
    print(f"Prix median me train : min={ts_train.min():.0f}e, max={ts_train.max():.0f}e, moyen={ts_train.mean():.0f}e")

    # Entraener le modele SARIMA
    try:
        # Recherche des meilleurs parametres
        print("Recherche des meilleurs parametres SARIMA...")
        best_order, best_seasonal_order, best_aic = find_best_sarima(ts_train, seasonal_period=12)

        if best_order is None:
            print(f"e  Impossible de trouver des parametres SARIMA pour {zone}")
            continue

        print(f"Meilleurs parametres : order={best_order}, seasonal_order={best_seasonal_order}, AIC={best_aic:.2f}")

        # Entraener le modele final avec contraintes
        model = SARIMAX(ts_train,
                       order=best_order,
                       seasonal_order=best_seasonal_order,
                       enforce_stationarity=True,
                       enforce_invertibility=True)

        fitted_model = model.fit(disp=False, maxiter=200, method='lbfgs')
        models_by_zone[zone] = fitted_model
        print(f" Modele SARIMA entraene")

        # Predictions avec gestion des erreurs
        forecast_steps = len(ts_test)
        forecast = fitted_model.forecast(steps=forecast_steps)

        # Verifier si les predictions sont aberrantes pour les prix
        max_reasonable = ts_train.max() * 1.5  # Au max 50% de plus que le max historique pour les prix
        min_reasonable = ts_train.min() * 0.5  # Au min 50% du min historique pour les prix

        if np.any(np.abs(forecast) > max_reasonable) or np.any(forecast < min_reasonable) or np.any(np.isnan(forecast)) or np.any(np.isinf(forecast)):
            print(f"e  Predictions aberrantes detectees, utilisation d'une approche plus simple...")
            # Fallback : utiliser une moyenne mobile simple
            forecast = np.full(forecast_steps, ts_train.rolling(window=6).mean().iloc[-1])

        # S'assurer que les predictions sont dans un intervalle raisonnable
        forecast = np.clip(forecast, min_reasonable, max_reasonable)

        forecasts_by_zone[zone] = pd.Series(forecast, index=ts_test.index)

        # Metriques
        mae = mean_absolute_error(ts_test.values, forecast)
        rmse = np.sqrt(mean_squared_error(ts_test.values, forecast))
        mape = np.mean(np.abs((ts_test.values - forecast) / ts_test.values)) * 100
        r2 = r2_score(ts_test.values, forecast)

        metrics_by_zone[zone] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'order': str(best_order),
            'seasonal_order': str(best_seasonal_order)
        }

        print(f"MAE: {mae:.0f}e/me | RMSE: {rmse:.0f}e/me | MAPE: {mape:.2f}% | Re: {r2:.4f}")

    except Exception as e:
        print(f"L Impossible d'entraener un modele SARIMA pour {zone}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Resume global
print("\n" + "="*50)
print("ReSUMe DES PERFORMANCES PAR ZONE (PRIX AU Me)")
print("="*50)
metrics_df = pd.DataFrame(metrics_by_zone).T
metrics_df = metrics_df.sort_values('MAPE')
print(metrics_df.to_string())
print("\n")

# Generer le graphique d'ensemble
print("Generation du graphique d'ensemble...")

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
    axes[idx].plot(ts_test.index, ts_test.values, label='Reel', color='#06A77D', linewidth=2, marker='o')
    axes[idx].plot(ts_test.index, forecast.values, label='Prediction', color='#D62828', linewidth=2, marker='^', linestyle='--')

    axes[idx].set_title(f'{dept}\nMAPE: {metrics["MAPE"]:.1f}% | Re: {metrics["R2"]:.2f}', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Prix median (e/me)', fontsize=9)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(fontsize=8, loc='best')
    axes[idx].tick_params(axis='x', rotation=45)

plt.suptitle('Vue d\'ensemble des Previsions de Prix - Tous Departements', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'forecast_prix_overview_all.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*50)
print("ANALYSE TERMINeE - PRIX AU Me (SARIMA)")
print("="*50)
print(f"\nGraphique genere dans le dossier : {RESULTS_DIR}/")
print(f"  - forecast_prix_overview_all.png")

# Sauvegarder les resultats detailles
print("\nSauvegarde des resultats...")
metrics_df.to_csv(RESULTS_DIR / 'metrics_sarima_prix.csv')
print(" Resultats sauvegardes")
