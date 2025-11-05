"""
Forecasting des PRIX AU M² avec Prophet
========================================
Ce script utilise Facebook Prophet pour prévoir l'évolution des prix médians au m²
dans les départements d'Île-de-France.

Différences avec forecasting.py (SARIMA):
- Prévoit les PRIX (€/m²) au lieu du VOLUME de transactions
- Utilise Prophet au lieu de SARIMA
- Ajoute des régresseurs externes (volume de transactions, tendances)
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =====================================
# CONFIGURATION
# =====================================

# Dossier de sauvegarde des résultats
RESULTS_DIR = Path("resultat_forecast")
RESULTS_DIR.mkdir(exist_ok=True)

# Périodes d'entraînement et de test
YEARS_TRAIN = ['2020-S2', '2021', '2022', '2023', '2024']
YEARS_TEST = ['2025-S1']

# Départements d'Île-de-France
DEPARTEMENTS_IDF = ['75', '77', '78', '91', '92', '93', '94', '95']

print("="*60)
print("FORECASTING DES PRIX AU M² AVEC PROPHET")
print("="*60)
print(f"\n📁 Résultats sauvegardés dans : {RESULTS_DIR}/")
print(f"📊 Départements analysés : {', '.join(DEPARTEMENTS_IDF)}")
print(f"🎓 Période train : {YEARS_TRAIN}")
print(f"🎯 Période test : {YEARS_TEST}")

# =====================================
# CHARGEMENT DES DONNÉES
# =====================================

print("\n" + "="*60)
print("CHARGEMENT DES DONNÉES DVF")
print("="*60)

# Charger données d'entraînement (2020-2024)
print("\n📥 Chargement des données train...")
df_train = pd.concat([
    pd.read_csv(f"data/ValeursFoncieres-{year}.txt", sep="|", low_memory=False)
    for year in YEARS_TRAIN
], ignore_index=True)
print(f"✓ Données train brutes : {df_train.shape[0]:,} lignes")

# Charger données de test (2025-S1)
print("\n📥 Chargement des données test...")
df_test = pd.concat([
    pd.read_csv(f"data/ValeursFoncieres-{year}.txt", sep="|", low_memory=False)
    for year in YEARS_TEST
], ignore_index=True)
print(f"✓ Données test brutes : {df_test.shape[0]:,} lignes")

# =====================================
# PRÉPARATION ET NETTOYAGE DES DONNÉES
# =====================================

def prepare_data(df):
    """
    Nettoie et prépare les données DVF pour l'analyse.

    Étapes :
    1. Filtrer sur l'Île-de-France
    2. Convertir les dates
    3. Convertir les colonnes numériques
    4. Filtrer les données aberrantes
    5. Calculer le prix au m²
    6. Créer la colonne 'zone' (département)

    Args:
        df (pd.DataFrame): DataFrame brut DVF

    Returns:
        pd.DataFrame: DataFrame nettoyé et enrichi
    """
    # ---- Étape 1 : Filtrer sur l'IDF ----
    df_idf = df[df['Code departement'].isin(DEPARTEMENTS_IDF)].copy()

    # ---- Étape 2 : Convertir les dates ----
    df_idf['Date mutation'] = pd.to_datetime(
        df_idf['Date mutation'],
        format='%d/%m/%Y',
        errors='coerce'
    )
    df_idf['annee'] = df_idf['Date mutation'].dt.year
    df_idf['mois'] = df_idf['Date mutation'].dt.month
    df_idf['trimestre'] = df_idf['Date mutation'].dt.quarter
    df_idf['annee_mois'] = df_idf['Date mutation'].dt.to_period('M')

    # ---- Étape 3 : Conversions numériques ----
    numeric_cols = [
        'Valeur fonciere',
        'Surface reelle bati',
        'Surface terrain',
        'Nombre pieces principales',
        'Nombre de lots'
    ]

    for col in numeric_cols:
        if df_idf[col].dtype == 'object':
            # Remplacer les virgules par des points
            df_idf[col] = df_idf[col].str.replace(',', '.').astype(float)
        else:
            df_idf[col] = pd.to_numeric(df_idf[col], errors='coerce')

    # ---- Étape 4 : Filtres qualité ----
    # Supprimer les transactions aberrantes
    df_idf = df_idf[
        (df_idf['Valeur fonciere'] > 1000) &                    # Min 1000€
        (df_idf['Surface reelle bati'] > 9) &                   # Min 9m²
        (df_idf['Surface reelle bati'] < 1000) &                # Max 1000m²
        (df_idf['Nombre pieces principales'] >= 1) &            # Min 1 pièce
        (df_idf['Nombre pieces principales'] <= 15)             # Max 15 pièces
    ].copy()

    # ---- Étape 5 : Calculer le prix au m² ----
    df_idf['prix_m2'] = df_idf['Valeur fonciere'] / df_idf['Surface reelle bati']

    # Filtrer les prix au m² aberrants
    df_idf = df_idf[
        (df_idf['prix_m2'] > 500) &      # Min 500€/m²
        (df_idf['prix_m2'] < 20000)      # Max 20 000€/m²
    ].copy()

    # ---- Étape 6 : Créer la zone (département) ----
    df_idf['zone'] = 'Dept ' + df_idf['Code departement'].astype(str)

    return df_idf


print("\n" + "="*60)
print("NETTOYAGE DES DONNÉES")
print("="*60)

df_train_clean = prepare_data(df_train)
df_test_clean = prepare_data(df_test)

print(f"\n✓ Train après nettoyage : {df_train_clean.shape[0]:,} lignes")
print(f"✓ Test après nettoyage : {df_test_clean.shape[0]:,} lignes")
print(f"✓ Taux de conservation train : {df_train_clean.shape[0]/df_train.shape[0]*100:.1f}%")
print(f"✓ Taux de conservation test : {df_test_clean.shape[0]/df_test.shape[0]*100:.1f}%")

# =====================================
# AGRÉGATION MENSUELLE PAR DÉPARTEMENT
# =====================================

print("\n" + "="*60)
print("AGRÉGATION MENSUELLE")
print("="*60)

# Agrégation TRAIN : par département et par mois
print("\n📊 Agrégation des données train...")
train_monthly = df_train_clean.groupby(['zone', 'annee_mois']).agg({
    'prix_m2': ['median', 'mean', 'std'],
    'Valeur fonciere': 'median',
    'Nombre pieces principales': 'mean',
    'Surface reelle bati': 'mean',
    'annee_mois': 'count'  # Compte le nombre de transactions
}).reset_index()

# Renommer les colonnes
train_monthly.columns = [
    'zone', 'annee_mois',
    'prix_median_m2', 'prix_moyen_m2', 'prix_std_m2',
    'valeur_mediane', 'nb_pieces_moy', 'surface_moy', 'nb_transactions'
]

# Agrégation TEST : même chose
print("📊 Agrégation des données test...")
test_monthly = df_test_clean.groupby(['zone', 'annee_mois']).agg({
    'prix_m2': ['median', 'mean', 'std'],
    'Valeur fonciere': 'median',
    'Nombre pieces principales': 'mean',
    'Surface reelle bati': 'mean',
    'annee_mois': 'count'
}).reset_index()

test_monthly.columns = [
    'zone', 'annee_mois',
    'prix_median_m2', 'prix_moyen_m2', 'prix_std_m2',
    'valeur_mediane', 'nb_pieces_moy', 'surface_moy', 'nb_transactions'
]

# Convertir en timestamp (requis par Prophet)
train_monthly['date'] = train_monthly['annee_mois'].dt.to_timestamp()
test_monthly['date'] = test_monthly['annee_mois'].dt.to_timestamp()

print(f"\n✓ Périodes train : {train_monthly['date'].min()} → {train_monthly['date'].max()}")
print(f"✓ Périodes test : {test_monthly['date'].min()} → {test_monthly['date'].max()}")
print(f"✓ Départements : {sorted(train_monthly['zone'].unique())}")

# =====================================
# CRÉATION DES SÉRIES TEMPORELLES PAR ZONE
# =====================================

print("\n" + "="*60)
print("PRÉPARATION DES SÉRIES TEMPORELLES")
print("="*60)

zones = sorted(train_monthly['zone'].unique())

# Dictionnaires pour stocker les DataFrames par département
train_data_by_zone = {}
test_data_by_zone = {}

for zone in zones:
    # --- TRAIN ---
    zone_train = train_monthly[train_monthly['zone'] == zone].copy()
    zone_train = zone_train.sort_values('date').reset_index(drop=True)
    train_data_by_zone[zone] = zone_train

    # --- TEST ---
    zone_test = test_monthly[test_monthly['zone'] == zone].copy()
    if len(zone_test) > 0:
        zone_test = zone_test.sort_values('date').reset_index(drop=True)
        test_data_by_zone[zone] = zone_test
    else:
        # Si pas de données test pour ce département
        test_data_by_zone[zone] = pd.DataFrame()

print(f"\n✓ {len(zones)} zones préparées")

# Afficher un aperçu pour un département
sample_zone = zones[0]
print(f"\n📋 Aperçu des données pour {sample_zone}:")
print(train_data_by_zone[sample_zone][['date', 'prix_median_m2', 'nb_transactions']].head())
print(f"   → {len(train_data_by_zone[sample_zone])} mois de données train")
print(f"   → {len(test_data_by_zone[sample_zone])} mois de données test")

# =====================================
# FONCTIONS POUR PROPHET
# =====================================

def prepare_prophet_data(df_zone, target='prix_median_m2', include_regressors=True):
    """
    Prépare les données au format Prophet.

    Prophet requiert :
    - Une colonne 'ds' (date)
    - Une colonne 'y' (variable cible)
    - Optionnellement des régresseurs externes

    Args:
        df_zone (pd.DataFrame): DataFrame pour une zone
        target (str): Nom de la colonne cible à prédire
        include_regressors (bool): Inclure ou non les régresseurs externes

    Returns:
        pd.DataFrame: DataFrame au format Prophet
    """
    df_prophet = pd.DataFrame()
    df_prophet['ds'] = df_zone['date']
    df_prophet['y'] = df_zone[target]

    if include_regressors:
        # Ajouter le volume de transactions comme régresseur
        df_prophet['nb_transactions'] = df_zone['nb_transactions']

        # Ajouter la surface moyenne comme régresseur
        df_prophet['surface_moy'] = df_zone['surface_moy']

        # Ajouter le nombre de pièces moyen
        df_prophet['nb_pieces_moy'] = df_zone['nb_pieces_moy']

    return df_prophet


def train_prophet_model(df_train, df_test, zone_name, use_regressors=True):
    """
    Entraîne un modèle Prophet pour une zone donnée.

    Args:
        df_train (pd.DataFrame): Données d'entraînement (format zone)
        df_test (pd.DataFrame): Données de test (format zone)
        zone_name (str): Nom de la zone
        use_regressors (bool): Utiliser ou non les régresseurs externes

    Returns:
        tuple: (model, forecast, metrics_dict) ou (None, None, None) si erreur
    """
    try:
        # Préparer les données au format Prophet
        train_prophet = prepare_prophet_data(df_train, include_regressors=use_regressors)
        test_prophet = prepare_prophet_data(df_test, include_regressors=use_regressors)

        # Vérifier qu'on a assez de données
        if len(train_prophet) < 24:
            print(f"  ⚠️  Pas assez de données ({len(train_prophet)} mois)")
            return None, None, None

        if len(test_prophet) == 0:
            print(f"  ⚠️  Pas de données test")
            return None, None, None

        # Créer le modèle Prophet
        model = Prophet(
            yearly_seasonality=True,      # Saisonnalité annuelle
            weekly_seasonality=False,     # Pas de saisonnalité hebdomadaire (données mensuelles)
            daily_seasonality=False,      # Pas de saisonnalité journalière
            seasonality_mode='multiplicative',  # Mode multiplicatif (mieux pour les prix)
            changepoint_prior_scale=0.05,      # Flexibilité des changements de tendance
            seasonality_prior_scale=10.0       # Poids de la saisonnalité
        )

        # Ajouter les régresseurs externes si demandé
        if use_regressors:
            model.add_regressor('nb_transactions', standardize=True)
            model.add_regressor('surface_moy', standardize=True)
            model.add_regressor('nb_pieces_moy', standardize=True)

        # Entraîner le modèle (suppression des messages)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_prophet)

        # Préparer le DataFrame de prédiction
        # On doit fournir les valeurs des régresseurs pour les dates futures
        future = test_prophet[['ds']].copy()
        if use_regressors:
            future['nb_transactions'] = test_prophet['nb_transactions'].values
            future['surface_moy'] = test_prophet['surface_moy'].values
            future['nb_pieces_moy'] = test_prophet['nb_pieces_moy'].values

        # Faire les prédictions
        forecast = model.predict(future)

        # Extraire les prédictions (colonne 'yhat')
        y_pred = forecast['yhat'].values
        y_true = test_prophet['y'].values

        # Calculer les métriques
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'nb_train': len(train_prophet),
            'nb_test': len(test_prophet)
        }

        return model, forecast, metrics

    except Exception as e:
        print(f"  ❌ Erreur : {str(e)}")
        return None, None, None

# =====================================
# ENTRAÎNEMENT DES MODÈLES PAR DÉPARTEMENT
# =====================================

print("\n" + "="*60)
print("ENTRAÎNEMENT DES MODÈLES PROPHET")
print("="*60)

# Dictionnaires pour stocker les résultats
models_by_zone = {}
forecasts_by_zone = {}
metrics_by_zone = {}

for zone in zones:
    print(f"\n{'='*60}")
    print(f"📍 Zone : {zone}")
    print(f"{'='*60}")

    # Récupérer les données
    df_train_zone = train_data_by_zone[zone]
    df_test_zone = test_data_by_zone[zone]

    print(f"  📊 {len(df_train_zone)} mois train | {len(df_test_zone)} mois test")

    # Entraîner le modèle
    model, forecast, metrics = train_prophet_model(
        df_train_zone,
        df_test_zone,
        zone,
        use_regressors=True
    )

    if model is not None:
        # Sauvegarder les résultats
        models_by_zone[zone] = model
        forecasts_by_zone[zone] = forecast
        metrics_by_zone[zone] = metrics

        # Afficher les performances
        print(f"  ✓ Modèle entraîné avec succès")
        print(f"  📈 MAE: {metrics['MAE']:.0f} €/m²")
        print(f"  📈 RMSE: {metrics['RMSE']:.0f} €/m²")
        print(f"  📈 MAPE: {metrics['MAPE']:.2f}%")
        print(f"  📈 R²: {metrics['R2']:.4f}")

print("\n" + "="*60)
print(f"✓ {len(models_by_zone)}/{len(zones)} modèles entraînés avec succès")
print("="*60)

# =====================================
# RÉSUMÉ DES PERFORMANCES
# =====================================

print("\n" + "="*60)
print("RÉSUMÉ DES PERFORMANCES PAR DÉPARTEMENT")
print("="*60)

if len(metrics_by_zone) > 0:
    # Créer un DataFrame récapitulatif
    metrics_df = pd.DataFrame(metrics_by_zone).T
    metrics_df = metrics_df.sort_values('MAPE')

    print("\n")
    print(metrics_df.to_string())

    # Statistiques globales
    print("\n" + "="*60)
    print("STATISTIQUES GLOBALES")
    print("="*60)
    print(f"MAPE moyen : {metrics_df['MAPE'].mean():.2f}%")
    print(f"MAPE médian : {metrics_df['MAPE'].median():.2f}%")
    print(f"R² moyen : {metrics_df['R2'].mean():.4f}")
    print(f"MAE moyen : {metrics_df['MAE'].mean():.0f} €/m²")

    # Sauvegarder les métriques
    metrics_df.to_csv(RESULTS_DIR / 'metrics_prophet.csv')
    print(f"\n✓ Métriques sauvegardées : {RESULTS_DIR / 'metrics_prophet.csv'}")
else:
    print("\n⚠️  Aucun modèle entraîné avec succès")

# =====================================
# GÉNÉRATION DES GRAPHIQUES
# =====================================

print("\n" + "="*60)
print("GÉNÉRATION DES GRAPHIQUES")
print("="*60)

if len(forecasts_by_zone) > 0:
    # Graphique d'ensemble : tous les départements
    n_zones = len(forecasts_by_zone)
    n_cols = 2
    n_rows = (n_zones + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_zones == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, zone in enumerate(sorted(forecasts_by_zone.keys())):
        ax = axes[idx]

        # Récupérer les données
        df_train_zone = train_data_by_zone[zone]
        df_test_zone = test_data_by_zone[zone]
        forecast = forecasts_by_zone[zone]
        metrics = metrics_by_zone[zone]

        # Historique (train)
        ax.plot(
            df_train_zone['date'],
            df_train_zone['prix_median_m2'],
            label='Historique',
            color='#2E86AB',
            linewidth=2
        )

        # Valeurs réelles (test)
        ax.plot(
            df_test_zone['date'],
            df_test_zone['prix_median_m2'],
            label='Réel',
            color='#06A77D',
            linewidth=2.5,
            marker='o',
            markersize=6
        )

        # Prédictions Prophet
        ax.plot(
            forecast['ds'],
            forecast['yhat'],
            label='Prophet',
            color='#D62828',
            linewidth=2.5,
            marker='^',
            markersize=6,
            linestyle='--'
        )

        # Intervalle de confiance (optionnel)
        ax.fill_between(
            forecast['ds'],
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            color='#D62828',
            alpha=0.2,
            label='Intervalle 95%'
        )

        # Titre et légende
        ax.set_title(
            f'{zone}\nMAPE: {metrics["MAPE"]:.1f}% | R²: {metrics["R2"]:.3f}',
            fontsize=11,
            fontweight='bold'
        )
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Prix médian (€/m²)', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, loc='best')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    # Masquer les axes inutilisés
    for idx in range(n_zones, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(
        'Prévisions des Prix au m² - Prophet avec Régresseurs Externes',
        fontsize=16,
        fontweight='bold',
        y=0.998
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'forecast_prix_prophet_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Graphique sauvegardé : {RESULTS_DIR / 'forecast_prix_prophet_overview.png'}")

    # ---- Graphique individuel pour le meilleur département ----
    best_zone = metrics_df.sort_values('MAPE').index[0]
    print(f"\n📊 Création d'un graphique détaillé pour le meilleur département : {best_zone}")

    df_train_zone = train_data_by_zone[best_zone]
    df_test_zone = test_data_by_zone[best_zone]
    forecast = forecasts_by_zone[best_zone]
    metrics = metrics_by_zone[best_zone]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Historique
    ax.plot(
        df_train_zone['date'],
        df_train_zone['prix_median_m2'],
        label='Données historiques (train)',
        color='#2E86AB',
        linewidth=2.5,
        marker='o',
        markersize=4
    )

    # Réel
    ax.plot(
        df_test_zone['date'],
        df_test_zone['prix_median_m2'],
        label='Valeurs réelles (test)',
        color='#06A77D',
        linewidth=3,
        marker='o',
        markersize=8
    )

    # Prédictions
    ax.plot(
        forecast['ds'],
        forecast['yhat'],
        label='Prédictions Prophet',
        color='#D62828',
        linewidth=3,
        marker='^',
        markersize=8,
        linestyle='--'
    )

    # Intervalle de confiance
    ax.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='#D62828',
        alpha=0.15,
        label='Intervalle de confiance 95%'
    )

    ax.set_title(
        f'Prévision des Prix au m² - {best_zone}\n'
        f'MAE: {metrics["MAE"]:.0f} €/m² | RMSE: {metrics["RMSE"]:.0f} €/m² | '
        f'MAPE: {metrics["MAPE"]:.2f}% | R²: {metrics["R2"]:.4f}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prix médian (€/m²)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.tick_params(axis='both', labelsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f'forecast_prix_prophet_{best_zone.replace(" ", "_")}.png'
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Graphique détaillé sauvegardé : {RESULTS_DIR / filename}")

else:
    print("\n⚠️  Aucun graphique généré (pas de prévisions)")

# =====================================
# FIN DU SCRIPT
# =====================================

print("\n" + "="*60)
print("✅ ANALYSE TERMINÉE - FORECASTING PRIX AU M² (PROPHET)")
print("="*60)
print(f"\n📁 Tous les résultats sont dans : {RESULTS_DIR}/")
print("   - metrics_prophet.csv (métriques détaillées)")
print("   - forecast_prix_prophet_overview.png (vue d'ensemble)")
print("   - forecast_prix_prophet_Dept_XX.png (graphique détaillé meilleur dept)")
print("\n" + "="*60)