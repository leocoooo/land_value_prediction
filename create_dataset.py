import pandas as pd

# Charger et concaténer
years = ['2020', '2021', '2022', '2023', '2024', '2025']
df_vf = pd.concat([
    pd.read_csv(f"data/final_data/DVF_{year}.csv", sep=",", low_memory=False)
    for year in years
], ignore_index=True)

# Filtrer Île-de-France
departements_idf = ['75', '77', '78', '91', '92', '93', '94', '95']
df_vf_idf = df_vf[df_vf['code_departement'].isin(departements_idf)].copy()

# Supprimer colonnes inutiles (adaptées aux nouvelles colonnes)
colonnes_a_supprimer = [
    'numero_disposition', 'adresse_suffixe', 'adresse_code_voie',
    'ancien_code_commune', 'ancien_nom_commune', 'ancien_id_parcelle',
    'numero_volume', 'code_nature_culture', 'code_nature_culture_speciale'
]
# Supprimer uniquement les colonnes qui existent
colonnes_a_supprimer = [col for col in colonnes_a_supprimer if col in df_vf_idf.columns]
df_vf_idf.drop(columns=colonnes_a_supprimer, inplace=True)

# Convertir date et créer variables temporelles
df_vf_idf['date_mutation'] = pd.to_datetime(df_vf_idf['date_mutation'], format='%Y-%m-%d', errors='coerce')
df_vf_idf['annee'] = df_vf_idf['date_mutation'].dt.year
df_vf_idf['mois'] = df_vf_idf['date_mutation'].dt.month
df_vf_idf['trimestre'] = df_vf_idf['date_mutation'].dt.quarter
df_vf_idf['annee_trimestre'] = df_vf_idf['annee'].astype(str) + '-T' + df_vf_idf['trimestre'].astype(str)

# Conversions numériques
numeric_cols = ['valeur_fonciere', 'surface_reelle_bati', 'surface_terrain',
                'nombre_pieces_principales', 'nombre_lots']
for col in numeric_cols:
    if col in df_vf_idf.columns:
        if df_vf_idf[col].dtype == 'object':
            df_vf_idf[col] = df_vf_idf[col].str.replace(',', '.').astype(float)
        else:
            df_vf_idf[col] = pd.to_numeric(df_vf_idf[col], errors='coerce')

# Filtres qualité
df_vf_idf_filtered = df_vf_idf[
    (df_vf_idf['valeur_fonciere'] > 1000) &
    (df_vf_idf['surface_reelle_bati'] > 9) &
    (df_vf_idf['surface_reelle_bati'] < 1000) &
    (df_vf_idf['nombre_pieces_principales'] >= 1) &
    (df_vf_idf['nombre_pieces_principales'] <= 15)
].copy()

# Prix au m²
df_vf_idf_filtered['prix_m2'] = df_vf_idf_filtered['valeur_fonciere'] / df_vf_idf_filtered['surface_reelle_bati']

# Filtrer prix au m² aberrants
df_vf_idf_filtered = df_vf_idf_filtered[
    (df_vf_idf_filtered['prix_m2'] > 500) &
    (df_vf_idf_filtered['prix_m2'] < 20000)
].copy()

# GÉOCODAGE : Créer identifiant de quartier/secteur
df_vf_idf_filtered['quartier'] = (
    df_vf_idf_filtered['nom_commune'].fillna('') +
    ' (' + df_vf_idf_filtered['code_postal'].astype(str).str.zfill(5) + ')'
)

# Pour Paris, utiliser l'arrondissement
df_vf_idf_filtered['arrondissement'] = df_vf_idf_filtered['code_postal'].astype(str).str[-2:].where(
    df_vf_idf_filtered['code_departement'] == '75',
    other=None
)
df_vf_idf_filtered['quartier_detaille'] = df_vf_idf_filtered.apply(
    lambda x: f"Paris {x['arrondissement']}e" if pd.notna(x['arrondissement']) else x['quartier'],
    axis=1
)

# PRIX MÉDIAN PAR QUARTIER ET TRIMESTRE
prix_median_quartier_trimestre = df_vf_idf_filtered.groupby(['quartier_detaille', 'annee_trimestre'])['prix_m2'].agg([
    ('prix_median_m2', 'median'),
    ('nb_transactions', 'count'),
    ('prix_moyen_m2', 'mean')
]).reset_index()

# Filtrer les quartiers avec au moins 5 transactions
prix_median_quartier_trimestre = prix_median_quartier_trimestre[
    prix_median_quartier_trimestre['nb_transactions'] >= 5
]

# Joindre le prix médian au dataframe principal
df_vf_idf_filtered = df_vf_idf_filtered.merge(
    prix_median_quartier_trimestre[['quartier_detaille', 'annee_trimestre', 'prix_median_m2']],
    on=['quartier_detaille', 'annee_trimestre'],
    how='left'
)

# Calculer l'écart par rapport au prix médian
df_vf_idf_filtered['ecart_prix_median_pct'] = (
    (df_vf_idf_filtered['prix_m2'] - df_vf_idf_filtered['prix_median_m2']) /
    df_vf_idf_filtered['prix_median_m2'] * 100
)

# Trier et réinitialiser l'index
df_vf_idf_filtered.sort_values('date_mutation', inplace=True)
df_vf_idf_filtered.reset_index(drop=True, inplace=True)

print(f"Dataset nettoyé : {df_vf_idf_filtered.shape}")
print(f"\nPrix médians calculés pour {prix_median_quartier_trimestre.shape[0]} combinaisons quartier-trimestre")
print("\nAperçu des colonnes créées :")
print(df_vf_idf_filtered[['date_mutation', 'quartier_detaille', 'annee_trimestre', 'prix_m2', 'prix_median_m2', 'ecart_prix_median_pct']].head(10))

df_vf_idf_filtered.to_csv("data/df_vf_idf.csv", index=False)