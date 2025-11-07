import pandas as pd

# Charger et concaténer
years = ['2020', '2021', '2022', '2023', '2024', '2025']
df_vf = pd.concat([
    pd.read_csv(f"data/final_data/DVF_{year}.csv", sep=",", low_memory=False)
    for year in years
], ignore_index=True)


# Filtrer Île-de-France
departements_idf = ['75', '77', '78', '91', '92', '93', '94', '95']
df_vf_idf = df_vf[df_vf['Code departement'].isin(departements_idf)].copy()

# Supprimer colonnes inutiles
colonnes_a_supprimer = [
    'Identifiant de document', 'Reference document', 'No disposition',
    '1 Articles CGI', '2 Articles CGI', '3 Articles CGI', 
    '4 Articles CGI', '5 Articles CGI'
]
df_vf_idf.drop(columns=colonnes_a_supprimer, inplace=True)


# Convertir date et créer variables temporelles
df_vf_idf['Date mutation'] = pd.to_datetime(df_vf_idf['Date mutation'], format='%d/%m/%Y', errors='coerce')
df_vf_idf['annee'] = df_vf_idf['Date mutation'].dt.year
df_vf_idf['mois'] = df_vf_idf['Date mutation'].dt.month
df_vf_idf['trimestre'] = df_vf_idf['Date mutation'].dt.quarter
df_vf_idf['annee_trimestre'] = df_vf_idf['annee'].astype(str) + '-T' + df_vf_idf['trimestre'].astype(str)

# Conversions numériques
numeric_cols = ['Valeur fonciere', 'Surface reelle bati', 'Surface terrain', 
                'Nombre pieces principales', 'Nombre de lots']
for col in numeric_cols:
       if df_vf_idf[col].dtype == 'object':
              df_vf_idf[col] = df_vf_idf[col].str.replace(',', '.').astype(float)

       else:
              df_vf_idf[col] = pd.to_numeric(df_vf_idf[col], errors='coerce')

# Filtres qualité
df_vf_idf_filtered = df_vf_idf[
    (df_vf_idf['Valeur fonciere'] > 1000) &
    (df_vf_idf['Surface reelle bati'] > 9) &
    (df_vf_idf['Surface reelle bati'] < 1000) &
    (df_vf_idf['Nombre pieces principales'] >= 1) &
    (df_vf_idf['Nombre pieces principales'] <= 15)
]


# Prix au m²
df_vf_idf['prix_m2'] = df_vf_idf['Valeur fonciere'] / df_vf_idf['Surface reelle bati']

# Filtrer prix au m² aberrants
df_vf_idf = df_vf_idf[
    (df_vf_idf['prix_m2'] > 500) & 
    (df_vf_idf['prix_m2'] < 20000)
].copy()

# GÉOCODAGE : Créer identifiant de quartier/secteur
# Option 1 : Par commune + code postal (simple et rapide)
df_vf_idf['quartier'] = df_vf_idf['Commune'].fillna('') + ' (' + df_vf_idf['Code postal'].astype(str).str.zfill(5) + ')'

# Option 2 : Pour Paris, utiliser l'arrondissement (plus précis)
df_vf_idf['arrondissement'] = df_vf_idf['Code postal'].astype(str).str[-2:].where(
    df_vf_idf['Code departement'] == '75',
    other=None
)
df_vf_idf['quartier_detaille'] = df_vf_idf.apply(
    lambda x: f"Paris {x['arrondissement']}e" if pd.notna(x['arrondissement']) else x['quartier'],
    axis=1
)


# PRIX MÉDIAN PAR QUARTIER ET TRIMESTRE
prix_median_quartier_trimestre = df_vf_idf.groupby(['quartier_detaille', 'annee_trimestre'])['prix_m2'].agg([
    ('prix_median_m2', 'median'),
    ('nb_transactions', 'count'),
    ('prix_moyen_m2', 'mean')
]).reset_index()


# Filtrer les quartiers avec au moins 5 transactions pour fiabilité
prix_median_quartier_trimestre = prix_median_quartier_trimestre[
    prix_median_quartier_trimestre['nb_transactions'] >= 5
]


# Joindre le prix médian au dataframe principal
df_vf_idf = df_vf_idf.merge(
    prix_median_quartier_trimestre[['quartier_detaille', 'annee_trimestre', 'prix_median_m2']],
    on=['quartier_detaille', 'annee_trimestre'],
    how='left'
)


# Calculer l'écart par rapport au prix médian du quartier
df_vf_idf['ecart_prix_median_pct'] = (
    (df_vf_idf['prix_m2'] - df_vf_idf['prix_median_m2']) / df_vf_idf['prix_median_m2'] * 100
)

# Trier et réinitialiser l'index
df_vf_idf.sort_values('Date mutation', inplace=True)
df_vf_idf.reset_index(drop=True, inplace=True)

print(f"Dataset nettoyé : {df_vf_idf.shape}")
print("\nPrix médians calculés pour {prix_median_quartier_trimestre.shape[0]} combinaisons quartier-trimestre")
print("\nAperçu des colonnes créées :")
print(df_vf_idf[['Date mutation', 'quartier_detaille', 'annee_trimestre', 'prix_m2', 'prix_median_m2', 'ecart_prix_median_pct']].head(10))
print(df_vf_idf)

df_vf_idf.to_csv("data/df_vf_idf.csv", index=False)