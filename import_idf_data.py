import pandas as pd

# Charger et concaténer
years = ['2020', '2021', '2022', '2023', '2024', '2025']
df_vf = pd.concat([
    pd.read_csv(f"data/raw/initial_data_files/DVF_{year}.csv", sep=",", low_memory=False)
    for year in years
], ignore_index=True)

# Filtrer Île-de-France
departements_idf = ['75', '77', '78', '91', '92', '93', '94', '95']
df_vf_idf = df_vf[df_vf['code_departement'].isin(departements_idf)].copy()

# Supprimer colonnes inutiles (revoir plus tard si on peut en réutiliser certaines)
colonnes_a_supprimer = [
    'numero_disposition', 'adresse_suffixe', 'adresse_code_voie',
    'ancien_code_commune', 'ancien_nom_commune', 'ancien_id_parcelle',
    'numero_volume', 'code_nature_culture', 'code_nature_culture_speciale', 
    'id_mutation', 'adresse_numero', 'id_parcelle',
    'lot1_numero', 'lot2_numero', 'lot3_numero',
    'lot4_numero', 'lot5_numero', 'lot1_surface_carrez',
    'lot2_surface_carrez', 'lot3_surface_carrez', 'lot4_surface_carrez', 
    'lot5_surface_carrez', 'nature_culture', 'nature_culture_speciale',
    ]

df_vf_idf.drop(columns=colonnes_a_supprimer, inplace=True)

# Export de la base brute 
df_vf_idf.to_parquet("data/processed/raw_idf_data/raw_idf_data.parquet")

