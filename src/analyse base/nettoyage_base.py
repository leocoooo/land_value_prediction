import numpy as np
import pandas as pd


df = pd.read_csv("../src/data/df_vf_idf.csv", sep=",", low_memory=False)
print(df.head(5))

df.shape

#print(df.dtypes)

# Voir valeur manquante 

valeurmanquante = df.isnull().sum()
print(valeurmanquante)

## beucoup de valeur manquante pour loi carrez, je supprime ces colonnes. En plus, on a deja les surfaces dans surface_reelle_bati (j'au comparé et c'est sensiblement pareil)

# Quppression des colonnes avec trop de VM et inutiles

df_clean = df.drop(columns=['id_mutation',
    'adresse_numero', 'id_parcelle', 
    'lot1_numero', 'lot2_numero', 'lot3_numero', 
    'lot4_numero', 'lot5_numero', 'lot1_surface_carrez', 'lot2_surface_carrez', 
    'lot3_surface_carrez','lot4_surface_carrez', 'lot5_surface_carrez', 'arrondissement', 
    'nature_culture', 'nature_culture_speciale', 'surface_terrain', 'annee_trimestre'
], errors='ignore') 

print(df_clean.dtypes)
valeurmanquante = df_clean.isnull().sum()
print(valeurmanquante)



# Pour le prix médian au m2, je regarde si il y a pour un même quartier_detaille déjà une valeur sinon on met le prix au m2

quartier_prix_median = df_clean[df_clean['prix_median_m2'].notna()].groupby('quartier_detaille')['prix_median_m2'].first().to_dict()

mask_missing = df_clean['prix_median_m2'].isna()

for idx in df_clean[mask_missing].index:
    quartier = df_clean.loc[idx, 'quartier_detaille']
    
    if quartier in quartier_prix_median:
        df_clean.loc[idx, 'prix_median_m2'] = quartier_prix_median[quartier]
    else:
        df_clean.loc[idx, 'prix_median_m2'] = df_clean.loc[idx, 'prix_m2']

# Recalculer ecart_prix_median_pct
df_clean['ecart_prix_median_pct'] = (
    (df_clean['prix_m2'] - df_clean['prix_median_m2']) / df_clean['prix_median_m2'] * 100
)

valeurmanquante = df_clean.isnull().sum()
print(valeurmanquante)
# Sauvegarde du dataframe nettoyé
#df_clean.to_csv("data/df_vf_idf_clean.csv", index=False)









