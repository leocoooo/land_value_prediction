from sklearn.neighbors import BallTree
import pandas as pd
import numpy as np


def delete_useless_columns(df, columns_to_drop=None):
    return df.drop(columns=columns_to_drop).copy()


def do_feature_engineering(df):

    df_feat = df.copy()

    # Ratio surface/pièces (taille moyenne des pièces)
    df_feat['surface_par_piece'] = (
        df_feat['surface_reelle_bati'] / df_feat['nombre_pieces_principales']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Ratio terrain/bâti (pour maisons)
    df_feat['ratio_terrain_bati'] = (
        df_feat['surface_terrain'] / df_feat['surface_reelle_bati']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Log des surfaces (relations non-linéaires)
    df_feat['log_surface_bati'] = np.log1p(df_feat['surface_reelle_bati'])
    df_feat['log_surface_terrain'] = np.log1p(df_feat['surface_terrain'])
    
    # Densité (nombre de lots peut indiquer copropriété)
    df_feat['densite_lots'] = np.log1p(df_feat['nombre_lots'])
    
    # Années depuis le début (tendance temporelle)
    df_feat['nb_jours_depuis_janvier_2021'] = (df_feat['date_mutation'] - pd.Timestamp("2021-01-01")).dt.days
    
    # Distance au centre (approximatif : Paris = 48.8566, 2.3522)
    paris_lat, paris_lon = 48.8566, 2.3522
    df_feat['distance_paris'] = np.sqrt(
        (df_feat['latitude'] - paris_lat)**2 + 
        (df_feat['longitude'] - paris_lon)**2
    )
    
    # Quadrant géographique (Nord/Sud/Est/Ouest de Paris)
    df_feat['nord_paris'] = (df_feat['latitude'] > paris_lat).astype(int)
    df_feat['est_paris'] = (df_feat['longitude'] > paris_lon).astype(int)
    df_feat['sud_paris'] = (df_feat['latitude'] <= paris_lat).astype(int)
    df_feat['ouest_paris'] = (df_feat['longitude'] <= paris_lon).astype(int)
            
    
    # Prix médian × distance (l'effet du quartier peut varier avec la distance)
    df_feat['prix_median_x_distance'] = (
        df_feat['prix_m2_median_maisons_voisines'] * df_feat['distance_paris']
    )
    
    # Surface × type de bien (déjà encodé dans code_type_local)
    df_feat['surface_x_type'] = (
        df_feat['surface_reelle_bati'] * df_feat['code_type_local'])
    

    df_feat = df_feat.drop(columns=["code_type_local"]).copy()
    
    return df_feat 


def fill_missing_surface_terrain(df):
    
    df['surface_terrain'] = df['surface_terrain'].fillna(0)
    df["log_surface_terrain"] = np.log1p(df["surface_terrain"])

    return df.copy()


def cast_variables(df):

    # convertir certaines colonnes en numériques (en gérant les valeurs manquantes)
    for col in ["commune_revenu_median_2020"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    
    # Cast des colonnes spécifiques en category
    for col in ["ouest_paris", "sud_paris", "est_paris", "nord_paris"]:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            print(f"{col} n'est pas dans les colonnes du dataframe")

    # Cast des colonnes object en category
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() > 300:
            print(f"La colonne {col} a trop de modalités ({df[col].nunique()}) pour être castée en 'category'.")
            continue  # Skip cette colonne
        df[col] = df[col].astype('category')

    # Cast des colonnes numériques
    for col in df.select_dtypes(include=np.number).columns:
        # Vérifier si la colonne contient des décimales
        if df[col].dtype in ['float64', 'float32'] and not (df[col] % 1 == 0).all():
            df[col] = df[col].astype('float32')
            continue
            
        if df[col].min() >= 0:
            if df[col].max() <= 255:
                df[col] = df[col].astype('uint8')
            elif df[col].max() <= 65535:
                df[col] = df[col].astype('uint16')
            elif df[col].max() <= 4294967295:
                df[col] = df[col].astype('uint32')
            else:
                df[col] = df[col].astype('float32')
        else:
            if df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype('int16')
            elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                df[col] = df[col].astype('int32')
            else:
                df[col] = df[col].astype('float32')
            
    return df.copy()


def impute_commune_features_by_neighbors(df, features_to_impute, radius_km=10):
    """
    Impute missing values for specified commune-level features by the mean of neighboring communes
    within a given radius (in kilometers), based on longitude and latitude.
    Prints for each variable and each commune the number of neighboring communes and the imputed value.
    """
    coords = np.radians(df[['latitude', 'longitude']].values)
    tree = BallTree(coords, metric='haversine')
    earth_radius = 6371  # km

    df_imputed = df.copy()
    for idx, row in df[df[features_to_impute].isnull().any(axis=1)].iterrows():
        distances, indices = tree.query_radius(
            [np.radians([row['latitude'], row['longitude']])],
            r=radius_km / earth_radius,
            return_distance=True
        )
        neighbors_idx = indices[0][distances[0] > 0]  # exclude self (distance==0)
        neighbors = df.iloc[neighbors_idx]
        for feature in features_to_impute:
            if pd.isnull(row[feature]):
                mean_val = neighbors[feature].mean()
                df_imputed.at[idx, feature] = mean_val
                print(f"Commune: {row['nom_commune']} | Variable: {feature} | #voisins: {len(neighbors_idx)} | Imputed value: {mean_val}")
    return df_imputed


def one_hot_encode_type_local(df):

    df["is_maison"] = (df["type_local"] == "Maison").astype('int32')
    df["is_appartement"] = (df["type_local"] == "Appartement").astype('int32')

    return df.copy()