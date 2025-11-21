import os
import urllib.request
import zipfile
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import Point
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION
# ========================================

CATEGORIES_POI = {
    'education': ['school', 'kindergarten', 'college', 'university', 'library'],
    'commerce': ['supermarket', 'convenience', 'mall', 'department_store', 'bakery', 'butcher', 'greengrocer', 'marketplace'],
    'restauration': ['restaurant', 'cafe', 'fast_food', 'bar', 'pub', 'food_court', 'ice_cream', 'biergarten'],
    'sante': ['hospital', 'clinic', 'doctors', 'dentist', 'pharmacy', 'veterinary', 'nursing_home'],
    'loisirs': ['cinema', 'theatre', 'arts_centre', 'nightclub', 'casino', 'community_centre', 'social_facility', 'park', 'playground', 'sports_centre', 'swimming_pool', 'pitch', 'stadium'],
    'services': ['bank', 'atm', 'post_office', 'post_box', 'police', 'fire_station', 'townhall', 'courthouse', 'embassy']
}

CATEGORIES_TRANSPORT = {
    'transport_lourd': ['railway_station', 'station', 'halt', 'tram_stop'],
    'bus': ['bus_stop', 'bus_station'],
    'transport_autre': ['taxi', 'ferry_terminal', 'aerodrome']
}

# ========================================
# 1. TÉLÉCHARGEMENT & EXTRACTION (Simplifié)
# ========================================

def telecharger_et_extraire_geofabrik(annee: int, output_dir: str = "data/raw/geofabrik") -> Tuple[str, str]:
    """
    Télécharge et extrait les données Géofabrik pour une année donnée.
    Retourne les chemins vers les shapefiles POI et transport.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construire l'URL
    yy = f"{annee % 100:02d}0101"
    filename = f"ile-de-france-{yy}-free.shp.zip"
    url = f"https://download.geofabrik.de/europe/france/{filename}"
    
    zip_path = output_dir / filename
    extract_dir = output_dir / f"idf_{annee}"
    
    # Télécharger si nécessaire
    if not zip_path.exists():
        print(f"Téléchargement {annee}... ", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("✓")
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return None, None
    else:
        print(f"{annee} déjà téléchargé")
    
    # Extraire si nécessaire
    if not extract_dir.exists():
        print(f"Extraction {annee}... ", end="", flush=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        print("✓")
    
    # Trouver les shapefiles
    poi_shp = extract_dir / "gis_osm_pois_a_free_1.shp"
    transport_shp = extract_dir / "gis_osm_transport_free_1.shp"
    
    # Vérifier que les fichiers existent
    if not poi_shp.exists():
        # Essayer l'autre nom possible
        poi_shp = extract_dir / "gis_osm_pois_free_1.shp"
    
    return str(poi_shp), str(transport_shp)


def preparer_toutes_annees(annees: List[int] = [2020, 2021, 2022, 2023, 2024]) -> None:
    """
    Télécharge et extrait toutes les années en une seule fois.
    """
    print("="*70)
    print("TÉLÉCHARGEMENT & EXTRACTION - DONNÉES GÉOFABRIK")
    print("="*70 + "\n")
    
    for annee in annees:
        telecharger_et_extraire_geofabrik(annee)
    
    print("\nToutes les données sont prêtes !")


# ========================================
# 2. CHARGEMENT DES DONNÉES (Optimisé)
# ========================================

def charger_pois_et_transport(annee: int) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Charge les POIs et transports pour une année donnée.
    Retourne les GeoDataFrames déjà convertis en EPSG:2154.
    EPSG:2154 est le code pour le système de coordonnées Lambert-93, qui est la projection officielle de la France métropolitaine essentiel pour déterminer les POIs à proximité.
    """
    poi_path, transport_path = telecharger_et_extraire_geofabrik(annee)
    
    # Charger et convertir directement en Lambert 93
    pois = gpd.read_file(poi_path).to_crs('EPSG:2154') if os.path.exists(poi_path) else gpd.GeoDataFrame()
    transport = gpd.read_file(transport_path).to_crs('EPSG:2154') if os.path.exists(transport_path) else gpd.GeoDataFrame()
    
    return pois, transport


# =========================
# 3. AGRÉGATION DES POIs 
# =========================

def agreger_features_geo(df: pd.DataFrame, pois: gpd.GeoDataFrame, transport: gpd.GeoDataFrame, rayon: int = 1000, batch_name: str = "batch") -> pd.DataFrame:
    """
    Pour chaque transaction, calcule :
    - Le nombre de POIs par catégorie dans le rayon
    - La distance au POI le plus proche par catégorie
    - Les mêmes stats pour les transports
    - Des métriques globales (total, diversité, densité relative)
    """
    print(f"\n Traitement {batch_name} ({len(df):,} transactions, rayon={rayon}m)")
    
    # 1. Nettoyage et préparation
    df = df.copy()
    #Masque des coordonnées valides (France métropolitaine approximative)
    mask_valid = (
        df['latitude'].notna() & df['longitude'].notna() &
        (df['latitude'] > 41) & (df['latitude'] < 52) &
        (df['longitude'] > -5) & (df['longitude'] < 10)
    )
    
    df_valid = df[mask_valid].copy()
    df_invalid = df[~mask_valid].copy()
    
    if len(df_valid) == 0:
        print("Aucune coordonnée valide !")
        return df
    
    print(f" {len(df_valid):,} coordonnées valides ({len(df_valid)/len(df)*100:.1f}%)")
    
    # 2. Créer GeoDataFrame des transactions
    geometry = [Point(lon, lat) for lon, lat in zip(df_valid['longitude'], df_valid['latitude'])]
    gdf_trans = gpd.GeoDataFrame(df_valid, geometry=geometry, crs='EPSG:4326').to_crs('EPSG:2154')
    coords_trans = np.array([[g.x, g.y] for g in gdf_trans.geometry])
    
    # 3. Fonction helper pour calculer les features d'une catégorie
    def calculer_features_categorie(gdf_source: gpd.GeoDataFrame, categories: Dict, prefix: str):
        for cat_name, cat_types in categories.items():
            if 'fclass' not in gdf_source.columns:
                continue
            # fclass indique la catégorie ou le type de point d'intérêt (POI)
            # Filtrer par catégorie
            mask = gdf_source['fclass'].isin(cat_types)
            gdf_cat = gdf_source[mask]
            
            if len(gdf_cat) > 0:
                # Gérer Points et Polygones : extraire centroïde si nécessaire
                # Vérifier si on a des Points ou des Polygones
                geom_types = gdf_cat.geometry.geom_type.unique()
                
                if len(geom_types) == 1 and geom_types[0] == 'Point':
                    # Optimisation : tous Points, extraction vectorisée
                    coords_cat = np.array([[g.x, g.y] for g in gdf_cat.geometry])
                else:
                    # Mélange de types ou Polygones : extraire centroïde au besoin
                    coords_cat = []
                    for geom in gdf_cat.geometry:
                        if geom.geom_type == 'Point':
                            coords_cat.append([geom.x, geom.y])
                        else:  # Polygon, MultiPolygon, LineString, etc.
                            centroid = geom.centroid
                            coords_cat.append([centroid.x, centroid.y])
                    coords_cat = np.array(coords_cat)
                
                tree = cKDTree(coords_cat)
                
                # Comptage dans le rayon
                indices = tree.query_ball_point(coords_trans, r=rayon)
                df_valid[f'nb_{cat_name}_{rayon}m'] = [len(idx) for idx in indices]
                
                # Distance au plus proche
                distances, _ = tree.query(coords_trans, k=1)
                df_valid[f'dist_{cat_name}'] = distances
            else:
                df_valid[f'nb_{cat_name}_{rayon}m'] = 0
                df_valid[f'dist_{cat_name}'] = np.nan
    
    # 4. Appliquer aux POIs et transports
    print(f"   → POIs...", end=" ", flush=True)
    calculer_features_categorie(pois, CATEGORIES_POI, 'poi')
    print("✓")
    
    print(f"   → Transports...", end=" ", flush=True)
    calculer_features_categorie(transport, CATEGORIES_TRANSPORT, 'transport')
    print("✓")
    
    # 5. Métriques globales
    cols_nb = [c for c in df_valid.columns if c.startswith('nb_') and c.endswith(f'{rayon}m')]
    df_valid[f'nb_total_{rayon}m'] = df_valid[cols_nb].sum(axis=1)
    df_valid[f'diversite_{rayon}m'] = (df_valid[cols_nb] > 0).sum(axis=1)
    
    # Densité relative (z-score)
    mean_total = df_valid[f'nb_total_{rayon}m'].mean()
    std_total = df_valid[f'nb_total_{rayon}m'].std()
    if std_total > 0:
        df_valid[f'densite_rel_{rayon}m'] = (df_valid[f'nb_total_{rayon}m'] - mean_total) / std_total
    else:
        df_valid[f'densite_rel_{rayon}m'] = 0
    
    print(f"   ✓ Total moyen: {mean_total:.1f} POIs")
    
    # 6. Réintégrer les invalides avec NaN
    if len(df_invalid) > 0:
        new_cols = [c for c in df_valid.columns if c not in df_invalid.columns]
        for col in new_cols:
            df_invalid[col] = np.nan
        result = pd.concat([df_valid, df_invalid]).sort_index()
    else:
        result = df_valid
    
    return result


# ========================================
# 4. PIPELINE COMPLET PAR ANNÉE 
# ========================================

def traiter_dataset_complet(
    df: pd.DataFrame,
    rayon: int = 1000,
    batch_size: int = 100000
) -> pd.DataFrame:
    """
    Pipeline complet : traite automatiquement les données par année.
    
    - Détecte automatiquement l'année de chaque transaction
    - Charge les données Géofabrik correspondantes
    - Agrège les features géographiques
    - Traite par batch si nécessaire pour éviter les problèmes de mémoire
    
    Args:
        df: DataFrame avec colonnes 'date_mutation', 'latitude', 'longitude'
        rayon: Rayon de recherche en mètres
        batch_size: Nombre de lignes par batch
    
    Returns:
        DataFrame enrichi avec toutes les features géographiques
    """
    print("\n" + "="*70)
    print("PIPELINE COMPLET - ENRICHISSEMENT GÉOGRAPHIQUE")
    print("="*70)
    
    # Extraire l'année
    annees = sorted(df['annee'].unique())
    
    print(f"\nAnnées détectées: {annees}")
    print(f"Total: {len(df):,} transactions")
    
    results = []
    
    for annee in annees:
        df_annee = df[df['annee'] == annee].copy()
        print(f"\n{'='*70}")
        print(f"ANNÉE {annee} - {len(df_annee):,} transactions")
        print('='*70)
        
        # Charger les données géographiques
        print(f"Chargement des données Géofabrik {annee}...")
        pois, transport = charger_pois_et_transport(annee)
        print(f"   ✓ {len(pois):,} POIs | {len(transport):,} transports")
        
        # Traiter par batches si nécessaire
        if len(df_annee) > batch_size:
            print(f"\n Traitement par batches de {batch_size:,} lignes...")
            batches = []
            for i in range(0, len(df_annee), batch_size):
                batch = df_annee.iloc[i:i+batch_size]
                batch_enriched = agreger_features_geo(
                    batch, pois, transport, rayon,
                    batch_name=f"{annee}_batch{i//batch_size+1}"
                )
                batches.append(batch_enriched)
            df_annee_enriched = pd.concat(batches)
        else:
            df_annee_enriched = agreger_features_geo(
                df_annee, pois, transport, rayon,
                batch_name=f"Année {annee}"
            )
        
        results.append(df_annee_enriched)
    
    # Combiner tous les résultats
    df_final = pd.concat(results).sort_index()
    
    print("\n" + "="*70)
    print("PIPELINE TERMINÉ !")
    print("="*70)
    print(f"\n📊 Résultat final: {len(df_final):,} transactions")
    
    # Stats des nouvelles colonnes
    new_cols = [c for c in df_final.columns if c.startswith(('nb_', 'dist_', 'diversite_', 'densite_'))]
    print(f"{len(new_cols)} nouvelles features créées")
    
    return df_final
