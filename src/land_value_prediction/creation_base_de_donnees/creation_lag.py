import pandas as pd
from typing import List, Literal
import gc

def creer_lags(
    df_base: pd.DataFrame, 
    df_macro: pd.DataFrame, 
    col_date_base: str, 
    col_date_macro: str, 
    colonnes_features: List[str], 
    lags: List[int] = [1, 2, 3, 4],
    frequence: Literal['mensuel', 'trimestriel', 'm', 't'] = 'mensuel'
) -> pd.DataFrame:
    """
    Ajoute des lags mensuels ou trimestriels pour éviter le look-ahead bias
    
    Parameters
    ----------
    df_base : pd.DataFrame
        DataFrame de base auquel ajouter les lags
    df_macro : pd.DataFrame
        DataFrame contenant les variables macro
    col_date_base : str
        Nom de la colonne de date dans df_base
    col_date_macro : str
        Nom de la colonne de date dans df_macro
    colonnes_features : List[str]
        Liste des colonnes à lagger
    lags : List[int]
        Liste des lags à créer (ex: [1, 2, 3, 6])
    frequence : Literal['mensuel', 'trimestriel', 'm', 't']
        Type de lag : 'mensuel'/'m' ou 'trimestriel'/'t'
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec les colonnes laggées ajoutées
    """
    
    # Normaliser la fréquence
    freq_map = {'mensuel': 'm', 'trimestriel': 't', 'm': 'm', 't': 't'}
    freq = freq_map.get(frequence.lower())
    
    if freq is None:
        raise ValueError("frequence doit être 'mensuel', 'trimestriel', 'm' ou 't'")
    
    # Paramètres selon la fréquence
    if freq == 'm':
        mois_par_lag = 1
        suffixe = 'm'
        ajustement_date = pd.offsets.MonthEnd(0)
        type_lag = 'mensuel'
    else:  # freq == 't'
        mois_par_lag = 3
        suffixe = 't'
        ajustement_date = pd.offsets.QuarterEnd(0)
        type_lag = 'trimestriel'
    
    df_result = df_base.copy()
    
    # Préparer les données macro
    df_macro = df_macro.copy()
    df_macro[col_date_macro] = pd.to_datetime(df_macro[col_date_macro], errors='coerce')
    
    # Convertir en numérique et float32
    for col in colonnes_features:
        if col in df_macro.columns:
            df_macro[col] = pd.to_numeric(df_macro[col], errors='coerce').astype('float32')
    
    # Pour chaque lag
    for lag in lags:
        print(f"→ Création lag{lag}{suffixe} ({type_lag})...", end=" ", flush=True)
        
        # Créer une colonne temporaire avec la date décalée
        col_temp = f'_temp_date_lag{lag}{suffixe}'
        df_result[col_temp] = pd.to_datetime(df_result[col_date_base]) - pd.DateOffset(months=lag * mois_par_lag)
        df_result[col_temp] = df_result[col_temp] + ajustement_date
        
        # Nombre de lignes AVANT le merge
        nb_avant = len(df_result)
        
        # Merge propre
        df_result = pd.merge(
            df_result,
            df_macro[[col_date_macro] + colonnes_features],
            left_on=col_temp,
            right_on=col_date_macro,
            how='left',
            suffixes=('', f'_lag{lag}{suffixe}')
        )
        
        # Nombre de lignes APRÈS le merge
        nb_apres = len(df_result)
        
        if nb_apres != nb_avant:
            print(f"DUPLICATION ! {nb_avant:,} → {nb_apres:,}")
            raise ValueError(f"Le merge a créé des duplications pour lag{lag}{suffixe}")
        
        # Renommer les colonnes ajoutées
        for col in colonnes_features:
            if col in df_result.columns:
                df_result.rename(columns={col: f'{col}_lag{lag}{suffixe}'}, inplace=True)
        
        # Nettoyer les colonnes temporaires
        df_result.drop(columns=[col_temp, col_date_macro], errors='ignore', inplace=True)
        
        gc.collect()
        print("✓")
    
    return df_result