import numpy as np
import pandas as pd
from typing import List


def charger_donnees(chemin: str) -> pd.DataFrame:
    return pd.read_csv(chemin, sep=",", low_memory=False)


def supprimer_colonnes_inutiles(df: pd.DataFrame) -> pd.DataFrame:
    colonnes_a_supprimer = [
        'id_mutation', 'adresse_numero', 'id_parcelle',
        'lot1_numero', 'lot2_numero', 'lot3_numero',
        'lot4_numero', 'lot5_numero', 'lot1_surface_carrez',
        'lot2_surface_carrez', 'lot3_surface_carrez',
        'lot4_surface_carrez', 'lot5_surface_carrez',
        'arrondissement', 'nature_culture', 'nature_culture_speciale',
        'annee_trimestre'
    ]
    return df.drop(columns=colonnes_a_supprimer, errors='ignore')


def remplir_prix_median_manquants(df: pd.DataFrame) -> pd.DataFrame:
    df_copie = df.copy()
    
    quartier_prix_median = (
        df_copie[df_copie['prix_median_m2'].notna()]
        .groupby('quartier_detaille')['prix_median_m2']
        .first()
        .to_dict()
    )
    
    mask_manquant = df_copie['prix_median_m2'].isna()
    
    for idx in df_copie[mask_manquant].index:
        quartier = df_copie.loc[idx, 'quartier_detaille']
        
        if quartier in quartier_prix_median:
            df_copie.loc[idx, 'prix_median_m2'] = quartier_prix_median[quartier]
        else:
            df_copie.loc[idx, 'prix_median_m2'] = df_copie.loc[idx, 'prix_m2']
    
    return df_copie


def recalculer_ecart_prix_median(df: pd.DataFrame) -> pd.DataFrame:
    df_copie = df.copy()
    df_copie['ecart_prix_median_pct'] = (
        (df_copie['prix_m2'] - df_copie['prix_median_m2']) / 
        df_copie['prix_median_m2'] * 100
    )
    return df_copie


def convertir_types(df: pd.DataFrame) -> pd.DataFrame:
    df_copie = df.copy()
    
    df_copie['date_mutation'] = pd.to_datetime(df_copie['date_mutation'])
    
    df_copie['code_postal'] = (
        df_copie['code_postal']
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )
    df_copie['code_postal'] = df_copie['code_postal'].replace('00000', np.nan)
    
    df_copie['code_type_local'] = df_copie['code_type_local'].astype('Int64')
    df_copie['code_departement'] = df_copie['code_departement'].astype(str)
    
    colonnes_categoriques = [
        'nature_mutation', 'type_local', 'adresse_nom_voie',
        'nom_commune', 'quartier', 'quartier_detaille'
    ]
    
    for col in colonnes_categoriques:
        if col in df_copie.columns:
            df_copie[col] = df_copie[col].astype('category')
    
    df_copie['mois'] = pd.Categorical(
        df_copie['mois'],
        categories=range(1, 13),
        ordered=True
    )
    df_copie['trimestre'] = pd.Categorical(
        df_copie['trimestre'],
        categories=[1, 2, 3, 4],
        ordered=True
    )
    
    return df_copie


def separer_train_test(
    df: pd.DataFrame,
    annees_train: List[int],
    annees_test: List[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df['annee'].isin(annees_train)].copy()
    test = df[df['annee'].isin(annees_test)].copy()
    return train, test


def preparer_features_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    colonne_cible: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = train.drop(columns=[colonne_cible])
    y_train = train[colonne_cible]
    X_test = test.drop(columns=[colonne_cible])
    y_test = test[colonne_cible]
    return X_train, y_train, X_test, y_test

def sauvegarder_dataframe(df: pd.DataFrame, chemin: str) -> None:
    df.to_csv(chemin, index=False)
    print(f"\n=== SAUVEGARDE ===")
    print(f"Dataframe sauvegardé dans : {chemin}")

def afficher_infos_datasets(
    df: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    print("=== TAILLES DES DATASETS ===")
    print(f"Train: {len(train)} lignes ({len(train)/len(df)*100:.1f}%)")
    print(f"Test: {len(test)} lignes ({len(test)/len(df)*100:.1f}%)")
    
    print("\n=== RÉPARTITION TEMPORELLE ===")
    print("Train:")
    print(train.groupby('annee').size())
    print("\nTest:")
    print(test.groupby('annee').size())
    
    print("\n=== SHAPES X et y ===")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    print("\n=== TARGET (prix_m2) ===")
    print(f"y_train - moyenne: {y_train.mean():.2f}€/m², min: {y_train.min():.2f}€, max: {y_train.max():.2f}€")
    print(f"y_test - moyenne: {y_test.mean():.2f}€/m², min: {y_test.min():.2f}€, max: {y_test.max():.2f}€")


def main() -> None:
    df = charger_donnees("data/df_vf_idf.csv")
    print(df.head(5))
    print(f"\nShape initiale: {df.shape}")
    
    print("\n=== VALEURS MANQUANTES INITIALES ===")
    print(df.isnull().sum())
    
    df_clean = supprimer_colonnes_inutiles(df)
    
    print("\n=== VALEURS MANQUANTES APRÈS SUPPRESSION ===")
    print(df_clean.isnull().sum())
    
    df_clean = remplir_prix_median_manquants(df_clean)
    df_clean = recalculer_ecart_prix_median(df_clean)
    
    print("\n=== VALEURS MANQUANTES APRÈS REMPLISSAGE ===")
    print(df_clean.isnull().sum())
    
    print("\n=== TYPES AVANT CONVERSION ===")
    print(df_clean.dtypes)
    
    df_clean = convertir_types(df_clean)
    
    print("\n=== TYPES APRÈS CONVERSION ===")
    print(df_clean.dtypes)
    
    train, test = separer_train_test(
        df_clean,
        annees_train=[2020, 2021, 2022, 2023, 2024],
        annees_test=[2025]
    )
    
    X_train, y_train, X_test, y_test = preparer_features_target(
        train,
        test,
        colonne_cible='prix_m2'
    )
    
    afficher_infos_datasets(df_clean, train, test, X_train, y_train, X_test, y_test)
    
    # Sauvegarde optionnelle
    sauvegarder_dataframe(df_clean, "data/df_vf_idf_clean.csv")

if __name__ == "__main__":
    main()