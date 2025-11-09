import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Tuple


def charger_donnees_clean(chemin: str) -> pd.DataFrame:
    return pd.read_csv(chemin, sep=",", low_memory=False)


def convertir_types_analyse(df: pd.DataFrame) -> pd.DataFrame:
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
    
    df_copie['code_type_local'] = df_copie['code_type_local'].astype('category')
    df_copie['code_departement'] = df_copie['code_departement'].astype(str)
    df_copie['code_commune'] = df_copie['code_commune'].astype(str)
    
    df_copie['annee_date'] = pd.to_datetime(df_copie['annee'].astype(str) + '-01-01')
    
    colonnes_categoriques = [
        'nature_mutation', 'type_local', 'adresse_nom_voie',
        'nom_commune', 'quartier', 'quartier_detaille', 'code_postal',
        'code_departement', 'code_commune'
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


def separer_train_test(df: pd.DataFrame, annees_train: List[int], annees_test: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df['annee'].isin(annees_train)].copy()
    test = df[df['annee'].isin(annees_test)].copy()
    return train, test


def preparer_features_target(train: pd.DataFrame, test: pd.DataFrame, colonne_cible: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = train.drop(columns=[colonne_cible, 'annee'])
    y_train = train[colonne_cible]
    X_test = test.drop(columns=[colonne_cible, 'annee'])
    y_test = test[colonne_cible]
    return X_train, y_train, X_test, y_test


def identifier_types_variables(X: pd.DataFrame) -> Dict[str, List[str]]:
    numeriques = X.select_dtypes(include=[np.number]).columns.tolist()
    categoriques = X.select_dtypes(include=['object', 'category']).columns.tolist()
    dates = X.select_dtypes(include=['datetime64']).columns.tolist()
    
    return {
        'numeriques': numeriques,
        'categoriques': categoriques,
        'dates': dates
    }


def calculer_stats_qualitatives(X: pd.DataFrame, col: str) -> Dict:
    value_counts = X[col].value_counts()
    value_counts_pct = X[col].value_counts(normalize=True) * 100
    
    return {
        'nb_categories': X[col].nunique(),
        'nb_valeurs_manquantes': X[col].isna().sum(),
        'top_10_effectifs': value_counts.head(10),
        'top_10_pourcentages': value_counts_pct.head(10)
    }


def afficher_stats_qualitative(X: pd.DataFrame, col: str) -> None:
    stats = calculer_stats_qualitatives(X, col)
    
    print(f"\n{'='*50}")
    print(f"Variable: {col}")
    print(f"{'='*50}")
    print(f"Nombre de catégories uniques: {stats['nb_categories']}")
    print(f"Valeurs manquantes: {stats['nb_valeurs_manquantes']}")
    print(f"\nTop 10 des valeurs:")
    
    df_top = pd.DataFrame({
        'Effectif': stats['top_10_effectifs'],
        'Pourcentage': stats['top_10_pourcentages']
    })
    print(df_top)


def plot_barplot_qualitative(X: pd.DataFrame, col: str, ax: plt.Axes) -> None:
    value_counts = X[col].value_counts().head(20)
    value_counts.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title(f'Distribution de {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Fréquence')
    ax.tick_params(axis='x', rotation=45)


def plot_pieplot_qualitative(X: pd.DataFrame, col: str, ax: plt.Axes) -> None:
    value_counts = X[col].value_counts().head(10)
    value_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
    ax.set_title(f'Répartition de {col} (Top 10)')
    ax.set_ylabel('')


def visualiser_qualitative(X: pd.DataFrame, col: str) -> None:
    if X[col].nunique() <= 20:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        plot_barplot_qualitative(X, col, axes[0])
        plot_pieplot_qualitative(X, col, axes[1])
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Trop de catégories ({X[col].nunique()}) pour visualiser\n")


def calculer_relation_quali_cible(X: pd.DataFrame, y: pd.Series, col: str) -> pd.DataFrame:
    data_temp = pd.DataFrame({col: X[col], 'prix_m2': y})
    
    stats_by_cat = data_temp.groupby(col)['prix_m2'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    return stats_by_cat


def test_anova(X: pd.DataFrame, y: pd.Series, col: str) -> Tuple[float, float, str]:
    data_temp = pd.DataFrame({col: X[col], 'prix_m2': y})
    categories = data_temp[col].unique()
    
    if len(categories) > 1:
        groupes = [data_temp[data_temp[col] == cat]['prix_m2'].dropna() for cat in categories]
        groupes = [g for g in groupes if len(g) > 0]
        
        if len(groupes) > 1:
            f_stat, p_value = stats.f_oneway(*groupes)
            significatif = 'Oui' if p_value < 0.05 else 'Non'
            return f_stat, p_value, significatif
    
    return 0.0, 1.0, 'Non'


def test_kruskal(X: pd.DataFrame, y: pd.Series, col: str) -> Tuple[float, float, str]:
    data_temp = pd.DataFrame({col: X[col], 'prix_m2': y})
    categories = data_temp[col].unique()
    
    if len(categories) > 1:
        groupes = [data_temp[data_temp[col] == cat]['prix_m2'].dropna() for cat in categories]
        groupes = [g for g in groupes if len(g) > 0]
        
        if len(groupes) > 1:
            h_stat, p_value = stats.kruskal(*groupes)
            significatif = 'Oui' if p_value < 0.05 else 'Non'
            return h_stat, p_value, significatif
    
    return 0.0, 1.0, 'Non'


def plot_boxplot_par_categorie(X: pd.DataFrame, y: pd.Series, col: str, ax: plt.Axes) -> None:
    data_temp = pd.DataFrame({col: X[col], 'prix_m2': y})
    data_temp.boxplot(column='prix_m2', by=col, ax=ax)
    ax.set_title(f'prix_m2 par {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('prix_m2')
    plt.sca(ax)
    plt.xticks(rotation=45)


def plot_barplot_moyennes(stats_by_cat: pd.DataFrame, col: str, ax: plt.Axes) -> None:
    stats_by_cat['mean'].head(20).plot(kind='bar', ax=ax, color='coral')
    ax.set_title(f'Moyenne prix_m2 par {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Moyenne prix_m2')
    ax.tick_params(axis='x', rotation=45)


def analyser_relation_quali_cible(X: pd.DataFrame, y: pd.Series, col: str) -> None:
    print(f"\n{'='*50}")
    print(f"Relation entre {col} et prix_m2")
    print(f"{'='*50}")
    
    stats_by_cat = calculer_relation_quali_cible(X, y, col)
    print("\nStatistiques descriptives par catégorie:")
    print(stats_by_cat.head(20))
    
    print("\n--- Tests d'association ---")
    
    f_stat, p_value_anova, sig_anova = test_anova(X, y, col)
    print(f"\nTest ANOVA (paramétrique):")
    print(f"  F-statistique: {f_stat:.4f}")
    print(f"  p-value: {p_value_anova:.4f}")
    print(f"  Différence significative: {sig_anova}")
    
    h_stat, p_value_kruskal, sig_kruskal = test_kruskal(X, y, col)
    print(f"\nTest de Kruskal-Wallis (non-paramétrique):")
    print(f"  H-statistique: {h_stat:.4f}")
    print(f"  p-value: {p_value_kruskal:.4f}")
    print(f"  Différence significative: {sig_kruskal}")
    
    if X[col].nunique() <= 20:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        plot_boxplot_par_categorie(X, y, col, axes[0])
        plot_barplot_moyennes(stats_by_cat, col, axes[1])
        
        plt.tight_layout()
        plt.show()


def main() -> None:
    df = charger_donnees_clean("data/df_vf_idf_clean.csv")
    df = convertir_types_analyse(df)
    
    train, test = separer_train_test(df, [2020, 2021, 2022, 2023, 2024], [2025])
    X_train, y_train, X_test, y_test = preparer_features_target(train, test, 'prix_m2')
    
    types_vars = identifier_types_variables(X_train)
    
    print("="*80)
    print("ANALYSE DESCRIPTIVE DES VARIABLES QUALITATIVES")
    print("="*80)
    
    for col in types_vars['categoriques']:
        afficher_stats_qualitative(X_train, col)
        visualiser_qualitative(X_train, col)
    
    print("\n" + "="*80)
    print("TESTS D'ASSOCIATION : VARIABLES QUALITATIVES vs CIBLE QUANTITATIVE")
    print("="*80)
    
    for col in types_vars['categoriques']:
        if X_train[col].nunique() < 100:
            analyser_relation_quali_cible(X_train, y_train, col)
    


if __name__ == "__main__":
    main()