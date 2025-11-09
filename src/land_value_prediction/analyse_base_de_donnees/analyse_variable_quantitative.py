import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict


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


def separer_train_test(df: pd.DataFrame, annees_train: List[int], annees_test: List[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df['annee'].isin(annees_train)].copy()
    test = df[df['annee'].isin(annees_test)].copy()
    return train, test


def preparer_features_target(train: pd.DataFrame, test: pd.DataFrame, colonne_cible: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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


def calculer_stats_descriptives(X: pd.DataFrame, colonnes: List[str]) -> pd.DataFrame:
    stats_dict = {}
    
    for col in colonnes:
        stats_dict[col] = {
            'Min': X[col].min(),
            'Q1': X[col].quantile(0.25),
            'Médiane': X[col].median(),
            'Moyenne': X[col].mean(),
            'Q3': X[col].quantile(0.75),
            'Max': X[col].max(),
            'Écart-type': X[col].std(),
            'NaN': X[col].isna().sum()
        }
    
    return pd.DataFrame(stats_dict).T


def plot_histogramme(X: pd.DataFrame, col: str, ax: plt.Axes) -> None:
    ax.hist(X[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
    ax.set_title(f'Distribution de {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Fréquence')


def plot_boxplot(X: pd.DataFrame, col: str, ax: plt.Axes) -> None:
    ax.boxplot(X[col].dropna(), vert=True)
    ax.set_title(f'Boxplot de {col}')
    ax.set_ylabel(col)


def plot_scatterplot(X: pd.DataFrame, y: pd.Series, col: str, ax: plt.Axes) -> None:
    ax.scatter(X[col], y, alpha=0.3, s=10)
    ax.set_title(f'{col} vs prix_m2')
    ax.set_xlabel(col)
    ax.set_ylabel('prix_m2')
    ax.grid(alpha=0.3)


def visualiser_par_batch(X: pd.DataFrame, y: pd.Series, colonnes: List[str], plot_func, titre: str) -> None:
    batch_size = 4
    
    for i in range(0, len(colonnes), batch_size):
        batch = colonnes[i:i + batch_size]
        n_cols = 2
        n_rows = int(np.ceil(len(batch) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = axes.flatten()
        
        for ax, col in zip(axes, batch):
            if plot_func.__name__ == 'plot_scatterplot':
                plot_func(X, y, col, ax)
            else:
                plot_func(X, col, ax)
        
        for ax in axes[len(batch):]:
            ax.axis('off')
        
        fig.suptitle(titre, fontsize=18, y=1.03)
        plt.tight_layout()
        plt.show()


def calculer_correlations(X: pd.DataFrame, y: pd.Series, colonnes: List[str]) -> pd.DataFrame:
    correlations = {}
    
    for col in colonnes:
        data_clean = pd.DataFrame({'x': X[col], 'y': y}).dropna()
        
        if len(data_clean) > 0:
            pearson_corr, pearson_p = stats.pearsonr(data_clean['x'], data_clean['y'])
            spearman_corr, spearman_p = stats.spearmanr(data_clean['x'], data_clean['y'])
            
            sample_size = min(5000, len(data_clean))
            _, shapiro_p = stats.shapiro(data_clean['x'].sample(sample_size))
            
            correlations[col] = {
                'Pearson': pearson_corr,
                'Pearson_p': pearson_p,
                'Spearman': spearman_corr,
                'Spearman_p': spearman_p,
                'Normal': 'Oui' if shapiro_p > 0.05 else 'Non'
            }
    
    corr_df = pd.DataFrame(correlations).T
    return corr_df.sort_values('Pearson', ascending=False, key=abs)


def visualiser_correlations_cible(corr_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    corr_df['Pearson'].plot(kind='barh', color='steelblue')
    plt.title('Corrélation de Pearson avec prix_m2')
    plt.xlabel('Coefficient de corrélation')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.show()


def visualiser_matrice_correlation(X: pd.DataFrame, y: pd.Series, colonnes: List[str]) -> None:
    train_with_target = X[colonnes].copy()
    train_with_target['prix_m2'] = y
    
    corr_matrix = train_with_target.corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matrice de corrélation complète (variables numériques)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = charger_donnees_clean("data/df_vf_idf_clean.csv")
    df = convertir_types_analyse(df)
    
    train, test = separer_train_test(df, [2020, 2021, 2022, 2023, 2024], [2025])
    X_train, y_train, X_test, y_test = preparer_features_target(train, test, 'prix_m2')
    
    types_vars = identifier_types_variables(X_train)
    
    print("="*80)
    print("IDENTIFICATION DES TYPES DE VARIABLES")
    print("="*80)
    print(f"\nVariables temporelles (dates): {types_vars['dates']}")
    print(f"Variables quantitatives (numériques): {types_vars['numeriques']}")
    print(f"Variables qualitatives (catégorielles): {types_vars['categoriques']}\n")
    
    print("="*80)
    print("ANALYSE DESCRIPTIVE DES VARIABLES QUANTITATIVES")
    print("="*80)
    
    print("\n📊 STATISTIQUES DESCRIPTIVES\n")
    stats_df = calculer_stats_descriptives(X_train, types_vars['numeriques'])
    print(stats_df)
    
    visualiser_par_batch(X_train, y_train, types_vars['numeriques'], plot_histogramme, "Histogrammes (max 4 par page)")
    visualiser_par_batch(X_train, y_train, types_vars['numeriques'], plot_boxplot, "Boxplots (max 4 par page)")
    visualiser_par_batch(X_train, y_train, types_vars['numeriques'], plot_scatterplot, "Scatterplots (max 4 par page)")
    
    print("\n🔗 CORRÉLATIONS AVEC LA VARIABLE CIBLE (prix_m2)\n")
    corr_df = calculer_correlations(X_train, y_train, types_vars['numeriques'])
    print(corr_df)
    
    visualiser_correlations_cible(corr_df)
    visualiser_matrice_correlation(X_train, y_train, types_vars['numeriques'])
    


if __name__ == "__main__":
    main()