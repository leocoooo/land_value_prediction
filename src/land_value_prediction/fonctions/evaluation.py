"""
Fonctions d'évaluation pour les modèles de prévision immobilière
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluer_modele(y_true, y_pred, dataset_name="", verbose=True):
    """
    Calcule les métriques de performance d'un modèle de régression

    Parameters
    ----------
    y_true : array-like
        Valeurs réelles
    y_pred : array-like
        Valeurs prédites
    dataset_name : str, optional
        Nom du dataset (ex: "TRAIN", "TEST")
    verbose : bool, optional
        Afficher les résultats

    Returns
    -------
    dict
        Dictionnaire contenant les métriques : rmse, mae, r2, mape
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE avec gestion des divisions par zéro
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

    if verbose:
        print(f"\n{dataset_name}")
        print(f"  RMSE : {rmse:.2f} €/m²")
        print(f"  MAE  : {mae:.2f} €/m²")
        print(f"  R²   : {r2:.4f}")
        print(f"  MAPE : {mape:.2f}%" if not np.isnan(mape) else "  MAPE : N/A")

    return metrics


def comparer_modeles(resultats_dict):
    """
    Compare plusieurs modèles sous forme de tableau

    Parameters
    ----------
    resultats_dict : dict
        Dictionnaire {nom_modele: {'train': metrics_train, 'test': metrics_test}}

    Returns
    -------
    pd.DataFrame
        Tableau comparatif des performances
    """
    comparaison = []

    for nom_modele, resultats in resultats_dict.items():
        for dataset, metrics in resultats.items():
            comparaison.append({
                'Modèle': nom_modele,
                'Dataset': dataset.upper(),
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'MAPE (%)': metrics['mape']
            })

    df_comp = pd.DataFrame(comparaison)
    return df_comp


def plot_predictions(y_true, y_pred, title="Prédictions vs Réalité", figsize=(10, 6)):
    """
    Graphique scatter des prédictions vs valeurs réelles

    Parameters
    ----------
    y_true : array-like
        Valeurs réelles
    y_pred : array-like
        Valeurs prédites
    title : str
        Titre du graphique
    figsize : tuple
        Taille de la figure
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)

    # Ligne de prédiction parfaite
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prédiction parfaite')

    plt.xlabel('Prix réel (€/m²)')
    plt.ylabel('Prix prédit (€/m²)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residus(y_true, y_pred, figsize=(14, 5)):
    """
    Visualisation complète des résidus

    Parameters
    ----------
    y_true : array-like
        Valeurs réelles
    y_pred : array-like
        Valeurs prédites
    figsize : tuple
        Taille de la figure
    """
    residus = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Résidus vs prédictions
    axes[0].scatter(y_pred, residus, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Prix prédit (€/m²)')
    axes[0].set_ylabel('Résidus (€/m²)')
    axes[0].set_title('Résidus vs Prédictions')
    axes[0].grid(alpha=0.3)

    # Distribution des résidus
    axes[1].hist(residus, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Résidus (€/m²)')
    axes[1].set_ylabel('Fréquence')
    axes[1].set_title('Distribution des résidus')
    axes[1].grid(alpha=0.3)

    # QQ-plot pour normalité
    from scipy import stats
    stats.probplot(residus, dist="norm", plot=axes[2])
    axes[2].set_title('QQ-Plot (normalité des résidus)')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyser_erreurs_par_segment(df, y_true, y_pred, segment_col, top_n=10):
    """
    Analyse des erreurs par segment (ex: par commune, par type de bien)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données originales
    y_true : array-like
        Valeurs réelles
    y_pred : array-like
        Valeurs prédites
    segment_col : str
        Colonne utilisée pour segmenter
    top_n : int
        Nombre de segments à afficher

    Returns
    -------
    pd.DataFrame
        Tableau des erreurs par segment
    """
    df_analyse = df.copy()
    df_analyse['erreur_abs'] = np.abs(y_true - y_pred)
    df_analyse['erreur_rel'] = np.abs((y_true - y_pred) / y_true) * 100

    analyse_segment = df_analyse.groupby(segment_col).agg({
        'erreur_abs': ['mean', 'median', 'std'],
        'erreur_rel': ['mean', 'median'],
        segment_col: 'count'
    }).round(2)

    analyse_segment.columns = ['MAE', 'Mediane_erreur', 'Std_erreur', 'MAPE', 'Mediane_MAPE', 'Nb_observations']
    analyse_segment = analyse_segment.sort_values('MAE', ascending=False).head(top_n)

    return analyse_segment