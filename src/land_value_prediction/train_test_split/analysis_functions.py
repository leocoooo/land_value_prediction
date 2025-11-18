import pandas as pd 
import numpy as np 

def comparer_train_test(train, test, round_decimals=3, include_diff_pct=True):
    """
    Compare les statistiques descriptives entre les ensembles train et test.
    
    Pour les variables numériques : moyenne, médiane, écart-type, min, max
    Pour les variables catégorielles : mode
    
    Parameters
    ----------
    train : pd.DataFrame
        DataFrame d'entraînement
    test : pd.DataFrame
        DataFrame de test
    round_decimals : int, optional
        Nombre de décimales pour l'arrondi (défaut: 3)
    include_diff_pct : bool, optional
        Inclure les différences en pourcentage (défaut: True)
    
    Returns
    -------
    pd.DataFrame
        Tableau comparatif des statistiques par variable
        
    Examples
    --------
    >>> comparison = comparer_train_test(X_train, X_test)
    >>> comparison[comparison['diff_mean'].abs() > 100]  # Variables avec grandes différences
    """
    comparison_list = []
    
    for col in train.columns:
        row = {'variable': col}
        
        if pd.api.types.is_numeric_dtype(train[col]):
            # Variables numériques
            train_mean = train[col].mean()
            test_mean = test[col].mean()
            train_median = train[col].median()
            test_median = test[col].median()
            train_std = train[col].std()
            test_std = test[col].std()
            
            row.update({
                'type': 'numeric',
                'train_mean': train_mean,
                'test_mean': test_mean,
                'diff_mean': train_mean - test_mean,
                'train_median': train_median,
                'test_median': test_median,
                'diff_median': train_median - test_median,
                'train_std': train_std,
                'test_std': test_std,
                'diff_std': train_std - test_std,
                'train_min': train[col].min(),
                'test_min': test[col].min(),
                'train_max': train[col].max(),
                'test_max': test[col].max()
            })
            
            # Différences en pourcentage
            if include_diff_pct:
                row['diff_mean_pct'] = ((train_mean - test_mean) / test_mean * 100) if test_mean != 0 else np.nan
                row['diff_median_pct'] = ((train_median - test_median) / test_median * 100) if test_median != 0 else np.nan
                row['diff_std_pct'] = ((train_std - test_std) / test_std * 100) if test_std != 0 else np.nan
        
        else:
            # Variables catégorielles
            try:
                train_mode = train[col].mode()[0] if len(train[col].mode()) > 0 else pd.NA
                test_mode = test[col].mode()[0] if len(test[col].mode()) > 0 else pd.NA
                
                row.update({
                    'type': 'categorical',
                    'train_mean': pd.NA,
                    'test_mean': pd.NA,
                    'diff_mean': pd.NA,
                    'train_median': train_mode,
                    'test_median': test_mode,
                    'diff_median': pd.NA,
                    'train_std': pd.NA,
                    'test_std': pd.NA,
                    'diff_std': pd.NA,
                    'train_min': pd.NA,
                    'test_min': pd.NA,
                    'train_max': pd.NA,
                    'test_max': pd.NA
                })
                
                if include_diff_pct:
                    row['diff_mean_pct'] = pd.NA
                    row['diff_median_pct'] = pd.NA
                    row['diff_std_pct'] = pd.NA
                    
            except Exception as e:
                print(f"Erreur sur la colonne {col}: {e}")
                continue
        
        comparison_list.append(row)
    
    # Créer le DataFrame et mettre la variable en index
    comparison = pd.DataFrame(comparison_list)
    comparison = comparison.set_index('variable')
    
    # Arrondir les valeurs numériques
    numeric_cols = comparison.select_dtypes(include=[np.number]).columns
    comparison[numeric_cols] = comparison[numeric_cols].round(round_decimals)
    
    return comparison


def identifier_differences_significatives(comparison_df, threshold_mean=0.1, threshold_std=0.2):
    """
    Identifie les variables avec des différences significatives entre train et test.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame retourné par comparer_train_test()
    threshold_mean : float, optional
        Seuil de différence relative acceptable pour la moyenne (défaut: 0.1 = 10%)
    threshold_std : float, optional
        Seuil de différence relative acceptable pour l'écart-type (défaut: 0.2 = 20%)
    
    Returns
    -------
    dict
        Dictionnaire avec les variables ayant des différences significatives
        
    Examples
    --------
    >>> comparison = comparer_train_test(X_train, X_test)
    >>> warnings = identifier_differences_significatives(comparison)
    >>> if warnings['mean']:
    ...     print(f"Attention aux variables: {warnings['mean']}")
    """
    warnings = {
        'mean': [],
        'std': [],
        'distribution': []
    }
    
    # Filtrer uniquement les variables numériques
    numeric_vars = comparison_df[comparison_df['type'] == 'numeric']
    
    for var in numeric_vars.index:
        row = numeric_vars.loc[var]
        
        # Vérifier la différence de moyenne
        if 'diff_mean_pct' in row.index and pd.notna(row['diff_mean_pct']):
            if abs(row['diff_mean_pct']) > threshold_mean * 100:
                warnings['mean'].append({
                    'variable': var,
                    'diff_pct': row['diff_mean_pct'],
                    'train_mean': row['train_mean'],
                    'test_mean': row['test_mean']
                })
        
        # Vérifier la différence d'écart-type
        if 'diff_std_pct' in row.index and pd.notna(row['diff_std_pct']):
            if abs(row['diff_std_pct']) > threshold_std * 100:
                warnings['std'].append({
                    'variable': var,
                    'diff_pct': row['diff_std_pct'],
                    'train_std': row['train_std'],
                    'test_std': row['test_std']
                })
        
        # Vérifier les différences de distribution (min/max)
        if pd.notna(row['train_min']) and pd.notna(row['test_min']):
            if row['test_min'] < row['train_min'] or row['test_max'] > row['train_max']:
                warnings['distribution'].append({
                    'variable': var,
                    'train_range': f"[{row['train_min']}, {row['train_max']}]",
                    'test_range': f"[{row['test_min']}, {row['test_max']}]"
                })
    
    return warnings