import pandas as pd

def ajouter_prix_maisons_voisines_semestre_prec(df, nb_min_transactions=5):
    
    # 1. Calcul des agrégations par niveau géographique
    prix_par_code_cadastral_semestre = df.groupby(['code_cadastral', 'semestre'], observed=False)['prix_m2'].agg([
        ('prix_m2_median_maisons_voisines', 'median'),
        ('prix_m2_moyen_maisons_voisines', 'mean'),
        ('nb_transactions', 'count'),
    ]).reset_index()

    prix_par_code_commune_semestre = df.groupby(['code_commune', 'semestre'], observed=False)['prix_m2'].agg([
        ('prix_m2_median_maisons_voisines', 'median'),
        ('prix_m2_moyen_maisons_voisines', 'mean'),
        ('nb_transactions', 'count'),
    ]).reset_index()

    prix_par_code_postal_semestre = df.groupby(['code_postal', 'semestre'], observed=False)['prix_m2'].agg([
        ('prix_m2_median_maisons_voisines', 'median'),
        ('prix_m2_moyen_maisons_voisines', 'mean'),
        ('nb_transactions', 'count'),
    ]).reset_index()

    prix_par_code_departement_semestre = df.groupby(['code_departement', 'semestre'], observed=False)['prix_m2'].agg([
        ('prix_m2_median_maisons_voisines', 'median'),
        ('prix_m2_moyen_maisons_voisines', 'mean'),
        ('nb_transactions', 'count'),
    ]).reset_index()

    # 2. Filtrage des codes cadastraux avec <= nb_min_transactions 
    mask_min_5_transac_cadastral = (prix_par_code_cadastral_semestre['nb_transactions'] <= nb_min_transactions)
    print(f"Il y a {mask_min_5_transac_cadastral.sum()} codes cadastraux avec {nb_min_transactions} transactions ou moins, "
          f"pour un total de {prix_par_code_cadastral_semestre.shape[0]} codes cadastraux.")
    print("Ces prix cadastraux ne seront pas utilisés, on utilisera plutôt le prix médian par code commune")
    prix_par_code_cadastral_semestre.loc[mask_min_5_transac_cadastral, 
                                          ['prix_m2_median_maisons_voisines', 'prix_m2_moyen_maisons_voisines']] = pd.NA

    # 3. Filtrage des codes communes avec <= nb_min_transactions transactions
    mask_min_5_transac_commune = (prix_par_code_commune_semestre['nb_transactions'] <= nb_min_transactions)
    print(f"Il y a {mask_min_5_transac_commune.sum()} codes communes avec {nb_min_transactions} transactions ou moins, "
          f"pour un total de {prix_par_code_commune_semestre.shape[0]} codes communes.")
    print("Ces prix communaux ne seront pas utilisés, on utilisera plutôt le prix médian par code postal")
    prix_par_code_commune_semestre.loc[mask_min_5_transac_commune, 
                                        ['prix_m2_median_maisons_voisines', 'prix_m2_moyen_maisons_voisines']] = pd.NA

    # 4. Filtrage des codes postaux avec <= nb_min_transactions transactions
    mask_min_5_transac_postal = (prix_par_code_postal_semestre['nb_transactions'] <= nb_min_transactions)
    print(f"Il y a {mask_min_5_transac_postal.sum()} codes postaux avec {nb_min_transactions} transactions ou moins, "
          f"pour un total de {prix_par_code_postal_semestre.shape[0]} codes postaux.")
    print("Ces prix postaux ne seront pas utilisés, on utilisera plutôt le prix médian par code département")
    prix_par_code_postal_semestre.loc[mask_min_5_transac_postal, 
                                       ['prix_m2_median_maisons_voisines', 'prix_m2_moyen_maisons_voisines']] = pd.NA

    # 5. Filtrage des codes départements avec <= nb_min_transactions transactions
    mask_min_5_transac_departement = (prix_par_code_departement_semestre['nb_transactions'] <= nb_min_transactions)
    print(f"Il y a {mask_min_5_transac_departement.sum()} codes départements avec {nb_min_transactions} transactions ou moins, "
          f"pour un total de {prix_par_code_departement_semestre.shape[0]} codes départements.")
    print("Ces prix départementaux ne seront pas utilisés")
    prix_par_code_departement_semestre.loc[mask_min_5_transac_departement, 
                                            ['prix_m2_median_maisons_voisines', 'prix_m2_moyen_maisons_voisines']] = pd.NA

    # 6. Merge niveau cadastral avec semestre_precedent
    df_prix_m2_maisons_voisines = df.merge(
        prix_par_code_cadastral_semestre[['code_cadastral', 'semestre', 'prix_m2_median_maisons_voisines', 'prix_m2_moyen_maisons_voisines']],
        left_on=['code_cadastral', 'semestre_precedent'],
        right_on=['code_cadastral', 'semestre'],
        how='left',
        suffixes=('', '_cadastral')
    ).drop(columns=['semestre_cadastral'])  # Supprimer la colonne semestre du df agrégé

    # 7. Compléter avec niveau commune si manquant
    mask_na = df_prix_m2_maisons_voisines['prix_m2_median_maisons_voisines'].isna()
    if mask_na.sum() > 0:
        temp_commune = df_prix_m2_maisons_voisines.loc[mask_na, ['code_commune', 'semestre_precedent']].merge(
            prix_par_code_commune_semestre[['code_commune', 'semestre', 'prix_m2_median_maisons_voisines', 'prix_m2_moyen_maisons_voisines']],
            left_on=['code_commune', 'semestre_precedent'],
            right_on=['code_commune', 'semestre'],
            how='left'
        )
        df_prix_m2_maisons_voisines.loc[mask_na, 'prix_m2_median_maisons_voisines'] = temp_commune['prix_m2_median_maisons_voisines'].values
        df_prix_m2_maisons_voisines.loc[mask_na, 'prix_m2_moyen_maisons_voisines'] = temp_commune['prix_m2_moyen_maisons_voisines'].values

    # 8. Compléter avec niveau postal si toujours manquant
    mask_na = df_prix_m2_maisons_voisines['prix_m2_median_maisons_voisines'].isna()
    if mask_na.sum() > 0:
        temp_postal = df_prix_m2_maisons_voisines.loc[mask_na, ['code_postal', 'semestre_precedent']].merge(
            prix_par_code_postal_semestre[['code_postal', 'semestre', 'prix_m2_median_maisons_voisines', 'prix_m2_moyen_maisons_voisines']],
            left_on=['code_postal', 'semestre_precedent'],
            right_on=['code_postal', 'semestre'],
            how='left'
        )
        df_prix_m2_maisons_voisines.loc[mask_na, 'prix_m2_median_maisons_voisines'] = temp_postal['prix_m2_median_maisons_voisines'].values
        df_prix_m2_maisons_voisines.loc[mask_na, 'prix_m2_moyen_maisons_voisines'] = temp_postal['prix_m2_moyen_maisons_voisines'].values

    # 9. Compléter avec niveau département si toujours manquant
    mask_na = df_prix_m2_maisons_voisines['prix_m2_median_maisons_voisines'].isna()
    if mask_na.sum() > 0:
        temp_departement = df_prix_m2_maisons_voisines.loc[mask_na, ['code_departement', 'semestre_precedent']].merge(
            prix_par_code_departement_semestre[['code_departement', 'semestre', 'prix_m2_median_maisons_voisines', 'prix_m2_moyen_maisons_voisines']],
            left_on=['code_departement', 'semestre_precedent'],
            right_on=['code_departement', 'semestre'],
            how='left'
        )
        df_prix_m2_maisons_voisines.loc[mask_na, 'prix_m2_median_maisons_voisines'] = temp_departement['prix_m2_median_maisons_voisines'].values
        df_prix_m2_maisons_voisines.loc[mask_na, 'prix_m2_moyen_maisons_voisines'] = temp_departement['prix_m2_moyen_maisons_voisines'].values

    return df_prix_m2_maisons_voisines