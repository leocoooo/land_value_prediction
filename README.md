
# Prédiction de la valeur foncière en Île-de-France

Ce projet vise à prédire la valeur des terrains et biens immobiliers en Île-de-France à partir de multiples sources de données et de techniques avancées de machine learning. Le flux de travail est structuré autour de plusieurs notebooks Jupyter, chacun dédié à une étape clé du pipeline de data science.

## Présentation des notebooks

**00_create_dataset.ipynb**
> Ce notebook importe et fusionne plusieurs tables de données afin de construire un dataset riche en variables pour la prédiction du prix au mètre carré en Île-de-France.

**01_train_test_split.ipynb**
> Ce notebook sépare les données en ensembles d'entraînement et de test tout en contrôlant le risque de data leakage.

**02_analyze_process_clean_data.ipynb**
> Ce notebook analyse les distributions et corrélations des variables, traite les valeurs manquantes par imputation géographique, et applique des transformations de feature engineering avant d'exporter les données nettoyées.

**03_selection_features.ipynb**
> Ce notebook sélectionne les variables les plus pertinentes en utilisant les valeurs SHAP issues d'un modèle LightGBM entraîné par validation croisée stratifiée, puis élimine les features fortement corrélées.

**04_modelisation_RF_XG_theo.ipynb**
> Ce notebook optimise les hyperparamètres de Random Forest et XGBoost via Optuna sur un échantillon stratifié de 50 000 lignes pour la rapidité tandis que l'entraînement final (fit) se fait sur l'intégralité du dataset train pour la performance maximale. On compare ensuite les 2 modèles à partir de 3 jeux de données : train, test et oos.

**04_modelisation_RF_XG.ipynb**
> Ce notebook entraîne 2 modèles : un Random Forest et un XGBoost. Pour la recherche d’hyperparamètres on utilise Optuna sur un échantillon stratifié de 50 000 lignes pour la rapidité tandis que l'entraînement final (fit) se fait sur l'intégralité du dataset train pour la performance maximale. On compare ensuite les 2 modèles à partir de 3 jeux de données : train, test et oos.

**04_modelisation_XGB_RF_Stacking_LigthGBM.ipynb**
> Ce notebook entraîne et compare les performances de XGBoost, Random Forest, LightGBM et un modèle de stacking sur les ensembles train, test et out-of-sample, avec possibilité d'exclure Paris.

**04_multiple_models.ipynb**
> Ce notebook optimise séparément des modèles XGBoost pour Paris et hors Paris via Optuna, évalue leurs performances par zone géographique, et compare les résultats à un modèle naïf basé sur la médiane.

**04_ridge_regression.ipynb**
> Ce notebook teste deux stratégies de sélection de variables pour une régression Ridge en supprimant les features fortement corrélées, optimise les hyperparamètres par validation croisée, et analyse l'interprétabilité des coefficients.


## Tableau récapitulatif des performances de chaque approche de modélisation
![texte alternatif](./data_valeurs_foncieres/doc/resultats.png)


## Structure des données
- **data/raw/** : Données brutes issues de différentes sources.
- **data/processed/** : Données nettoyées et prêtes pour l'analyse et la modélisation.
- **src/** : Code source pour le traitement, l'ingénierie des features et la modélisation.
- **notebooks/** : Notebooks Jupyter pour chaque étape du workflow.

## Pour commencer
1. Clonez le dépôt.
2. Installez les dépendances nécessaires (voir `pyproject.toml`).
3. Exécutez les notebooks dans l'ordre pour suivre le workflow complet, de la donnée brute à l'évaluation des modèles.

## Documentation
Des documents complémentaires sont disponibles dans le dossier `doc/`, incluant des dictionnaires de variables et la description des sources de données.
