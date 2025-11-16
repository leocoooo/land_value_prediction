# Dictionnaire des données - Variables par commune

Ce document décrit les variables finales du dataset `final_data_par_commune` utilisées pour la prédiction du prix au m² des biens immobiliers en Île-de-France.

## 📋 Table des matières
- [Identifiant](#identifiant)
- [Indicateurs démographiques](#indicateurs-démographiques)
- [Indicateurs économiques et revenus](#indicateurs-économiques-et-revenus)
- [Indicateurs du marché immobilier](#indicateurs-du-marché-immobilier)
- [Indicateurs d'emploi](#indicateurs-demploi)
- [Indicateurs d'activité économique](#indicateurs-dactivité-économique)
- [Indicateurs de développement humain](#indicateurs-de-développement-humain)
- [Indicateurs de sécurité](#indicateurs-de-sécurité)
- [Densité de population par année](#densité-de-population-par-année)

---
## Warnings 
Il y a certaines leaks de data mais qui ne sont pas trop dérangeant à priori. Par exemple on va prendre le taux de criminalité moyen sur les années 2016-2024 pour des ventes avant 2024. 


## Identifiant

| Variable | Description | Type | Source |
|----------|-------------|------|--------|
| `code_commune` | Code INSEE de la commune (identifiant unique) | string | INSEE |

---

## Indicateurs démographiques

| Variable | Description | Type | Unité | Source | Années |
|----------|-------------|------|-------|--------|--------|
| `taux_croissance_pop_annuel_2016_2019` | Taux de croissance annuel moyen de la population entre 2016 et 2019 | float | % | INSEE | 2016-2019 |
| `taux_croissance_pop_annuel_2016_2020` | Taux de croissance annuel moyen de la population entre 2016 et 2020 | float | % | INSEE | 2016-2020 |
| `taux_croissance_pop_annuel_2016_2021` | Taux de croissance annuel moyen de la population entre 2016 et 2021 | float | % | INSEE | 2016-2021 |
| `taux_croissance_pop_annuel_2016_2022` | Taux de croissance annuel moyen de la population entre 2016 et 2022 | float | % | INSEE | 2016-2022 |
| `taux_croissance_pop_annuel_2016_2023` | Taux de croissance annuel moyen de la population entre 2016 et 2023 | float | % | INSEE | 2016-2023 |
| `taux_croissance_pop_annuel_2016_2024` | Taux de croissance annuel moyen de la population entre 2016 et 2024 | float | % | INSEE | 2016-2024 |
| `densite_population_active_2020` | Densité de la population active (15-64 ans) en 2020 | float | hab/km² | Calcul : population_15_64_2020 / superficie | 2020 |

**Formule de calcul du taux de croissance** :
```
taux = ((pop_année_finale / pop_2016) ^ (1/nb_années) - 1) × 100
```

---

## Indicateurs économiques et revenus

| Variable | Description | Type | Unité | Source | Année |
|----------|-------------|------|-------|--------|-------|
| `revenu_fiscal_reference_moyen_2020` | Revenu fiscal de référence moyen par ménage fiscal | float | € | INSEE | 2020 |
| `revenu_median_2020` | Revenu médian par unité de consommation | float | € | INSEE | 2020 |
| `taux_pauvrete_60pct_2020` | Taux de pauvreté (seuil à 60% du revenu médian) | float | % | INSEE | 2020 |

---

## Indicateurs du marché immobilier

| Variable | Description | Type | Unité | Source | Année |
|----------|-------------|------|-------|--------|-------|
| `densite_residences_principales_2020` | Densité de résidences principales | float | logements/km² | Calcul : nb_residences_principales_2020 / superficie | 2020 |
| `taux_residences_secondaires` | Part des résidences secondaires dans le total des logements | float | ratio (0-1) | Calcul : nb_residences_secondaires / nb_logements_total | 2020 |
| `taux_logements_vacants` | Part des logements vacants dans le total des logements | float | ratio (0-1) | Calcul : nb_logements_vacants / nb_logements_total | 2020 |
| `taux_proprietaires` | Part des propriétaires parmi les résidences principales | float | ratio (0-1) | Calcul : nb_proprietaires / nb_residences_principales | 2020 |
| `ratio_residences_secondaires_population` | Ratio résidences secondaires / population active | float | ratio | Calcul : nb_residences_secondaires / population_15_64 | 2020 |

---

## Indicateurs d'emploi

| Variable | Description | Type | Unité | Source | Année |
|----------|-------------|------|-------|--------|-------|
| `emploi_salarie_2020` | Nombre total d'emplois salariés | int | emplois | INSEE | 2020 |
| `nb_actifs_15_64_2020` | Nombre d'actifs âgés de 15 à 64 ans | int | personnes | INSEE | 2020 |
| `taux_chomage_15_64` | Taux de chômage de la population 15-64 ans | float | ratio (0-1) | Calcul : chomeurs_15_64 / nb_actifs_15_64 | 2020 |
| `taux_emploi` | Taux d'emploi (rapport emploi/population active) | float | ratio (0-1) | Calcul : emploi_total / population_15_64 | 2020 |
| `evolution_emploi_2014_2020` | Évolution de l'emploi total entre 2014 et 2020 | float | % | Calcul : (emploi_2020 - emploi_2014) / emploi_2014 | 2014-2020 |

---

## Indicateurs d'activité économique

### Nombre d'établissements par secteur

| Variable | Description | Type | Unité | Source | Année |
|----------|-------------|------|-------|--------|-------|
| `etablissements_agriculture_2021` | Nombre d'établissements du secteur agricole | int | établissements | INSEE | 2021 |
| `etablissements_industrie_2021` | Nombre d'établissements du secteur industriel | int | établissements | INSEE | 2021 |
| `etablissements_construction_2021` | Nombre d'établissements du secteur construction | int | établissements | INSEE | 2021 |
| `etablissements_1_salarie_2021` | Nombre d'établissements avec 1 salarié | int | établissements | INSEE | 2021 |
| `etablissements_10_plus_salaries_2021` | Nombre d'établissements avec 10 salariés ou plus | int | établissements | INSEE | 2021 |

### Structure de l'activité économique

| Variable | Description | Type | Unité | Calcul | Année |
|----------|-------------|------|-------|--------|-------|
| `part_services_entreprises` | Part des établissements de services aux entreprises | float | ratio (0-1) | etabl_services_entreprises / etabl_total | 2021 |
| `part_commerce_tourisme` | Part des établissements de commerce et transport | float | ratio (0-1) | etabl_commerce_transport / etabl_total | 2021 |
| `part_admin_sante` | Part des établissements de services publics et santé | float | ratio (0-1) | etabl_services_publics_sante / etabl_total | 2021 |
| `etablissements_par_menage` | Nombre d'établissements par ménage | float | ratio | etabl_total / nb_menages | 2021/2020 |
| `taux_etablissements_10_plus` | Part des établissements avec 10 salariés ou plus | float | ratio (0-1) | etabl_10_plus / etabl_total | 2021 |

---

## Indicateurs de développement humain

| Variable | Description | Type | Unité | Source | Année |
|----------|-------------|------|-------|--------|-------|
| `sante_score_2013_commune` | Score de santé de la commune (composante IDH2) | float | score normalisé | Région Île-de-France | 2013 |
| `education_score_2013_commune` | Score d'éducation de la commune (composante IDH2) | float | score normalisé | Région Île-de-France | 2013 |
| `revenu_score_2013_commune` | Score de revenu de la commune (composante IDH2) | float | score normalisé | Région Île-de-France | 2013 |
| `idh2_2013_commune` | Indice de Développement Humain 2 (synthèse des 3 scores) | float | score normalisé | Région Île-de-France | 2013 |

**Note sur l'IDH2** : L'IDH2 est un indicateur composite calculé à partir des scores de santé, éducation et revenu, adapté au contexte francilien.

---

## Indicateurs de sécurité

| Variable | Description | Type | Unité | Source | Période |
|----------|-------------|------|-------|--------|---------|
| `taux_criminalite_moyen` | Taux moyen de criminalité (nombre de crimes/population) | float | ratio | Calcul : nombre crimes total / population moyenne | Moyenne multi-annuelle |

**Méthode de calcul** : Agrégation du nombre total de crimes sur toutes les années disponibles, divisé par la population moyenne.

---

## Densité de population par année

Densité calculée pour chaque année de 2016 à 2024 :

| Variable | Description | Type | Unité | Calcul | Année |
|----------|-------------|------|-------|--------|-------|
| `densite_pop_2016` | Densité de population en 2016 | float | hab/km² | population_2016 / superficie | 2016 |
| `densite_pop_2017` | Densité de population en 2017 | float | hab/km² | population_2017 / superficie | 2017 |
| `densite_pop_2018` | Densité de population en 2018 | float | hab/km² | population_2018 / superficie | 2018 |
| `densite_pop_2019` | Densité de population en 2019 | float | hab/km² | population_2019 / superficie | 2019 |
| `densite_pop_2020` | Densité de population en 2020 | float | hab/km² | population_2020 / superficie | 2020 |
| `densite_pop_2021` | Densité de population en 2021 | float | hab/km² | population_2021 / superficie | 2021 |
| `densite_pop_2022` | Densité de population en 2022 | float | hab/km² | population_2022 / superficie | 2022 |
| `densite_pop_2023` | Densité de population en 2023 | float | hab/km² | population_2023 / superficie | 2023 |
| `densite_pop_2024` | Densité de population en 2024 | float | hab/km² | population_2024 / superficie | 2024 |

---

## 📊 Statistiques du dataset

- **Nombre total de variables** : 44
- **Périmètre géographique** : Île-de-France (région 11)
- **Nombre de communes** : ~1300 communes
- **Période couverte** : 2013-2024 (selon les indicateurs)

## 🔗 Sources principales

1. **INSEE** : Population, démographie, emploi, établissements économiques
2. **Région Île-de-France** : IDH2 et ses composantes
3. **data.gouv.fr** : Données de criminalité
4. **Calculs dérivés** : Variables de ratio, taux et densité calculées à partir des données brutes

---

## 📝 Notes d'utilisation

### Valeurs manquantes
- Certaines communes peuvent avoir des valeurs manquantes pour les indicateurs IDH2 (2013)
- Les taux de criminalité sont calculés sur plusieurs années pour lisser les variations annuelles

### Cohérence temporelle
- La plupart des indicateurs économiques sont de 2020-2021
- Les densités de population sont disponibles annuellement de 2016 à 2024
- L'IDH2 date de 2013 (dernière mise à jour disponible)

### Utilisation pour la prédiction
Ces variables servent de features pour prédire le `prix_m2` des biens immobiliers. Elles capturent :
- La dynamique démographique (croissance, densité)
- L'attractivité économique (emploi, revenus, établissements)
- La tension du marché immobilier (taux de vacance, propriétaires)
- La qualité de vie (IDH2, sécurité)

---

*Document généré le 16/11/2025*  
*Projet : land_value_prediction - Prédiction des prix immobiliers en Île-de-France*
