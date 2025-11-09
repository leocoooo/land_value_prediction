import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("data/df_vf_idf_clean.csv", sep=",", low_memory=False)
#print(df.head(5))

# Analyse des données 

## Statistiques descriptives

print(df.describe())

# Distribution des prix au m2
import matplotlib.pyplot as plt
plt.hist(df['prix_m2'], bins=50, color='blue', alpha=0.7)
plt.title('Distribution des prix au m2')
plt.xlabel('Prix au m2')
plt.ylabel('Fréquence')
plt.show()

# distribution des prix médian au m2
plt.hist(df['prix_median_m2'], bins=50, color='green', alpha=0.7)
plt.title('Distribution des prix médian au m2')
plt.xlabel('Prix médian au m2')
plt.ylabel('Fréquence')
plt.show()

# Analyse des corrélations
correlation_matrix = df.select_dtypes(include=[np.number]).corr()


plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f', cbar_kws={"shrink": 0.8})
plt.title('Matrice de corrélation', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# detection des outliers pour le prix au m2
# Colonnes à analyser
cols_to_check = ['prix_m2']

# Analyse des outliers
for col in cols_to_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    pct = len(outliers) / len(df) * 100
    print(f"{col}:")
    print(f"   • Outliers: {len(outliers):,} ({pct:.2f}%)")
    print(f"   • Bornes: [{lower:.2f}, {upper:.2f}]")
    print(f"   • Min: {df[col].min():.2f} | Max: {df[col].max():.2f}\n")

# Visualisation avec boxplots
fig, axes = plt.subplots(1, len(cols_to_check), figsize=(14, 10))

# Si une seule colonne, axes n'est pas un array
if len(cols_to_check) == 1:
    axes = [axes]

for i, col in enumerate(cols_to_check):
    axes[i].boxplot(df[col].dropna(), vert=True)
    axes[i].set_title(col, fontweight='bold', fontsize=12)
    axes[i].set_ylabel('Valeur')
    axes[i].grid(alpha=0.3)

plt.suptitle('Détection des outliers (Boxplots)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# Analyse de ou c'est le plus cher 
if 'code_departement' in df.columns:
    prix_par_dept = df.groupby('code_departement')['prix_m2'].agg(['mean', 'median', 'count'])
    prix_par_dept = prix_par_dept.sort_values('mean', ascending=False)
    print("Prix au m² par département (Île-de-France) :")
    print(prix_par_dept)
    

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(prix_par_dept.index.astype(str), prix_par_dept['mean'], color='teal', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Département')
    ax.set_ylabel('Prix moyen au m² (€)')
    ax.set_title('Prix moyen au m² par département', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


if 'quartier_detaille' in df.columns:
    top_quartiers = df.groupby('quartier_detaille')['prix_m2'].agg(['mean', 'count'])
    top_quartiers = top_quartiers[top_quartiers['count'] >= 50]  
    top_quartiers = top_quartiers.sort_values('mean', ascending=False).head(10)

    print(top_quartiers)
