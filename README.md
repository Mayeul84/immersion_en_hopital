# Impact de la sparsité sur l'informativité du transcriptome single-cell

**Immersion en hôpital — MVA**  
Mayeul Lambert & Céline Liu

---

## Contexte

Ce projet étudie l'impact de la profondeur de lecture (sparsité) sur la capacité à identifier des sous-types cellulaires dans des données de transcriptomique single-cell (scRNA-seq).

Le jeu de données porte sur des **tumeurs de la corticosurrénale** et comprend 168 927 cellules × 16 638 gènes, annotées à trois échelles : clinique (*Tumor Type*), macroscopique (*Cell Types*) et microscopique (*Microenvironment States*).

**Problématique centrale :** jusqu'à quel seuil de dégradation du signal (sparsité croissante) peut-on encore identifier les sous-types cellulaires par clustering non supervisé ?

---

## Méthode

### Simulation de la sparsité — Thinning de Poisson

Réduire la profondeur de lecture revient à sous-échantillonner binomialement la matrice de counts avec un taux $p \in (0, 1)$, ce qui correspond à un **thinning de Poisson** :

$$\tilde{X}_{ij} \sim \text{NB}(p \cdot \epsilon_j \cdot \mu_{ij},\; \theta_i)$$

On fait varier $p$ de 1 (données complètes) vers 0 (données très sparses) pour simuler différentes profondeurs de lecture.

### Pipeline de clustering

Pour chaque niveau de sparsité $p$ :

1. Thinning de Poisson sur la matrice brute
2. Normalisation SCT (correction des biais techniques de capture)
3. Réduction de dimension par PCA
4. Construction du graphe des $k$ plus proches voisins
5. Clustering de Leiden avec optimisation de la résolution (Optuna / Golden section search)
6. Évaluation par ARI, homogénéité et complétude

### Évaluation

- **Homogénéité** : un cluster ne contient-il qu'une seule classe d'origine ?
- **Complétude** : une classe est-elle bien regroupée dans un seul cluster ?
- **ARI (Adjusted Rand Index)** : accord entre la partition prédite et le *ground truth*, corrigé du hasard

Le paramètre $k$ (nombre de voisins dans le graphe) permet de définir des **seuils critiques** de sparsité en dessous desquels le clustering s'effondre.

---

## Structure du projet

```
immersion_en_hopital/
│
├── cell_types.py          # Étude de l'impact de la sparsité sur tous les types cellulaires
├── endothelial.py         # Étude sur les cellules endothéliales
├── fibroblast.py          # Étude sur les fibroblastes
├── myeloid.py             # Étude sur les cellules myéloïdes
│
├── scripts/
│   ├── clustering.py      # Fonctions de clustering (Leiden, optimisation de résolution)
│   ├── gene_subsampling.py# Thinning de Poisson, sous-échantillonnage de gènes et cellules
│   ├── scoring.py         # Métriques d'évaluation (ARI, homogénéité, complétude)
│   ├── studies.py         # Études expérimentales (one-vs-all, paires de labels, impact de k)
│   ├── studies_opt.py     # Version parallélisée (CPU) des études
│   └── utils.py           # Fonctions utilitaires (AnnData, affichage, UMAP)
│
├── plots/                 # Résultats (figures et labels Leiden)
│   ├── cell_types/
│   ├── endothelial/
│   ├── fibroblast/
│   └── myeloid/
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Dépendances principales : `scanpy`, `anndata`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `optuna`, `matplotlib`, `tqdm`, `joblib`, `psutil`.

---

## Usage

Chaque script de niveau racine correspond à une population cellulaire. Lancer directement via :

```bash
python myeloid.py
python fibroblast.py
python endothelial.py
python cell_types.py
```

Les résultats (figures `.png`, labels Leiden `.parquet`/`.csv`) sont sauvegardés dans le dossier `plots/` correspondant.

### Parallélisation

Les études sont conçues pour tourner sur 16 cœurs CPU. Le nombre de threads est configurable en tête de chaque script :

```python
N = 16  # Nombre de CPUs à utiliser
```

---

## Types cellulaires étudiés

| Type cellulaire   | Nombre de cellules | Sous-types analysés |
|-------------------|--------------------|---------------------|
| Stéroïdes         | 133 501            | —                   |
| **Myéloïdes**     | **16 698**         | Macrophages, TAM    |
| **Endothéliales** | **8 540**          | —                   |
| **Fibroblastes**  | **7 942**          | CAF, Résidents      |
| Lymphoïdes        | 2 054              | —                   |
| Chromaffines      | 192                | —                   |

Les types en gras sont les populations d'intérêt principal.

---

## Résultats principaux

Seuils critiques d'homogénéité ($h = 0.4$) en niveau de sparsité et nombre moyen de gènes détectés par cellule, estimés sur 25 runs :

| Type cellulaire   | $k=15$       | $k=25$       | $k=50$       | $k=100$      | $k=200$       |
|-------------------|--------------|--------------|--------------|--------------|---------------|
| Endothéliales     | 0.06 / 166   | 0.07 / 190   | 0.21 / 483   | 0.25 / 553   | 0.46 / 853    |
| Fibroblastes      | 0.13 / 327   | 0.25 / 553   | 0.19 / 446   | 0.24 / 536   | 0.77 / 1160   |
| Myéloïdes         | 0.19 / 446   | 0.25 / 553   | 0.20 / 465   | 0.19 / 446   | 0.22 / 501    |

---

## Données

Les données ne sont pas incluses dans ce dépôt. Le jeu de données est un fichier AnnData (`.h5ad`) à placer selon le chemin défini dans `scripts/utils.py` (`DATA_PATH`).
