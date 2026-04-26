#!/usr/bin/env python
# coding: utf-8

import os

# NOMBRE DE CPUS A UTILISER EN PARALLELE
N = 16

os.environ["OMP_NUM_THREADS"] = str(N)
os.environ["OPENBLAS_NUM_THREADS"] = str(N)
os.environ["MKL_NUM_THREADS"] = str(N)
os.environ["NUMEXPR_NUM_THREADS"] = str(N)

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc

sc.settings.n_jobs = N

from scripts.utils import (DATA_PATH, ANNDATA_MAP,
    check_data, preprocess_data, save_data,
    plot_UMAP, update_data, format_time
)

from scripts.gene_subsampling import neyman_subsample

from scripts.studies_opt import (
    study_complete_sparsity,
    study_group_sparsity,
    study_group_sparsity_exclude,
    study_group_sparsity_onevsall,
    study_sparsity_degs
)

start_time_global = time.time()

# CHEMIN DE SAUVEGARDE DES RESULTATS
SAVE_DIR = "plots/fibroblast"
os.makedirs(SAVE_DIR, exist_ok=True)

print("="*50)
print("SCRIPT FIBROBLAST")
print("="*50)


# CHARGEMENT DES DONNEES & PRÉTRAITEMENT
celltype = "Fibroblasts"
print(f"[1/3] Chargement des données de type {celltype}...")

# Tentative de chargement des données prétraitées depuis le disque 
fibroblast_cells = check_data(celltype=celltype, data_path=DATA_PATH)
if fibroblast_cells is None:
    # Construction du dataset à partir des données brutes : filtrage + PCA / graph / UMAP
    fibroblast_cells = preprocess_data(celltype=celltype, with_subsets_config=True, n_neighbors=15, n_comps=100, random_state=42, verbose=False)
    save_data(data=fibroblast_cells, data_path=f"{DATA_PATH}/{ANNDATA_MAP[celltype]}")

# SOUS-ÉCHANTILLONNAGE
# Réduction du nombre de cellules tout en conservant la distribution des labels
n_cells = 2e3
print(f"[2/3] Échantillonnage de {int(n_cells)} cellules...")
fibroblast_cells = neyman_subsample(
    data=fibroblast_cells, 
    target_labels=["Fibroblasts"], 
    label_col="celltype_label", 
    stratify_by=["cellstates_tme"], 
    n_target=int(n_cells)
)

# Recalcul de la normalisation + PCA + graphe + UMAP après sous-échantillonnage
fibroblast_cells = update_data(data=fibroblast_cells)

# UMAP 
print("[3/3] Calcul de l'UMAP.")
plot_UMAP(fibroblast_cells)
plt.savefig(os.path.join(SAVE_DIR, "UMAP_fibroblasts.png"), dpi=200, bbox_inches='tight')
plt.close()

# EXPÉRIENCES
labels = fibroblast_cells.obs["cellstates_tme"]
n_points_ratio = 10
n_runs = 25
n_neighbors_candidates = [15, 25, 50, 100, 200]
n_neighbors = 20

"""Évaluation systématique de la robustesse du clustering selon :
  - différents niveaux de sous-échantillonnage (sparsité des données)
  - différentes tailles de voisinage (construction du graphe)

Objectif :
Comprendre la sensibilité du clustering à :
  1) la sparsité des données (niveau de thinning)
  2) la résolution du graphe de voisinage (k)
"""

# --- EXPÉRIENCE 1 ---
print("\n" + "="*50)
print("EXPÉRIENCE 1/8 : Sparsité complète (Ratio 0.01 -> 1.0)")
print("="*50)
t0_exp1 = time.time()
ratio_candidates_full = np.linspace(0.01, 1, n_points_ratio)
study_complete_sparsity(
    fibroblast_cells, labels=labels, ratio_candidates=ratio_candidates_full, 
    n_runs=n_runs, n_neighbors_candidates=n_neighbors_candidates, 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_complete_sparsity_full.png"
)
print(f"Temps : {format_time(time.time() - t0_exp1)}")

# --- EXPÉRIENCE 2 ---
print("\n" + "="*50)
print("EXPÉRIENCE 2/8 : Sparsité complète (Ratio 0.01 -> 0.25)")
print("="*50)
t0_exp2 = time.time()
ratio_candidates_low = np.linspace(0.01, 0.25, n_points_ratio)
study_complete_sparsity(
    fibroblast_cells, labels=labels, ratio_candidates=ratio_candidates_low, 
    n_runs=n_runs, n_neighbors_candidates=n_neighbors_candidates, 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_complete_sparsity_low.png"
)
print(f"Temps : {format_time(time.time() - t0_exp2)}")

"""study_group_sparsity :
Évalue la dégradation de la stabilité du clustering lorsque les données deviennent plus sparse,
en se concentrant sur la séparabilité par paires de types cellulaires (les autres classes sont ignorées).

Cela produit des scores pairwise qui mesurent :
  → la capacité de séparation entre deux types cellulaires en fonction de la qualité des données

Contrairement aux métriques globales, cela met en évidence :
  - les paires de types cellulaires qui se confondent en premier
  - les relations robustes ou fragiles face à la réduction de signal
"""

# --- EXPÉRIENCE 3 ---
print("\n" + "="*50)
print("EXPÉRIENCE 3/8 : Sparsité groupée (Ratio 0.01 -> 1.0, k=20)")
print("="*50)
t0_exp3 = time.time()
study_group_sparsity(
    fibroblast_cells, labels=labels, ratio_candidates=ratio_candidates_full, 
    n_runs=n_runs, n_neighbors=n_neighbors, search_resolution_method="optuna", 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_group_sparsity_full.png"
)
print(f"Temps : {format_time(time.time() - t0_exp3)}")

"""Évaluation "exclude-one" :
Pour chaque label L, toutes les cellules de type L sont retirées,
et le clustering est évalué sur le reste des données.

Cela mesure l'influence de chaque type cellulaire sur la structure globale du clustering.
Si les performances s'améliorent lorsque L est retiré, cela indique que L introduit
de l'ambiguïté ou des recouvrements avec d'autres populations.
"""

# --- EXPÉRIENCE 4 ---
print("\n" + "="*50)
print("EXPÉRIENCE 4/8 : Exclusion groupée (Ratio 0.01 -> 1.0, k=20)")
print("="*50)
t0_exp4 = time.time()
study_group_sparsity_exclude(
    fibroblast_cells, labels=labels, ratio_candidates=ratio_candidates_full, 
    n_runs=n_runs, n_neighbors=n_neighbors, search_resolution_method="optuna", 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_group_sparsity_exclude_full.png"
)
print(f"Temps : {format_time(time.time() - t0_exp4)}")

# --- EXPÉRIENCE 5 ---
print("\n" + "="*50)
print("EXPÉRIENCE 5/8 : Exclusion groupée (Ratio 0.01 -> 0.25, k=20)")
print("="*50)
t0_exp5 = time.time()
study_group_sparsity_exclude(
    fibroblast_cells, labels=labels, ratio_candidates=ratio_candidates_low, 
    n_runs=n_runs, n_neighbors=n_neighbors, search_resolution_method="optuna", 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_group_sparsity_exclude_low.png"
)
print(f"Temps : {format_time(time.time() - t0_exp5)}")

"""Évaluation one-vs-all :
Pour chaque label L, le problème est réduit à une classification binaire :
  - classe L vs toutes les autres cellules regroupées en "other".

Cela permet de mesurer la capacité du clustering à isoler un type cellulaire spécifique,
indépendamment de la structure interne des autres classes.

Un score élevé indique que le type cellulaire forme un cluster distinct et bien séparé.
"""

# --- EXPE 6 ---
print("\n" + "="*50)
print("EXPÉRIENCE 6/8 : OneVsAll groupé (Ratio 0.01 -> 1.0, k=20)")
print("="*50)
t0_exp6 = time.time()
study_group_sparsity_onevsall(
    fibroblast_cells, labels=labels, ratio_candidates=ratio_candidates_full, 
    n_runs=n_runs, n_neighbors=n_neighbors, search_resolution_method="optuna", 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_group_sparsity_onevsall_full.png"
)
print(f"Temps : {format_time(time.time() - t0_exp6)}")

# --- EXPÉRIENCE 7 ---
print("\n" + "="*50)
print("EXPÉRIENCE 7/8 : OneVsAll groupé (Ratio 0.01 -> 0.25, k=20)")
print("="*50)
t0_exp7 = time.time()
study_group_sparsity_onevsall(
    fibroblast_cells, labels=labels, ratio_candidates=ratio_candidates_low, 
    n_runs=n_runs, n_neighbors=n_neighbors, search_resolution_method="optuna", 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_group_sparsity_onevsall_low.png"
)
print(f"Temps : {format_time(time.time() - t0_exp7)}")

"""Analyse des gènes différentiellement exprimés (DEGs) sous sparsité :
Pour chaque niveau de sous-échantillonnage et chaque valeur de k, on applique le clustering Leiden
puis on identifie les gènes différentiellement exprimés entre clusters."""

# --- EXPÉRIENCE 8 ---
print("\n" + "="*50)
print("EXPÉRIENCE 8/8 : Sparsité + Leiden + gènes différentiellement exprimés (Ratio 0.01 -> 1.0)")
print("="*50)
t0_exp8 = time.time()
study_sparsity_degs(
    fibroblast_cells, labels=labels, ratio_candidates=ratio_candidates_full,
    n_neighbors_candidates=n_neighbors_candidates, search_resolution_method="optuna",
    n_top_genes=100, deg_method="wilcoxon", log2fc_min=0.25, pval_cutoff=0.05,
    show=False, save=True, save_dir=SAVE_DIR
)
print(f"Temps : {format_time(time.time() - t0_exp8)}")

print("\n" + "="*50)
print(f"Graphiques sauvegardés dans : {SAVE_DIR}")
print("="*50)