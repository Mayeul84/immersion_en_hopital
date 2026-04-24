#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import psutil
import numpy as np
import scanpy as sc

import time
import datetime

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

start_time_global = time.time()

N = 16  
os.environ["OMP_NUM_THREADS"] = str(N)
os.environ["OPENBLAS_NUM_THREADS"] = str(N)
os.environ["MKL_NUM_THREADS"] = str(N)
os.environ["NUMEXPR_NUM_THREADS"] = str(N)
sc.settings.n_jobs = N

# ==========================================
# IMPORTS 
# ==========================================
from scripts.utils import check_data, preprocess_data, save_data, plot_UMAP, update_data
from scripts.utils import PROJECT_PATH, DATA_PATH, RESULTS_PATH, CELLTYPE_MAP, HISTOTYPE_MAP, ANNDATA_MAP, SUBSETS_CONFIG
from scripts.gene_subsampling import neyman_subsample
from scripts.clustering import cluster_data, find_best_resolution, find_best_resolution_linspace, best_leiden_run
from studies_opt import (study_sparsity, study_sparsity_with_trajectories, 
                                 study_complete_sparsity, study_group_sparsity, 
                                 study_group_sparsity_exclude, study_group_sparsity_onevsall, study_sparsity_degs)

SAVE_DIR = "plots/fibroblast"
os.makedirs(SAVE_DIR, exist_ok=True)

print("="*50)
print("SCRIPT FIBROBLAST")
print("="*50)

# ==========================================
# 1. LOADING & PREPROCESS
# ==========================================
celltype = "Fibroblasts"
print(f"[1/3] Loading data of type {celltype}...")

fibroblast_cells = check_data(celltype=celltype, data_path=DATA_PATH)
if fibroblast_cells is None:
    fibroblast_cells = preprocess_data(celltype=celltype, with_subsets_config=True, n_neighbors=15, n_comps=100, random_state=42, verbose=False)
    save_data(data=fibroblast_cells, data_path=f"{DATA_PATH}/{ANNDATA_MAP[celltype]}")

# ==========================================
# 2. UNDERSAMPLING
# ==========================================
n_cells = 2e3
print(f"[2/3] Sampling {int(n_cells)} cellules...")
fibroblast_cells = neyman_subsample(data=fibroblast_cells, target_labels=["Fibroblasts"], label_col="celltype_label", stratify_by=["cellstates_tme"], n_target=int(n_cells))
fibroblast_cells = update_data(data=fibroblast_cells)

# ==========================================
# 3. UMAP 
# ==========================================
print("[3/3] Computing UMAP.")
plot_UMAP(fibroblast_cells)
plt.savefig(os.path.join(SAVE_DIR, "UMAP_fibroblasts.png"), dpi=200, bbox_inches='tight')
plt.close()

# ==========================================
# EXPERIMENTS
# ==========================================
labels = fibroblast_cells.obs["cellstates_tme"]
n_points_ratio = 10
n_runs = 25
n_neighbors_candidates = [15, 25, 50, 100, 200]
n_neighbors = 20

# --- EXPE 1 ---
# print("\n" + "="*50)
# print("EXPERIENCE 1/5: Complete Sparsity (Ratio 0.01 -> 1.0)")
# print("="*50)
t0_exp1 = time.time()
ratio_candidates_full = np.linspace(0.01, 1, n_points_ratio)
# study_complete_sparsity(
#     fibroblast_cells, labels=labels, ratio_candidates=ratio_candidates_full, 
#     n_runs=n_runs, n_neighbors_candidates=n_neighbors_candidates, 
#     show=False, save=True, save_dir=SAVE_DIR, save_name="study_complete_sparsity_full.png"
# )
# print(f"Time : {format_time(time.time() - t0_exp1)}")

# --- EXPE 2 ---
#print("\n" + "="*50)
#print("EXPERIENCE 2/5: Complete Sparsity (Ratio 0.01 -> 0.25)")
#print("="*50)
#t0_exp2 = time.time()
ratio_candidates_low = np.linspace(0.01, 0.25, n_points_ratio)
#study_complete_sparsity(
#    fibroblast_cells, labels=labels, ratio_candidates=ratio_candidates_low, 
# #  



print("\n" + "="*50)
print("EXPERIENCE DEGs: Sparsity + Leiden + Differentially Expressed Genes (Ratio 0.01 -> 1.0)")
print("="*50)
t0_degs = time.time()
study_sparsity_degs(
    fibroblast_cells, labels=labels, ratio_candidates=ratio_candidates_full,
    n_neighbors_candidates=n_neighbors_candidates, search_resolution_method="optuna",
    n_top_genes=100, deg_method="wilcoxon", log2fc_min=0.25, pval_cutoff=0.05,
    show=False, save=True, save_dir=SAVE_DIR
)
print(f"Time : {format_time(time.time() - t0_degs)}")

print("\n" + "="*50)
print(f"Graphs saved in: {SAVE_DIR}")
print("="*50)