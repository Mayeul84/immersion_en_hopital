#!/usr/bin/env python
# coding: utf-8

import os

N = 16
os.environ["OMP_NUM_THREADS"] = str(N)
os.environ["OPENBLAS_NUM_THREADS"] = str(N)
os.environ["MKL_NUM_THREADS"] = str(N)
os.environ["NUMEXPR_NUM_THREADS"] = str(N)

import time
import psutil
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

from studies_opt import (
    study_complete_sparsity,
    study_group_sparsity,
    study_group_sparsity_exclude,
    study_group_sparsity_onevsall,
    study_sparsity_degs
)

start_time_global = time.time()

SAVE_DIR = "plots/endothelial"
os.makedirs(SAVE_DIR, exist_ok=True)

print("="*50)
print("SCRIPT ENDOTHELIAL")
print("="*50)

# LOADING & PREPROCESS
celltype="Endothelial cells"
print(f"[1/3] Loading data of type {celltype}...")

endothelial_cells = check_data(celltype=celltype, data_path=DATA_PATH)
if endothelial_cells is None:
    endothelial_cells = preprocess_data(celltype=celltype, with_subsets_config=True, n_neighbors=15, n_comps=100, random_state=42, verbose=False)
    save_data(data=endothelial_cells, data_path=f"{DATA_PATH}/{ANNDATA_MAP[celltype]}")

# UNDERSAMPLING
n_cells = 2e3
print(f"[2/3] Sampling {int(n_cells)} cellules...")
endothelial_cells = neyman_subsample(data=endothelial_cells, target_labels=["Endothelial cells"], label_col="celltype_label", stratify_by=["cellstates_tme"], n_target=int(n_cells))
endothelial_cells = update_data(data=endothelial_cells)

# UMAP 
print("[3/3] Computing UMAP.")
plot_UMAP(endothelial_cells)
plt.savefig(os.path.join(SAVE_DIR, "UMAP_endothelial.png"), dpi=200, bbox_inches='tight')
plt.close()

# EXPERIMENTS
labels = endothelial_cells.obs["cellstates_tme"]
n_points_ratio = 10
n_runs = 25
n_neighbors_candidates = [15, 25, 50, 100, 200]
n_neighbors = 20

# --- EXPE 1 ---
print("\n" + "="*50)
print("EXPERIENCE 1/8: Complete Sparsity (Ratio 0.01 -> 1.0)")
print("="*50)
t0_exp1 = time.time()
ratio_candidates_full = np.linspace(0.01, 1, n_points_ratio)
study_complete_sparsity(
    endothelial_cells, labels=labels, ratio_candidates=ratio_candidates_full, 
    n_runs=n_runs, n_neighbors_candidates=n_neighbors_candidates, 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_complete_sparsity_full.png"
)
print(f"Time : {format_time(time.time() - t0_exp1)}")

# --- EXPE 2 ---
print("\n" + "="*50)
print("EXPERIENCE 2/8: Complete Sparsity (Ratio 0.01 -> 0.25)")
print("="*50)
t0_exp2 = time.time()
ratio_candidates_low = np.linspace(0.01, 0.25, n_points_ratio)
study_complete_sparsity(
    endothelial_cells, labels=labels, ratio_candidates=ratio_candidates_low, 
    n_runs=n_runs, n_neighbors_candidates=n_neighbors_candidates, 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_complete_sparsity_low.png"
)
print(f"Time : {format_time(time.time() - t0_exp2)}")

# --- EXPE 3 ---
print("\n" + "="*50)
print("EXPERIENCE 3/8: Group Sparsity (Ratio 0.01 -> 1.0, k=20)")
print("="*50)
t0_exp3 = time.time()
study_group_sparsity(
    endothelial_cells, labels=labels, ratio_candidates=ratio_candidates_full, 
    n_runs=n_runs, n_neighbors=n_neighbors, search_resolution_method="optuna", 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_group_sparsity_full.png"
)
print(f"Time : {format_time(time.time() - t0_exp3)}")

# --- EXPE 4 ---
print("\n" + "="*50)
print("EXPERIENCE 4/8: Group Sparsity Exclude (Ratio 0.01 -> 1.0, k=20)")
print("="*50)
t0_exp4 = time.time()
study_group_sparsity_exclude(
    endothelial_cells, labels=labels, ratio_candidates=ratio_candidates_full, 
    n_runs=n_runs, n_neighbors=n_neighbors, search_resolution_method="optuna", 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_group_sparsity_exclude_full.png"
)
print(f"Time : {format_time(time.time() - t0_exp4)}")

# --- EXPE 5 ---
print("\n" + "="*50)
print("EXPERIENCE 5/8: Group Sparsity Exclude (Ratio 0.01 -> 0.25, k=20)")
print("="*50)
t0_exp5 = time.time()
study_group_sparsity_exclude(
    endothelial_cells, labels=labels, ratio_candidates=ratio_candidates_low, 
    n_runs=n_runs, n_neighbors=n_neighbors, search_resolution_method="optuna", 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_group_sparsity_exclude_low.png"
)
print(f"Time : {format_time(time.time() - t0_exp5)}")

# --- EXPE 6 ---
print("\n" + "="*50)
print("EXPERIENCE 6/8: Group Sparsity OneVsAll (Ratio 0.01 -> 1.0, k=20)")
print("="*50)
t0_exp6 = time.time()
study_group_sparsity_onevsall(
    endothelial_cells, labels=labels, ratio_candidates=ratio_candidates_full, 
    n_runs=n_runs, n_neighbors=n_neighbors, search_resolution_method="optuna", 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_group_sparsity_onevsall_full.png"
)
print(f"Time : {format_time(time.time() - t0_exp6)}")

# --- EXPE 7 ---
print("\n" + "="*50)
print("EXPERIENCE 7/8: Group Sparsity OneVsAll (Ratio 0.01 -> 0.25, k=20)")
print("="*50)
t0_exp7 = time.time()
study_group_sparsity_onevsall(
    endothelial_cells, labels=labels, ratio_candidates=ratio_candidates_low, 
    n_runs=n_runs, n_neighbors=n_neighbors, search_resolution_method="optuna", 
    show=False, save=True, save_dir=SAVE_DIR, save_name="study_group_sparsity_onevsall_low.png"
)
print(f"Time : {format_time(time.time() - t0_exp7)}")

# --- EXPE 8 ---
print("\n" + "="*50)
print("EXPERIENCE 8/8: Sparsity + Leiden + Differentially Expressed Genes (Ratio 0.01 -> 1.0)")
print("="*50)
t0_exp8 = time.time()
study_sparsity_degs(
    endothelial_cells, labels=labels, ratio_candidates=ratio_candidates_full,
    n_neighbors_candidates=n_neighbors_candidates, search_resolution_method="optuna",
    n_top_genes=100, deg_method="wilcoxon", log2fc_min=0.25, pval_cutoff=0.05,
    show=False, save=True, save_dir=SAVE_DIR
)
print(f"Time : {format_time(time.time() - t0_exp8)}")

print("\n" + "="*50)
print(f"Graphs saved in: {SAVE_DIR}")
print("="*50)