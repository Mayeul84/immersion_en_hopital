import os
import gc

from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import adjusted_rand_score
import scanpy as sc

from matplotlib.lines import Line2D
from scripts.gene_subsampling import thinning

from scripts.scoring import balanced_correctcells_score, compute_all_scores
from scripts.utils import update_data
from scripts.utils import MAX_RNG_RANGE

from scripts.clustering import find_best_resolution, average_leiden_run, best_leiden_run

# Ce script répertorie les mêmes fonctions que dans le script 'studies.py' mais construistent de sorte à pouvoir lancer le calcul
# en parallèle sur CPU.
# Ici, les fonctions sont cassées en sous-fonctions (fonction lancée sur un CPU spécifique) et en fonctions principales (lancement des sous-fonctions sur CPUs et récupération des calculs)

# Deux nouvelles fonctions: Expérience 'One vs All' et Analyse de gènes dicriminants. Tout le reste est identique au script 'studies.py', à l'exception de compute_n_genes_sparsity dont
# on a considéré qu'une optimisation n'était pas nécessaire (temps de calcul déjà faible)

N_JOBS = 16 # Nombre de CPUs utilisés en parallèle par défaut
DEFAULT_SAVE_DIR = "/plots" # Chemin de sauvegarde par défaut (modifiable)

# ==============================================================================
# SOUS-FONCTIONS (VOIR FONCTIONS PRINCIPALES POUR PLUS DE DETAILS)
# ==============================================================================

def _get_optimal_resolution(cells, labels, ratio, n_neighbors, normalization, search_resolution_method):
    """
    Calcule la résolution optimale UNE SEULE FOIS pour un ratio donné.

    cells: objet anndata/scanpy (dataframe).
    labels: list/array des labels ground truth. Un label est un entier entre 0 et n_class-1.
    ratio: ratio de sparsité appliqué
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden).
    normalization: choix de la normalisation des données brutes.
    search_resolution_method: méthode d'optimisation pour la recherche de la résolution optimale de Leiden.
    """
    if ratio < 1:
        temp_cells = thinning(cells, reduction_ratio=ratio, same_reads=False, copy=True)
        temp_cells = update_data(temp_cells, n_neighbors=n_neighbors, n_comps=100, random_state=42, normalization=normalization)
    else:
        temp_cells = cells.copy()
        temp_cells = update_data(temp_cells, n_neighbors=n_neighbors, n_comps=100, random_state=42, normalization=normalization)
    
    if 'neighbors' not in temp_cells.uns:
        sc.pp.neighbors(temp_cells, n_neighbors=n_neighbors, n_pcs=100)
        
    results = find_best_resolution(data=temp_cells, true_labels=labels, n_neighbors=None, n_trials=50, method=search_resolution_method, show=False)
    resolution = results["resolution"]
    
    del temp_cells
    gc.collect()
    return resolution

def _run_stdthinning(cells, labels, ratio, n_neighbors, normalization, resolution):
    """
    Lance le thinning, le partitionnement et le calcul de scores.

    cells: objet anndata/scanpy (dataframe).
    labels: list/array des labels ground truth. Un label est un entier entre 0 et n_class-1.
    ratio: ratio de sparsité appliqué
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden).
    normalization: choix de la normalisation des données brutes.
    resolution: résolution de leiden appliqué.
    """
    seed = np.random.randint(0, MAX_RNG_RANGE)
    if ratio < 1:
        thinned_cells = thinning(cells, reduction_ratio=ratio, same_reads=False, copy=True)
        thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=seed, normalization=normalization)
    else:
        thinned_cells = cells.copy()
        thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=seed, normalization=normalization)

    sc.tl.leiden(thinned_cells, resolution=resolution, key_added='leiden_temp', random_state=seed)
    leiden_labels = thinned_cells.obs['leiden_temp'].copy()

    scores = compute_all_scores(true_labels=labels, cluster_labels=leiden_labels)
    del thinned_cells
    return scores

def _run_trajectories(thinned_cells, labels, resolution, seed):
    """
    Lance le partitionnement et le calcul de scores à partir de thinned cells.

    thinned_cells: objet anndata/scanpy (dataframe). Cellules auxquels 'thinning' a déjà été appliqué.
    labels: list/array des labels ground truth. Un label est un entier entre 0 et n_class-1.
    resolution: résolution de leiden appliqué.
    seed: graine aléatoire du leiden (l'algorithme de leiden est en partie aléatoire).
    """
    # Travaille sur une copie superficielle pour éviter les conflits de mutation entre workers
    adata_tmp = thinned_cells.copy()
    sc.tl.leiden(adata_tmp, resolution=resolution, key_added='leiden_temp', random_state=seed)
    leiden_labels = adata_tmp.obs['leiden_temp'].copy()
    
    h, c, v = homogeneity_completeness_v_measure(labels, leiden_labels)
    ari = adjusted_rand_score(labels, leiden_labels)
    correct, correct_detailed = balanced_correctcells_score(labels, leiden_labels)
    
    del adata_tmp
    return h, c, v, ari, correct, correct_detailed

# def _run_group_nostdthinning(thinned_cells, labels, resolution, combination_labels, seed):
#     adata_tmp = thinned_cells.copy()
#     sc.tl.leiden(adata_tmp, resolution=resolution, key_added='leiden_temp', random_state=seed)
#     leiden_labels = adata_tmp.obs['leiden_temp'].copy()

#     h_runs, c_runs, v_runs, a_runs, ct_runs = {}, {}, {}, {}, {}
#     for label1, label2 in combination_labels:
#         mask = (labels == label1) | (labels == label2)
#         if mask.sum() == 0:
#             continue

#         scores = compute_all_scores(true_labels=labels[mask], cluster_labels=leiden_labels[mask])
#         pair_key = f"{label1} vs {label2}"
#         h_runs[pair_key] = scores["homogeneity"]
#         c_runs[pair_key] = scores["completness"]
#         v_runs[pair_key] = scores["v"]
#         a_runs[pair_key] = scores["ari"]
#         ct_runs[pair_key] = scores["correct"]
        
#     del adata_tmp
#     return h_runs, c_runs, v_runs, a_runs, ct_runs

def _run_group(cells, labels, ratio, n_neighbors, normalization, resolution, combination_labels):
    """
    Lance une étude 'groups' où on évalue les scores de partitionnement de chaque couple de labels ground truth. (Voir fonction principale)

    cells: objet anndata/scanpy (dataframe)
    labels: list/array des labels ground truth. Un label est un entier entre 0 et n_class-1.
    ratio: ratio de sparsité appliqué
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden).
    normalization: choix de la normalisation des données brutes.
    resolution: résolution de leiden appliqué.
    combination_labels: l'ensemble des couples de labels étudiés
    """
    seed = np.random.randint(0, MAX_RNG_RANGE)
    if ratio < 1:
        thinned_cells = thinning(cells, reduction_ratio=ratio, same_reads=False, copy=True)
        thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=seed, normalization=normalization)
    else:
        thinned_cells = cells.copy()
        thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=seed, normalization=normalization)

    sc.tl.leiden(thinned_cells, resolution=resolution, key_added='leiden_temp', random_state=seed)
    leiden_labels = thinned_cells.obs['leiden_temp'].copy()

    h_runs, c_runs, v_runs, a_runs, ct_runs = {}, {}, {}, {}, {}
    for label1, label2 in combination_labels:
        mask = (labels == label1) | (labels == label2)
        if mask.sum() == 0:
            continue

        scores = compute_all_scores(true_labels=labels[mask], cluster_labels=leiden_labels[mask])
        pair_key = f"{label1} vs {label2}"
        h_runs[pair_key] = scores["homogeneity"]
        c_runs[pair_key] = scores["completness"]
        v_runs[pair_key] = scores["v"]
        a_runs[pair_key] = scores["ari"]
        ct_runs[pair_key] = scores["correct"]

    del thinned_cells
    return h_runs, c_runs, v_runs, a_runs, ct_runs

# def _run_group_exclude_nostdthinning(thinned_cells, labels, resolution, unique_labels, seed):
#     adata_tmp = thinned_cells.copy()
#     sc.tl.leiden(adata_tmp, resolution=resolution, key_added='leiden_temp', random_state=seed)
#     leiden_labels = adata_tmp.obs['leiden_temp'].copy()

#     h_runs, c_runs, v_runs, a_runs, ct_runs = {}, {}, {}, {}, {}
#     for label in unique_labels:
#         mask = (labels != label)
#         if mask.sum() == 0:
#             continue

#         scores = compute_all_scores(true_labels=labels[mask], cluster_labels=leiden_labels[mask])
#         h_runs[label] = scores["homogeneity"]
#         c_runs[label] = scores["completness"]
#         v_runs[label] = scores["v"]
#         a_runs[label] = scores["ari"]
#         ct_runs[label] = scores["correct"]
        
#     del adata_tmp
#     return h_runs, c_runs, v_runs, a_runs, ct_runs

def _run_group_exclude(cells, labels, ratio, n_neighbors, normalization, resolution, unique_labels):
    """
    Lance une étude 'exclude' où on évalue les scores de partitionnement lorsqu'on exclu un label ground truth, pour chaque label. (Voir fonction principale)

    cells: objet anndata/scanpy (dataframe)
    labels: list/array des labels ground truth. Un label est un entier entre 0 et n_class-1.
    ratio: ratio de sparsité appliqué
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden).
    normalization: choix de la normalisation des données brutes.
    resolution: résolution de leiden appliqué.
    unique_labels: nom des labels ground truth avec d'éventuels autres labels (exemple: 'no-exclusion', qui désigne le fait de ne pas filtrer)
    """

    seed = np.random.randint(0, MAX_RNG_RANGE)
    if ratio < 1:
        thinned_cells = thinning(cells, reduction_ratio=ratio, same_reads=False, copy=True)
        thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=seed, normalization=normalization)
    else:
        thinned_cells = cells.copy()
        thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=seed, normalization=normalization)

    sc.tl.leiden(thinned_cells, resolution=resolution, key_added='leiden_temp', random_state=seed)
    leiden_labels = thinned_cells.obs['leiden_temp'].copy()

    h_runs, c_runs, v_runs, a_runs, ct_runs = {}, {}, {}, {}, {}
    for label in unique_labels:
        mask = (labels != label)
        if mask.sum() == 0:
            continue

        scores = compute_all_scores(true_labels=labels[mask], cluster_labels=leiden_labels[mask])
        h_runs[label] = scores["homogeneity"]
        c_runs[label] = scores["completness"]
        v_runs[label] = scores["v"]
        a_runs[label] = scores["ari"]
        ct_runs[label] = scores["correct"]

    del thinned_cells
    return h_runs, c_runs, v_runs, a_runs, ct_runs

def _run_group_onevsall(cells, labels, ratio, n_neighbors, normalization, resolution, unique_labels):
    """
    Lance une étude 'one vs all' où on évalue les scores de partitionnement de chaque label ground truth contre toutes les autres. (Voir fonction principale)

    cells: objet anndata/scanpy (dataframe)
    labels: list/array des labels ground truth. Un label est un entier entre 0 et n_class-1.
    ratio: ratio de sparsité appliqué
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden).
    normalization: choix de la normalisation des données brutes.
    resolution: résolution de leiden appliqué.
    unique_labels: nom des labels ground truth avec d'éventuels autres labels (exemple: 'no-exclusion', qui désigne le fait de ne pas filtrer)
    """
    seed = np.random.randint(0, MAX_RNG_RANGE)
    if ratio < 1:
        thinned_cells = thinning(cells, reduction_ratio=ratio, same_reads=False, copy=True)
        thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=seed, normalization=normalization)
    else:
        thinned_cells = cells.copy()
        thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=seed, normalization=normalization)

    sc.tl.leiden(thinned_cells, resolution=resolution, key_added='leiden_temp', random_state=seed)
    leiden_labels = thinned_cells.obs['leiden_temp'].copy()

    h_runs, c_runs, v_runs, a_runs, ct_runs = {}, {}, {}, {}, {}
    for label in unique_labels:

        if label == "no-exclusion":
            onevsall_labels = labels
        else:
            onevsall_labels = np.where(labels == label, label, "other")

        scores = compute_all_scores(true_labels=onevsall_labels, cluster_labels=leiden_labels)
        h_runs[label] = scores["homogeneity"]
        c_runs[label] = scores["completness"]
        v_runs[label] = scores["v"]
        a_runs[label] = scores["ari"]
        ct_runs[label] = scores["correct"]

    del thinned_cells
    return h_runs, c_runs, v_runs, a_runs, ct_runs

# ==============================================================================
# FONCTIONS PRINCIPALES
# ==============================================================================

def study_sparsity(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", stats="average", show=False, save=True, save_dir=DEFAULT_SAVE_DIR, save_name="study_sparsity.png", ax=None, legend=True):
    """
    Lance une étude de la sparsité sur le anndata 'cells' à partir des 'labels' ground truth.
    On calcule les scores de chaque ratio candidat figurant dans 'ratio_candidates'.
    Retourne les historiques de scores globaux.

    cells: objet anndata/scanpy (dataframe)
    labels: list/array des labels ground truth. Un label est un entier entre 0 et n_class-1.
    ratio_candidates: list/array des ratio de sparsité candidats.
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden).
    n_runs: nombre d'instances à lancer pour trouver le meilleur partitionnement
    normalization: choix de la normalisation des données brutes. Par défaut, normalisation SCT.
    search_resolution_method: choix de la méthode d'optimisation pour trouver la meilleure résolution de leiden
    stats: manière d'obtenir le partitionnement de leiden après 'n_runs'. Choix: 'average' ou 'highest'.
    show: True alors on affiche l'évolution des scores selon le ratio de sparsité.
    ax: objet de dessin matplotlib 
    legend: True alors on affiche la légende sur le graphique.
    save: Sauvegarde ?
    save_dir: Répertoire où seront sauvegardés les résultats.
    save_name: Nom du fichier sauvegardé.
    n_jobs: nombre de CPUs qui seront affectés au calcul.
    """
    
    if ratio_candidates is None:
        ratio_candidates = np.linspace(0.01, 1, 20)

    if len(cells) != len(labels):
        raise ValueError("Number of cells and labels should be the same!")

    homogeneity_history, homogeneity_std = [], []
    completness_history, completness_std = [], []
    ari_history, ari_std = [], []
    v_history, v_std = [], []
    correct_history, correct_std = [], []

    for ratio in ratio_candidates:
        print(f"Ratio={ratio:.3f}")
        if ratio < 1:
            thinned_cells = thinning(cells, reduction_ratio=ratio, same_reads=False, copy=True)
            thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=42, normalization=normalization)
        else:
            thinned_cells = cells.copy()
            thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=42, normalization=normalization)

        results = find_best_resolution(data=thinned_cells, true_labels=labels, n_neighbors=None, n_trials=50, method=search_resolution_method, show=False)
        resolution = results["resolution"]

        if stats == "average":
            output = average_leiden_run(data=thinned_cells, true_labels=labels, n_runs=n_runs, resolution=resolution, show=False)
            scores = output["scores"]
            scores_std = output["scores_std"]
            
            homogeneity_history.append(scores["homogeneity"]); completness_history.append(scores["completness"]); ari_history.append(scores["ari"]); v_history.append(scores["v"]); correct_history.append(scores["correct"])
            homogeneity_std.append(scores_std["homogeneity"]); completness_std.append(scores_std["homogeneity"]); ari_std.append(scores_std["ari"]); v_std.append(scores_std["v"]); correct_std.append(scores_std["correct"])
        
        elif stats == "highest":
            output = best_leiden_run(data=thinned_cells, true_labels=labels, n_runs=n_runs, score_key='ari', resolution=resolution, show=False)
            scores = output["scores"]
            
            homogeneity_history.append(scores["homogeneity"]); completness_history.append(scores["completness"]); ari_history.append(scores["ari"]); v_history.append(scores["v"]); correct_history.append(scores["correct"])
            homogeneity_std.append(0); completness_std.append(0); ari_std.append(0); v_std.append(0); correct_std.append(0)

        del thinned_cells
        gc.collect()

    if show or save:
        is_local_ax = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            is_local_ax = True
        else:
            fig = ax.get_figure()

        r = ratio_candidates

        def plot_with_band(x, mean, std, color, label, linestyle="-"):
            mean, std = np.array(mean), np.array(std)
            ci = 1.96 * std / np.sqrt(n_runs)
            ax.plot(x, mean, color=color, linestyle=linestyle, label=label)
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.2)

        plot_with_band(r, homogeneity_history, homogeneity_std, "blue",   "homogeneity")
        plot_with_band(r, completness_history, completness_std, "red",    "completness")
        plot_with_band(r, correct_history,     correct_std,     "orange", "correctly classified cells")
        plot_with_band(r, ari_history,         ari_std,         "green",  "ari", linestyle="--")
        plot_with_band(r, v_history,           v_std,           "black",  "v",   linestyle="--")

        ax.invert_xaxis()
        ax.set_title(f"k={n_neighbors}")
        if legend:
            ax.legend()
        ax.grid(True, alpha=0.6)
        ax.set_xlabel("r")
        ax.set_ylabel("score")

        if is_local_ax:
            if save:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, save_name), dpi=200, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)

    return {
        "homogeneity": homogeneity_history,
        "completness": completness_history,
        "ari": ari_history,
        "v": v_history,
        "correct": correct_history,
    }


def study_sparsity_stdthinning(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", show=False, save=True, save_dir=DEFAULT_SAVE_DIR, save_name="study_sparsity_stdthinning.png", ax=None, legend=True, n_jobs=N_JOBS):
    """
    Lance une étude de la sparsité sur le anndata 'cells' à partir des 'labels' ground truth.
    On calcule les scores de chaque ratio candidat figurant dans 'ratio_candidates'.
    Retourne les historiques de scores globaux.

    Le thinning est appliqué de nouveau à chaque run, de sorte à prendre en compte la variance due à la méthode de thinning.

    cells: objet anndata/scanpy (dataframe)
    labels: list/array des labels ground truth. Un label est un entier entre 0 et n_class-1.
    ratio_candidates: list/array des ratio de sparsité candidats.
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden).
    n_runs: nombre d'instances à lancer pour trouver le meilleur partitionnement
    normalization: choix de la normalisation des données brutes. Par défaut, normalisation SCT.
    search_resolution_method: choix de la méthode d'optimisation pour trouver la meilleure résolution de leiden
    stats: manière d'obtenir le partitionnement de leiden après 'n_runs'. Choix: 'average' ou 'highest'.
    show: True alors on affiche l'évolution des scores selon le ratio de sparsité.
    ax: objet de dessin matplotlib 
    legend: True alors on affiche la légende sur le graphique.
    save: Sauvegarde ?
    save_dir: Répertoire où seront sauvegardés les résultats.
    save_name: Nom du fichier sauvegardé.
    n_jobs: nombre de CPUs qui seront affectés au calcul.
    """
    if ratio_candidates is None:
        ratio_candidates = np.linspace(0.01, 1, 20)

    if len(cells) != len(labels):
        raise ValueError("Number of cells and labels should be the same!")

    homogeneity_history, homogeneity_std = [], []
    completness_history, completness_std = [], []
    ari_history, ari_std = [], []
    v_history, v_std = [], []
    correct_history, correct_std = [], []

    for ratio in ratio_candidates:
        print(f"Ratio={ratio:.3f}")

        resolution = _get_optimal_resolution(cells, labels, ratio, n_neighbors, normalization, search_resolution_method)

        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_stdthinning)(cells, labels, ratio, n_neighbors, normalization, resolution)
            for _ in range(n_runs)
        )

        h_runs = [r["homogeneity"] for r in results]
        c_runs = [r["completness"] for r in results]
        v_runs = [r["v"] for r in results]
        a_runs = [r["ari"] for r in results]
        ct_runs = [r["correct"] for r in results]

        homogeneity_history.append(np.mean(h_runs));  homogeneity_std.append(np.std(h_runs))
        completness_history.append(np.mean(c_runs));  completness_std.append(np.std(c_runs))
        ari_history.append(np.mean(a_runs));          ari_std.append(np.std(a_runs))
        v_history.append(np.mean(v_runs));            v_std.append(np.std(v_runs))
        correct_history.append(np.mean(ct_runs));     correct_std.append(np.std(ct_runs))
        print("\n")

    if show or save:
        is_local_ax = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            is_local_ax = True
        else:
            fig = ax.get_figure()

        r = ratio_candidates

        def plot_with_band(x, mean, std, color, label, linestyle="-"):
            mean, std = np.array(mean), np.array(std)
            ci = 1.96 * std / np.sqrt(n_runs)
            ax.plot(x, mean, color=color, linestyle=linestyle, label=label)
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.2)

        plot_with_band(r, homogeneity_history, homogeneity_std, "blue",   "homogeneity")
        plot_with_band(r, completness_history, completness_std, "red",    "completness")
        plot_with_band(r, correct_history,     correct_std,     "orange", "correctly classified cells")
        plot_with_band(r, ari_history,         ari_std,         "green",  "ari", linestyle="--")
        plot_with_band(r, v_history,           v_std,           "black",  "v",   linestyle="--")

        ax.invert_xaxis()
        ax.set_title(f"k={n_neighbors}")
        if legend:
            ax.legend()
        ax.grid(True, alpha=0.6)
        ax.set_xlabel("r")
        ax.set_ylabel("score")

        if is_local_ax:
            if save:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, save_name), dpi=200, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)

    return {
        "homogeneity": homogeneity_history,
        "completness": completness_history,
        "ari": ari_history,
        "v": v_history,
        "correct": correct_history,
    }

def study_sparsity_with_trajectories(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", show=False, save=True, save_dir=DEFAULT_SAVE_DIR, save_name="study_sparsity_with_trajectories.png", ax=None, n_jobs=N_JOBS):
    """
    Lance une étude de la sparsité sur le anndata 'cells' à partir des 'labels' ground truth.
    On calcule les scores de chaque ratio candidat figurant dans 'ratio_candidates'.
    Retourne les historiques de scores par partitionnement de leiden (au total 'n_runs')

    cells: objet anndata/scanpy (dataframe)
    labels: list/array des labels ground truth. Un label est un entier entre 0 et n_class-1.
    ratio_candidates: list/array des ratio de sparsité candidats.
    n_runs: nombre d'instances à lancer pour trouver le meilleur partitionnement
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden).
    normalization: choix de la normalisation des données brutes. Par défaut, normalisation SCT.
    search_resolution_method: choix de la méthode d'optimisation pour trouver la meilleure résolution de leiden
    show: True alors on affiche l'évolution des scores selon le ratio de sparsité.
    save: Sauvegarde ?
    save_dir: Répertoire où seront sauvegardés les résultats.
    save_name: Nom du fichier sauvegardé.
    ax: objet de dessin matplotlib 
    n_jobs: nombre de CPUs qui seront affectés au calcul.
    """
    if ratio_candidates is None:
        ratio_candidates = np.linspace(0.01, 1, 20)

    if len(cells) != len(labels):
        raise ValueError("Number of cells and labels should be the same!")

    homogeneity_history = [[] for _ in range(n_runs)]
    completness_history = [[] for _ in range(n_runs)]
    ari_history = [[] for _ in range(n_runs)]
    v_history = [[] for _ in range(n_runs)]
    correct_history = [[] for _ in range(n_runs)]
    correct_detailed_history = [[] for _ in range(n_runs)]

    for ratio in ratio_candidates:
        print(f"Ratio={ratio:.3f}")
        
        if ratio < 1:
            thinned_cells = thinning(cells, reduction_ratio=ratio, same_reads=False, copy=True)
            thinned_cells = update_data(thinned_cells, normalization=normalization)
        else:
            thinned_cells = cells.copy()
            thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=42, normalization=normalization)

        sc.pp.neighbors(thinned_cells, n_neighbors=n_neighbors, n_pcs=100)
        results = find_best_resolution(data=thinned_cells, true_labels=labels, n_neighbors=None, n_trials=50, method=search_resolution_method, show=False)
        resolution = results["resolution"]

        seeds = [np.random.randint(0, MAX_RNG_RANGE) for _ in range(n_runs)]
        run_results = Parallel(n_jobs=n_jobs)(
            delayed(_run_trajectories)(thinned_cells, labels, resolution, seed)
            for seed in seeds
        )

        for i, (h, c, v, ari, correct, correct_detailed) in enumerate(run_results):
            homogeneity_history[i].append(h)
            completness_history[i].append(c)
            v_history[i].append(v)
            ari_history[i].append(ari)
            correct_history[i].append(correct)
            correct_detailed_history[i].append(correct_detailed)
            
        print("\n")
        del thinned_cells
        gc.collect()

    if show or save:
        is_local_ax = False
        if ax is None:
            fig, ax = plt.subplots()
            is_local_ax = True
        else:
            fig = ax.get_figure()

        r = ratio_candidates

        def plot_trajectories(x, trajectories, color, label, linestyle="-"):
            for seed in range(n_runs):
                ax.plot(x, trajectories[seed], color=color, linestyle=linestyle, alpha=0.15)

        plot_trajectories(r, homogeneity_history, "blue", "homogeneity")
        plot_trajectories(r, completness_history, "red", "completness")
        plot_trajectories(r, correct_history, "orange", "correctly classified cells")
        plot_trajectories(r, ari_history, "green",  "ari", linestyle="-")
        plot_trajectories(r, v_history, "black",  "v", linestyle="-")

        ax.invert_xaxis()
        ax.set_title(f"k={n_neighbors}")

        legend_elements = [
            Line2D([0], [0], color="blue",   label="homogeneity"),
            Line2D([0], [0], color="red",    label="completness"),
            Line2D([0], [0], color="orange", label="correctly classified cells"),
            Line2D([0], [0], color="green",  linestyle="-", label="ari"),
            Line2D([0], [0], color="black",  linestyle="-", label="v"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        ax.set_xlabel("r")
        ax.set_ylabel("score")

        if is_local_ax:
            if save:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, save_name), dpi=200, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)

    return {
        "homogeneity": homogeneity_history,
        "completness": completness_history,
        "ari": ari_history,
        "v": v_history,
        "correct": correct_history,
        "correct_detailed": correct_detailed_history
    }

def study_complete_sparsity(cells, labels, ratio_candidates=None, n_runs=50,
                            n_neighbors_candidates=None, search_resolution_method="optuna",
                            stats="average", runs_on_thinning=True, show=False, save=True, save_dir=DEFAULT_SAVE_DIR, save_name="study_complete_sparsity.png",
                            n_jobs=N_JOBS):
    """
    Lance une étude de la sparsité sur le anndata 'cells' à partir des 'labels' ground truth pour chaque n_neighbors de 'n_neighbors_candidates'
    On calcule les scores de chaque ratio candidat figurant dans 'ratio_candidates'.
    Retourne les historiques de scores globaux pour chaque n_neighbors de 'n_neighbors_candidates'.

    cells: objet anndata/scanpy (dataframe)
    labels: list/array des labels ground truth. Un label est un entier entre 0 et n_class-1.
    n_runs: nombre d'instances à lancer pour trouver le meilleur partitionnement
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden).
    search_resolution_method: choix de la méthode d'optimisation pour trouver la meilleure résolution de leiden
    stats: manière d'obtenir le partitionnement de leiden après 'n_runs'. Choix: 'average' ou 'highest'.
    runs_on_thinning: Si True alors le thinning se fera à chaque run de Leiden plutôt qu'en amont des runs.
    show: True alors on affiche l'évolution des scores selon le ratio de sparsité.
    save: Sauvegarde ?
    save_dir: Répertoire où seront sauvegardés les résultats.
    save_name: Nom du fichier sauvegardé.
    n_jobs: nombre de CPUs qui seront affectés au calcul.
    """
    if n_neighbors_candidates is None:
        n_neighbors_candidates = [15, 50, 100, 200]

    if show or save:
        n = len(n_neighbors_candidates)
        n_cols = int(np.ceil(np.sqrt(n)))
        n_rows = int(np.ceil(n / n_cols))
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*3 + 1, n_rows*3))
        axs = axs.flatten()
    else:
        axs = [None] * len(n_neighbors_candidates)

    results = []
    for i, k in enumerate(n_neighbors_candidates):
        print(f"\n--- n_neighbors ={k} ---")
        current_ax = axs[i] if (show or save) else None
        
        if runs_on_thinning:
            res = study_sparsity_stdthinning(
                cells=cells, labels=labels, ratio_candidates=ratio_candidates,
                n_runs=n_runs, n_neighbors=k, search_resolution_method=search_resolution_method,
                show=False, save=True, ax=current_ax, legend=False, n_jobs=n_jobs 
            )
        else:
            res = study_sparsity(
                cells=cells, labels=labels, ratio_candidates=ratio_candidates,
                n_runs=n_runs, n_neighbors=k, search_resolution_method=search_resolution_method,
                stats=stats, show=False, save=True, ax=current_ax, legend=False
            )
        results.append(res)

    scores_history = dict(zip(n_neighbors_candidates, results))

    # Mise en forme et Sauvegarde finale
    if show or save:
        for i, n_neighbors in enumerate(n_neighbors_candidates):
            axs[i].set_title(f"k={n_neighbors}")
        
        for j in range(len(n_neighbors_candidates), len(axs)):
            axs[j].set_visible(False)
        
        plt.tight_layout()
        plt.grid(alpha=0.6)
        
        handles, labels_ = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc='upper right', ncol=5)
        plt.subplots_adjust(top=0.75, wspace=0.5)

        if save:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, save_name), dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    return scores_history


# def study_group_sparsity_nostdthinning(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", stats="average", show=False, save=True, save_dir=DEFAULT_SAVE_DIR, save_name="study_group_sparsity_nostdthinning.png", n_jobs=N_JOBS):
#     if ratio_candidates is None:
#         ratio_candidates = np.linspace(0.01, 1, 20)

#     if len(cells) != len(labels):
#         raise ValueError("Number of cells and labels should be the same!")

#     unique_labels = np.unique(labels)
#     combination_labels = list(combinations(unique_labels, 2))
#     pairs_key = [f'{label1} vs {label2}' for label1, label2 in combination_labels]

#     homogeneity_history, homogeneity_std = {label: [] for label in pairs_key}, {label: [] for label in pairs_key}
#     completness_history, completness_std = {label: [] for label in pairs_key}, {label: [] for label in pairs_key}
#     ari_history, ari_std = {label: [] for label in pairs_key}, {label: [] for label in pairs_key}
#     v_history, v_std = {label: [] for label in pairs_key}, {label: [] for label in pairs_key}
#     correct_history, correct_std = {label: [] for label in pairs_key}, {label: [] for label in pairs_key}

#     for ratio in ratio_candidates:
#         print(f"Ratio={ratio:.3f}")
#         if ratio < 1:
#             thinned_cells = thinning(cells, reduction_ratio=ratio, same_reads=False, copy=True)
#             thinned_cells = update_data(thinned_cells)
#         else:
#             thinned_cells = cells.copy()
#             thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=42, normalization=normalization)

#         sc.pp.neighbors(thinned_cells, n_neighbors=n_neighbors, n_pcs=100)
#         results = find_best_resolution(data=thinned_cells, true_labels=labels, n_neighbors=None, n_trials=50, method=search_resolution_method, show=False)
#         resolution = results["resolution"]

#         seeds = [np.random.randint(0, MAX_RNG_RANGE) for _ in range(n_runs)]
#         run_results = Parallel(n_jobs=n_jobs)(
#             delayed(_run_group_nostdthinning)(thinned_cells, labels, resolution, combination_labels, seed)
#             for seed in seeds
#         )

#         h_runs = {k: [res[0][k] for res in run_results if k in res[0]] for k in pairs_key}
#         c_runs = {k: [res[1][k] for res in run_results if k in res[1]] for k in pairs_key}
#         v_runs = {k: [res[2][k] for res in run_results if k in res[2]] for k in pairs_key}
#         a_runs = {k: [res[3][k] for res in run_results if k in res[3]] for k in pairs_key}
#         ct_runs = {k: [res[4][k] for res in run_results if k in res[4]] for k in pairs_key}

#         for label in pairs_key:
#             if len(h_runs[label]) > 0:
#                 homogeneity_history[label].append(np.mean(h_runs[label]));  homogeneity_std[label].append(np.std(h_runs[label]))
#                 completness_history[label].append(np.mean(c_runs[label]));  completness_std[label].append(np.std(c_runs[label]))
#                 ari_history[label].append(np.mean(a_runs[label]));          ari_std[label].append(np.std(a_runs[label]))
#                 v_history[label].append(np.mean(v_runs[label]));            v_std[label].append(np.std(v_runs[label]))
#                 correct_history[label].append(np.mean(ct_runs[label]));     correct_std[label].append(np.std(ct_runs[label]))
#         print("\n")
#         del thinned_cells
#         gc.collect()

#     if show or save:
#         n_scores = 5
#         n_cols = int(np.ceil(np.sqrt(n_scores)))
#         n_rows = int(np.ceil(n_scores / n_cols))
#         fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*3 + 8, n_rows*3))
#         axs = axs.flatten()

#         r = ratio_candidates

#         def plot_with_band(x, mean, std, label, color=None, linestyle="-", ax=None):
#             if len(mean) == 0: return
#             mean, std = np.array(mean), np.array(std)
#             ci = 1.96 * std / np.sqrt(n_runs)
#             ax.plot(x, mean, color=color, linestyle=linestyle, label=label)
#             ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15)

#         scores_history = [(homogeneity_history, homogeneity_std), (completness_history, completness_std), (correct_history, correct_std), (ari_history, ari_std), (v_history, v_std)]
#         colors = ["blue", "red", "orange", "green", "black"]
#         score_labels = ["homogeneity", "completness", "correctly classified cells", "ari", "v"]
        
#         for i in range(n_scores):
#             score_to_plot, std_to_plot = scores_history[i]
#             for label in pairs_key:
#                 plot_with_band(r, score_to_plot[label], std_to_plot[label], color=None, label=label, ax=axs[i])

#             axs[i].invert_xaxis()
#             axs[i].set_title(score_labels[i])
#             axs[i].grid(True, alpha=0.6)
#             axs[i].set_xlabel("r")
#             axs[i].set_ylabel("score")

#         handles, legend_labels = axs[0].get_legend_handles_labels()
#         fig.legend(handles, legend_labels, loc='upper right')
#         plt.suptitle(f"k={n_neighbors}")

#         plt.subplots_adjust(left=None, right=None, top=0.55, bottom=None, wspace=0.4, hspace=None)

#         for j in range(n_scores, len(axs)):
#             axs[j].set_visible(False)
            
#         if save:
#             os.makedirs(save_dir, exist_ok=True)
#             fig.savefig(os.path.join(save_dir, save_name), dpi=200, bbox_inches='tight')
#         if show:
#             plt.show()
#         else:
#             plt.close(fig)

#     return {
#         "homogeneity": homogeneity_history,
#         "completness": completness_history,
#         "ari": ari_history,
#         "v": v_history,
#         "correct": correct_history,
#     }


def study_group_sparsity(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", show=False, save=True, save_dir=DEFAULT_SAVE_DIR, save_name="study_group_sparsity.png", n_jobs=N_JOBS):
    """
    Lance une étude de la sparsité sur le anndata 'cells' à partir des 'labels' ground truth.
    On calcule les scores de chaque ratio candidat figurant dans 'ratio_candidates'.
    Retourne les historiques de scores pour chaque couple de label ground truth.

    Ces scores sont calculés à partir de liste de labels filtrés obtenues en filtrant les labels obtenus par leiden sur tous les couples possibles.
    Par exemple, si labels=[0,2,1,2,0] et leiden_labels=[0,0,2,2,1], alors on peut calculer les scores du couple (0,1) (ground truth) en prenant
    le sous-tableau sublabels = [0,1,0] et le sous-tableau leiden_sublabels=[0,2,1].

    cells: objet anndata/scanpy (dataframe)
    labels: list/array des labels ground truth. Un label est un entier entre 0 et n_class-1.
    n_runs: nombre d'instances à lancer pour trouver le meilleur partitionnement
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden).
    search_resolution_method: choix de la méthode d'optimisation pour trouver la meilleure résolution de leiden
    stats: manière d'obtenir le partitionnement de leiden après 'n_runs'. Choix: 'average' ou 'highest'.
    runs_on_thinning: Si True alors le thinning se fera à chaque run de Leiden plutôt qu'en amont des runs.
    show: True alors on affiche l'évolution des scores selon le ratio de sparsité.
    save: Sauvegarde ?
    save_dir: Répertoire où seront sauvegardés les résultats.
    save_name: Nom du fichier sauvegardé.
    n_jobs: nombre de CPUs qui seront affectés au calcul.
    """
    if ratio_candidates is None:
        ratio_candidates = np.linspace(0.01, 1, 20)

    if len(cells) != len(labels):
        raise ValueError("Number of cells and labels should be the same!")

    unique_labels = np.unique(labels)
    combination_labels = list(combinations(unique_labels, 2))
    pairs_key = [f'{label1} vs {label2}' for label1, label2 in combination_labels]

    homogeneity_history, homogeneity_std = {label: [] for label in pairs_key}, {label: [] for label in pairs_key}
    completness_history, completness_std = {label: [] for label in pairs_key}, {label: [] for label in pairs_key}
    ari_history, ari_std = {label: [] for label in pairs_key}, {label: [] for label in pairs_key}
    v_history, v_std = {label: [] for label in pairs_key}, {label: [] for label in pairs_key}
    correct_history, correct_std = {label: [] for label in pairs_key}, {label: [] for label in pairs_key}

    for ratio in ratio_candidates:
        print(f"Ratio={ratio:.3f}")

        # 1. OPTIMISATION : Résolution
        resolution = _get_optimal_resolution(cells, labels, ratio, n_neighbors, normalization, search_resolution_method)

        # 2. PARALLÉLISATION
        run_results = Parallel(n_jobs=n_jobs)(
            delayed(_run_group)(cells, labels, ratio, n_neighbors, normalization, resolution, combination_labels)
            for _ in range(n_runs)
        )

        h_runs = {k: [res[0][k] for res in run_results if k in res[0]] for k in pairs_key}
        c_runs = {k: [res[1][k] for res in run_results if k in res[1]] for k in pairs_key}
        v_runs = {k: [res[2][k] for res in run_results if k in res[2]] for k in pairs_key}
        a_runs = {k: [res[3][k] for res in run_results if k in res[3]] for k in pairs_key}
        ct_runs = {k: [res[4][k] for res in run_results if k in res[4]] for k in pairs_key}

        for label in pairs_key:
            if len(h_runs[label]) > 0:
                homogeneity_history[label].append(np.mean(h_runs[label]));  homogeneity_std[label].append(np.std(h_runs[label]))
                completness_history[label].append(np.mean(c_runs[label]));  completness_std[label].append(np.std(c_runs[label]))
                ari_history[label].append(np.mean(a_runs[label]));          ari_std[label].append(np.std(a_runs[label]))
                v_history[label].append(np.mean(v_runs[label]));            v_std[label].append(np.std(v_runs[label]))
                correct_history[label].append(np.mean(ct_runs[label]));     correct_std[label].append(np.std(ct_runs[label]))
        print("\n")

    if show or save:
        n_scores = 5
        n_cols = int(np.ceil(np.sqrt(n_scores)))
        n_rows = int(np.ceil(n_scores / n_cols))
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*3 + 8, n_rows*3))
        axs = axs.flatten()

        r = ratio_candidates

        def plot_with_band(x, mean, std, label, color=None, linestyle="-", ax=None):
            if len(mean) == 0: return
            mean, std = np.array(mean), np.array(std)
            ci = 1.96 * std / np.sqrt(n_runs)
            ax.plot(x, mean, color=color, linestyle=linestyle, label=label)
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15)

        scores_history = [(homogeneity_history, homogeneity_std), (completness_history, completness_std), (correct_history, correct_std), (ari_history, ari_std), (v_history, v_std)]
        score_labels = ["homogeneity", "completness", "correctly classified cells", "ari", "v"]

        for i in range(n_scores):
            score_to_plot, std_to_plot = scores_history[i]
            for label in pairs_key:
                plot_with_band(r, score_to_plot[label], std_to_plot[label], color=None, label=label, ax=axs[i])

            axs[i].invert_xaxis()
            axs[i].set_title(score_labels[i])
            axs[i].grid(True, alpha=0.6)
            axs[i].set_xlabel("r")
            axs[i].set_ylabel("score")

        handles, legend_labels = axs[0].get_legend_handles_labels()
        nrow = 3
        fig.legend(handles, legend_labels, loc='upper right', ncol=int(np.ceil(len(pairs_key) / nrow)))

        plt.subplots_adjust(left=None, right=None, top=0.8, bottom=None, wspace=0.4, hspace=None)

        for j in range(n_scores, len(axs)):
            axs[j].set_visible(False)
            
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, save_name), dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    return {
        "homogeneity": homogeneity_history,
        "completness": completness_history,
        "ari": ari_history,
        "v": v_history,
        "correct": correct_history,
    }


# def study_group_sparsity_exclude_nostdthinning(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", stats="average", show=False, save=True, save_dir=DEFAULT_SAVE_DIR, save_name="study_group_sparsity_exclude_nostdthinning.png", n_jobs=N_JOBS):
#     if ratio_candidates is None:
#         ratio_candidates = np.linspace(0.01, 1, 20)

#     if len(cells) != len(labels):
#         raise ValueError("Number of cells and labels should be the same!")

#     unique_labels = np.append(np.unique(labels), 'no-exclusion')
#     homogeneity_history, homogeneity_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}
#     completness_history, completness_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}
#     ari_history, ari_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}
#     v_history, v_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}
#     correct_history, correct_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}

#     for ratio in ratio_candidates:
#         print(f"Ratio={ratio:.3f}")
#         if ratio < 1:
#             thinned_cells = thinning(cells, reduction_ratio=ratio, same_reads=False, copy=True)
#             thinned_cells = update_data(thinned_cells)
#         else:
#             thinned_cells = cells.copy()
#             thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=42, normalization=normalization)

#         sc.pp.neighbors(thinned_cells, n_neighbors=n_neighbors, n_pcs=100)
#         results = find_best_resolution(data=thinned_cells, true_labels=labels, n_neighbors=None, n_trials=50, method=search_resolution_method, show=False)
#         resolution = results["resolution"]

#         seeds = [np.random.randint(0, MAX_RNG_RANGE) for _ in range(n_runs)]
#         run_results = Parallel(n_jobs=n_jobs)(
#             delayed(_run_group_exclude_nostdthinning)(thinned_cells, labels, resolution, unique_labels, seed)
#             for seed in seeds
#         )

#         h_runs = {k: [res[0][k] for res in run_results if k in res[0]] for k in unique_labels}
#         c_runs = {k: [res[1][k] for res in run_results if k in res[1]] for k in unique_labels}
#         v_runs = {k: [res[2][k] for res in run_results if k in res[2]] for k in unique_labels}
#         a_runs = {k: [res[3][k] for res in run_results if k in res[3]] for k in unique_labels}
#         ct_runs = {k: [res[4][k] for res in run_results if k in res[4]] for k in unique_labels}

#         for label in unique_labels:
#             if len(h_runs[label]) > 0:
#                 homogeneity_history[label].append(np.mean(h_runs[label]));  homogeneity_std[label].append(np.std(h_runs[label]))
#                 completness_history[label].append(np.mean(c_runs[label]));  completness_std[label].append(np.std(c_runs[label]))
#                 ari_history[label].append(np.mean(a_runs[label]));          ari_std[label].append(np.std(a_runs[label]))
#                 v_history[label].append(np.mean(v_runs[label]));            v_std[label].append(np.std(v_runs[label]))
#                 correct_history[label].append(np.mean(ct_runs[label]));     correct_std[label].append(np.std(ct_runs[label]))
#         print("\n")
#         del thinned_cells
#         gc.collect()

#     if show or save:
#         n_scores = 5
#         n_cols = int(np.ceil(np.sqrt(n_scores)))
#         n_rows = int(np.ceil(n_scores / n_cols))
#         fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*3 + 1, n_rows*3))
#         axs = axs.flatten()

#         r = ratio_candidates

#         def plot_with_band(x, mean, std, label, color=None, linestyle="-", ax=None):
#             if len(mean) == 0: return
#             mean, std = np.array(mean), np.array(std)
#             ci = 1.96 * std / np.sqrt(n_runs)
#             ax.plot(x, mean, color=color, linestyle=linestyle, label=label)
#             ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15)

#         scores_history = [(homogeneity_history, homogeneity_std), (completness_history, completness_std), (correct_history, correct_std), (ari_history, ari_std), (v_history, v_std)]
#         score_labels = ["homogeneity", "completness", "correctly classified cells", "ari", "v"]
        
#         for i in range(n_scores):
#             score_to_plot, std_to_plot = scores_history[i]
#             for label in unique_labels:
#                 linestyle = "--" if label == "no-exclusion" else "-"
#                 plot_with_band(r, score_to_plot[label], std_to_plot[label], color=None, label=label, linestyle=linestyle, ax=axs[i])

#             axs[i].invert_xaxis()
#             axs[i].set_title(score_labels[i])
#             axs[i].grid(True, alpha=0.6)
#             axs[i].set_xlabel("r")
#             axs[i].set_ylabel("score")

#         handles, legend_labels = axs[0].get_legend_handles_labels()
#         fig.legend(handles, legend_labels, loc='upper right', ncol=len(unique_labels))

#         plt.subplots_adjust(left=None, right=None, top=0.9, bottom=None, wspace=0.4, hspace=0.25)

#         for j in range(n_scores, len(axs)):
#             axs[j].set_visible(False)
            
#         if save:
#             os.makedirs(save_dir, exist_ok=True)
#             fig.savefig(os.path.join(save_dir, save_name), dpi=200, bbox_inches='tight')
#         if show:
#             plt.show()
#         else:
#             plt.close(fig)

#     return {
#         "homogeneity": homogeneity_history,
#         "completness": completness_history,
#         "ari": ari_history,
#         "v": v_history,
#         "correct": correct_history,
#     }


def study_group_sparsity_exclude(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", show=False, save=True, save_dir=DEFAULT_SAVE_DIR, save_name="study_group_sparsity_exclude.png", n_jobs=N_JOBS):
    """
    Lance une étude de la sparsité sur le anndata 'cells' à partir des 'labels' ground truth.
    On calcule les scores de chaque ratio candidat figurant dans 'ratio_candidates'.
    Retourne les historiques de scores d'exclusion. Les scores d'exclusion sont calculés à partir de liste de labels filtrés
    obtenus en filtrant les cellules d'un label ground truth spécifique (on exclut un type cellulaire et on observe les scores du partitionnement sur les cellules restantes),
    et ceci pour chaque label ground truth.
    
    cells: objet anndata/scanpy (dataframe)
    labels: list/array des labels ground truth. Un label est un entier entre 0 et n_class-1.
    n_runs: nombre d'instances à lancer pour trouver le meilleur partitionnement
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden).
    search_resolution_method: choix de la méthode d'optimisation pour trouver la meilleure résolution de leiden
    stats: manière d'obtenir le partitionnement de leiden après 'n_runs'. Choix: 'average' ou 'highest'.
    runs_on_thinning: Si True alors le thinning se fera à chaque run de Leiden plutôt qu'en amont des runs.
    show: True alors on affiche l'évolution des scores selon le ratio de sparsité.
    save: Sauvegarde ?
    save_dir: Répertoire où seront sauvegardés les résultats.
    save_name: Nom du fichier sauvegardé.
    n_jobs: nombre de CPUs qui seront affectés au calcul.
    """
    if ratio_candidates is None:
        ratio_candidates = np.linspace(0.01, 1, 20)

    if len(cells) != len(labels):
        raise ValueError("Number of cells and labels should be the same!")

    unique_labels = np.append(np.unique(labels), 'no-exclusion')
    homogeneity_history, homogeneity_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}
    completness_history, completness_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}
    ari_history, ari_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}
    v_history, v_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}
    correct_history, correct_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}

    for ratio in ratio_candidates:
        print(f"Ratio={ratio:.3f}")

        # OPTIMISATION : Résolution
        resolution = _get_optimal_resolution(cells, labels, ratio, n_neighbors, normalization, search_resolution_method)

        # PARALLÉLISATION
        run_results = Parallel(n_jobs=n_jobs)(
            delayed(_run_group_exclude)(cells, labels, ratio, n_neighbors, normalization, resolution, unique_labels)
            for _ in range(n_runs)
        )

        h_runs = {k: [res[0][k] for res in run_results if k in res[0]] for k in unique_labels}
        c_runs = {k: [res[1][k] for res in run_results if k in res[1]] for k in unique_labels}
        v_runs = {k: [res[2][k] for res in run_results if k in res[2]] for k in unique_labels}
        a_runs = {k: [res[3][k] for res in run_results if k in res[3]] for k in unique_labels}
        ct_runs = {k: [res[4][k] for res in run_results if k in res[4]] for k in unique_labels}

        for label in unique_labels:
            if len(h_runs[label]) > 0:
                homogeneity_history[label].append(np.mean(h_runs[label]));  homogeneity_std[label].append(np.std(h_runs[label]))
                completness_history[label].append(np.mean(c_runs[label]));  completness_std[label].append(np.std(c_runs[label]))
                ari_history[label].append(np.mean(a_runs[label]));          ari_std[label].append(np.std(a_runs[label]))
                v_history[label].append(np.mean(v_runs[label]));            v_std[label].append(np.std(v_runs[label]))
                correct_history[label].append(np.mean(ct_runs[label]));     correct_std[label].append(np.std(ct_runs[label]))
        print("\n")

    if show or save:
        n_scores = 5
        n_cols = int(np.ceil(np.sqrt(n_scores)))
        n_rows = int(np.ceil(n_scores / n_cols))
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*3 + 1, n_rows*3))
        axs = axs.flatten()

        r = ratio_candidates

        def plot_with_band(x, mean, std, label, color=None, linestyle="-", ax=None):
            if len(mean) == 0: return
            mean, std = np.array(mean), np.array(std)
            ci = 1.96 * std / np.sqrt(n_runs)
            ax.plot(x, mean, color=color, linestyle=linestyle, label=label)
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15)

        scores_history = [(homogeneity_history, homogeneity_std), (completness_history, completness_std), (correct_history, correct_std), (ari_history, ari_std), (v_history, v_std)]
        score_labels = ["homogeneity", "completness", "correctly classified cells", "ari", "v"]

        for i in range(n_scores):
            score_to_plot, std_to_plot = scores_history[i]
            for label in unique_labels:
                linestyle = "--" if label == "no-exclusion" else "-"
                plot_with_band(r, score_to_plot[label], std_to_plot[label], color=None, label=label, linestyle=linestyle, ax=axs[i])

            axs[i].invert_xaxis()
            axs[i].set_title(score_labels[i])
            axs[i].grid(True, alpha=0.6)
            axs[i].set_xlabel("r")
            axs[i].set_ylabel("score")

        handles, legend_labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper right', ncol=len(unique_labels))

        plt.subplots_adjust(left=None, right=None, top=0.9, bottom=None, wspace=0.4, hspace=0.25)

        for j in range(n_scores, len(axs)):
            axs[j].set_visible(False)
            
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, save_name), dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    return {
        "homogeneity": homogeneity_history,
        "completness": completness_history,
        "ari": ari_history,
        "v": v_history,
        "correct": correct_history,
    }

def study_group_sparsity_onevsall(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", show=False, save=True, save_dir=DEFAULT_SAVE_DIR, save_name="study_group_sparsity_exclude.png", n_jobs=N_JOBS):
    """
    Lance une étude de la sparsité sur le anndata 'cells' à partir des 'labels' ground truth.
    On calcule les scores de chaque ratio candidat figurant dans 'ratio_candidates'.
    Retourne les historiques de scores 'one vs all'. Les scores 'one vs all' sont calculés à partir de liste de labels filtrés, on vient filtrer
    les cellules autres qu'un certain label et on les remplace par un label commun, de sorte à ce que le score évalue la qualité de partitionnement
    entre le label étudié et toutes les autres cellules, et ceci pour chaque label ground truth.
    
    cells: objet anndata/scanpy (dataframe)
    labels: list/array des labels ground truth. Un label est un entier entre 0 et n_class-1.
    n_runs: nombre d'instances à lancer pour trouver le meilleur partitionnement
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden).
    search_resolution_method: choix de la méthode d'optimisation pour trouver la meilleure résolution de leiden
    stats: manière d'obtenir le partitionnement de leiden après 'n_runs'. Choix: 'average' ou 'highest'.
    runs_on_thinning: Si True alors le thinning se fera à chaque run de Leiden plutôt qu'en amont des runs.
    show: True alors on affiche l'évolution des scores selon le ratio de sparsité.
    save: Sauvegarde ?
    save_dir: Répertoire où seront sauvegardés les résultats.
    save_name: Nom du fichier sauvegardé.
    n_jobs: nombre de CPUs qui seront affectés au calcul.
    """  
    if ratio_candidates is None:
        ratio_candidates = np.linspace(0.01, 1, 20)

    if len(cells) != len(labels):
        raise ValueError("Number of cells and labels should be the same!")

    unique_labels = np.append(np.unique(labels), 'no-exclusion')
    homogeneity_history, homogeneity_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}
    completness_history, completness_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}
    ari_history, ari_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}
    v_history, v_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}
    correct_history, correct_std = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}

    for ratio in ratio_candidates:
        print(f"Ratio={ratio:.3f}")

        # OPTIMISATION : Résolution
        resolution = _get_optimal_resolution(cells, labels, ratio, n_neighbors, normalization, search_resolution_method)

        # PARALLÉLISATION
        run_results = Parallel(n_jobs=n_jobs)(
            delayed(_run_group_onevsall)(cells, labels, ratio, n_neighbors, normalization, resolution, unique_labels)
            for _ in range(n_runs)
        )

        h_runs = {k: [res[0][k] for res in run_results if k in res[0]] for k in unique_labels}
        c_runs = {k: [res[1][k] for res in run_results if k in res[1]] for k in unique_labels}
        v_runs = {k: [res[2][k] for res in run_results if k in res[2]] for k in unique_labels}
        a_runs = {k: [res[3][k] for res in run_results if k in res[3]] for k in unique_labels}
        ct_runs = {k: [res[4][k] for res in run_results if k in res[4]] for k in unique_labels}

        for label in unique_labels:
            if len(h_runs[label]) > 0:
                homogeneity_history[label].append(np.mean(h_runs[label]));  homogeneity_std[label].append(np.std(h_runs[label]))
                completness_history[label].append(np.mean(c_runs[label]));  completness_std[label].append(np.std(c_runs[label]))
                ari_history[label].append(np.mean(a_runs[label]));          ari_std[label].append(np.std(a_runs[label]))
                v_history[label].append(np.mean(v_runs[label]));            v_std[label].append(np.std(v_runs[label]))
                correct_history[label].append(np.mean(ct_runs[label]));     correct_std[label].append(np.std(ct_runs[label]))
        print("\n")

    # Affichage ?
    if show or save:
        n_scores = 5
        n_cols = int(np.ceil(np.sqrt(n_scores)))
        n_rows = int(np.ceil(n_scores / n_cols))
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*3 + 1, n_rows*3))
        axs = axs.flatten()

        r = ratio_candidates

        def plot_with_band(x, mean, std, label, color=None, linestyle="-", ax=None):
            if len(mean) == 0: return
            mean, std = np.array(mean), np.array(std)
            ci = 1.96 * std / np.sqrt(n_runs)
            ax.plot(x, mean, color=color, linestyle=linestyle, label=label)
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15)

        scores_history = [(homogeneity_history, homogeneity_std), (completness_history, completness_std), (correct_history, correct_std), (ari_history, ari_std), (v_history, v_std)]
        score_labels = ["homogeneity", "completness", "correctly classified cells", "ari", "v"]

        for i in range(n_scores):
            score_to_plot, std_to_plot = scores_history[i]
            for label in unique_labels:
                linestyle = "--" if label == "no-exclusion" else "-"
                plot_with_band(r, score_to_plot[label], std_to_plot[label], color=None, label=label, linestyle=linestyle, ax=axs[i])

            axs[i].invert_xaxis()
            axs[i].set_title(score_labels[i])
            axs[i].grid(True, alpha=0.6)
            axs[i].set_xlabel("r")
            axs[i].set_ylabel("score")

        handles, legend_labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper right', ncol=len(unique_labels))

        plt.subplots_adjust(left=None, right=None, top=0.9, bottom=None, wspace=0.4, hspace=0.25)

        for j in range(n_scores, len(axs)):
            axs[j].set_visible(False)
            
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, save_name), dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    return {
        "homogeneity": homogeneity_history,
        "completness": completness_history,
        "ari": ari_history,
        "v": v_history,
        "correct": correct_history,
    }


import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


DEFAULT_SAVE_DIR = "results_sparsity_degs"


def study_sparsity_degs(
    cells,
    labels,
    ratio_candidates=None,
    n_neighbors_candidates=None,
    normalization="sct",
    search_resolution_method="optuna",
    n_top_genes=100,
    deg_method="wilcoxon",
    log2fc_min=0.25,
    pval_cutoff=0.05,
    show=False,
    save=True,
    save_dir=DEFAULT_SAVE_DIR,
):
    """
    Variante single-run de study_complete_sparsity :
    - Un seul thinning + un seul clustering Leiden par (ratio, n_neighbors)
    - Pas d'intervalles de confiance
    - Calcule les gènes différenciants (DEGs) pour chaque combinaison
    - Exporte :
        * Top {n_top_genes} DEGs par cluster vers CSV (un fichier par combinaison)
        * Labels Leiden trouvés pour chaque n_neighbors vers CSV (un fichier par k)
        * Heatmap des top DEGs vers PNG (si show ou save)

    cells : AnnData
        Objet AnnData contenant la matrice d'expression.
    labels : array-like
        Labels de référence (ground truth), même longueur que cells.
    ratio_candidates : array-like, optional
        Ratios de thinning à tester. Défaut : np.linspace(0.01, 1, 20).
    n_neighbors_candidates : list of int, optional
        Valeurs de n_neighbors à tester. Défaut : [15, 50, 100, 200].
    normalization : str
        Méthode de normalisation passée à update_data (ex : "sct", "log1p").
    search_resolution_method : str
        Méthode de recherche de résolution Leiden (ex : "optuna", "grid").
    n_top_genes : int
        Nombre de top gènes discriminants à exporter par cluster.
    deg_method : str
        Méthode pour rank_genes_groups : "wilcoxon" | "t-test" | "logreg".
    log2fc_min : float
        Seuil minimum de log2 fold-change pour filtrer les DEGs.
    pval_cutoff : float
        Seuil de p-valeur ajustée (FDR) pour filtrer les DEGs.
    save:
        Sauvegarde ?
    save_dir: 
        Répertoire où seront sauvegardés les résultats.
    save_name:
        Nom du fichier sauvegardé.
    """
    if ratio_candidates is None:
        ratio_candidates = np.linspace(0.01, 1, 20)
    if n_neighbors_candidates is None:
        n_neighbors_candidates = [15, 50, 100, 200]

    if len(cells) != len(labels):
        raise ValueError("cells et labels doivent avoir la même longueur.")

    if save:
        os.makedirs(save_dir, exist_ok=True)

    all_results = {}

    for k in n_neighbors_candidates:
        print(f"\n{'='*50}")
        print(f"  n_neighbors = {k}")
        print(f"{'='*50}")

        k_results = {
            "ratio": [],
            "n_clusters": [],
            "degs": [],          # liste de DataFrames (un par ratio)
            "leiden_labels": {}  # {ratio_str: Series de labels}
        }

        for ratio in ratio_candidates:
            ratio_str = f"r{ratio:.3f}"
            print(f"\n  Ratio = {ratio:.3f}")

            # Thinning
            if ratio < 1.0:
                thinned = thinning(cells, reduction_ratio=ratio, same_reads=False, copy=True)
            else:
                thinned = cells.copy()

            # Preprocessing / graph
            thinned = update_data(
                thinned,
                n_neighbors=k,
                n_comps=100,
                random_state=42,
                normalization=normalization,
            )

            # Sauvegarder les counts normalisés avant scaling pour les DEGs
            if thinned.raw is None:
                thinned.raw = thinned

            # Résolution optimale + Leiden (single run)
            res_results = find_best_resolution(
                data=thinned,
                true_labels=labels,
                n_neighbors=None,
                n_trials=50,
                method=search_resolution_method,
                show=False,
            )
            resolution = res_results["resolution"]

            sc.tl.leiden(thinned, resolution=resolution, random_state=42)
            leiden_col = thinned.obs["leiden"].copy()
            leiden_col.name = ratio_str

            n_clusters = leiden_col.nunique()
            print(f"    → {n_clusters} clusters (résolution={resolution:.4f})")

            k_results["ratio"].append(ratio)
            k_results["n_clusters"].append(n_clusters)
            k_results["leiden_labels"][ratio_str] = leiden_col

            # Calcul des DEGs
            sc.tl.rank_genes_groups(
                thinned,
                groupby="leiden",
                method=deg_method,
                use_raw=True,
                pts=True,
                tie_correct=True,
                key_added="rank_genes_groups",
            )

            # Extraction sous forme de DataFrame
            deg_df = sc.get.rank_genes_groups_df(
                thinned,
                group=None,           # tous les clusters
                key="rank_genes_groups",
                pval_cutoff=pval_cutoff,
                log2fc_min=log2fc_min,
            )

            # Top n_top_genes par cluster (trié par score décroissant)
            top_df = (
                deg_df
                .sort_values("scores", ascending=False)
                .groupby("group", sort=False)
                .head(n_top_genes)
                .reset_index(drop=True)
            )
            # Ajout de métadonnées contextuelles
            top_df.insert(0, "n_neighbors", k)
            top_df.insert(1, "ratio", round(ratio, 4))
            top_df.insert(2, "resolution", round(resolution, 4))

            # Renommage pour clarté
            top_df = top_df.rename(columns={
                "group":          "leiden_cluster",
                "names":          "gene",
                "scores":         "score",
                "logfoldchanges": "log2fc",
                "pvals":          "pval",
                "pvals_adj":      "pval_adj",
                "pts":            "pct_in",      # % cellules du cluster
                "pts_rest":       "pct_out",     # % cellules hors cluster
            })

            k_results["degs"].append(top_df)

            # Sauvegarde par combinaison (CSV)
            if save:
                csv_name = f"DEGs_k{k}_{ratio_str}.csv"
                top_df.to_csv(os.path.join(save_dir, csv_name), index=False)
                print(f"    → DEGs sauvegardés : {csv_name}")

            # Figure heatmap
            if show or save:
                try:
                    sc.tl.dendrogram(thinned, groupby="leiden")
                    sc.pl.rank_genes_groups_heatmap(
                        thinned,
                        n_genes=min(10, n_top_genes),
                        groupby="leiden",
                        use_raw=True,
                        show_gene_labels=True,
                        show=False,
                    )
                    fig_h = plt.gcf()
                    fig_h.suptitle(f"Top DEGs — k={k}, ratio={ratio:.3f}, res={resolution:.3f}", y=1.01, fontsize=10)
                    if save:
                        fig_h.savefig(
                            os.path.join(save_dir, f"heatmap_k{k}_{ratio_str}.png"),
                            dpi=150, bbox_inches="tight",
                        )
                    if show:
                        plt.show()
                    plt.close(fig_h)
                except Exception as e:
                    print(f"    ⚠ Heatmap ignorée : {e}")

            del thinned
            gc.collect()

        # Consolidation des labels Leiden pour ce k
        leiden_labels_df = pd.DataFrame(k_results["leiden_labels"])
        leiden_labels_df.index = cells.obs_names
        leiden_labels_df.index.name = "cell_barcode"
        k_results["leiden_labels"] = leiden_labels_df

        # Export des labels Leiden (Parquet = format optimal)
        if save:
            parquet_name = f"leiden_labels_k{k}.parquet"
            leiden_labels_df.to_parquet(os.path.join(save_dir, parquet_name))
            print(f"\n  Labels Leiden sauvegardés : {parquet_name}")

            # CSV de secours (plus universel)
            csv_labels_name = f"leiden_labels_k{k}.csv"
            leiden_labels_df.to_csv(os.path.join(save_dir, csv_labels_name))

        # Consolidation de tous les DEGs pour ce k → un seul CSV
        if save and k_results["degs"]:
            all_degs_k = pd.concat(k_results["degs"], ignore_index=True)
            master_csv = f"DEGs_ALL_k{k}.csv"
            all_degs_k.to_csv(os.path.join(save_dir, master_csv), index=False)
            print(f"  DEGs consolidés : {master_csv}")

        all_results[k] = k_results

    # Vue d'ensemble : courbe n_clusters vs ratio pour chaque k
    if show or save:
        fig_ov, ax_ov = plt.subplots(figsize=(8, 5))
        for k in n_neighbors_candidates:
            ax_ov.plot(
                all_results[k]["ratio"],
                all_results[k]["n_clusters"],
                marker="o", markersize=4, label=f"k={k}",
            )
        ax_ov.invert_xaxis()
        ax_ov.set_xlabel("Ratio de thinning (r)")
        ax_ov.set_ylabel("Nombre de clusters Leiden")
        ax_ov.set_title("Nombre de clusters en fonction du thinning et de k")
        ax_ov.legend()
        ax_ov.grid(True, alpha=0.5)
        plt.tight_layout()
        if save:
            fig_ov.savefig(
                os.path.join(save_dir, "overview_n_clusters.png"),
                dpi=200, bbox_inches="tight",
            )
        if show:
            plt.show()
        plt.close(fig_ov)

    print(f"\n✓ Terminé. Résultats dans : {os.path.abspath(save_dir)}")
    return all_results