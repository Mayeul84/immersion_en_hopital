from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import adjusted_rand_score
import numpy as np
import scanpy as sc

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .gene_subsampling import thinning

from typing import List, Optional

from .scoring import balanced_correctcells_score, compute_all_scores
from .utils import sctransform_manual, update_data
from .utils import MAX_RNG_RANGE

from .clustering import find_best_resolution, cluster_data, average_leiden_run, best_leiden_run

def study_sparsity(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna",stats="average",show=False, ax=None, legend=True):

    if ratio_candidates is None:
        ratio_candidates = np.linspace(0.01,1,20)

    if len(cells) != len(labels):
        raise ValueError("Number of cells and labels should be the same!")

    homogeneity_history, homogeneity_std = [], []
    completness_history, completness_std = [], []
    ari_history, ari_std = [], []
    v_history, v_std = [], []
    correct_history, correct_std = [], []

    for ratio in ratio_candidates:
        print(f"Ratio={ratio:.3f}")
        if(ratio < 1):
            thinned_cells = thinning(cells,reduction_ratio=ratio,same_reads=False,copy=True)
            thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors,n_comps=100,random_state=42,normalization=normalization)
        else:
            thinned_cells = cells
            thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors,n_comps=100,random_state=42,normalization=normalization)

        results = find_best_resolution(data=thinned_cells,true_labels=labels,n_neighbors=None,n_trials=50,method=search_resolution_method,show=False)
        resolution = results["resolution"]

        if stats=="average":
            output = average_leiden_run(data=thinned_cells,true_labels=labels,n_runs=n_runs,resolution=resolution,show=False)
            scores = output["scores"]
            scores_std = output["scores_std"]
            
            homogeneity_history.append(scores["homogeneity"]),completness_history.append(scores["completness"]),ari_history.append(scores["ari"]),v_history.append(scores["v"]),correct_history.append(scores["correct"])
            homogeneity_std.append(scores_std["homogeneity"]),completness_std.append(scores_std["homogeneity"]),ari_std.append(scores_std["ari"]),v_std.append(scores_std["v"]),correct_std.append(scores_std["correct"])
        
        elif stats=="highest":
            output = best_leiden_run(data=thinned_cells,true_labels=labels,n_runs=n_runs,score_key='ari',resolution=resolution,show=False)
            scores = output["scores"]
            
            homogeneity_history.append(scores["homogeneity"]),completness_history.append(scores["completness"]),ari_history.append(scores["ari"]),v_history.append(scores["v"]),correct_history.append(scores["correct"])
            homogeneity_std.append(0),completness_std.append(0),ari_std.append(0),v_std.append(0),correct_std.append(0)

    if show:
        if ax is None:
            _, ax = plt.subplots(figsize=(6,5))

        r = ratio_candidates

        def plot_with_band(x, mean, std, color, label, linestyle="-"):
            mean, std = np.array(mean), np.array(std)
            ci = 1.96 * std / np.sqrt(n_runs)  # intervalle de confiance à 95%
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
        ax.grid(True,alpha=0.6)
        ax.set_xlabel("r")
        ax.set_ylabel("score")

    return {
        "homogeneity": homogeneity_history,
        "completness": completness_history,
        "ari": ari_history,
        "v": v_history,
        "correct": correct_history,
    }

def study_sparsity_stdthinning(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", show=False, ax=None, legend=True):

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

        h_runs, c_runs, v_runs, a_runs, ct_runs = [], [], [], [], []

        for seed in range(n_runs):
            if ratio < 1:
                thinned_cells = thinning(cells, reduction_ratio=ratio, same_reads=False, copy=True)
                thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=np.random.randint(0, MAX_RNG_RANGE), normalization=normalization)
            else:
                thinned_cells = cells
                thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=np.random.randint(0, MAX_RNG_RANGE), normalization=normalization)

            results = find_best_resolution(data=thinned_cells, true_labels=labels, n_neighbors=None, n_trials=50, method=search_resolution_method, show=False)
            resolution = results["resolution"]

            sc.tl.leiden(thinned_cells, resolution=resolution, key_added='leiden_temp', random_state=np.random.randint(0, MAX_RNG_RANGE))
            leiden_labels = thinned_cells.obs['leiden_temp']

            scores = compute_all_scores(true_labels=labels, cluster_labels=leiden_labels)
            h_runs.append(scores["homogeneity"])
            c_runs.append(scores["completness"])
            v_runs.append(scores["v"])
            a_runs.append(scores["ari"])
            ct_runs.append(scores["correct"])

        homogeneity_history.append(np.mean(h_runs));  homogeneity_std.append(np.std(h_runs))
        completness_history.append(np.mean(c_runs));  completness_std.append(np.std(c_runs))
        ari_history.append(np.mean(a_runs));          ari_std.append(np.std(a_runs))
        v_history.append(np.mean(v_runs));            v_std.append(np.std(v_runs))
        correct_history.append(np.mean(ct_runs));     correct_std.append(np.std(ct_runs))
        print("\n")

    if show:
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

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

    return {
        "homogeneity": homogeneity_history,
        "completness": completness_history,
        "ari": ari_history,
        "v": v_history,
        "correct": correct_history,
    }

def study_sparsity_with_trajectories(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", show=False, ax=None):

    if ratio_candidates is None:
        ratio_candidates = np.linspace(0.01,1,20)

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
        if(ratio < 1):
            thinned_cells = thinning(cells,reduction_ratio=ratio,same_reads=False,copy=True)
            thinned_cells = update_data(thinned_cells, normalization=normalization)
        else:
            thinned_cells = cells
            thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors,n_comps=100,random_state=42,normalization=normalization)


        sc.pp.neighbors(thinned_cells,n_neighbors=n_neighbors,n_pcs=100)
        results = find_best_resolution(data=thinned_cells,true_labels=labels,n_neighbors=None,n_trials=50,method=search_resolution_method,show=False)
        resolution = results["resolution"]

        for seed in range(n_runs):
            sc.tl.leiden(thinned_cells,resolution=resolution,key_added='leiden_temp', random_state=np.random.randint(0, MAX_RNG_RANGE))
            leiden_labels = thinned_cells.obs['leiden_temp']

            h, c, v = homogeneity_completeness_v_measure(labels,leiden_labels)
            ari = adjusted_rand_score(labels, leiden_labels)
            correct,correct_detailed = balanced_correctcells_score(labels,leiden_labels)

            homogeneity_history[seed].append(h)
            completness_history[seed].append(c)
            v_history[seed].append(v)
            ari_history[seed].append(ari)
            correct_history[seed].append(correct)
            correct_detailed_history[seed].append(correct_detailed)
        print("\n")

    if show:
        if ax is None:
            _, ax = plt.subplots()

        r = ratio_candidates

        def plot_trajectories(x, trajectories, color, label, linestyle="-"):
            mean, std = np.mean(trajectories), np.std(trajectories)
            ci = 1.96 * std / np.sqrt(n_runs)  # intervalle de confiance à 95%

            for seed in range(n_runs):
                ax.plot(x, trajectories[seed], color=color, linestyle=linestyle,alpha=0.15)
                #ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=x0.15)

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

    return {
        "homogeneity": homogeneity_history,
        "completness": completness_history,
        "ari": ari_history,
        "v": v_history,
        "correct": correct_history,
        "correct_detailed": correct_detailed_history
    }

def study_complete_sparsity(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors_candidates=None, search_resolution_method="optuna",stats="average", runs_on_thinning=True, show=False):
    
    if n_neighbors_candidates is None:
        n_neighbors_candidates = [15,50,100,200]

    if show:
        n = len(n_neighbors_candidates)
        n_cols = int(np.ceil(np.sqrt(n)))
        n_rows = int(np.ceil(n / n_cols))
        fig, axs = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(n_cols*3 + 1,n_rows*3))
        axs = axs.flatten()

    scores_history = {}
    for i, n_neighbors in enumerate(n_neighbors_candidates):

        if runs_on_thinning:
            k_scores = study_sparsity_stdthinning(cells=cells,labels=labels,ratio_candidates=ratio_candidates,n_runs=n_runs,n_neighbors=n_neighbors,search_resolution_method=search_resolution_method,show=show,ax=axs[i],legend=False)
        else:
            k_scores = study_sparsity(cells=cells,labels=labels,ratio_candidates=ratio_candidates,n_runs=n_runs,n_neighbors=n_neighbors,search_resolution_method=search_resolution_method,stats=stats,show=show,ax=axs[i],legend=False)
        scores_history["k"] = k_scores

        if show:
            axs[i].set_title(f"k={n_neighbors}")

    if show:
        for j in range(i + 1, len(axs)):
            axs[j].set_visible(False)

            plt.tight_layout()
            plt.grid(alpha=0.6)

            axs[j].show()
        
        handles, labels = axs[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=5)
        plt.subplots_adjust(
            left=None,
            bottom=None,
            right=None,
            top=0.75,
            wspace=0.5,
        )
    
    return scores_history


def study_group_sparsity_nostdthinning(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", stats="average", show=False):
    from itertools import combinations

    if ratio_candidates is None:
        ratio_candidates = np.linspace(0.01,1,20)

    if len(cells) != len(labels):
        raise ValueError("Number of cells and labels should be the same!")

    unique_labels = np.unique(labels)
    combination_labels = list(combinations(unique_labels, 2))
    pairs_key = [f'{label1} vs {label2}' for label1,label2 in combination_labels]

    homogeneity_history, homogeneity_std = {label:[] for label in pairs_key}, {label:[] for label in pairs_key}
    completness_history, completness_std = {label:[] for label in pairs_key}, {label:[] for label in pairs_key}
    ari_history, ari_std = {label:[] for label in pairs_key}, {label:[] for label in pairs_key}
    v_history, v_std = {label:[] for label in pairs_key}, {label:[] for label in pairs_key}
    correct_history, correct_std = {label:[] for label in pairs_key}, {label:[] for label in pairs_key}

    for ratio in ratio_candidates:
        print(f"Ratio={ratio:.3f}")
        if(ratio < 1):
            thinned_cells = thinning(cells,reduction_ratio=ratio,same_reads=False,copy=True)
            thinned_cells = update_data(thinned_cells)
        else:
            thinned_cells = cells
            thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors,n_comps=100,random_state=42,normalization=normalization)

        sc.pp.neighbors(thinned_cells,n_neighbors=n_neighbors,n_pcs=100)
        results = find_best_resolution(data=thinned_cells,true_labels=labels,n_neighbors=None,n_trials=50,method=search_resolution_method,show=False)
        resolution = results["resolution"]

        h_runs, c_runs, v_runs, a_runs, ct_runs = {label:[] for label in pairs_key}, {label:[] for label in pairs_key}, {label:[] for label in pairs_key}, {label:[] for label in pairs_key}, {label:[] for label in pairs_key}
    
        for seed in range(n_runs):
            sc.tl.leiden(thinned_cells,resolution=resolution,key_added='leiden_temp', random_state=np.random.randint(0, MAX_RNG_RANGE))
            leiden_labels = thinned_cells.obs['leiden_temp']
            
            for label1, label2 in combination_labels:
                mask = (labels == label1) | (labels == label2)
                if mask.sum() == 0:
                    continue

                scores = compute_all_scores(true_labels=labels[mask],cluster_labels=leiden_labels[mask])
                h,c,v,ari,correct = scores["homogeneity"], scores["completness"], scores["v"], scores["ari"], scores["correct"]

                pair_key = f"{label1} vs {label2}"
                h_runs[pair_key].append(h)
                c_runs[pair_key].append(c)
                v_runs[pair_key].append(v)
                a_runs[pair_key].append(ari)
                ct_runs[pair_key].append(correct)

        for label in pairs_key:
            homogeneity_history[label].append(np.mean(h_runs[label]));  homogeneity_std[label].append(np.std(h_runs[label]))
            completness_history[label].append(np.mean(c_runs[label]));  completness_std[label].append(np.std(c_runs[label]))
            ari_history[label].append(np.mean(a_runs[label]));          ari_std[label].append(np.std(a_runs[label]))
            v_history[label].append(np.mean(v_runs[label]));            v_std[label].append(np.std(v_runs[label]))
            correct_history[label].append(np.mean(ct_runs[label]));     correct_std[label].append(np.std(ct_runs[label]))
        print("\n")

    if show:
        n_scores = 5
        n_cols = int(np.ceil(np.sqrt(n_scores)))
        n_rows = int(np.ceil(n_scores / n_cols))
        fig, axs = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(n_cols*3 + 8,n_rows*3))
        axs = axs.flatten()

        r = ratio_candidates

        def plot_with_band(x, mean, std, label, color=None, linestyle="-",ax=None):
            mean, std = np.array(mean), np.array(std)
            ci = 1.96 * std / np.sqrt(n_runs)  # intervalle de confiance à 95%
            ax.plot(x, mean, color=color, linestyle=linestyle, label=label)
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15)

        scores_history = [(homogeneity_history,homogeneity_std),(completness_history,completness_std),(correct_history,correct_std),(ari_history,ari_std),(v_history,v_std)]
        colors = ["blue", "red", "orange","green", "black"]
        score_labels = ["homogeneity","completness","correctly classified cells","ari","v"]
        for i in range(n_scores):

            score_to_plot, std_to_plot = scores_history[i]
            color = colors[i]
            score_label = score_labels[i]

            for label in pairs_key:
                plot_with_band(r, score_to_plot[label], std_to_plot[label], color=None, label=label, ax=axs[i])

            axs[i].invert_xaxis()
            axs[i].set_title(score_labels[i])
            #axs[i].legend()
            axs[i].grid(True,alpha=0.6)
            axs[i].set_xlabel("r")
            axs[i].set_ylabel("score")

        handles, legend_labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper right')
        plt.suptitle(f"k={n_neighbors}")

        plt.subplots_adjust(left=None, right=None, top=0.55, bottom=None, wspace=0.4, hspace=None)

        for j in range(n_scores, len(axs)):
            axs[j].set_visible(False)
        plt.show()


    return {
        "homogeneity": homogeneity_history,
        "completness": completness_history,
        "ari": ari_history,
        "v": v_history,
        "correct": correct_history,
    }

def study_group_sparsity(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", show=False):
    from itertools import combinations

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

        h_runs, c_runs, v_runs, a_runs, ct_runs = {label: [] for label in pairs_key}, {label: [] for label in pairs_key}, {label: [] for label in pairs_key}, {label: [] for label in pairs_key}, {label: [] for label in pairs_key}

        for seed in range(n_runs):
            if ratio < 1:
                thinned_cells = thinning(cells, reduction_ratio=ratio, same_reads=False, copy=True)
                thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=np.random.randint(0, MAX_RNG_RANGE), normalization=normalization)
            else:
                thinned_cells = cells
                thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=np.random.randint(0, MAX_RNG_RANGE), normalization=normalization)

            results = find_best_resolution(data=thinned_cells, true_labels=labels, n_neighbors=None, n_trials=50, method=search_resolution_method, show=False)
            resolution = results["resolution"]

            sc.tl.leiden(thinned_cells, resolution=resolution, key_added='leiden_temp', random_state=np.random.randint(0, MAX_RNG_RANGE))
            leiden_labels = thinned_cells.obs['leiden_temp']

            for label1, label2 in combination_labels:
                mask = (labels == label1) | (labels == label2)
                if mask.sum() == 0:
                    continue

                scores = compute_all_scores(true_labels=labels[mask], cluster_labels=leiden_labels[mask])
                pair_key = f"{label1} vs {label2}"
                h_runs[pair_key].append(scores["homogeneity"])
                c_runs[pair_key].append(scores["completness"])
                v_runs[pair_key].append(scores["v"])
                a_runs[pair_key].append(scores["ari"])
                ct_runs[pair_key].append(scores["correct"])

        for label in pairs_key:
            homogeneity_history[label].append(np.mean(h_runs[label]));  homogeneity_std[label].append(np.std(h_runs[label]))
            completness_history[label].append(np.mean(c_runs[label]));  completness_std[label].append(np.std(c_runs[label]))
            ari_history[label].append(np.mean(a_runs[label]));          ari_std[label].append(np.std(a_runs[label]))
            v_history[label].append(np.mean(v_runs[label]));            v_std[label].append(np.std(v_runs[label]))
            correct_history[label].append(np.mean(ct_runs[label]));     correct_std[label].append(np.std(ct_runs[label]))
        print("\n")

    if show:
        n_scores = 5
        n_cols = int(np.ceil(np.sqrt(n_scores)))
        n_rows = int(np.ceil(n_scores / n_cols))
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*3 + 8, n_rows*3))
        axs = axs.flatten()

        r = ratio_candidates

        def plot_with_band(x, mean, std, label, color=None, linestyle="-", ax=None):
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
            #axs[i].legend()
            axs[i].grid(True,alpha=0.6)
            axs[i].set_xlabel("r")
            axs[i].set_ylabel("score")

        handles, legend_labels = axs[0].get_legend_handles_labels()
        nrow=3
        fig.legend(handles, legend_labels, loc='upper right',ncol = int(np.ceil(len(pairs_key) / nrow)))
        #plt.suptitle(f"k={n_neighbors}")

        plt.subplots_adjust(left=None, right=None, top=0.8, bottom=None, wspace=0.4, hspace=None)

        for j in range(n_scores, len(axs)):
            axs[j].set_visible(False)
        plt.show()

    return {
        "homogeneity": homogeneity_history,
        "completness": completness_history,
        "ari": ari_history,
        "v": v_history,
        "correct": correct_history,
    }

def study_group_sparsity_exclude_nostdthinning(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", stats="average", show=False):
    from itertools import combinations

    if ratio_candidates is None:
        ratio_candidates = np.linspace(0.01,1,20)

    if len(cells) != len(labels):
        raise ValueError("Number of cells and labels should be the same!")

    unique_labels = np.append(np.unique(labels), 'no-exclusion')
    homogeneity_history, homogeneity_std = {label:[] for label in unique_labels}, {label:[] for label in unique_labels}
    completness_history, completness_std = {label:[] for label in unique_labels}, {label:[] for label in unique_labels}
    ari_history, ari_std = {label:[] for label in unique_labels}, {label:[] for label in unique_labels}
    v_history, v_std = {label:[] for label in unique_labels}, {label:[] for label in unique_labels}
    correct_history, correct_std = {label:[] for label in unique_labels}, {label:[] for label in unique_labels}

    for ratio in ratio_candidates:
        print(f"Ratio={ratio:.3f}")
        if(ratio < 1):
            thinned_cells = thinning(cells,reduction_ratio=ratio,same_reads=False,copy=True)
            thinned_cells = update_data(thinned_cells)
        else:
            thinned_cells = cells
            thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors,n_comps=100,random_state=42,normalization=normalization)

        sc.pp.neighbors(thinned_cells,n_neighbors=n_neighbors,n_pcs=100)
        results = find_best_resolution(data=thinned_cells,true_labels=labels,n_neighbors=None,n_trials=50,method=search_resolution_method,show=False)
        resolution = results["resolution"]

        h_runs, c_runs, v_runs, a_runs, ct_runs = {label:[] for label in unique_labels}, {label:[] for label in unique_labels}, {label:[] for label in unique_labels}, {label:[] for label in unique_labels}, {label:[] for label in unique_labels}
    
        for seed in range(n_runs):
            sc.tl.leiden(thinned_cells,resolution=resolution,key_added='leiden_temp', random_state=np.random.randint(0, MAX_RNG_RANGE))
            leiden_labels = thinned_cells.obs['leiden_temp']
            
            for label in unique_labels:
                mask = (labels != label)
                if mask.sum() == 0:
                    continue

                scores = compute_all_scores(true_labels=labels[mask],cluster_labels=leiden_labels[mask])
                h,c,v,ari,correct = scores["homogeneity"], scores["completness"], scores["v"], scores["ari"], scores["correct"]

                h_runs[label].append(h)
                c_runs[label].append(c)
                v_runs[label].append(v)
                a_runs[label].append(ari)
                ct_runs[label].append(correct)

        for label in unique_labels:
            homogeneity_history[label].append(np.mean(h_runs[label]));  homogeneity_std[label].append(np.std(h_runs[label]))
            completness_history[label].append(np.mean(c_runs[label]));  completness_std[label].append(np.std(c_runs[label]))
            ari_history[label].append(np.mean(a_runs[label]));          ari_std[label].append(np.std(a_runs[label]))
            v_history[label].append(np.mean(v_runs[label]));            v_std[label].append(np.std(v_runs[label]))
            correct_history[label].append(np.mean(ct_runs[label]));     correct_std[label].append(np.std(ct_runs[label]))
        print("\n")

    if show:
        n_scores = 5
        n_cols = int(np.ceil(np.sqrt(n_scores)))
        n_rows = int(np.ceil(n_scores / n_cols))
        fig, axs = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(n_cols*3 + 1,n_rows*3))
        axs = axs.flatten()

        r = ratio_candidates

        def plot_with_band(x, mean, std, label, color=None, linestyle="-",ax=None):
            mean, std = np.array(mean), np.array(std)
            ci = 1.96 * std / np.sqrt(n_runs)  # intervalle de confiance à 95%
            ax.plot(x, mean, color=color, linestyle=linestyle, label=label)
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15)

        scores_history = [(homogeneity_history,homogeneity_std),(completness_history,completness_std),(correct_history,correct_std),(ari_history,ari_std),(v_history,v_std)]
        colors = ["blue", "red", "orange","green", "black"]
        score_labels = ["homogeneity","completness","correctly classified cells","ari","v"]
        
        for i in range(n_scores):

            score_to_plot, std_to_plot = scores_history[i]
            color = colors[i]
            score_label = score_labels[i]

            for label in unique_labels:
                if label=="no-exclusion":
                    plot_with_band(r, score_to_plot[label], std_to_plot[label], color=None, label=label, linestyle="--", ax=axs[i])
                else:
                    plot_with_band(r, score_to_plot[label], std_to_plot[label], color=None, label=label, linestyle="-", ax=axs[i])

            axs[i].invert_xaxis()
            axs[i].set_title(score_labels[i])
            #axs[i].legend()
            axs[i].grid(True,alpha=0.6)
            axs[i].set_xlabel("r")
            axs[i].set_ylabel("score")

        handles, legend_labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper right',ncol=len(unique_labels))
        #plt.suptitle(f"k={n_neighbors}")

        plt.subplots_adjust(left=None, right=None, top=0.9, bottom=None, wspace=0.4, hspace=0.25)

        for j in range(n_scores, len(axs)):
            axs[j].set_visible(False)
        plt.show()


    return {
        "homogeneity": homogeneity_history,
        "completness": completness_history,
        "ari": ari_history,
        "v": v_history,
        "correct": correct_history,
    }

def study_group_sparsity_exclude(cells, labels, ratio_candidates=None, n_runs=50, n_neighbors=15, normalization="sct", search_resolution_method="optuna", show=False):

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

        h_runs, c_runs, v_runs, a_runs, ct_runs = {label: [] for label in unique_labels}, {label: [] for label in unique_labels}, {label: [] for label in unique_labels}, {label: [] for label in unique_labels}, {label: [] for label in unique_labels}

        for seed in range(n_runs):
            if ratio < 1:
                thinned_cells = thinning(cells, reduction_ratio=ratio, same_reads=False, copy=True)
                thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=np.random.randint(0, MAX_RNG_RANGE), normalization=normalization)
            else:
                thinned_cells = cells
                thinned_cells = update_data(thinned_cells, n_neighbors=n_neighbors, n_comps=100, random_state=np.random.randint(0, MAX_RNG_RANGE), normalization=normalization)

            results = find_best_resolution(data=thinned_cells, true_labels=labels, n_neighbors=None, n_trials=50, method=search_resolution_method, show=False)
            resolution = results["resolution"]

            sc.tl.leiden(thinned_cells, resolution=resolution, key_added='leiden_temp', random_state=np.random.randint(0, MAX_RNG_RANGE))
            leiden_labels = thinned_cells.obs['leiden_temp']

            for label in unique_labels:
                mask = (labels != label)
                if mask.sum() == 0:
                    continue

                scores = compute_all_scores(true_labels=labels[mask], cluster_labels=leiden_labels[mask])
                h_runs[label].append(scores["homogeneity"])
                c_runs[label].append(scores["completness"])
                v_runs[label].append(scores["v"])
                a_runs[label].append(scores["ari"])
                ct_runs[label].append(scores["correct"])

        for label in unique_labels:
            homogeneity_history[label].append(np.mean(h_runs[label]));  homogeneity_std[label].append(np.std(h_runs[label]))
            completness_history[label].append(np.mean(c_runs[label]));  completness_std[label].append(np.std(c_runs[label]))
            ari_history[label].append(np.mean(a_runs[label]));          ari_std[label].append(np.std(a_runs[label]))
            v_history[label].append(np.mean(v_runs[label]));            v_std[label].append(np.std(v_runs[label]))
            correct_history[label].append(np.mean(ct_runs[label]));     correct_std[label].append(np.std(ct_runs[label]))
        print("\n")

    if show:
        n_scores = 5
        n_cols = int(np.ceil(np.sqrt(n_scores)))
        n_rows = int(np.ceil(n_scores / n_cols))
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*3 + 1, n_rows*3))
        axs = axs.flatten()

        r = ratio_candidates

        def plot_with_band(x, mean, std, label, color=None, linestyle="-", ax=None):
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
        plt.show()

    return {
        "homogeneity": homogeneity_history,
        "completness": completness_history,
        "ari": ari_history,
        "v": v_history,
        "correct": correct_history,
    }