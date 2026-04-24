import scanpy as sc
import matplotlib.pyplot as plt

import numpy as np

from .utils import MAX_RNG_RANGE
from .utils import plot_custom_UMAP

from .scoring import compute_all_scores

import tqdm as tqdm
import optuna

import scipy
from scipy.optimize import minimize_scalar

def cluster_data(data,true_labels=None,n_neighbors=None,resolution=0.5,n_comps=100,random_state=None,show=True):

    if random_state is None:
        random_state = np.random.randint(0, MAX_RNG_RANGE)

    if n_neighbors is not None:
        sc.pp.neighbors(data,n_neighbors=n_neighbors,n_pcs=n_comps)

    sc.tl.leiden(data,resolution=resolution,key_added='leiden_temp', random_state=random_state)

    leiden_labels = data.obs['leiden_temp']
    output = {"leiden_labels":leiden_labels}

    if show:
        plot_custom_UMAP(data=data,labels=leiden_labels)

    if not true_labels is None:
        scores = compute_all_scores(true_labels=true_labels, cluster_labels=leiden_labels)
        output["scores"] = scores

    return output

def find_best_resolution_linspace(data,resolution_range=[0.2,1.5],resolution_step=0.05,true_labels=None,n_neighbors=None,n_comps=100,n_seeds=3,random_state=None,show=True):

    resolutions = np.arange(resolution_range[0],resolution_range[-1],resolution_step)

    best_cluster = None
    best_score = -np.inf
    best_resolution = 0.0

    scores_history = []
    pbar = tqdm.tqdm(resolutions, desc="clustering")
    for resolution in pbar:
        
        s = []
        for seed in range(n_seeds):
            output = cluster_data(data=data,true_labels=true_labels,n_neighbors=n_neighbors,resolution=resolution,n_comps=n_comps,random_state=random_state,show=False)
            s.append(output["scores"]["ari"])

        score = np.mean(s)
        scores_history.append(score)

        if best_score < score:
            best_cluster = output["leiden_labels"]
            best_score = score
            best_resolution = resolution

        pbar.set_postfix({'score': f'{score:.2e}', 
                        'best_score': f'{best_score:.2e}'})

    if show:
        plt.figure()
        plt.plot(resolutions,scores_history)
        plt.grid(alpha=0.6)
        plt.show()

        plt.figure
        plot_custom_UMAP(data=data,labels=best_cluster)
    
    return {"leiden_labels":best_cluster, "score":best_score, "resolution":best_resolution}


def objective(trial, data, true_labels, n_neighbors=None, n_comps=100, use_rep="X_pca", n_seeds=3, method="optuna"):
    
    if method=="optuna":
        resolution  = trial.suggest_float("resolution", 0.01, 2.0)
    elif method=="golden":
        resolution = trial
    else:
        raise ValueError("Choose a correct method.")
    
    scores = []
    for seed in range(n_seeds):
        output = cluster_data(data=data, true_labels=true_labels, n_neighbors=n_neighbors,resolution=resolution,n_comps=n_comps,random_state=seed,show=False)
        score = output["scores"]["ari"]
        scores.append(score)

    return np.mean(scores)

def early_stopping_callback(study, trial, patience=20):
    if trial.number < patience:
        return
    
    recent_values = [t.value for t in study.trials[-patience:]
                     if t.value is not None]
    
    if max(recent_values) < study.best_value:
        study.stop()

def find_best_resolution(data,true_labels,n_neighbors=15,n_trials=100,n_comps=100,method="optuna",random_state=None,show=True):
    
    if method=="golden":
        result = minimize_scalar(
            lambda res: -objective(res, data, true_labels=true_labels, n_neighbors=n_neighbors, method=method, n_comps=n_comps, n_seeds=n_trials),
            bounds=(0.01, 3.0),
            method="bounded",
            options={"xatol": 5e-2}  # tolérance de convergence
        )

        best_resolution = result.x

    elif method=="optuna":        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state)
        )

        study.optimize(
            lambda trial: objective(trial, data, true_labels=true_labels, n_neighbors=n_neighbors, n_comps=n_comps, n_seeds=1),
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[lambda study, trial: early_stopping_callback(study, trial, patience=20)]
        )

        best = study.best_params
        best_resolution = best["resolution"]

    output = cluster_data(data=data, true_labels=true_labels, n_neighbors=n_neighbors,resolution=best_resolution,n_comps=n_comps,random_state=random_state,show=show)
    return {"leiden_labels":output["leiden_labels"], "scores":output["scores"], "resolution":best_resolution}


def best_leiden_run(data, true_labels, resolution, n_neighbors=None, n_comps=100, n_runs=50,score_key="ari",show=False):
    
    if n_neighbors is not None:
        sc.pp.neighbors(data, n_neighbors=n_neighbors)
    
    best_score = -np.inf
    best_scores_dict = None
    best_labels = None
    for run in range(n_runs):
        seed = np.random.randint(0, MAX_RNG_RANGE)
        results = cluster_data(data=data,true_labels=true_labels,n_neighbors=None,resolution=resolution,n_comps=n_comps,random_state=seed,show=False)
        score = results["scores"][score_key]
        if score > best_score:
            best_score = score
            best_scores_dict = results["scores"]
            best_labels = results["leiden_labels"]
    
    if show:
        plot_custom_UMAP(data=data,labels=best_labels)
    
    return {"leiden_labels":best_labels, "scores":best_scores_dict}

def average_leiden_run(data, true_labels, resolution, n_neighbors=None, n_comps=100, n_runs=50,show=False):
    
    if n_neighbors is not None:
        sc.pp.neighbors(data, n_neighbors=n_neighbors)

    h_runs, c_runs, v_runs, a_runs, ct_runs, leiden_runs = [], [], [], [], [], []
    for run in range(n_runs):

        seed = np.random.randint(0, MAX_RNG_RANGE)
        output = cluster_data(data=data,true_labels=true_labels,n_neighbors=None,n_comps=n_comps,resolution=resolution,random_state=seed,show=False)
        leiden_labels = output["leiden_labels"]
        scores = output["scores"]
        h,c,v,ari,correct = scores["homogeneity"], scores["completness"], scores["v"], scores["ari"], scores["correct"]

        leiden_runs.append(leiden_labels)
        h_runs.append(h)
        c_runs.append(c)
        v_runs.append(v)
        a_runs.append(ari)
        ct_runs.append(correct)
    
    all_leiden_labels = np.array(leiden_runs).astype(int) # shape (n_runs, n_cells)
    majority_labels = scipy.stats.mode(all_leiden_labels, axis=0).mode.astype(str) # majority

    if show:
        plot_custom_UMAP(data=data,labels=majority_labels)

    output = {
        "leiden_labels": majority_labels,

        "scores":{
            "homogeneity":np.mean(h_runs),
            "completness":np.mean(c_runs),
            "ari":np.mean(a_runs),
            "v":np.mean(v_runs),
            "correct":np.mean(ct_runs)
        },

        "scores_std":{
            "homogeneity":np.std(h_runs),
            "completness":np.std(c_runs),
            "ari":np.std(a_runs),
            "v":np.std(v_runs),
            "correct":np.std(ct_runs)
        },
    }

    return output