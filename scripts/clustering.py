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

# Ce script répertorie toutes les fonctions utiles ou nécessaires pour la clusterisation/partitionnement d'un jeu de données.


def cluster_data(data,true_labels=None,n_neighbors=None,resolution=0.5,n_comps=100,random_state=None,show=True):
    """
        Partitionne data.X (les données normalisées) selon l'algorithme de leiden de paramètres 'resolution' à partir du graphe de voisinages de paramètres ('resolution', 'n_comps').
        La fonction retourne les labels assignés par le partionnement de leiden ainsi que les scores du partionnement vis à vis de 'true_labels' (la ground truth).

        data: objet scanpy (dataframe)
        true_labels: list/array de labels. Un label est un entier entre 0 et n_class-1.
        n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden). Si n_neighbors==None, alors le graphe de voisinage n'est pas actualisée
            et le clustering de leiden s'effectue à partir de celui déjà enregistré dans 'data'. Si aucun n'est enregistré -> erreur.
        resolution: paramètre de résolution de leiden.
        n_comps: nombre de composantes de pca pour le calcul du graphe de voisinage.
        random_state: seed de l'algorithme de leiden (l'algorithme est en partie aléatoire).
        show: True alors on affiche la UMAP avec la ground truth et les nouveaux labels issus de leiden.
    """

    # Si aucun random_state imposé, on en prend aléatoirement.
    if random_state is None:
        random_state = np.random.randint(0, MAX_RNG_RANGE)

    # Calcul du graphe de voisinage (graphe de k-Nearest-Neighbors)
    if n_neighbors is not None:
        sc.pp.neighbors(data,n_neighbors=n_neighbors,n_pcs=n_comps)

    # Partitionnement de leiden.
    sc.tl.leiden(data,resolution=resolution,key_added='leiden_temp', random_state=random_state)

    leiden_labels = data.obs['leiden_temp']
    output = {"leiden_labels":leiden_labels}

    # Affichage de la UMAP ?
    if show:
        plot_custom_UMAP(data=data,labels=leiden_labels)

    # Calcul du score ?
    if not true_labels is None:
        scores = compute_all_scores(true_labels=true_labels, cluster_labels=leiden_labels)
        output["scores"] = scores

    return output

def find_best_resolution_linspace(data,resolution_range=[0.2,1.5],resolution_step=0.05,true_labels=None,n_neighbors=None,n_comps=100,n_seeds=3,random_state=None,show=True):
    """
    Trouve la meilleure résolution de leiden vis à vis de la ground truth 'true_labels' (Maximisation du score d'ARI) en discrétisant l'espace des résolutions candidates.
    On se fixe un intervalle de résolution et on essaye de trouver la meilleure résolution en calculant le score de chaque résolution (calcul exhaustif et donc coûteux).

    data: objet scanpy (dataframe)
    resolution_range: plage sur laquelle on se permet de chercher la meilleure résolution.
    resolution_step: discrétisation de la plage 'resolution_range'.
    true_labels: list/array de labels. Un label est un entier entre 0 et n_class-1.
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden). Si n_neighbors==None, alors le graphe de voisinage n'est pas actualisée
            et le clustering de leiden s'effectue à partir de celui déjà enregistré dans 'data'. Si aucun n'est enregistré -> erreur.
    n_comps: nombre de composantes de pca pour le calcul du graphe de voisinage.
    n_seeds: nombre d'initialisation aléatoires à essayer pour chaque résolution candidate (algorithme de leiden est en partie aléatoire).
    random_state: seed de l'algorithme de leiden (l'algorithme est en partie aléatoire).
    show: True alors on affiche l'évolution du score selon la résolution.
    """

    # Résolutions candidates
    resolutions = np.arange(resolution_range[0],resolution_range[-1],resolution_step)

    best_cluster = None
    best_score = -np.inf
    best_resolution = 0.0

    scores_history = []
    pbar = tqdm.tqdm(resolutions, desc="clustering")
    for resolution in pbar:
        
        s = []
        # On essaie plusieurs instances pour la même résolution.
        for seed in range(n_seeds):
            output = cluster_data(data=data,true_labels=true_labels,n_neighbors=n_neighbors,resolution=resolution,n_comps=n_comps,random_state=random_state,show=False)
            s.append(output["scores"]["ari"])

        # On prend le score moyen de la résolution candidate
        score = np.mean(s)
        scores_history.append(score)

        # Est-elle meilleure que celle déjà enregistrée ?
        if best_score < score:
            best_cluster = output["leiden_labels"]
            best_score = score
            best_resolution = resolution

        pbar.set_postfix({'score': f'{score:.2e}', 
                        'best_score': f'{best_score:.2e}'})

    # Affichage de l'évolution du score selon la résolution.
    if show:
        plt.figure()
        plt.plot(resolutions,scores_history)
        plt.grid(alpha=0.6)
        plt.show()

        plt.figure
        plot_custom_UMAP(data=data,labels=best_cluster)
    
    return {"leiden_labels":best_cluster, "score":best_score, "resolution":best_resolution}


# Fonction d'optimisation.
def objective(trial, data, true_labels, n_neighbors=None, n_comps=100, n_seeds=3, method="optuna"):
    """
    Fonction objectif à maximiser.
    Cette fonction est une brique pour calculer la meilleure résolution selon la méthode d'optimisation 'method'.

    trial: état candidat (correspond à une résolution candidate)
    data: objet scanpy (dataframe)
    true_labels: list/array de labels. Un label est un entier entre 0 et n_class-1.
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden). Si n_neighbors==None, alors le graphe de voisinage n'est pas actualisée
            et le clustering de leiden s'effectue à partir de celui déjà enregistré dans 'data'. Si aucun n'est enregistré -> erreur.
    n_comps: nombre de composantes de pca pour le calcul du graphe de voisinage.
    n_seeds: nombre d'initialisation aléatoires à essayer pour chaque résolution candidate (algorithme de leiden est en partie aléatoire).
    method: méthode d'optimisation utilisée.
        Optuna est une méthode boîte noire qui compile plusieurs méthodes d'optimisation simultanément, https://optuna.readthedocs.io/en/stable/.
        Golden est une approche d'optimisation où l'on suppose que la fonction objective est unimodale (un seul maximum local).
        En pratique, on utilisera plus Optuna car notre fonction objective est bruitée (et donc très probablement multimodales à moins de lisser suffisamment la courbe en augmentant n_seeds).
    """
    
    # Choix de la méthode d'optimisation
    if method=="optuna":
        resolution  = trial.suggest_float("resolution", 0.01, 2.0)
    elif method=="golden":
        resolution = trial
    else:
        raise ValueError("Choose a correct method.")
    
    scores = []
    # Calcul du score de partionnement sur plusieurs seeds
    for seed in range(n_seeds):
        output = cluster_data(data=data, true_labels=true_labels, n_neighbors=n_neighbors,resolution=resolution,n_comps=n_comps,random_state=seed,show=False)
        score = output["scores"]["ari"]
        scores.append(score)

    # On retourne la moyenne des scores
    return np.mean(scores)

def early_stopping_callback(study, trial, patience=20):
    """
    Fonction callback d'Optuna. On stoppe l'étude si les 'patience' dernières résolutions candidates n'ont pas été meilleures que le dernier meilleur score enregistré.
    """
    if trial.number < patience:
        return
    
    recent_values = [t.value for t in study.trials[-patience:]
                     if t.value is not None]
    
    if max(recent_values) < study.best_value:
        study.stop()

def find_best_resolution(data,true_labels,n_neighbors=15,n_trials=100,n_comps=100,method="optuna",random_state=None,show=True):
    """
    Trouve la meilleure résolution de leiden vis à vis de la ground truth 'true_labels' (Maximisation du score d'ARI) en utilisant la méthode d'optimisation 'method'.

    data: objet scanpy (dataframe)
    true_labels: list/array de labels. Un label est un entier entre 0 et n_class-1.
    n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden). Si n_neighbors==None, alors le graphe de voisinage n'est pas actualisée
            et le clustering de leiden s'effectue à partir de celui déjà enregistré dans 'data'. Si aucun n'est enregistré -> erreur.
    n_trials:
        Pour Optuna, nombre de résolutions candidates maximales à explorer avant arrêt et envoie de la dernière meilleure résolution.
        Pour Golden, nombres de seeds sur lesquels calculer le score (pour lisser le bruit de la fonction objective).
    n_comps: nombre de composantes de pca pour le calcul du graphe de voisinage.
    random_state: seed de l'algorithme de leiden (l'algorithme est en partie aléatoire).
    method: méthode d'optimisation utilisée.
        Optuna est une méthode boîte noire qui compile plusieurs méthodes d'optimisation simultanément, https://optuna.readthedocs.io/en/stable/.
        Golden est une approche d'optimisation où l'on suppose que la fonction objective est unimodale (un seul maximum local).
        En pratique, on utilisera plus Optuna car notre fonction objective est bruitée (et donc très probablement multimodales à moins de lisser suffisamment la courbe en augmentant n_seeds).
    show: True alors on affiche la UMAP avec la ground truth et les nouveaux labels issus de leiden.
    """
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
    """
        Partitionne data.X (les données normalisées) selon l'algorithme de leiden de paramètres 'resolution' à partir du graphe de voisinages de paramètres ('resolution', 'n_comps').
        La fonction retourne les labels assignés par le meilleur partionnement de leiden trouvé sur 'n_runs' instances (l'algorithme de leiden est en partie aléatoire)
        ainsi que les scores du partionnement vis à vis de 'true_labels' (la ground truth).

        data: objet anndata/scanpy (dataframe)
        true_labels: list/array de labels. Un label est un entier entre 0 et n_class-1.
        resolution: paramètre de résolution de leiden.
        n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden). Si n_neighbors==None, alors le graphe de voisinage n'est pas actualisée
            et le clustering de leiden s'effectue à partir de celui déjà enregistré dans 'data'. Si aucun n'est enregistré -> erreur.
        n_comps: nombre de composantes de pca pour le calcul du graphe de voisinage.
        n_runs: nombre d'instances à lancer pour trouver le meilleur partitionnement
        score_key: score sur lequel évaluer.
        show: True alors on affiche la UMAP avec la ground truth et les nouveaux labels issus de leiden.
    """
    if n_neighbors is not None:
        sc.pp.neighbors(data, n_neighbors=n_neighbors)
    
    best_score = -np.inf
    best_scores_dict = None
    best_labels = None

    # Evaluer chaque instance selon 'score_key'
    for run in range(n_runs):
        seed = np.random.randint(0, MAX_RNG_RANGE)
        results = cluster_data(data=data,true_labels=true_labels,n_neighbors=None,resolution=resolution,n_comps=n_comps,random_state=seed,show=False)
        score = results["scores"][score_key]
        if score > best_score:
            best_score = score
            best_scores_dict = results["scores"]
            best_labels = results["leiden_labels"]
    
    # Affichage de la UMAP ?
    if show:
        plot_custom_UMAP(data=data,labels=best_labels)
    
    return {"leiden_labels":best_labels, "scores":best_scores_dict}

def average_leiden_run(data, true_labels, resolution, n_neighbors=None, n_comps=100, n_runs=50,show=False):
    """
        Partitionne data.X (les données normalisées) selon l'algorithme de leiden de paramètres 'resolution' à partir du graphe de voisinages de paramètres ('resolution', 'n_comps').
        La fonction retourne les labels assignés par un partionnement moyen de différents partitionnement leiden calculés 'n_runs' instances (l'algorithme de leiden est en partie aléatoire)
        ainsi que les scores du partionnement vis à vis de 'true_labels' (la ground truth).

        data: objet anndata/scanpy (dataframe)
        true_labels: list/array de labels. Un label est un entier entre 0 et n_class-1.
        resolution: paramètre de résolution de leiden.
        n_neighbors: nombre de plus proches voisins dans le graphe de k-NearestNeighbors (voir algorithme de Leiden). Si n_neighbors==None, alors le graphe de voisinage n'est pas actualisée
            et le clustering de leiden s'effectue à partir de celui déjà enregistré dans 'data'. Si aucun n'est enregistré -> erreur.
        n_comps: nombre de composantes de pca pour le calcul du graphe de voisinage.
        n_runs: nombre d'instances à lancer pour trouver le meilleur partitionnement
        show: True alors on affiche la UMAP avec la ground truth et les nouveaux labels issus de leiden.
    """

    # Calcul du graphe kNN ?
    if n_neighbors is not None:
        sc.pp.neighbors(data, n_neighbors=n_neighbors)

    # Calcule de chaque partitionnement
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

    # Un label est assigné à une cellule par un vote à la majorité de chaque partitionnement de leiden
    majority_labels = scipy.stats.mode(all_leiden_labels, axis=0).mode.astype(str)

    # Affichage de la UMAP ?
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