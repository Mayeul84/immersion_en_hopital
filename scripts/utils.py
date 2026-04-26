import os
import datetime

import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as sp

from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt

# Ce script répertorie toutes les petites fonctions utiles au projet mais qui ne sont pas le coeur du travail.

# Etat aléatoirement maximale dans les algorithmes de générations de nombres aléatories.
MAX_RNG_RANGE = 10000

# Chemins d'accès

### Chemin principal
PROJECT_PATH = ... # A CHANGER, METTRE LE CHEMIN ACTUEL DU PROJET. Exemple: r"C:\Users\mayeu\Desktop\TRAVAIL\MVA\Cours\Projet médecin"

### Pour l'exportation
RESULTS_PATH = PROJECT_PATH + r"\results"
FIGURES_PATH = RESULTS_PATH + r"\figures"

### Pour charger et exporter les données .h5ad
DATA_PATH = PROJECT_PATH + r"\data"

### Jeu de données originel (au format .h5ad)
SINGLE_CELLS_PATH = DATA_PATH + r"\single_cells.h5ad"

# Lien entre les labels numériques et labels textuelles (pour rendre le jeu de données plus interprétable)
CELLTYPE_MAP = {
    0: 'Steroid cells',
    1: 'Chromaffin cells',
    2: 'Endothelial cells',
    3: 'Fibroblasts',
    4: 'Myeloid cells',
    5: 'Lymphoid cells'
}

# Lien entre les labels numériques et labels textuelles (pour rendre le jeu de données plus interprétable)
HISTOTYPE_MAP = {
    0: 'ACC C1A',
    1: 'ACC C1B',
    2: 'ACA',
    3: 'PBMAH',
    4: 'Normal'
}

# Sous-ensemble de types cellulaires intéressants à étudier (fourni par Anne)
SUBSETS_CONFIG = {
    "Steroid cells":    {'labels': ['ACC C1A', 'ACC C1B', 'ACA', 'PBMAH', 'Normal'], 'col': 'histotype_label'},
    "Endothelial cells": {'labels': ['TEC2', 'EC-venous'], 'col': 'cellstates_tme'},
    "Fibroblasts": {'labels': ['CAF1', 'CAF2', 'CAF3', 'Resident fibroblasts 1', 'Resident fibroblasts 2'], 'col': 'cellstates_tme'},
    "Myeloid cells": {'labels': ['TAM1', 'TAM2', 'Inflammatory macrophages', 'Resident macrophages 1', 'Resident macrophages 2'], 'col': 'cellstates_tme'}
}

# IMPORTANT: Lien entre les sous-ensembles de types cellulaires étudiés et leur nom de fichier auquel ils sont ou seront enregistrés.
### all: le jeu de données complet, 4principals: les 4 principaux types cellulaires, No steroid: sans cellules stéroïdes, le reste est limpide.
ANNDATA_MAP = {"all": "all.h5ad", "4principals": "4principals.h5ad", "No steroid": "nosteroid_cells.h5ad", "Myeloid cells": "myeloid_cells.h5ad", "Steroid cells": "steroid_cells.h5ad", "Fibroblasts": "fibroblast.h5ad", "Endothelial cells": "endothelial_cells.h5ad"}

def format_time(seconds):
    """
    Convertit un temps en secondes en date.
    """
    return str(datetime.timedelta(seconds=int(seconds)))

def load_data(data_path):
    """
    Charger la donnée situé au chemin 'data_path'.

    Retourne un dataframe anndata.
    """
    data = sc.read_h5ad(data_path)    
    if data.raw is not None and "_index" in data.raw.var.columns:
        data.raw.var.rename(columns={"_index": "gene_index"}, inplace=True)

    return data

def check_data(celltype,data_path=DATA_PATH):
    """
    Vérifier si le 'celltype' sélectionné a son propre dataframe (filtre du dataframe de base) enregistré en local au format .h5ad.
    Retourne le dataframe s'il existe, sinon renvoie None.

    On enregistre des sous-dataframes du dataframe initial afin de ne pas avoir à le recharger complètement systématiquement (coûteux en mémoire)

    celltype: str, nom du type cellulaire choisi.
    data_path: str, chemin vers un dossier de dataframes anndata au format .h5ad.
    """

    # Le lien entre 'celltype' et le nom du fichier .h5ad est fait par ANNDATA_MAP
    file_to_find = ANNDATA_MAP[celltype]
    
    files = [f for f in os.listdir(data_path) if f.endswith(".h5ad")]

    # Existe-t-il ?
    if file_to_find in files:
        return load_data(f"{data_path}\\{file_to_find}")
    else:
        return None
    
def save_data(data, data_path=f"{DATA_PATH}\\data.h5ad"):
    """
    Sauvegarde la 'data' au format .h5ad au chemin 'data_path'.
    """

    # Création de la colonne 'gene_index' dans le tableau de données brutes (important)
    if data.raw is not None and "_index" in data.raw.var.columns:
        data.raw.var.rename(columns={"_index": "gene_index"}, inplace=True)

    # Export
    data.write_h5ad(data_path, compression="gzip")

def preprocess_data(celltype="Steroid cells", with_subsets_config=False, n_neighbors=15, n_comps=100, random_state=42, verbose=False):
    """
    Preprocessing du jeu de données originale. Selon, le sous ensemble 'celltype' de types cellulaires sélectionné, on construit un sous-dataframe anndata.

    celltype: str, nom du sous-ensemble de type cellulaires.
    with_subsets_config: Si True, alors on filtre aussi sur le sous-ensemble de types cellulaires qu'Anne a indiqué.
    n_neighbors: pour le graphe kNN, dans l'actualisation du jeu de données.
    n_comps: nombre de composantes PCA,
    random_state: seed pour calculs aléatoires
    """
    data = ad.read_h5ad(SINGLE_CELLS_PATH,backed="r")
    if data.raw is not None and "_index" in data.raw.var.columns:
        data.raw.var.rename(columns={"_index": "gene_index"}, inplace=True)

    # On renomme les colonnes pour que la donnée soit mieux interprétable.
    data.obs['celltype_label'] = data.obs['celltype'].map(CELLTYPE_MAP).astype('category')
    data.obs['histotype_label'] = data.obs['histotype'].map(HISTOTYPE_MAP).astype('category')
    
    # Quel sous-ensemble de types cellulaires ?
    if celltype == "all":
        data = data.to_memory()
        return data
    elif celltype=="No steroid":
        mask = data.obs["celltype_label"] != "Steroid cells"
    elif celltype=="4principals":
        mask = data.obs["celltype_label"].isin(["Steroid cells", "Myeloid cells", "Fibroblasts", "Endothelial cells"])
    else:
        mask = data.obs["celltype_label"] == celltype

    # Filtrage supplémentaire de types cellulaires qu'Anne a indiqué
    if with_subsets_config:
        mask = mask & (data.obs[SUBSETS_CONFIG[celltype]['col']].isin(SUBSETS_CONFIG[celltype]['labels']))

    data = data[mask]
    data = data.to_memory()

    # Suprression de données inutiles et lourdes.
    del data.obsm
    del data.obsp

    # Actualisation du jeu de données normalisées.
    data = update_data(data, n_neighbors=n_neighbors, n_comps=n_comps, random_state=random_state, verbose=verbose)

    return data
    
def sctransform_manual(adata, n_cells=5000, n_genes=2000, clip_range=None,verbose=False):
    """
    Implémentation manuelle de SCTransform
    
    Parameters:
    -----------
    adata : AnnData
        Objet AnnData avec des counts bruts
    n_cells : int
        Nombre de cellules pour estimer les paramètres
    n_genes : int
        Nombre de gènes variables à retourner
    clip_range : tuple
        Range pour clipper les résidus (défaut: (-sqrt(n), sqrt(n)))
    """
    
    counts = sp.csr_array(adata.X).copy() # normalement c'était déjà csr_array
    
    n_obs, n_vars = counts.shape

    # 2. Calculer la profondeur de séquençage par cellule (UMI totaux)
    umi_per_cell = counts.sum(axis=1)
    log_umi_per_cell = np.log10(umi_per_cell + 1)
    
    # 3. Calculer les statistiques par gène
    gene_mean = counts.mean(axis=0)
    gene_detection_rate = (counts > 0).mean(axis=0)
    
    # Log des moyennes pour la régression
    log_gene_mean = np.log10(gene_mean + 1)
    
    # 4. Échantillonner des cellules si trop nombreuses
    if n_obs > n_cells:
        cell_idx = np.random.choice(n_obs, n_cells, replace=False)
        counts_sample = counts[cell_idx, :]
        umi_sample = umi_per_cell[cell_idx]
        log_umi_sample = log_umi_per_cell[cell_idx]
    else:
        counts_sample = counts
        umi_sample = umi_per_cell
        log_umi_sample = log_umi_per_cell
    
    # 5. Pour chaque gène, estimer les paramètres du modèle binomial négatif
    if(verbose):
        print("Estimating model parameters...")
    
    theta_estimates = np.zeros(n_vars)  # Paramètre de dispersion
    mu_models = np.zeros((n_obs, n_vars))  # Moyennes prédites
    
    for g in range(n_vars):
        if g % 500 == 0 and verbose:
            print(f"  Gene {g}/{n_vars}")
        
        gene_counts = counts_sample[:, g]
        
        # Régression pour modéliser mu en fonction de log_umi
        # log(mu) = beta0 + beta1 * log(umi)
        if gene_mean[g] > 0:
            # Régression linéaire simple
            X_reg = np.column_stack([np.ones(len(log_umi_sample)), log_umi_sample])
            y_reg = np.log10(gene_counts.toarray() + 1)
            
            try:
                beta = np.linalg.lstsq(X_reg, y_reg, rcond=None)[0]
            except:
                beta = np.array([np.log10(gene_mean[g] + 1), 0])
            
            # Prédire mu pour toutes les cellules
            mu_pred = 10 ** (beta[0] + beta[1] * log_umi_per_cell) - 1
            mu_pred = np.maximum(mu_pred, 1e-6)
            mu_models[:, g] = mu_pred
            
            # Estimer theta (dispersion) via method of moments
            gene_var = gene_counts.todense().var()
            gene_mu = gene_counts.mean()
            
            if gene_var > gene_mu:
                # Variance = mu + mu^2/theta
                theta_est = gene_mu ** 2 / (gene_var - gene_mu)
                theta_est = np.clip(theta_est, 0.01, 100)
            else:
                theta_est = 100  # Faible dispersion
                
            theta_estimates[g] = theta_est
        else:
            mu_models[:, g] = 1e-6
            theta_estimates[g] = 100
    
    # 6. Calculer les résidus de Pearson
    if(verbose):
        print("Computing Pearson residuals...")
    
    residuals = np.zeros(counts.shape, dtype=np.float32)
    
    for g in range(n_vars):
        mu = mu_models[:, g]
        theta = theta_estimates[g]
        
        # Variance attendue sous le modèle binomial négatif
        # var = mu + mu^2/theta
        expected_var = mu + (mu ** 2) / theta
        expected_var = np.maximum(expected_var, 1e-6)
        
        # Résidu de Pearson: (obs - expected) / sqrt(var)        
        residuals[:, g] = (counts[:, g].toarray().ravel() - mu) / np.sqrt(expected_var)
    
    # 7. Clipper les résidus
    if clip_range is None:
        clip_value = np.sqrt(n_obs)
        clip_range = (-clip_value, clip_value)
    
    residuals = np.clip(residuals, clip_range[0], clip_range[1])
    
    # 8. Identifier les gènes hautement variables
    residual_var = residuals.var(axis=0)
    top_genes_idx = np.argsort(residual_var)[-n_genes:]
    
    if(verbose):
        print(f"Done! Selected {n_genes} highly variable genes")
    
    return residuals, top_genes_idx, {
        'theta': theta_estimates,
        'mu_models': mu_models,
        'residual_variance': residual_var
    }

def update_data(data,n_neighbors=15,n_comps=100,var_names="X",random_state=42, normalization="sct",verbose=False):
    """
    Actualisation du jeu de données normalisées.
    Lorsqu'on modifie le tableau de données brutes (accessibles via data.raw.X), les données normalisées (accessibles via data.X) ne sont pas modifiées en conséquence.
    Pour que cela le soit, il faut appeler cette fonction.

    data: objet anndata/scanpy (dataframe)
    n_neighbors: pour le graphe kNN, dans l'actualisation du jeu de données.
    n_comps: nombre de composantes PCA
    var_names: où trouver le nom des gènes ?
    normalization: méthode de normalisation appliquée (par défaut "sct"). Possibilités: "sct", "log1p".
    """

    if var_names=="raw":
        gene_indices = data.raw.var.index.astype("int").tolist()
    elif var_names == "X":
        gene_indices = data.raw.var[data.raw.var['gene_index'].isin(data.var_names)].index.astype("int").tolist()
        
    temp_data = ad.AnnData(
        X=np.expm1(data.raw.X[:, gene_indices].copy()),
        obs=data.obs.copy(),
        var=data.raw.var.iloc[gene_indices].copy()
    )
    
    if normalization == "sct":
        residuals, top_genes, params = sctransform_manual(
            temp_data,
            n_cells=temp_data.n_obs,
            n_genes=len(gene_indices),
            verbose=verbose
        )
        data.X = residuals

    elif normalization == "log1p":
        sc.pp.normalize_total(temp_data)
        sc.pp.log1p(temp_data)
        #sc.pp.highly_variable_genes(temp_data, n_top_genes=len(gene_indices))
        #data.X = temp_data[:, temp_data.var['highly_variable']].X
        data.X = temp_data.X
        
    else:
        raise ValueError(f"normalization must be 'sct' or 'log1p', got '{normalization}'")

    sc.tl.pca(data, n_comps=n_comps)
    sc.pp.neighbors(data,n_neighbors=n_neighbors,n_pcs=n_comps)
    sc.tl.umap(data, random_state=random_state)

    return data

def plot_UMAP(data,save_path=None):
    """
    Affichage de la UMAP selon les étiquettes originales.
    'data' doit avoir tous les objets nécessaires à la réalisation de la UMAP déjà calculés.s

    data: objet anndata/scanpy (dataframe)
    save_path: str, chemin auquel enregistré la figure. Si None, pas de sauvegarde.
    """

    data.obs['celltype_label'] = data.obs['celltype'].map(CELLTYPE_MAP).astype('category')
    data.obs['histotype_label'] = data.obs['histotype'].map(HISTOTYPE_MAP).astype('category')
    

    if(save_path):
        full_path = os.path.join(FIGURES_PATH,save_path)
        os.makedirs(os.path.dirname(full_path),exist_ok=True)
        sc.settings.figdir = os.path.dirname(full_path)
        sc.settings.set_figure_params(dpi=300, format='pdf')
    

    if save_path:
        sc.pl.umap(
        data, 
        color=['celltype_label', 'histotype_label', 'cellstates_tme'],       
        legend_fontsize=8, 
        legend_fontoutline=2,
        ncols=2, 
        wspace=0.5,
        title=['Cell Types', 'Tumor Type', 'Microenvironment States'],
        save=os.path.basename(full_path)
        )
        
    else:
        sc.pl.umap(
        data, 
        color=['celltype_label', 'histotype_label', 'cellstates_tme'],       
        legend_fontsize=8, 
        legend_fontoutline=2,
        ncols=2, 
        wspace=0.5,
        title=['Cell Types', 'Tumor Type', 'Microenvironment States'],
        )


    plt.show()


def plot_custom_UMAP(data,labels,save_path=None):
    """
    Affichage de la UMAP selon les étiquettes originales et des étiquettes custom données via 'labels'.
    'data' doit avoir tous les objets nécessaires à la réalisation de la UMAP déjà calculés.

    data: objet anndata/scanpy (dataframe)
    labels: array/list de labels.
    save_path: str, chemin auquel enregistré la figure. Si None, pas de sauvegarde.
    """

    data.obs['celltype_label'] = data.obs['celltype'].map(CELLTYPE_MAP).astype('category')
    data.obs['histotype_label'] = data.obs['histotype'].map(HISTOTYPE_MAP).astype('category')


    # Custom UMAP Labels
    data.obs['custom_umap_label'] = labels
    data.obs['custom_umap_label'] = data.obs['custom_umap_label'].astype('category')

    cats = data.obs['custom_umap_label'].cat.categories
    random_colors = np.random.rand(len(cats),3)
    data.uns['custom_umap_label_colors'] = [tuple(c) for c in random_colors]

    if(save_path):
        full_path = os.path.join(FIGURES_PATH,save_path)
        os.makedirs(os.path.dirname(full_path),exist_ok=True)
        sc.settings.figdir = os.path.dirname(full_path)
        sc.settings.set_figure_params(dpi=300, format='pdf')
    

    if save_path:
        sc.pl.umap(
        data, 
        color=['celltype_label', 'histotype_label', 'cellstates_tme', 'custom_umap_label'],       
        legend_fontsize=8, 
        legend_fontoutline=2,
        ncols=2, 
        wspace=0.5,
        title=['Cell Types', 'Tumor Type', 'Microenvironment States', 'Custom Clustering'],
        save=os.path.basename(full_path)
        )
        
    else:
        sc.pl.umap(
        data, 
        color=['celltype_label', 'histotype_label', 'cellstates_tme', 'custom_umap_label'],       
        legend_fontsize=8, 
        legend_fontoutline=2,
        ncols=2, 
        wspace=0.5,
        title=['Cell Types', 'Tumor Type', 'Microenvironment States', 'Custom Clustering'],
        )


    plt.show()

def plot_new_UMAP(
    data,
    overwrite=False,
    save_path=None,
    n_comps=100,
    n_neighbors=None,
    n_pcs=None,
    use_rep=None,
    random_state=0
):
    """
    Calcul et affichage de la UMAP selon les étiquettes originales et des étiquettes custom données via 'labels'.

    data: objet anndata/scanpy (dataframe)
    overwrite: Si True alors les calculs de UMAP écrasent les données relatives à UMAP enregistrées dans 'data'.
    save_path: str, chemin auquel enregistré la figure. Si None, pas de sauvegarde.
    n_comps: nombre de composantes PCA (PCA nécessaire pour UMAP)
    n_neighbors: nombre de voisins pour le graphe kNN (graphe kNN nécessaire pour UMAP)
    n_pcs: nombre de composantes PCA utilisé pour le graphe kNN.
    use_rep: quelle représentation ? Par défaut, la PCA.
    random_state: seed pour calculs aléatoires
    """

    data.obs['celltype_label'] = data.obs['celltype'].map(CELLTYPE_MAP).astype('category')
    data.obs['histotype_label'] = data.obs['histotype'].map(HISTOTYPE_MAP).astype('category')

    # récupérer paramètres sauvegardés si non fournis
    neighbors_params = data.uns.get('neighbors', {}).get('params', {})
    final_n_neighbors = n_neighbors if n_neighbors is not None else neighbors_params.get('n_neighbors', 15)
    final_n_pcs = n_pcs if n_pcs is not None else neighbors_params.get('n_pcs')
    final_use_rep = use_rep if use_rep is not None else neighbors_params.get('use_rep')

    # sauvegarde éventuelle de l'UMAP existante
    umap_backup = None
    if not overwrite and 'X_umap' in data.obsm:
        umap_backup = data.obsm['X_umap'].copy()

    # recalcul TOUJOURS
    sc.tl.pca(data, n_comps=n_comps)
    sc.pp.neighbors(
        data,
        n_neighbors=final_n_neighbors,
        n_pcs=final_n_pcs if final_use_rep is None else None,
        use_rep=final_use_rep
    )

    sc.tl.umap(data, random_state=random_state)

    # gestion sauvegarde figure
    if save_path:
        full_path = os.path.join(FIGURES_PATH, save_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        sc.settings.figdir = os.path.dirname(full_path)
        sc.settings.set_figure_params(dpi=300, format='pdf')

    # affichage (identique à ta fonction)
    sc.pl.umap(
        data,
        color=['celltype_label', 'histotype_label', 'cellstates_tme'],
        legend_fontsize=8,
        legend_fontoutline=2,
        ncols=2,
        wspace=0.5,
        title=['Cell Types', 'Tumor Type', 'Microenvironment States'],
        save=os.path.basename(full_path) if save_path else None
    )

    plt.show()

    # restauration si overwrite=False
    if not overwrite:
        if umap_backup is not None:
            data.obsm['X_umap'] = umap_backup
        else:
            del data.obsm['X_umap']

        print("Nouvelle UMAP affichée mais NON sauvegardée (overwrite=False)")
    else:
        print("Nouvelle UMAP calculée et sauvegardée (overwrite=True)")


def gridsearch_leiden(
    adata,
    true_key,
    neighbors_grid,
    resolution_grid,
    n_pcs=100,
    n_runs=5,
    random_state=0
):
    """
    Quadrille l'espace (n_neighbors,resolution) à partir des deux grids 'neighbors_grid' et 'resolution_grid'
    et calcule les scores de partitionnement de 'data' en chaque point (n_neighbors,resolution).

    adata: objet anndata/scanpy (dataframe)
    true_key: nom de la colonne de adata qui correspond aux labels considérés comme ground truth. (Exemple: celltype_label)
    neighbors_grid: grille de neighbors candidats
    resolution_grid: grille de résolution candidates
    n_pcs: nombre de composantes PCA utilisé pour le graphe kNN.
    n_runs: nombre d'instances à lancer pour moyenner les scores de partitionnement.
    random_state: seed pour calculs aléatoires
    """
    
    true_labels = adata.obs[true_key]
    results = []
    
    for k in neighbors_grid:
        # Parcours n_neighbors
        sc.pp.neighbors(adata, n_neighbors=k, n_pcs=n_pcs)
        
        for res in resolution_grid:
            # Parcours resolution
            
            v_scores = []
            ari_scores = []
            
            # runs sur lesquels on moyennera les scores
            for seed in range(n_runs):
                sc.tl.leiden(
                    adata,
                    resolution=res,
                    key_added="leiden_temp",
                    random_state=random_state + seed
                )
                
                pred = adata.obs["leiden_temp"]
                
                _, _, v = homogeneity_completeness_v_measure(true_labels, pred)
                ari = adjusted_rand_score(true_labels, pred)
                
                v_scores.append(v)
                ari_scores.append(ari)
            
            results.append({
                "k": k,
                "resolution": res,
                "V_mean": np.mean(v_scores), # moyennes des scores suivant les runs
                "ARI_mean": np.mean(ari_scores),
                "V_std": np.std(v_scores),
                "ARI_std": np.std(ari_scores),
                "score": (np.mean(v_scores) + np.mean(ari_scores)) / 2
            })
    
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    return results[0], results