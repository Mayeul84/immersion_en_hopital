import numpy as np
import scipy.sparse as sp
import scanpy as sc
import anndata as ad

import pandas as pd
from typing import List

# Ce script répertorie toutes les fonctions qui permettent de sous-échantillonner les gènes et/ou les cellules d'un jeu de données anndata.

def neyman_subsample(
    data: ad.AnnData,
    target_labels: List[str],
    stratify_by: List[str],
    n_target: int = None,
    n_target_total: int = None,
    variance_col: str = "nCount_SCT",
    label_col: str = "celltype_label",
    seed: int = 42,
) -> ad.AnnData:
    """
    Sous-échantillonne les cellules appartenant à certains labels de `label_col`
    via une allocation de Neyman (On tire un certain nombre de samples proportionnellement à la variancede chaque strate), stratifiée sur une ou plusieurs colonnes obs.
    Les cellules hors `target_labels` sont conservées intégralement.
    Retourne un nouvel objet AnnData avec les cellules cibles sous-échantillonnées
        et les autres cellules intactes.
    
    data : AnnData
        Objet AnnData source.
    target_labels : list of str
        Labels dans `label_col` à sous-échantillonner.
        Ex: ["Steroid cells", "T cells"]
    stratify_by : list of str
        Colonnes de data.obs utilisées pour définir les strata.
        Ex: ["cellstates_tme", "histotype_label"]
    n_target : int, optional
        Nombre de cellules à garder PAR label dans `target_labels`.
        Mutuellement exclusif avec `n_target_total`.
    n_target_total : int, optional
        Nombre TOTAL de cellules à garder pour l'ensemble des `target_labels`.
        Mutuellement exclusif avec `n_target`.
    variance_col : str, default "nCount_SCT"
        Colonne obs utilisée comme proxy de variance pour l'allocation de Neyman.
    label_col : str, default "celltype_label"
        Colonne obs contenant les labels cibles.
    seed : int, default 42
        Graine pour la reproductibilité du sampling.

    ValueError
        Si n_target et n_target_total sont tous les deux fournis ou absents,
        ou si un label de `target_labels` n'existe pas dans `label_col`,
        ou si une colonne de `stratify_by` n'existe pas dans obs,
        ou si `n_target`/`n_target_total` dépasse le nombre de cellules disponibles.
    """

    # Validation n_target / n_target_total
    if (n_target is None) == (n_target_total is None):
        raise ValueError(
            "Exactly one of `n_target` or `n_target_total` must be provided."
        )

    rng = np.random.default_rng(seed)
    obs = data.obs.copy()

    # Validation des entrées
    missing_labels = set(target_labels) - set(obs[label_col].unique())
    if missing_labels:
        raise ValueError(
            f"Labels introuvables dans obs['{label_col}']: {missing_labels}\n"
            f"Labels disponibles: {sorted(obs[label_col].unique())}"
        )

    missing_cols = set(stratify_by) - set(obs.columns)
    if missing_cols:
        raise ValueError(
            f"Colonnes introuvables dans obs: {missing_cols}\n"
            f"Colonnes disponibles: {list(obs.columns)}"
        )

    if variance_col not in obs.columns:
        raise ValueError(
            f"Colonne de variance '{variance_col}' introuvable dans obs.\n"
            f"Colonnes disponibles: {list(obs.columns)}"
        )

    # Indices des cellules NON cibles (conservées intégralement)
    # Lors du déroulement de l'algorithme idx_keep va s'extend pour contenir toutes les cellules restantes après notre subsampling
    # (Voir la fonction _run_neyman)
    mask_non_target = ~obs[label_col].isin(target_labels)
    idx_keep = list(obs.index[mask_non_target])

    # Résolution du n_target effectif
    subset = obs.loc[obs[label_col].isin(target_labels)]

    if n_target_total is not None:
        # Mode total : on travaille sur l'ensemble des target_labels en une passe
        _run_neyman(
            subset=subset,
            stratify_by=stratify_by,
            n_target=n_target_total,
            variance_col=variance_col,
            target_labels=target_labels,
            rng=rng,
            idx_keep=idx_keep,
        )
    else:
        # Mode par label : une passe de Neyman par label
        for label in target_labels:
            label_subset = obs.loc[obs[label_col] == label]
            _run_neyman(
                subset=label_subset,
                stratify_by=stratify_by,
                n_target=n_target,
                variance_col=variance_col,
                target_labels=[label],
                rng=rng,
                idx_keep=idx_keep,
            )

    return data[idx_keep]


def _run_neyman(
    subset: pd.DataFrame,
    stratify_by: List[str],
    n_target: int,
    variance_col: str,
    target_labels: List[str],
    rng: np.random.Generator,
    idx_keep: list,
) -> None:
    """
    Effectue l'allocation de Neyman sur 'subset' et étend 'idx_keep' in-place pour qu'à la fin 'idx_keep' contienne toutes les cellules samplées (+ celles qu'on ne devait pas toucher)
    """
    if n_target > len(subset):
        print(
            f"n_target={n_target} dépasse le nombre de cellules disponibles "
            f"pour '{target_labels}' ({len(subset)})."
        )
        n_target = len(subset)

    subset = subset.copy()
    subset["_stratum"] = (
        subset[stratify_by].astype(str).agg("__".join, axis=1)
    )

    stratum_stats = (
        subset.groupby("_stratum")[variance_col]
        .agg(["std", "count"])
        .rename(columns={"count": "n_stratum"})
    )
    stratum_stats["std"] = stratum_stats["std"].fillna(0.0)

    # Poids de sampling
    stratum_stats["weight"] = stratum_stats["n_stratum"] * stratum_stats["std"]
    total_weight = stratum_stats["weight"].sum()
    
    if total_weight == 0:
        stratum_stats["allocation"] = (
            stratum_stats["n_stratum"] / stratum_stats["n_stratum"].sum() * n_target
        )
    else:
        stratum_stats["allocation"] = (
            stratum_stats["weight"] / total_weight * n_target
        )

    stratum_stats["allocation_int"] = np.floor(stratum_stats["allocation"]).astype(int)
    remainder = n_target - stratum_stats["allocation_int"].sum()

    residuals = stratum_stats["allocation"] - stratum_stats["allocation_int"]
    top_idx = residuals.nlargest(int(remainder)).index
    stratum_stats.loc[top_idx, "allocation_int"] += 1

    stratum_stats["allocation_int"] = stratum_stats[
        ["allocation_int", "n_stratum"]
    ].min(axis=1)

    # On extend idx_keep en lui donnant les nouvelles cellules samplées par strata
    for stratum_name, row in stratum_stats.iterrows():
        n_sample = int(row["allocation_int"])
        stratum_cells = subset.loc[subset["_stratum"] == stratum_name]

        if n_sample == 0:
            continue

        sampled_idx = rng.choice(stratum_cells.index, size=n_sample, replace=False)
        idx_keep.extend(sampled_idx.tolist())


def thinning_novec(adata, reduction_ratio=0.25, same_reads=False, copy=True):
    """
    Thinning + re-séquençage : même nombre de reads. Le thinning s'applique seulement sur les données brutes.
    Pour obtenir un tableau normalisé post thinning, il faudra passer ensuite sur la fonction 'update_data'.

    Le thinning ne se fait pas vectoriellement (d'où le 'novec'). Moins efficace.

    adata: objet Anndata/Scanpy
    reduction_ratio: le ratio de sparsité i.e la probabilité à laquelle on pioche un read.
    same_reads: on force le tableau à avoir le même nombre de reads.
    copy: artefact de code
    """        
    
    X = adata.raw.X.expm1().copy() if hasattr(adata.X, 'toarray') else np.expm1(adata.raw.X).copy()
    n_cells, n_genes = X.shape

    X_new = sp.lil_matrix(X.shape,dtype=X.dtype)
    #R_target = int(X.sum(axis=1).mean())

    for c in range(X.shape[0]):
        # Étape 1 : Thinning (capture)
        Y = np.random.binomial(np.round(X[c,:].data).astype(int), reduction_ratio)
        cols_indices = X[c,:].indices

        # Étape 2 : Re-séquençage (même reads/cellule)
        if same_reads:
            L_c = sum(Y)
            R_target = sum(X[c,:].data)
            if L_c > 0:
                p_g = Y / L_c
                X_new[c, cols_indices] = np.random.multinomial(R_target, p_g).tolist()

        else:
            X_new[c,cols_indices] = Y
    
    # On efface les objets inutiles
    del X
    del Y

    # Nouvelle donnée brute
    for i in range(len(X_new.data)):
        X_new.data[i] = np.log1p(X_new.data[i]).tolist()


    res_adata = adata.copy()
    res_adata.raw = sc.AnnData(
        X=X_new.tocsr(),
        var=adata.raw.var.copy(),
    )

    del X_new

    return res_adata



def thinning(adata, reduction_ratio=0.25, same_reads=False, copy=True):
    """
    Thinning + re-séquençage : même nombre de reads. Le thinning s'applique seulement sur les données brutes.
    Pour obtenir un tableau normalisé post thinning, il faudra passer ensuite sur la fonction 'update_data'.

    adata: objet Anndata/Scanpy
    reduction_ratio: le ratio de sparsité i.e la probabilité à laquelle on pioche un read.
    same_reads: on force le tableau à avoir le même nombre de reads.
    copy: artefact de code
    """      

    X = adata.raw.X.copy()
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr()

    # Inverser log1p (counts bruts)
    X.data = np.expm1(X.data)

    # Thinning vectorisé
    # Binomial sur tous les non-zéros d'un coup
    X_new = X.copy()
    X_new.data = np.random.binomial(
        np.round(X.data).astype(int),
        reduction_ratio
    ).astype(X.data.dtype)

    if same_reads:
        # Re-séquençage par cellule, difficile à vectoriser complètement
        # mais on évite les conversions dense
        X_new = X_new.tolil()
        for c in range(X.shape[0]):
            row = X_new.data[c]
            if len(row) == 0:
                continue
            L_c = sum(row)
            if L_c > 0:
                R_target = X[c].data.sum()
                p_g = np.array(row) / L_c
                X_new.data[c] = np.random.multinomial(R_target, p_g).tolist()
        X_new = X_new.tocsr()

    # log1p vectorisé sur les non-zéros uniquement
    X_new = X_new.tocsr()
    X_new.data = np.log1p(X_new.data)

    # Mettre à zéro les éléments nuls après thinning
    X_new.eliminate_zeros()

    res_adata = adata.copy() if copy else adata
    res_adata.raw = sc.AnnData(
        X=X_new,
        var=adata.raw.var.copy(),
    )

    return res_adata