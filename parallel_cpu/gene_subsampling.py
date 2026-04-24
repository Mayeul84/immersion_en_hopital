import numpy as np
import scipy.sparse as sp
import scanpy as sc
import anndata as ad

from typing import List

def neyman_subsample(
    data: ad.AnnData,
    target_labels: List[str],
    stratify_by: List[str],
    n_target: int,
    variance_col: str = "nCount_SCT",
    label_col: str = "celltype_label",
    seed: int = 42,
) -> ad.AnnData:
    """
    Sous-échantillonne les cellules appartenant à certains labels de `label_col`
    via une allocation de Neyman, stratifiée sur une ou plusieurs colonnes obs.
    Les cellules hors `target_labels` sont conservées intégralement.

    Parameters
    ----------
    data : AnnData
        Objet AnnData source.
    target_labels : list of str
        Labels dans `label_col` à sous-échantillonner.
        Ex: ["Steroid cells", "T cells"]
    stratify_by : list of str
        Colonnes de data.obs utilisées pour définir les strata.
        Ex: ["cellstates_tme", "histotype_label"]
    n_target : int
        Nombre total de cellules à garder par label dans `target_labels`.
    variance_col : str, default "nCount_SCT"
        Colonne obs utilisée comme proxy de variance pour l'allocation de Neyman.
    label_col : str, default "celltype_label"
        Colonne obs contenant les labels cibles.
    seed : int, default 42
        Graine pour la reproductibilité du sampling.

    Returns
    -------
    AnnData
        Nouvel objet AnnData avec les cellules cibles sous-échantillonnées
        et les autres cellules intactes.

    Raises
    ------
    ValueError
        Si un label de `target_labels` n'existe pas dans `label_col`,
        ou si une colonne de `stratify_by` n'existe pas dans obs,
        ou si `n_target` dépasse le nombre de cellules disponibles pour un label.
    """

    rng = np.random.default_rng(seed)
    obs = data.obs.copy()

    # --- Validation des entrées ---
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

    # --- Indices des cellules NON cibles (conservées intégralement) ---
    mask_non_target = ~obs[label_col].isin(target_labels)
    idx_keep = list(obs.index[mask_non_target])

    # --- Allocation de Neyman par label cible ---
    #for label in target_labels:
    subset = obs.loc[obs[label_col].isin(target_labels)]

    if n_target > len(subset):
        # raise ValueError(
        #     f"n_target={n_target} dépasse le nombre de cellules "
        #     f"disponibles pour '{target_labels}' ({len(subset)})."
        # )
        print(f"n_target={n_target} dépasse le nombre de cellules disponibles pour '{target_labels}' ({len(subset)}).")
        n_target=len(subset)

    # Création des strata via produit cartésien des colonnes stratify_by
    subset["_stratum"] = (
        subset[stratify_by].astype(str).agg("__".join, axis=1)
    )

    # Calcul de la std du proxy de variance par stratum
    stratum_stats = (
        subset.groupby("_stratum")[variance_col]
        .agg(["std", "count"])
        .rename(columns={"count": "n_stratum"})
    )
    # Strata avec std=NaN (un seul élément) → on met std=0
    stratum_stats["std"] = stratum_stats["std"].fillna(0.0)

    # Poids de Neyman : n_i * sigma_i
    stratum_stats["weight"] = (
        stratum_stats["n_stratum"] * stratum_stats["std"]
    )
    total_weight = stratum_stats["weight"].sum()

    # Cas edge : si tous les std sont 0 → allocation proportionnelle simple
    if total_weight == 0:
        stratum_stats["allocation"] = (
            stratum_stats["n_stratum"] / stratum_stats["n_stratum"].sum()
            * n_target
        )
    else:
        stratum_stats["allocation"] = (
            stratum_stats["weight"] / total_weight * n_target
        )

    # Arrondi avec correction pour atteindre exactement n_target
    stratum_stats["allocation_int"] = np.floor(
        stratum_stats["allocation"]
    ).astype(int)
    remainder = n_target - stratum_stats["allocation_int"].sum()

    # On distribue le reste aux strata avec les plus grands résidus fractionnaires
    residuals = stratum_stats["allocation"] - stratum_stats["allocation_int"]
    top_idx = residuals.nlargest(int(remainder)).index
    stratum_stats.loc[top_idx, "allocation_int"] += 1

    # Clamp : on ne peut pas sampler plus que ce qui existe dans un stratum
    stratum_stats["allocation_int"] = stratum_stats[
        ["allocation_int", "n_stratum"]
    ].min(axis=1)

    # --- Sampling dans chaque stratum ---
    for stratum_name, row in stratum_stats.iterrows():
        n_sample = int(row["allocation_int"])
        stratum_cells = subset.loc[subset["_stratum"] == stratum_name]

        if n_sample == 0:
            continue

        sampled_idx = rng.choice(
            stratum_cells.index, size=n_sample, replace=False
        )
        idx_keep.extend(sampled_idx.tolist())

    # --- Retour du nouvel AnnData ---
    return data[idx_keep]


def thinning_novec(adata, reduction_ratio=0.25, same_reads=False, copy=True):
    """
    Thinning + re-séquençage : même nombre de reads, η plus faible.
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
    
    del X
    del Y

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

    X = adata.raw.X.copy()
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr()

    # Inverser log1p → counts bruts
    X.data = np.expm1(X.data)

    # ---- Thinning vectorisé ----
    # Binomial sur tous les non-zéros d'un coup
    X_new = X.copy()
    X_new.data = np.random.binomial(
        np.round(X.data).astype(int),
        reduction_ratio
    ).astype(X.data.dtype)

    if same_reads:
        # Re-séquençage par cellule — difficile à vectoriser complètement
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