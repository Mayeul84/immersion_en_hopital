import numpy as np
from collections import Counter

from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import adjusted_rand_score

from sklearn.metrics import confusion_matrix

# Ce script répertorie toutes les fonctions qui permettent de scorer les partitionnements leiden vis à vis de labels ground truth (les labels originaux qu'on aimerait obtenir).

def balanced_correctcells_score(true_labels, cluster_labels):
    """
    On considère qu'un cluster a un label correspondant au label ground truth majoritaire.
    On calcule ensuite la fraction de bonne classification de chaque cluster: n_majority / n_total.
    La fonction retourne la moyenne des fractions.

    true_labels: array/list des labels ground truth.
    cluster_labels: array/list des labels 
    """
    conf_mat = confusion_matrix(true_labels, cluster_labels)
    
    # fraction "bien classifiée" par cluster détecté (majorité dans chaque colonne)
    max_per_col = conf_mat.max(axis=0)
    dominant_label_per_cluster = conf_mat.argmax(axis=0)  # index du true_label majoritaire par cluster
    cluster_sizes = conf_mat.sum(axis=0)
    
    fraction_per_col = np.divide(max_per_col, cluster_sizes,
                                 out=np.zeros_like(max_per_col, dtype=float),
                                 where=cluster_sizes!=0)
    
    # Grouper les fractions par true_label assigné, puis moyenner
    unique_labels = np.unique(dominant_label_per_cluster)
    fraction_per_label = {}
    for label in unique_labels:
        mask = dominant_label_per_cluster == label
        fraction_per_label[label] = fraction_per_col[mask].mean()
    
    # Moyenne finale sur les true_labels
    balanced_score = np.mean(list(fraction_per_label.values()))
    
    return balanced_score, fraction_per_label


def compute_all_scores(true_labels,cluster_labels):
    """
    Calcule l'ensemble des scores intéressants pour nos études, c'est à dire:
        - homogéneité
        - complétude
        - v (moyenne harmonique des deux précédentes)
        - 'balanced_correctcells_score' total et par cluster.
        - ARI
    """

    h, c, v = homogeneity_completeness_v_measure(true_labels,cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    correct, correct_detailed = balanced_correctcells_score(true_labels,cluster_labels)

    scores = {
        "homogeneity": h,
        "completness": c,
        "correct": correct,
        "correct_detailed": correct_detailed,
        "v": v,
        "ari": ari,
    }

    return scores