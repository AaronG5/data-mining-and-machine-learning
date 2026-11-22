import numpy as np
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.metrics.cluster import pair_confusion_matrix


def jaccard_poru(a, b):
    (_, fp), (fn, tp) = pair_confusion_matrix(a, b)
    return tp / (tp + fp + fn)


def isoriniai(a, b):
    return {"Jaccard": jaccard_poru(a, b),
            "ARI": adjusted_rand_score(a, b),
            "NMI": normalized_mutual_info_score(a, b)}


def vidiniai(X, klasteriu_zymes):
    if len(set(klasteriu_zymes)) < 2:
        return {"Silhouette": np.nan, "Calinski-Harabasz": np.nan, "Davies-Bouldin": np.nan}
    return {"Silhouette": silhouette_score(X, klasteriu_zymes),
            "Calinski-Harabasz": calinski_harabasz_score(X, klasteriu_zymes),
            "Davies-Bouldin": davies_bouldin_score(X, klasteriu_zymes)}


def matai(X, klasteriu_zymes, klasiu_zymes):
    return {**vidiniai(X, klasteriu_zymes), **isoriniai(klasiu_zymes, klasteriu_zymes)}
