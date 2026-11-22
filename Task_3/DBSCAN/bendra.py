"""Bendros funkcijos: duomenys, erdvės (originali / PCA / UMAP), kelio eps, DBSCAN, matai, stabilumas."""
import os
import warnings

import numpy as np
import pandas as pd
import umap
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.neighbors import NearestNeighbors

# Be ekstremalių išskirčių ir koreliuotų požymių, nenormuota (normuojama normuoti())
FAILAS = "../apdoroti_duomenys/steel_plates_3_klases_be_koreliuotu_nenormuota.csv"

RANDOM_STATE = 42
# PCA 5D (87,9 % dispersijos) – tiek pat dimensijų kaip UMAP 5D
ERDVES = ["originali", "pca5", "pca2", "umap5", "umap2"]
# Tie patys UMAP parametrai kaip ../dimensijos_mazinimas/UMAP/umap_tyrimas.py
UMAP_PARAMETRAI = dict(n_neighbors=25, min_dist=0.05, metric="euclidean")

# Tiriami min_samples (Sander et al.: min_samples ≈ 2·D)
MIN_SAMPLES = {
    "originali": [5, 10, 16, 32],
    "pca5": [5, 10, 16, 32],
    "pca2": [4, 10, 25, 50],
    "umap5": [5, 10, 25, 50],
    "umap2": [4, 10, 25, 50],
}
# Pasirinktas variantas: (min_samples, eps daugiklis kelio eps atžvilgiu)
PASIRINKTI = {
    "originali": (5, 1.0),
    "pca5": (10, 1.0),
    "pca2": (10, 1.0),
    "umap5": (25, 1.0),
    "umap2": (25, 1.0),
}
ERDVIU_PAVADINIMAI = {"originali": "Originali (16D)", "pca5": "PCA (5D)", "pca2": "PCA (2D)",
                      "umap5": "UMAP (5D)", "umap2": "UMAP (2D)"}


def rez_katalogas(erdve):
    kelias = f"res/{erdve}"
    os.makedirs(kelias, exist_ok=True)
    return kelias


def ikelti():
    """Grąžina (nenormuoti požymiai, klasių kodai, klasių pavadinimai)."""
    duomenys = pd.read_csv(FAILAS)
    klase = duomenys["class"].astype("category")
    return duomenys.drop(columns="class"), klase.cat.codes.values, klase.cat.categories


def normuoti(df):
    """Robust normavimas kaip ../dimensijos_mazinimas/paruosimas.py; Outside_Global_Index nenormuojamas."""
    df = df.copy()
    poz = df.columns.drop("Outside_Global_Index", errors="ignore")
    iqr = df[poz].quantile(0.75) - df[poz].quantile(0.25)
    df[poz] = (df[poz] - df[poz].median()) / iqr.replace(0, 1)
    return df


def erdves_matrica(X_nenorm, erdve, random_state=RANDOM_STATE):
    """Normuoja ir projektuoja į erdvę, kurioje vykdomas DBSCAN."""
    X = normuoti(X_nenorm)
    if erdve == "originali":
        return X.to_numpy()
    d = int(erdve[-1])
    if erdve.startswith("pca"):
        return PCA(n_components=d, random_state=random_state).fit_transform(X)
    warnings.filterwarnings("ignore", message="n_jobs value .* overridden")
    return umap.UMAP(n_components=d, random_state=random_state, compatibility_layout=True,
                     **UMAP_PARAMETRAI).fit_transform(X)


def visos_erdves(X_nenorm):
    return {erdve: erdves_matrica(X_nenorm, erdve) for erdve in ERDVES}


def kelio_eps(X, min_samples):
    """Grąžina (surikiuoti k-atstumai, kelio indeksas, eps)."""
    # kneighbors grąžina ir patį tašką, kaip ir DBSCAN min_samples jį įskaičiuoja
    atst = np.sort(NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)[0][:, -1])
    kelias = KneeLocator(np.arange(len(atst)), atst, curve="convex", direction="increasing").knee
    return atst, kelias, atst[kelias]


def dbscan_kelio(X, min_samples, daugiklis=1.0):
    """DBSCAN su eps = kelio eps × daugiklis. Grąžina (žymės, eps)."""
    eps = kelio_eps(X, min_samples)[2] * daugiklis
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X), eps


def jaccard_poru(a, b):
    (_, fp), (fn, tp) = pair_confusion_matrix(a, b)
    return tp / (tp + fp + fn)


def isoriniai(a, b, priesaga=""):
    return {f"Jaccard{priesaga}": jaccard_poru(a, b),
            f"ARI{priesaga}": adjusted_rand_score(a, b),
            f"NMI{priesaga}": normalized_mutual_info_score(a, b)}


def vidiniai(X, zymes):
    """Klasterių sk., triukšmas ir vidiniai matai (be triukšmo taškų)."""
    be_triuksmo = zymes != -1
    n_klast = len(set(zymes[be_triuksmo]))
    rez = {"klasteriai": n_klast, "triukšmas": int((~be_triuksmo).sum()),
           "triukšmas %": 100 * (~be_triuksmo).mean()}
    if n_klast >= 2:
        Xk, zk = X[be_triuksmo], zymes[be_triuksmo]
        rez["Silhouette"] = silhouette_score(Xk, zk)
        rez["Calinski-Harabasz"] = calinski_harabasz_score(Xk, zk)
        rez["Davies-Bouldin"] = davies_bouldin_score(Xk, zk)
    else:
        rez["Silhouette"] = rez["Calinski-Harabasz"] = rez["Davies-Bouldin"] = np.nan
    return rez


def matai(X, zymes, tikros):
    """Vidiniai (be triukšmo) ir išoriniai (triukšmas – atskira grupė) matai."""
    return {**vidiniai(X, zymes), **isoriniai(tikros, zymes)}


def imtys(n, kiekis=20, dalis=0.8):
    """Imtis i traukiama su random_state = i."""
    return [np.sort(np.random.default_rng(i).choice(n, int(dalis * n), replace=False))
            for i in range(kiekis)]


def stabilumas(X_nenorm, erdve, variantai, imciu_sarasas):
    """Kiekviena imtis atskirai normuojama, projektuojama (random_state = i) ir klasterizuojama;
    imčių poros lyginamos tik bendriems objektams.
    Grąžina {(min_samples, daugiklis): (porų lentelė, imčių lentelė)}."""
    zymes = {v: np.full((len(imciu_sarasas), len(X_nenorm)), np.nan) for v in variantai}
    imciu_eilutes = {v: [] for v in variantai}
    for i, idx in enumerate(imciu_sarasas):
        Xv = erdves_matrica(X_nenorm.iloc[idx], erdve, random_state=i)
        for v in variantai:
            z, eps = dbscan_kelio(Xv, *v)
            zymes[v][i, idx] = z
            imciu_eilutes[v].append({"imtis": i, "eps": eps, **vidiniai(Xv, z)})
    rez = {}
    for v, z in zymes.items():
        poros = []
        for i in range(len(z)):
            for j in range(i + 1, len(z)):
                bendri = ~np.isnan(z[i]) & ~np.isnan(z[j])
                poros.append({"i": i, "j": j, **isoriniai(z[i, bendri], z[j, bendri])})
        rez[v] = (pd.DataFrame(poros), pd.DataFrame(imciu_eilutes[v]))
    return rez
