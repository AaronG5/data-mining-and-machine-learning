"""Bendros funkcijos: duomenų įkėlimas, klasterizavimo erdvės (originali / PCA / UMAP),
kelio (knee) eps paieška, pasirinktas DBSCAN variantas ir kokybės matai.

Erdvės (3 užd. 17 skaidrė: geriausias 2 užd. metodas + PCA kaip globali atskaita):
  originali – 16 normuotų požymių (atskaitos suskirstymas),
  pca       – globalus metodas, komponentės išlaikančios ≥ 80 % dispersijos,
  umap      – lokalus metodas, geriausias 2 užduotyje (tie patys parametrai)."""
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

FAILAS = "../apdoroti_duomenys/steel_plates_3_klases_normuota_robust.csv"
# Tie patys objektai ta pačia tvarka, tik nenormuoti – profiliams originaliais vienetais
FAILAS_NENORMUOTA = "../apdoroti_duomenys/steel_plates_3_klases_be_koreliuotu_nenormuota.csv"

RANDOM_STATE = 42
ERDVES = ["originali", "pca", "umap"]
PCA_DISPERSIJA = 0.80  # → 4 komponentės (83,7 %)
# Tie patys UMAP parametrai kaip ../dimensijos_mazinimas/UMAP/umap_tyrimas.py
UMAP_PARAMETRAI = dict(n_neighbors=25, min_dist=0.05, n_components=2, metric="euclidean")

# Tiriami min_samples kiekvienai erdvei (Sander et al.: min_samples ≈ 2·D)
MIN_SAMPLES = {
    "originali": [5, 10, 16, 32],  # D=16
    "pca": [4, 8, 16, 32],         # D=4
    "umap": [4, 10, 25, 50],       # D=2
}
# Pasirinktas variantas kiekvienai erdvei: (min_samples, eps daugiklis kelio eps atžvilgiu).
# Pagrindimas – dbscan_eksperimentai.py ir dbscan_stabilumas.py rezultatai.
PASIRINKTI = {
    "originali": (5, 1.0),
    "pca": (8, 1.0),
    "umap": (25, 1.0),
}
ERDVIU_PAVADINIMAI = {"originali": "Originali (16D)", "pca": "PCA (4D)", "umap": "UMAP (2D)"}


def rez_katalogas(erdve):
    kelias = f"res/{erdve}"
    os.makedirs(kelias, exist_ok=True)
    return kelias


def ikelti(failas=FAILAS):
    duomenys = pd.read_csv(failas)
    klase = duomenys["class"].astype("category")
    return duomenys.drop(columns="class"), klase.cat.codes.values, klase.cat.categories


# --- Erdvės -----------------------------------------------------------------
def pca_projekcija(X, n_components=PCA_DISPERSIJA):
    return PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(X)


def umap_projekcija(X):
    warnings.filterwarnings("ignore", message="n_jobs value .* overridden")
    return umap.UMAP(random_state=RANDOM_STATE, compatibility_layout=True,
                     **UMAP_PARAMETRAI).fit_transform(X)


def visos_erdves(X):
    """{erdvė: matrica, kurioje vykdomas DBSCAN}."""
    return {"originali": np.asarray(X), "pca": pca_projekcija(X), "umap": umap_projekcija(X)}


# --- Kelio metodas ir DBSCAN ------------------------------------------------
def kelio_eps(X, min_samples):
    """Grąžina (surikiuoti k-atstumai, kelio indeksas, eps)."""
    # DBSCAN min_samples įskaičiuoja ir patį tašką; kneighbors taip pat grąžina patį tašką (atstumas 0),
    # todėl paskutinis stulpelis = atstumas iki (min_samples-1)-ojo kaimyno.
    atst = np.sort(NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)[0][:, -1])
    kelias = KneeLocator(np.arange(len(atst)), atst, curve="convex", direction="increasing").knee
    return atst, kelias, atst[kelias]


def pasirinktas_dbscan(Xv, erdve):
    """Pasirinktas tos erdvės variantas: grąžina (žymės, eps, min_samples)."""
    ms, daugiklis = PASIRINKTI[erdve]
    eps = kelio_eps(Xv, ms)[2] * daugiklis
    return DBSCAN(eps=eps, min_samples=ms).fit_predict(Xv), eps, ms


# --- Kokybės matai ----------------------------------------------------------
def jaccard_poru(a, b):
    # Porų Jaccard: TP / (TP + FP + FN), kur TP – poros, kurios abiejuose suskirstymuose kartu.
    (_, fp), (fn, tp) = pair_confusion_matrix(a, b)
    return tp / (tp + fp + fn)


def isoriniai(a, b, priesaga=""):
    return {f"Jaccard{priesaga}": jaccard_poru(a, b),
            f"ARI{priesaga}": adjusted_rand_score(a, b),
            f"NMI{priesaga}": normalized_mutual_info_score(a, b)}


def imtys(n, kiekis=50, dalis=0.8):
    """`kiekis` atsitiktinių imčių po `dalis` objektų (be grąžinimo), fiksuotas RANDOM_STATE."""
    rng = np.random.default_rng(RANDOM_STATE)
    return [np.sort(rng.choice(n, int(dalis * n), replace=False)) for _ in range(kiekis)]


def stabilumas(Xv, eps, min_samples, imciu_sarasas):
    """Kiekviena imtis klasterizuojama tais pačiais parametrais; visų imčių poros lyginamos
    tik bendriems objektams. Triukšmas (-1) – atskira grupė, todėl perėjimas
    klasteris <-> triukšmas laikomas nestabilumu.
    Grąžina (porų matų lentelė, klasterių skaičius kiekvienoje imtyje)."""
    zymes = np.full((len(imciu_sarasas), len(Xv)), np.nan)
    klasteriu_sk = []
    for i, idx in enumerate(imciu_sarasas):
        zymes[i, idx] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xv[idx])
        klasteriu_sk.append(len(set(zymes[i, idx]) - {-1}))
    poros = []
    for i in range(len(zymes)):
        for j in range(i + 1, len(zymes)):
            bendri = ~np.isnan(zymes[i]) & ~np.isnan(zymes[j])
            poros.append(isoriniai(zymes[i, bendri], zymes[j, bendri]))
    return pd.DataFrame(poros), klasteriu_sk


def matai(X, zymes, tikros):
    """Vidiniai matai skaičiuojami be triukšmo taškų (-1 nėra klasteris).
    Išoriniai – visiems objektams, triukšmą laikant atskira grupe."""
    be_triuksmo = zymes != -1
    n_klast = len(set(zymes[be_triuksmo]))
    rez = {
        "klasteriai": n_klast,
        "triukšmas": int((~be_triuksmo).sum()),
        "triukšmas %": 100 * (~be_triuksmo).mean(),
    }
    if n_klast >= 2:
        Xk, zk = X[be_triuksmo], zymes[be_triuksmo]
        rez["Silhouette"] = silhouette_score(Xk, zk)
        rez["Calinski-Harabasz"] = calinski_harabasz_score(Xk, zk)
        rez["Davies-Bouldin"] = davies_bouldin_score(Xk, zk)
    else:
        rez["Silhouette"] = rez["Calinski-Harabasz"] = rez["Davies-Bouldin"] = np.nan
    return {**rez, **isoriniai(tikros, zymes)}
