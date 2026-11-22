"""Bendros funkcijos: duomenų įkėlimas, klasterizavimo erdvės (originali / PCA / UMAP),
kelio (knee) eps paieška, pasirinktas DBSCAN variantas ir kokybės matai.

Erdvės (3 užd. 17–18 skaidrės: geriausias 2 užd. metodas + PCA kaip globali atskaita,
kiekvienas dviem dimensijomis):
  originali – 16 normuotų požymių (atskaitos suskirstymas),
  pca5      – globalus metodas, 5 komponentės (87,9 % dispersijos; tiek pat dimensijų kaip UMAP 5D),
  pca2      – PCA 2 komponentės (68,5 % dispersijos),
  umap5     – lokalus metodas, geriausias 2 užduotyje, 5 komponentės,
  umap2     – tas pats UMAP, 2 komponentės (kaip 2 užduotyje)."""
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
ERDVES = ["originali", "pca5", "pca2", "umap5", "umap2"]
PCA_KOMPONENTES = 5  # 87,9 % dispersijos, kaip UMAP 5D – vienodas dimensijų skaičius palyginimui
# Tie patys UMAP parametrai kaip ../dimensijos_mazinimas/UMAP/umap_tyrimas.py (keičiamas tik n_components)
UMAP_PARAMETRAI = dict(n_neighbors=25, min_dist=0.05, metric="euclidean")

# Tiriami min_samples kiekvienai erdvei (Sander et al.: min_samples ≈ 2·D)
MIN_SAMPLES = {
    "originali": [5, 10, 16, 32],  # D=16
    "pca5": [5, 10, 16, 32],       # D=5
    "pca2": [4, 10, 25, 50],       # D=2
    "umap5": [5, 10, 25, 50],      # D=5
    "umap2": [4, 10, 25, 50],      # D=2
}
# Pasirinktas variantas kiekvienai erdvei: (min_samples, eps daugiklis kelio eps atžvilgiu).
# Pagrindimas – dbscan_eksperimentai.py ir dbscan_stabilumas.py rezultatai.
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


def ikelti(failas=FAILAS):
    duomenys = pd.read_csv(failas)
    klase = duomenys["class"].astype("category")
    return duomenys.drop(columns="class"), klase.cat.codes.values, klase.cat.categories


# --- Normavimas ir erdvės ---
def normuoti(df):
    """Robust normavimas kaip ../dimensijos_mazinimas/paruosimas.py: (x - mediana) / IQR
    (IQR = 0 -> 1); Outside_Global_Index nenormuojamas. Statistikos – tik iš pateiktų eilučių."""
    df = df.copy()
    poz = df.columns.drop("Outside_Global_Index", errors="ignore")
    iqr = df[poz].quantile(0.75) - df[poz].quantile(0.25)
    df[poz] = (df[poz] - df[poz].median()) / iqr.replace(0, 1)
    return df


# --- Erdvės -----------------------------------------------------------------
def pca_projekcija(X, n_components=PCA_KOMPONENTES, random_state=RANDOM_STATE):
    return PCA(n_components=n_components, random_state=random_state).fit_transform(X)


def umap_projekcija(X, n_components=2, random_state=RANDOM_STATE):
    warnings.filterwarnings("ignore", message="n_jobs value .* overridden")
    return umap.UMAP(n_components=n_components, random_state=random_state,
                     compatibility_layout=True, **UMAP_PARAMETRAI).fit_transform(X)


def erdves_matrica(X, erdve, random_state=RANDOM_STATE):
    """Normuotų duomenų matrica toje erdvėje, kurioje vykdomas DBSCAN."""
    if erdve == "originali":
        return np.asarray(X)
    metodas, d = erdve[:-1], int(erdve[-1])
    projekcija = pca_projekcija if metodas == "pca" else umap_projekcija
    return projekcija(X, d, random_state=random_state)


def visos_erdves(X):
    """{erdvė: matrica, kurioje vykdomas DBSCAN}."""
    return {erdve: erdves_matrica(X, erdve) for erdve in ERDVES}


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


def imtys(n, kiekis=20, dalis=0.8):
    """`kiekis` atsitiktinių imčių po `dalis` objektų (be grąžinimo). Imtis i traukiama su
    random_state = i (0, 1, …, N−1): skirtingos, bet užfiksuotos – rezultatai atkartojami."""
    return [np.sort(np.random.default_rng(i).choice(n, int(dalis * n), replace=False))
            for i in range(kiekis)]


def stabilumas(X_nenorm, erdve, variantai, imciu_sarasas):
    """Stabilumas, kai kiekviena imtis apdorojama nuo pradžių: imama iš NEnormuotų duomenų,
    normuojama (robust) tik pagal tos imties statistikas, jai iš naujo skaičiuojama projekcija
    (PCA / UMAP) ir kelio eps; tada DBSCAN. Taip į imtį nepatenka informacija iš likusių 20 %.
    Visų imčių poros lyginamos tik bendriems objektams. Triukšmas (-1) – atskira grupė, todėl
    perėjimas klasteris <-> triukšmas laikomas nestabilumu.
    variantai – [(min_samples, eps daugiklis), ...].
    Imčiai i projekcijos random_state = i.
    Grąžina {variantas: (porų matų lentelė su imčių indeksais i, j;
                         imčių lentelė: eps, klasteriai, triukšmas %, Silhouette, CH, DB)}."""
    X_nenorm = pd.DataFrame(X_nenorm).reset_index(drop=True)
    zymes = {v: np.full((len(imciu_sarasas), len(X_nenorm)), np.nan) for v in variantai}
    imciu_eilutes = {v: [] for v in variantai}
    for i, idx in enumerate(imciu_sarasas):
        Xv = erdves_matrica(normuoti(X_nenorm.iloc[idx]), erdve, random_state=i)
        for ms, dg in variantai:
            eps = kelio_eps(Xv, ms)[2] * dg
            z = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xv)
            zymes[(ms, dg)][i, idx] = z
            imciu_eilutes[(ms, dg)].append({"imtis": i, "eps": eps, **vidiniai(Xv, z)})
    rez = {}
    for v, z in zymes.items():
        poros = []
        for i in range(len(z)):
            for j in range(i + 1, len(z)):
                bendri = ~np.isnan(z[i]) & ~np.isnan(z[j])
                poros.append({"i": i, "j": j, **isoriniai(z[i, bendri], z[j, bendri])})
        rez[v] = (pd.DataFrame(poros), pd.DataFrame(imciu_eilutes[v]))
    return rez


def vidiniai(X, zymes):
    """Klasterių sk., triukšmas ir vidiniai matai (be triukšmo taškų: -1 nėra klasteris)."""
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
    return rez


def matai(X, zymes, tikros):
    """Vidiniai matai – be triukšmo taškų; išoriniai – visiems objektams, triukšmą laikant
    atskira grupe."""
    return {**vidiniai(X, zymes), **isoriniai(tikros, zymes)}
