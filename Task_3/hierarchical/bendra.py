"""Hierarchinio klasterizavimo nustatymai ir pagalbinės funkcijos."""
import itertools
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering

sys.path.append(str(Path(__file__).resolve().parent.parent))
from erdves import ERDVIU_PAVADINIMAI, RANDOM_STATE, UMAP_PARAMETRAI, ikelti, visos_erdves  # noqa: E402,F401
from matai import isoriniai, vidiniai  # noqa: E402,F401

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

REZ = Path(__file__).resolve().parent / "res"
REZ.mkdir(exist_ok=True)

K_RIBOS = range(2, 11)
IMTYS, IMTIES_DALIS, N_TIKRINIMUI = 20, 0.8, [5, 10, 15, 20]
ISORINIAI = ["ARI", "NMI", "Jaccard"]
VIDINIAI = ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]
MAZIAUSIAS_KLASTERIS = 0.01  # mažesnis klasteris (dalis objektų) – variantas išsigimęs

# erdvė: (jungimas, atstumas, k)
PASIRINKTI = {
    "originali": ("ward", "euclidean", 3),
    "pca5": ("ward", "euclidean", 3),
    "pca2": ("ward", "euclidean", 3),
    "umap5": ("average", "euclidean", 6),
    "umap2": ("average", "euclidean", 6),
}

JUNGIMAI = {"ward": "Ward", "complete": "tolimiausio kaimyno", "average": "vidutinio ryšio",
            "single": "artimiausio kaimyno"}
ATSTUMAI = {"euclidean": "euklidinis", "cityblock": "Manheteno", "cosine": "kosinusinis"}


def jungimo_variantai():
    """Visi jungimo ir atstumo deriniai; Ward apibrėžtas tik euklidiniam atstumui."""
    return [(j, a) for j, a in itertools.product(JUNGIMAI, ATSTUMAI) if j != "ward" or a == "euclidean"]


def klasteriai(Z, jungimas, atstumas, k):
    return AgglomerativeClustering(n_clusters=k, linkage=jungimas, metric=atstumas).fit_predict(Z)


def vardas(jungimas, atstumas, k=None):
    tekstas = f"{JUNGIMAI[jungimas]} ({ATSTUMAI[atstumas]})"
    return tekstas if k is None else f"{tekstas}, k={k}"


def dydziai(zymes):
    return "/".join(map(str, np.sort(np.bincount(zymes))[::-1]))


def vid_sn(reiksmes):
    return f"{reiksmes.mean():.3f} ± {reiksmes.std():.3f}"


def suderinti(atskaita, zymes):
    """Klasteriai pernumeruojami pagal atskaitą (Vengrų metodas); be poros likę gauna tolesnius numerius."""
    lentele = pd.crosstab(zymes, atskaita)
    eil, stulp = linear_sum_assignment(-lentele.values)
    atitikmenys = dict(zip(lentele.index[eil], lentele.columns[stulp]))
    likusieji = lentele.index.difference(list(atitikmenys))
    atitikmenys |= {z: atskaita.max() + 1 + i for i, z in enumerate(likusieji)}
    return np.array([atitikmenys[z] for z in zymes])


def visu_erdviu_klasteriai(erdves):
    """Pasirinkto varianto žymės kiekvienoje erdvėje, suderintos su originalios erdvės rezultatu (atskaita)."""
    atskaita = klasteriai(erdves["originali"], *PASIRINKTI["originali"])
    return atskaita, {e: suderinti(atskaita, klasteriai(Z, *PASIRINKTI[e])) for e, Z in erdves.items()}


def k_kreive(Z, jungimas, atstumas):
    """Vidiniai matai ir dendrogramos sujungimų aukščiai bei šuoliai, kai k = 2–10."""
    aukstis = linkage(Z, method=jungimas, metric=atstumas)[::-1, 2]
    return pd.DataFrame({k: {**vidiniai(Z, klasteriai(Z, jungimas, atstumas, k)),
                             "sujungimo aukštis": aukstis[k - 2], "šuolis": aukstis[k - 2] - aukstis[k - 1]}
                         for k in K_RIBOS}).T.rename_axis("k")


def issaugoti(failas):
    plt.tight_layout()
    plt.savefig(REZ / failas, dpi=150)
    plt.close()
