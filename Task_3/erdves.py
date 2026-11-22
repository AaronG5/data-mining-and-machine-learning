import os
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

FAILAS = (Path(__file__).resolve().parent / "apdoroti_duomenys"
          / "steel_plates_3_klases_be_koreliuotu_nenormuota.csv")

# random_state fiksuojamas specialiai: visi metodai turi matyti tą pačią UMAP erdvę,
# o rezultatai turi būti atkartojami
RANDOM_STATE = 42
# erdvė: (dimensijos mažinimo metodas, dimensijos)
ERDVES = {
    "originali": (None, None),
    "pca5": ("pca", 5),
    "pca2": ("pca", 2),
    "umap5": ("umap", 5),
    "umap2": ("umap", 2),
}
ERDVIU_PAVADINIMAI = {"originali": "Originali (16D)", "pca5": "PCA (5D)", "pca2": "PCA (2D)",
                      "umap5": "UMAP (5D)", "umap2": "UMAP (2D)"}
# Tie patys UMAP parametrai kaip dimensijos_mazinimas/UMAP/umap_tyrimas.py
UMAP_PARAMETRAI = dict(n_neighbors=25, min_dist=0.05, metric="euclidean")
# Robust normavimas taikomas visiems požymiams, išskyrus šiuos
NENORMUOJAMI = ["Outside_Global_Index"]

warnings.filterwarnings("ignore", message="n_jobs value .* overridden")


class Duomenys(NamedTuple):
    pozymiai_nenormuoti: pd.DataFrame
    klasiu_zymes: np.ndarray
    klasiu_pavadinimai: pd.Index


def rez_katalogas(erdve):
    kelias = f"res/{erdve}"
    os.makedirs(kelias, exist_ok=True)
    return kelias


def ikelti():
    duomenys = pd.read_csv(FAILAS)
    klase = duomenys["class"].astype("category")
    return Duomenys(pozymiai_nenormuoti=duomenys.drop(columns="class"),
                    klasiu_zymes=klase.cat.codes.values,
                    klasiu_pavadinimai=klase.cat.categories)


#Robust scaling
def normuoti(df):
    df = df.copy()
    normuojami = df.columns.drop(NENORMUOJAMI, errors="ignore")
    mediana = df[normuojami].median()
    iqr = df[normuojami].quantile(0.75) - df[normuojami].quantile(0.25)
    df[normuojami] = (df[normuojami] - mediana) / iqr.replace(0, 1)
    return df


def projektuoti(X, metodas, dimensijos, random_state=RANDOM_STATE):
    if metodas is None:
        return X.to_numpy()
    if metodas == "pca":
        return PCA(n_components=dimensijos, random_state=random_state).fit_transform(X)
    if metodas == "umap":
        return umap.UMAP(n_components=dimensijos, random_state=random_state,
                         compatibility_layout=True, **UMAP_PARAMETRAI).fit_transform(X)
    raise ValueError(f"Nežinomas metodas: {metodas}")


def normuoti_ir_projektuoti(X_nenorm, erdve, random_state=RANDOM_STATE):
    return projektuoti(normuoti(X_nenorm), *ERDVES[erdve], random_state=random_state)


def visos_erdves(X_nenorm):
    X = normuoti(X_nenorm)
    return {erdve: projektuoti(X, metodas, dimensijos)
            for erdve, (metodas, dimensijos) in ERDVES.items()}
