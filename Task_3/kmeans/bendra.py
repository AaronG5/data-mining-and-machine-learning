"""Bendros k-means funkcijos: k-means paleidimas, WCSS, klasterių numerių suderinimas,
sutapimo lentelės ir aprašomosios statistikos."""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from erdves import RANDOM_STATE

REZ = Path(__file__).resolve().parent / "res"
ZINGSNIAI = {1: "1_k_selection", 2: "2_original", 3: "3_reduced", 4: "4_stability", 5: "5_results"}

TVARKA = ["originali", "pca2", "pca5", "umap2", "umap5"]
SUMAZINTOS = TVARKA[1:]

K_REIKSMES = list(range(2, 11))
SEKLOS = list(range(10))
VARIANTAI = list(product(["euclidean", "cosine"], ["k-means++", "random"], [1, 5, 10, 15, 20]))

# Pasirinkta pagal k_selection_*.csv (k) ir parameters_*.csv (atstumas, init, n_init)
PASIRINKTAS_K = {"originali": 3, "pca2": 3, "pca5": 3, "umap2": 5, "umap5": 5}
PASIRINKTI = {erdve: ("euclidean", "k-means++", 20) for erdve in TVARKA}


def rez_katalogas(zingsnis):
   kelias = REZ / ZINGSNIAI[zingsnis]
   kelias.mkdir(parents=True, exist_ok=True)
   return kelias


def kmeans(X, k, atstumas="euclidean", init="k-means++", n_init=10, seed=RANDOM_STATE):
   """Grąžina (žymės, iteracijų sk.). Kosinusinis variantas: eilutės normuojamos iki vienetinio ilgio."""
   Xk = normalize(X) if atstumas == "cosine" else X
   modelis = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=seed).fit(Xk)  # pyright: ignore[reportArgumentType]
   return modelis.labels_, modelis.n_iter_


def wcss(X, zymes):
   return sum(((X[zymes == z] - X[zymes == z].mean(axis=0)) ** 2).sum() for z in np.unique(zymes))


def vid_sn(x):
   return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}"


def parametrai_tekstu(erdve):
   atstumas, init, n_init = PASIRINKTI[erdve]
   return f"k={PASIRINKTAS_K[erdve]}, {atstumas}, init={init}, n_init={n_init}"


def issaugoti_zymes(kelias, zymes, kodai, klasiu_pavadinimai):
   pd.DataFrame({"id": np.arange(len(zymes)), "klasteris": zymes, "klasė": klasiu_pavadinimai[kodai]}).to_csv(
      kelias, index=False
   )


def suderinti(ref, zymes):
   """Vengrų algoritmu sunumeruoja klasterius pagal ref. Klasteriai be poros gauna numerius nuo
   max(ref) + 1 ir priskiriami labiausiai persidengiančiam ref klasteriui (skilimas = sutapimas).
   Grąžina (suderintos žymės, atitikmenys {suderintas klasteris: ref klasteris})."""
   ref_kl, kl = np.unique(ref), np.unique(zymes)
   kryz = pd.crosstab(zymes, ref).reindex(index=kl, columns=ref_kl, fill_value=0).to_numpy()
   eilutes, stulpeliai = linear_sum_assignment(-kryz)
   naujas = {int(kl[e]): int(ref_kl[s]) for e, s in zip(eilutes, stulpeliai)}
   atitikmenys = {v: v for v in naujas.values()}
   kitas = int(ref_kl.max()) + 1
   # Nuo didžiausio, kad numeriai sutaptų tarp erdvių
   for i in np.argsort(-kryz.sum(axis=1), kind="stable"):
      if kl[i] not in naujas:
         naujas[int(kl[i])] = kitas
         atitikmenys[kitas] = int(ref_kl[kryz[i].argmax()])
         kitas += 1
   return np.array([naujas[z] for z in zymes]), atitikmenys


def sutapimo_lentele(ref, suderintos, atitikmenys):
   """Kiek kiekvieno originalaus klasterio objektų liko tame pačiame klasteryje."""
   sutampa = np.array([atitikmenys[z] for z in suderintos]) == ref
   lentele = pd.Series(sutampa).groupby(ref).agg(objektu="size", sutampa="sum")
   lentele.loc["viso"] = lentele.sum()
   lentele["skiriasi"] = lentele["objektu"] - lentele["sutampa"]  # pyright: ignore[reportIndexIssue]
   lentele["sutampa %"] = (100 * lentele["sutampa"] / lentele["objektu"]).round(2)  # pyright: ignore[reportIndexIssue]
   return lentele.rename_axis("originalus klasteris")


def perkelimo_lentele(ref, suderintos, sutapimas):
   """Į kuriuos klasterius pateko originalių klasterių objektai ir kiek perkelta."""
   lentele = pd.crosstab(pd.Series(ref, name="originalus klasteris"), pd.Series(suderintos, name="klasteris"))
   lentele["perkelta"] = sutapimas["skiriasi"]
   lentele["perkelta %"] = (100 - sutapimas["sutampa %"]).round(2)
   return lentele


def statistikos(X_nenorm, zymes):
   """Mediana, vidurkis, IQR ir SN originaliais vienetais: visai aibei ir kiekvienam klasteriui."""
   grupes = {"visi": X_nenorm, **{f"klasteris {z}": X_nenorm[zymes == z] for z in np.unique(zymes)}}
   return pd.concat(
      {
         g: pd.DataFrame(
            {
               "mediana": df.median(),
               "vidurkis": df.mean(),
               "IQR": df.quantile(0.75) - df.quantile(0.25),
               "SN": df.std(),
            }
         )
         for g, df in grupes.items()
      },
      names=["grupė", "požymis"],
   )
