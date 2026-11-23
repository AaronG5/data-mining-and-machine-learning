"""k parinkimas: Elbow (WCSS), vidutinis siluetas ir empirinis sqrt(n/2)."""

import numpy as np
import pandas as pd
from bendra import K_REIKSMES, kmeans, wcss
from kneed import KneeLocator
from matai import vidiniai


def k_kreive(X):
   """Grąžina (lentelė k, WCSS, Silhouette; {k: žymės}). k-means++, n_init=10, random_state=42."""
   zymes = {k: kmeans(X, k)[0] for k in K_REIKSMES}
   lentele = pd.DataFrame(
      [{"k": k, "WCSS": wcss(X, z), "Silhouette": vidiniai(X, z)["Silhouette"]} for k, z in zymes.items()]
   )
   return lentele, zymes


def metodu_k(lentele, n):
   """Kiekvieno metodo siūlomas k."""
   return {
      "Elbow": KneeLocator(lentele["k"], lentele["WCSS"], curve="convex", direction="decreasing").knee,
      "Silhouette": int(lentele.loc[lentele["Silhouette"].idxmax(), "k"]),
      "Empirinis sqrt(n/2)": round(np.sqrt(n / 2)),
   }


def k_lentele(lentele, n):
   """k reikšmių lentelė, o apačioje atskirai kiekvieno metodo siūlomas k."""
   metodai = pd.DataFrame([{"metodas": m, "k": k} for m, k in metodu_k(lentele, n).items()])
   return pd.concat([lentele, metodai], ignore_index=True)
