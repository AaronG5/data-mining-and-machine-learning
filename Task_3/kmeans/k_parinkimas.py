"""k parinkimas: Elbow (paaiškinta dispersija %), vidutinis siluetas ir empirinis sqrt(n/2)."""

import numpy as np
import pandas as pd
from bendra import K_REIKSMES, kmeans, wcss
from kneed import KneeLocator
from matai import vidiniai

DISPERSIJA = "Paaiškinta dispersija %"


def paaiskinta_dispersija(X, zymes):
   """Klasterių paaiškinta dispersijos dalis procentais: (1 - WCSS / TSS) * 100,
   kur TSS - visų taškų kvadratinių atstumų iki bendro centro suma."""
   X = np.asarray(X)
   tss = ((X - X.mean(axis=0)) ** 2).sum()
   return 100 * (1 - wcss(X, zymes) / tss)


def k_kreive(X):
   """Grąžina (lentelė k, WCSS, paaiškinta dispersija %, Silhouette; {k: žymės}). k-means++, n_init=10, random_state=42."""
   zymes = {k: kmeans(X, k)[0] for k in K_REIKSMES}
   lentele = pd.DataFrame(
      [
         {
            "k": k,
            "WCSS": wcss(X, z),
            DISPERSIJA: paaiskinta_dispersija(X, z),
            "Silhouette": vidiniai(X, z)["Silhouette"],
         }
         for k, z in zymes.items()
      ]
   )
   return lentele, zymes


def metodu_k(lentele, n):
   """Kiekvieno metodo siūlomas k."""
   return {
      "Elbow": KneeLocator(lentele["k"], lentele[DISPERSIJA], curve="concave", direction="increasing").knee,
      "Silhouette": int(lentele.loc[lentele["Silhouette"].idxmax(), "k"]),
      "Empirinis sqrt(n/2)": round(np.sqrt(n / 2)),
   }


def k_lentele(lentele, n):
   """k reikšmių lentelė, o apačioje atskirai kiekvieno metodo siūlomas k."""
   metodai = pd.DataFrame([{"metodas": m, "k": k} for m, k in metodu_k(lentele, n).items()])
   return pd.concat([lentele, metodai], ignore_index=True)
