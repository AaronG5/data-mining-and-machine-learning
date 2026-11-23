"""k-means parametrų lentelė: VARIANTAI x 10 sėklų (0-9)."""

import numpy as np
import pandas as pd
from bendra import PASIRINKTI, SEKLOS, VARIANTAI, kmeans, vid_sn, wcss
from matai import isoriniai, vidiniai
from stabilumas import palyginti_imciu_poras


def parametru_lentele(X, k, erdve):
   """Grąžina (lentelė, atskaitos žymės). Pirma lentelė be atskaitos stulpelių, tada atskaita
   (PASIRINKTI parametrai, random_state=42), tada atskaitos stulpeliai."""
   visi = np.arange(len(X))
   eilutes, zymes = [], {}
   for v in VARIANTAI:
      paleidimai = [kmeans(X, k, *v, seed=s) for s in SEKLOS]
      zymes[v] = [z for z, _ in paleidimai]
      vid = pd.DataFrame([vidiniai(X, z) for z in zymes[v]])
      # Visi objektai kiekviename paleidime, todėl lyginamos visos 45 sėklų poros
      poros = palyginti_imciu_poras([visi] * len(SEKLOS), zymes[v])
      eilutes.append(
         {
            "atstumas": v[0],
            "init": v[1],
            "n_init": v[2],
            "WCSS": vid_sn([wcss(X, z) for z in zymes[v]]),
            **{m: vid_sn(vid[m]) for m in vid},
            "ARI tarp sėklų": vid_sn(poros["ARI"]),
            "NMI tarp sėklų": vid_sn(poros["NMI"]),
            "iteracijos vid": np.mean([it for _, it in paleidimai]),
         }
      )
   lentele = pd.DataFrame(eilutes)

   atskaita = kmeans(X, k, *PASIRINKTI[erdve])[0]
   for m in ["ARI", "NMI"]:
      lentele[f"{m} su atskaita"] = [vid_sn([isoriniai(atskaita, z)[m] for z in zymes[v]]) for v in VARIANTAI]
   lentele["pasirinktas"] = [v == PASIRINKTI[erdve] for v in VARIANTAI]
   return lentele, atskaita
