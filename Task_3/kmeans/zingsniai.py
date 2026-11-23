"""k-means 1-5 žingsniai. Paleidimas: python zingsniai.py (visi) arba python zingsniai.py 3 4 5."""

import sys

import matplotlib.pyplot as plt
import pandas as pd
from bendra import (
   PASIRINKTAS_K,
   PASIRINKTI,
   SUMAZINTOS,
   TVARKA,
   issaugoti_zymes,
   kmeans,
   parametrai_tekstu,
   perkelimo_lentele,
   rez_katalogas,
   statistikos,
   suderinti,
   sutapimo_lentele,
   vid_sn,
)
from erdves import ERDVIU_PAVADINIMAI, ikelti, visos_erdves
from grafikai import issaugoti, k_kreives, piesti, silueto_grafikas, spalva
from k_parinkimas import k_kreive, k_lentele, metodu_k
from matai import isoriniai, vidiniai
from parametrai import parametru_lentele
from stabilumas import imciu_pakankamumas, imtys, palyginti_imciu_poras, projektuoti_imtis

N_TIKRINIMUI = [5, 10, 15, 20]


def skaityti_labels_ref():
   return pd.read_csv(rez_katalogas(2) / "labels_ref_original.csv")["klasteris"].to_numpy()


def zingsnis1(erdves):
   """1. k parinkimas originalioje erdvėje (k = 2-10)."""
   rez = rez_katalogas(1)
   X = erdves["originali"]
   lentele, zymes = k_kreive(X)
   k_lentele(lentele, len(X)).to_csv(rez / "k_selection_original.csv", index=False)
   print(k_lentele(lentele, len(X)).to_string(index=False))

   _, asys = plt.subplots(1, 2, figsize=(12, 4.5))
   k_kreives(*asys, lentele, metodu_k(lentele, len(X)), PASIRINKTAS_K["originali"])  # pyright: ignore[reportCallIssue]
   plt.suptitle(f"{ERDVIU_PAVADINIMAI['originali']}: optimalaus k parinkimas")
   issaugoti(rez / "k_selection.png")

   _, asys = plt.subplots(2, 2, figsize=(12, 10))
   for ax, k in zip(asys.ravel(), [2, 3, 4, 5]):
      silueto_grafikas(ax, X, zymes[k], f"k = {k}")
   plt.suptitle(f"{ERDVIU_PAVADINIMAI['originali']}: silueto diagramos")
   issaugoti(rez / "silhouette.png")


def zingsnis2(erdves, X_nenorm, kodai, klases):
   """2. k-means originalioje erdvėje: parametrų lentelė, labels_ref, projekcijos, statistikos."""
   rez = rez_katalogas(2)
   lentele, labels_ref = parametru_lentele(erdves["originali"], PASIRINKTAS_K["originali"], "originali")
   lentele.to_csv(rez / "parameters_original.csv", index=False)
   print(lentele.to_string(index=False))
   issaugoti_zymes(rez / "labels_ref_original.csv", labels_ref, kodai, klases)
   statistikos(X_nenorm, labels_ref).to_csv(rez / "statistics_original.csv")

   _, asys = plt.subplots(2, 2, figsize=(12, 10))
   for eil, proj in zip(asys, ["PCA", "UMAP"]):
      Z = erdves[f"{proj.lower()}2"]
      piesti(eil[0], Z, labels_ref, f"k-means klasteriai ({proj} 2D)", proj)
      piesti(eil[1], Z, kodai, f"Tikrosios klasės ({proj} 2D)", proj, klases)
   plt.suptitle(f"{ERDVIU_PAVADINIMAI['originali']}: k-means {parametrai_tekstu('originali')}")
   issaugoti(rez / "projection_original.png")


def zingsnis3(erdves, X_nenorm, kodai, klases):
   """3. k-means sumažintose erdvėse (PCA ir UMAP, 2D ir 5D). Reikia 2 žingsnio labels_ref."""
   rez = rez_katalogas(3)
   labels_ref = skaityti_labels_ref()

   k_lenteles = [
      k_lentele(k_kreive(erdves[e])[0], len(X_nenorm)).assign(erdvė=ERDVIU_PAVADINIMAI[e]) for e in SUMAZINTOS
   ]
   pd.concat(k_lenteles).to_csv(rez / "k_selection_reduced.csv", index=False)

   zymes = {"originali": labels_ref}
   for erdve in SUMAZINTOS:
      lentele, z = parametru_lentele(erdves[erdve], PASIRINKTAS_K[erdve], erdve)
      lentele.to_csv(rez / f"parameters_{erdve}.csv", index=False)
      zymes[erdve], atitikmenys = suderinti(labels_ref, z)
      issaugoti_zymes(rez / f"labels_{erdve}.csv", zymes[erdve], kodai, klases)
      statistikos(X_nenorm, zymes[erdve]).to_csv(rez / f"statistics_{erdve}.csv")
      sutapimas = sutapimo_lentele(labels_ref, zymes[erdve], atitikmenys)
      sutapimas.to_csv(rez / f"matching_clusters_{erdve}.csv")
      perkelimo_lentele(labels_ref, zymes[erdve], sutapimas).to_csv(rez / f"separated_clusters_{erdve}.csv")
      print(f"{ERDVIU_PAVADINIMAI[erdve]}:\n{sutapimas}\n")

   for proj in ["PCA", "UMAP"]:
      Z = erdves[f"{proj.lower()}2"]
      _, asys = plt.subplots(2, 2, figsize=(12, 10))
      for ax, d in zip(asys[0], [2, 5]):
         piesti(ax, Z, zymes[f"{proj.lower()}{d}"], f"k-means klasteriai, rasti {proj} {d}D", proj)
      piesti(asys[1, 0], Z, labels_ref, "Originalios erdvės klasteriai (labels_ref)", proj)
      piesti(asys[1, 1], Z, kodai, "Tikrosios klasės", proj, klases)
      plt.suptitle(f"k-means {proj} erdvėse ({proj} 2D projekcija)")
      issaugoti(rez / f"projection_{proj.lower()}.png")

   # Originali ir PCA rodomos PCA 2D projekcijoje, UMAP - UMAP 2D
   _, asys = plt.subplots(1, len(TVARKA), figsize=(4.6 * len(TVARKA), 4.8))
   for ax, erdve in zip(asys, TVARKA):
      Z, z = erdves["umap2" if erdve.startswith("umap") else "pca2"], zymes[erdve]
      ax.scatter(Z[:, 0], Z[:, 1], s=6, color=[spalva(v) for v in z])
      ax.set_title(
         f"{ERDVIU_PAVADINIMAI[erdve]}\nk={len(set(z))}  "
         f"Sil={vidiniai(erdves[erdve], z)['Silhouette']:.2f}  "
         f"ARI(atsk.)={isoriniai(labels_ref, z)['ARI']:.2f}",
         fontsize=10,
      )
      ax.set_xticks([])
      ax.set_yticks([])
   plt.suptitle("k-means rezultatai skirtingose erdvėse", fontweight="bold")
   issaugoti(rez / "clustering.png")


def zingsnis4(X_nenorm):
   """4. Stabilumas: 20 imčių po 80 %, kiekviena atskirai normuojama, projektuojama ir
   klasterizuojama (random_state = i); imčių poros lyginamos tik bendriems objektams."""
   rez = rez_katalogas(4)
   imciu_sarasas = imtys(len(X_nenorm), 20)

   santrauka, poros = [], {}
   _, ax = plt.subplots(figsize=(7, 4.5))
   for erdve in TVARKA:
      zymes, vid = [], []
      for i, Xi in enumerate(projektuoti_imtis(X_nenorm, erdve, imciu_sarasas)):
         zymes.append(kmeans(Xi, PASIRINKTAS_K[erdve], *PASIRINKTI[erdve], seed=i)[0])
         vid.append(vidiniai(Xi, zymes[-1]))
      vid = pd.DataFrame(vid)
      p = poros[ERDVIU_PAVADINIMAI[erdve]] = palyginti_imciu_poras(imciu_sarasas, zymes)
      santrauka.append(
         {
            "erdvė": ERDVIU_PAVADINIMAI[erdve],
            **{m: vid_sn(vid[m]) for m in vid},
            **{m: vid_sn(p[m]) for m in ["ARI", "NMI", "Jaccard"]},
            "stabilumas (ARI vid ± SN)": vid_sn(p["ARI"]),
         }
      )

      pakank = pd.DataFrame(imciu_pakankamumas(p, N_TIKRINIMUI))
      ax.errorbar(
         pakank["N"], pakank["ARI vid"], yerr=pakank["ARI SN"], marker="o", capsize=3, label=ERDVIU_PAVADINIMAI[erdve]
      )
   ax.set(
      xlabel="Imčių skaičius N",
      ylabel="ARI vidurkis ± SN",
      xticks=N_TIKRINIMUI,
      title="Imčių skaičiaus pakankamumas (k-means)",
   )
   ax.grid(alpha=0.3)
   ax.legend(fontsize=8)
   issaugoti(rez / "stability_sufficiency.png")

   santrauka = pd.DataFrame(santrauka)
   santrauka.to_csv(rez / "stability_metrics.csv", index=False)
   print(santrauka.to_string(index=False))

   _, ax = plt.subplots(figsize=(8, 4.2))
   ax.boxplot([p["ARI"] for p in poros.values()], tick_labels=list(poros), orientation="horizontal")
   ax.set(title="k-means stabilumas: ARI tarp 20 × 80 % imčių porų (bendri objektai)", xlabel="ARI")
   ax.grid(alpha=0.3, axis="x")
   issaugoti(rez / "stability.png")


def zingsnis5(erdves):
   """5. Rezultatų suvestinė iš 2-4 žingsnių CSV failų."""
   rez, rez3 = rez_katalogas(5), rez_katalogas(3)
   stabilumas = pd.read_csv(rez_katalogas(4) / "stability_metrics.csv").set_index("erdvė")
   zymes = {
      "originali": skaityti_labels_ref(),
      **{e: pd.read_csv(rez3 / f"labels_{e}.csv")["klasteris"].to_numpy() for e in SUMAZINTOS},
   }

   eksperimentai = pd.DataFrame(
      [
         {
            "erdvė": ERDVIU_PAVADINIMAI[e],
            "parametrai": parametrai_tekstu(e),
            "k": PASIRINKTAS_K[e],
            **vidiniai(erdves[e], zymes[e]),
            **isoriniai(zymes["originali"], zymes[e]),
            "stabilumas (ARI vid ± SN)": stabilumas.loc[ERDVIU_PAVADINIMAI[e], "stabilumas (ARI vid ± SN)"],
         }
         for e in TVARKA
      ]
   ).round(3)
   eksperimentai.to_csv(rez / "experiments.csv", index=False)
   print(eksperimentai.to_string(index=False))

   sutapimai = pd.DataFrame(
      {
         ERDVIU_PAVADINIMAI[e]: pd.read_csv(rez3 / f"matching_clusters_{e}.csv", index_col=0)["sutampa %"]
         for e in SUMAZINTOS
      }
   ).T
   sutapimai.columns = [f"klasteris {c} sutampa %" if c != "viso" else "viso sutampa %" for c in sutapimai.columns]
   sutapimai.rename_axis("erdvė").to_csv(rez / "matching_clusters.csv")

   perkelimai = pd.DataFrame(
      {
         ERDVIU_PAVADINIMAI[e]: pd.read_csv(rez3 / f"separated_clusters_{e}.csv", index_col=0)["perkelta %"]
         for e in SUMAZINTOS
      }
   ).T
   perkelimai.columns = [f"iš klasterio {c} perkelta %" for c in perkelimai.columns]
   perkelimai.rename_axis("erdvė").to_csv(rez / "separated_clusters.csv")
   print(f"\n{sutapimai}\n\n{perkelimai}")


if __name__ == "__main__":
   zingsniai = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5]
   X_nenorm, kodai, klases = ikelti()
   erdves = visos_erdves(X_nenorm)
   if 1 in zingsniai:
      zingsnis1(erdves)
   if 2 in zingsniai:
      zingsnis2(erdves, X_nenorm, kodai, klases)
   if 3 in zingsniai:
      zingsnis3(erdves, X_nenorm, kodai, klases)
   if 4 in zingsniai:
      zingsnis4(X_nenorm)
   if 5 in zingsniai:
      zingsnis5(erdves)
