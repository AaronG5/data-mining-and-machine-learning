"""5. Erdvių palyginimas (3 užd. 16 ir 18 skaidrės): pasirinktas DBSCAN originalioje, PCA ir UMAP
erdvėse. Išoriniai matai skaičiuojami dviem būdais:
  • su tikromis klasėmis,
  • su originalios erdvės DBSCAN rezultatu (atskaitos suskirstymu) – ar dimensijos mažinimas
    pakeitė klasterizavimą.
Pastaba (17 skaidrė): UMAP erdvėje vidiniai matai išauga dirbtinai – jų su kitomis erdvėmis nelyginti."""
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd

from bendra import (ERDVES, ERDVIU_PAVADINIMAI, RANDOM_STATE, ikelti, imtys, isoriniai, matai,
                    pasirinktas_dbscan, pca_projekcija, rez_katalogas, stabilumas, visos_erdves)

N_IMCIU, DALIS = 50, 0.8
REZ = rez_katalogas("palyginimas")

X, kodai, klases = ikelti()
erdves = visos_erdves(X)
imciu_sarasas = imtys(len(X), N_IMCIU, DALIS)

zymes, eilutes = {}, []
for erdve in ERDVES:
    Xv = erdves[erdve]
    zymes[erdve], eps, ms = pasirinktas_dbscan(Xv, erdve)
    stab, _ = stabilumas(Xv, eps, ms, imciu_sarasas)
    m = matai(Xv, zymes[erdve], kodai)
    eilutes.append({
        "Erdvė": ERDVIU_PAVADINIMAI[erdve], "Algoritmas": "DBSCAN",
        "Parametrai": f"eps={eps:.3f}, min_samples={ms}",
        "k": m["klasteriai"], "triukšmas %": m["triukšmas %"],
        "Silhouette ↑": m["Silhouette"], "CH ↑": m["Calinski-Harabasz"], "DB ↓": m["Davies-Bouldin"],
        "ARI (klasės)": m["ARI"], "NMI (klasės)": m["NMI"], "Jaccard (klasės)": m["Jaccard"],
        **isoriniai(zymes["originali"], zymes[erdve], " (atskaita)"),
        "Stabilumas ARI": f"{stab['ARI'].mean():.3f} ± {stab['ARI'].std():.3f}",
    })

lentele = pd.DataFrame(eilutes)
lentele.to_csv(f"{REZ}/eksperimentu_lentele.csv", index=False)
pd.set_option("display.width", 250)
print(f"Data: {date.today()}, random_state={RANDOM_STATE}, stabilumas: {N_IMCIU} × {DALIS:.0%} imtys")
print("(atskaita) – lyginama su originalios erdvės DBSCAN rezultatu\n")
print(lentele.round(3).T.to_string(header=False))

# Sutampančių taškų klasteriuose skaičius: originalios erdvės klasteriai vs sumažintos erdvės
for erdve in ERDVES[1:]:
    kryz = pd.crosstab(pd.Series(zymes["originali"], name="originali"),
                       pd.Series(zymes[erdve], name=erdve))
    kryz.to_csv(f"{REZ}/sutapimas_originali_vs_{erdve}.csv")
    print(f"\nSutampantys taškai: originali (eilutės) vs {ERDVIU_PAVADINIMAI[erdve]} (stulpeliai)\n{kryz}")

# Vizualizacija kaip 18 skaidrėje: originali ir PCA – PCA 2D projekcijoje, UMAP – UMAP 2D
rodymas = {"originali": pca_projekcija(X, 2), "pca": pca_projekcija(X, 2), "umap": erdves["umap"]}
_, asys = plt.subplots(1, len(ERDVES), figsize=(5.5 * len(ERDVES), 5.2))
for ax, erdve, e in zip(asys, ERDVES, eilutes):
    Z, z = rodymas[erdve], zymes[erdve]
    triuksmas = z == -1
    ax.scatter(Z[triuksmas, 0], Z[triuksmas, 1], s=10, c="lightgrey", marker="x")
    ax.scatter(Z[~triuksmas, 0], Z[~triuksmas, 1], s=6, c=z[~triuksmas], cmap="tab10",
               vmin=0, vmax=9)
    ax.set_title(ERDVIU_PAVADINIMAI[erdve], fontweight="bold")
    ax.text(0.02, 0.97, f"k={e['k']}  Sil={e['Silhouette ↑']:.2f}  ARI(atsk.)={e['ARI (atskaita)']:.2f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle("DBSCAN rezultatai skirtingose erdvėse", fontweight="bold")
plt.figtext(0.5, 0.01, "Originali ir PCA 4D – PCA 2D projekcijoje; UMAP 2D – UMAP projekcijoje. "
            "Pilki kryželiai – triukšmo taškai.", ha="center", fontsize=8, color="grey")
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(f"{REZ}/erdviu_palyginimas.png", dpi=150)
plt.close()
