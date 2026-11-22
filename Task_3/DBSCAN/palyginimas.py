"""5. Erdvių palyginimas: pasirinktas DBSCAN kiekvienoje erdvėje, išoriniai matai su klasėmis ir
su originalios erdvės rezultatu. Stabilumas imamas iš dbscan_stabilumas.py (paleisti pirmiau)."""
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd

from bendra import (ERDVES, ERDVIU_PAVADINIMAI, PASIRINKTI, RANDOM_STATE, dbscan_kelio, ikelti,
                    isoriniai, matai, rez_katalogas, visos_erdves)

REZ = rez_katalogas("palyginimas")
STABILUMAS = {"Stabilumas ARI": "ARI vid ± SN", "Stabilumas NMI": "NMI vid ± SN",
              "Stabilumas Jaccard": "Jaccard vid ± SN", "k (imtys)": "klasteriai vid ± SN (imtys)",
              "Silhouette (imtys)": "Silhouette vid ± SN (imtys)",
              "CH (imtys)": "Calinski-Harabasz vid ± SN (imtys)",
              "DB (imtys)": "Davies-Bouldin vid ± SN (imtys)"}

X, kodai, klases = ikelti()
erdves = visos_erdves(X)

zymes, eilutes = {}, []
for erdve in ERDVES:
    Xv = erdves[erdve]
    ms = PASIRINKTI[erdve][0]
    zymes[erdve], eps = dbscan_kelio(Xv, *PASIRINKTI[erdve])
    stab = pd.read_csv(f"{rez_katalogas(erdve)}/stabilumas.csv").iloc[0]  # 1 eilutė – pasirinktas
    m = matai(Xv, zymes[erdve], kodai)
    eilutes.append({
        "Erdvė": ERDVIU_PAVADINIMAI[erdve], "Algoritmas": "DBSCAN",
        "Parametrai": f"eps={eps:.3f}, min_samples={ms}",
        "k": m["klasteriai"], "triukšmas %": m["triukšmas %"],
        "Silhouette ↑": m["Silhouette"], "CH ↑": m["Calinski-Harabasz"], "DB ↓": m["Davies-Bouldin"],
        "ARI (klasės)": m["ARI"], "NMI (klasės)": m["NMI"], "Jaccard (klasės)": m["Jaccard"],
        **isoriniai(zymes["originali"], zymes[erdve], " (atskaita)"),
        **{pav: stab[st] for pav, st in STABILUMAS.items()},
    })

lentele = pd.DataFrame(eilutes)
lentele.to_csv(f"{REZ}/eksperimentu_lentele.csv", index=False)
pd.set_option("display.width", 250)
print(f"Data: {date.today()}, random_state={RANDOM_STATE}; stabilumas – iš res/<erdvė>/stabilumas.csv")
print("(imtys) – vidurkis ± SN per stabilumo imtis")
print("(atskaita) – lyginama su originalios erdvės DBSCAN rezultatu\n")
print(lentele.round(3).T.to_string(header=False))

# Sutampantys taškai: originalios erdvės klasteriai vs sumažintos erdvės
for erdve in ERDVES[1:]:
    kryz = pd.crosstab(pd.Series(zymes["originali"], name="originali"),
                       pd.Series(zymes[erdve], name=erdve))
    kryz.to_csv(f"{REZ}/sutapimas_originali_vs_{erdve}.csv")
    print(f"\nSutampantys taškai: originali (eilutės) vs {ERDVIU_PAVADINIMAI[erdve]} (stulpeliai)\n{kryz}")

# Originali ir PCA rodomos PCA 2D projekcijoje, UMAP – UMAP 2D
rodymas = {e: erdves["umap2"] if e.startswith("umap") else erdves["pca2"] for e in ERDVES}
_, asys = plt.subplots(1, len(ERDVES), figsize=(4.6 * len(ERDVES), 4.8))
for ax, erdve, e in zip(asys, ERDVES, eilutes):
    Z, z = rodymas[erdve], zymes[erdve]
    triuksmas = z == -1
    ax.scatter(Z[triuksmas, 0], Z[triuksmas, 1], s=10, c="lightgrey", marker="x")
    ax.scatter(Z[~triuksmas, 0], Z[~triuksmas, 1], s=6, c=z[~triuksmas], cmap="tab10",
               vmin=0, vmax=9)
    ax.set_title(f"{ERDVIU_PAVADINIMAI[erdve]}\n"
                 f"k={e['k']}  Sil={e['Silhouette ↑']:.2f}  ARI(atsk.)={e['ARI (atskaita)']:.2f}",
                 fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle("DBSCAN rezultatai skirtingose erdvėse", fontweight="bold")
plt.figtext(0.5, 0.01, "Originali, PCA 5D ir PCA 2D – PCA 2D projekcijoje; UMAP 5D ir UMAP 2D – UMAP 2D projekcijoje. "
            "Pilki kryželiai – triukšmo taškai.", ha="center", fontsize=8, color="grey")
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(f"{REZ}/erdviu_palyginimas.png", dpi=150)
plt.close()
