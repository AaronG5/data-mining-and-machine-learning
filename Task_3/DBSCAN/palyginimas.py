"""5. Erdvių palyginimas (3 užd. 16 ir 18 skaidrės): pasirinktas DBSCAN originalioje, PCA 5D/2D ir
UMAP 5D/2D erdvėse. Išoriniai matai skaičiuojami dviem būdais:
  • su tikromis klasėmis,
  • su originalios erdvės DBSCAN rezultatu (atskaitos suskirstymu) – ar dimensijos mažinimas
    pakeitė klasterizavimą.
Pastaba (17 skaidrė): UMAP erdvėje vidiniai matai išauga dirbtinai – jų su kitomis erdvėmis nelyginti."""
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd

from bendra import (ERDVES, ERDVIU_PAVADINIMAI, FAILAS_NENORMUOTA, PASIRINKTI, RANDOM_STATE, ikelti,
                    imtys, isoriniai, matai, pasirinktas_dbscan, rez_katalogas, stabilumas,
                    visos_erdves)

N_IMCIU, DALIS = 20, 0.8  # skaidrėje rekomenduojama N = 10–20
REZ = rez_katalogas("palyginimas")

X, kodai, klases = ikelti()
X_nenorm, _, _ = ikelti(FAILAS_NENORMUOTA)  # stabilumui: kiekviena imtis normuojama atskirai
erdves = visos_erdves(X)
imciu_sarasas = imtys(len(X), N_IMCIU, DALIS)

zymes, eilutes = {}, []
for erdve in ERDVES:
    Xv = erdves[erdve]
    zymes[erdve], eps, ms = pasirinktas_dbscan(Xv, erdve)
    stab, stab_imtys = stabilumas(X_nenorm, erdve, [PASIRINKTI[erdve]], imciu_sarasas)[PASIRINKTI[erdve]]
    m = matai(Xv, zymes[erdve], kodai)
    eilutes.append({
        "Erdvė": ERDVIU_PAVADINIMAI[erdve], "Algoritmas": "DBSCAN",
        "Parametrai": f"eps={eps:.3f}, min_samples={ms}",
        "k": m["klasteriai"], "triukšmas %": m["triukšmas %"],
        "Silhouette ↑": m["Silhouette"], "CH ↑": m["Calinski-Harabasz"], "DB ↓": m["Davies-Bouldin"],
        "ARI (klasės)": m["ARI"], "NMI (klasės)": m["NMI"], "Jaccard (klasės)": m["Jaccard"],
        **isoriniai(zymes["originali"], zymes[erdve], " (atskaita)"),
        **{f"Stabilumas {mt}": f"{stab[mt].mean():.3f} ± {stab[mt].std():.3f}"
           for mt in ["ARI", "NMI", "Jaccard"]},
        **{f"{pav} (imtys)": f"{stab_imtys[mt].mean():.3f} ± {stab_imtys[mt].std():.3f}"
           for pav, mt in [("k", "klasteriai"), ("Silhouette", "Silhouette"),
                           ("CH", "Calinski-Harabasz"), ("DB", "Davies-Bouldin")]},
    })

lentele = pd.DataFrame(eilutes)
lentele.to_csv(f"{REZ}/eksperimentu_lentele.csv", index=False)
pd.set_option("display.width", 250)
print(f"Data: {date.today()}, random_state={RANDOM_STATE} (visi duomenys); stabilumas: {N_IMCIU} × "
      f"{DALIS:.0%} imtys, kiekviena normuojama ir projektuojama atskirai, random_state = 0…{N_IMCIU - 1}")
print("(imtys) – vidurkis ± SN per stabilumo imtis\n")
print("(atskaita) – lyginama su originalios erdvės DBSCAN rezultatu\n")
print(lentele.round(3).T.to_string(header=False))

# Sutampančių taškų klasteriuose skaičius: originalios erdvės klasteriai vs sumažintos erdvės
for erdve in ERDVES[1:]:
    kryz = pd.crosstab(pd.Series(zymes["originali"], name="originali"),
                       pd.Series(zymes[erdve], name=erdve))
    kryz.to_csv(f"{REZ}/sutapimas_originali_vs_{erdve}.csv")
    print(f"\nSutampantys taškai: originali (eilutės) vs {ERDVIU_PAVADINIMAI[erdve]} (stulpeliai)\n{kryz}")

# Vizualizacija kaip 18 skaidrėje: originali ir PCA – PCA 2D projekcijoje, UMAP – UMAP 2D
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
