"""3. Pasirinktas DBSCAN variantas kiekvienoje erdvėje: matai, klasterių profiliai ir vizualizacija
(PCA 2D ir UMAP 2D projekcijose)."""
import matplotlib.pyplot as plt
import pandas as pd

from bendra import (ERDVES, ERDVIU_PAVADINIMAI, FAILAS_NENORMUOTA, ikelti, matai,
                    pasirinktas_dbscan, rez_katalogas, visos_erdves)

X, kodai, klases = ikelti()
X_orig, _, _ = ikelti(FAILAS_NENORMUOTA)
erdves = visos_erdves(X)
PROJEKCIJOS = {"PCA 2D": erdves["pca2"], "UMAP 2D": erdves["umap2"]}


def piesti(ax, Z, grupes, pavadinimas, asiu_vardas):
    for g in sorted(set(grupes)):
        m = grupes == g
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.7, label=g,
                   c="lightgrey" if g == "triukšmas" else None)
    ax.set(title=pavadinimas, xlabel=f"{asiu_vardas} 1", ylabel=f"{asiu_vardas} 2")
    ax.legend(markerscale=2, fontsize=8)


for erdve in ERDVES:
    Xv, rez = erdves[erdve], rez_katalogas(erdve)
    zymes, eps, ms = pasirinktas_dbscan(Xv, erdve)
    pavad = f"{ERDVIU_PAVADINIMAI[erdve]}: DBSCAN eps={eps:.3f}, min_samples={ms}"

    print(f"\n{'=' * 70}\n{pavad}")
    print(pd.Series(matai(Xv, zymes, kodai)).round(3).to_string())

    kryz = pd.crosstab(pd.Series(zymes, name="klasteris"), pd.Series(klases[kodai], name="klasė"))
    kryz.to_csv(f"{rez}/klasteriai_vs_klases.csv")
    print("\nKlasteriai vs tikrosios klasės:\n", kryz)

    profilis = X.groupby(zymes).median().T
    profilis.to_csv(f"{rez}/klasteriu_profiliai.csv")
    X_orig.groupby(zymes).agg(["median", "mean", "std"]).T.to_csv(
        f"{rez}/klasteriu_statistikos_originalios.csv")
    print("\nKlasterių medianos (originalūs vienetai):\n",
          X_orig.groupby(zymes).median().T.round(3).to_string())

    grupes = pd.Series(zymes).map(lambda k: "triukšmas" if k == -1 else f"klasteris {k}").values
    _, asys = plt.subplots(2, 2, figsize=(12, 10))
    for eil, (proj, Z) in zip(asys, PROJEKCIJOS.items()):
        piesti(eil[0], Z, grupes, f"DBSCAN klasteriai ({proj})", proj.split()[0])
        piesti(eil[1], Z, klases[kodai], f"Tikrosios klasės ({proj})", proj.split()[0])
    plt.suptitle(pavad)
    plt.tight_layout()
    plt.savefig(f"{rez}/dbscan_projekcijos.png", dpi=150)
    plt.close()
