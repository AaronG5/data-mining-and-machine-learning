"""4. Klasterių stabilumas kiekvienoje erdvėje: N kartų imama 80 % duomenų, kiekviena imtis
klasterizuojama, visų imčių porų suskirstymai lyginami tik bendriems objektams (ARI, NMI, Jaccard).
Projekcija (PCA / UMAP) apskaičiuojama vieną kartą visiems duomenims, imtys imamos iš jos –
tikrinamas DBSCAN stabilumas toje erdvėje."""
import matplotlib.pyplot as plt
import pandas as pd

from bendra import (ERDVES, ERDVIU_PAVADINIMAI, PASIRINKTI, ikelti, imtys, kelio_eps,
                    rez_katalogas, stabilumas, visos_erdves)

N = 50
DALIS = 0.8
# (min_samples, eps daugiklis): pasirinktas variantas (pirmas) ir keli palyginimui.
# eps fiksuojamas pagal visų duomenų kelio tašką: tikrinamas konkretaus modelio stabilumas.
VARIANTAI = {
    "originali": [PASIRINKTI["originali"], (5, 0.8), (10, 1.0), (32, 0.6)],
    "pca": [PASIRINKTI["pca"], (4, 1.0), (16, 1.2), (32, 0.6)],
    "umap": [PASIRINKTI["umap"], (10, 1.0), (25, 0.6), (50, 0.8)],
}

X, _, _ = ikelti()
erdves = visos_erdves(X)
imciu_sarasas = imtys(len(X), N, DALIS)
pd.set_option("display.width", 200)
print(f"Stabilumas: {N} imčių po {DALIS:.0%}, {N * (N - 1) // 2} poros kiekvienam variantui")

for erdve in ERDVES:
    Xv, rez = erdves[erdve], rez_katalogas(erdve)
    poros, klasteriu_sk = [], []
    for ms, dg in VARIANTAI[erdve]:
        eps = kelio_eps(Xv, ms)[2] * dg
        pav = f"ms={ms}, eps={eps:.2f}"
        p, k = stabilumas(Xv, eps, ms, imciu_sarasas)
        poros.append(p.assign(variantas=pav))
        klasteriu_sk += [{"variantas": pav, "klasteriai": x} for x in k]
    poros = pd.concat(poros)

    santrauka = poros.groupby("variantas", sort=False).agg(["mean", "std", "min"])
    santrauka.to_csv(f"{rez}/stabilumas.csv")
    print(f"\n{ERDVIU_PAVADINIMAI[erdve]}:\n{santrauka.round(3).to_string()}")
    print(f"\nKlasterių skaičius {N} imčių (kiek imčių davė tiek klasterių):")
    print(pd.DataFrame(klasteriu_sk).groupby("variantas", sort=False)["klasteriai"]
          .value_counts().unstack(fill_value=0).sort_index(axis=1))

    _, asys = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, matas in zip(asys, ["ARI", "NMI", "Jaccard"]):
        grupes = [g[matas].values for _, g in poros.groupby("variantas", sort=False)]
        ax.boxplot(grupes, tick_labels=poros["variantas"].unique(), orientation="horizontal")
        ax.set_title(matas)
        ax.grid(alpha=0.3, axis="x")
    plt.suptitle(f"{ERDVIU_PAVADINIMAI[erdve]}: DBSCAN stabilumas "
                 f"({N} × {DALIS:.0%} imtys, lyginami bendri objektai)")
    plt.tight_layout()
    plt.savefig(f"{rez}/stabilumas.png", dpi=150)
    plt.close()
