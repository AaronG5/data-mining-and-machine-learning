"""4. Klasterių stabilumas kiekvienoje erdvėje (3 užd. skaidrė „Stabilumo eksperimentas“):
N kartų imama 80 % NEnormuotų duomenų (be ekstremalių išskirčių ir koreliuotų požymių).
Kiekviena imtis atskirai normuojama (robust), jai iš naujo skaičiuojama projekcija (PCA / UMAP,
random_state = imties nr.) ir kelio eps, tada klasterizuojama. Visų N·(N−1)/2 imčių porų
suskirstymai lyginami tik bendriems objektams (ARI, NMI, Jaccard) – tikrinamas visos procedūros
(normavimas -> projekcija -> eps parinkimas -> DBSCAN) stabilumas.
Pateikiama: išoriniai matai vidurkis ± SN ir min, vidiniai matai (Silhouette, CH, DB) vidurkis ± SN
per N imčių, boxplot ir imčių skaičiaus pakankamumas (ARI vidurkis, kai N = 5, 10, 15, 20)."""
import matplotlib.pyplot as plt
import pandas as pd

from bendra import (ERDVES, ERDVIU_PAVADINIMAI, FAILAS_NENORMUOTA, PASIRINKTI, ikelti, imtys,
                    rez_katalogas, stabilumas)

N = 20                         # skaidrėje rekomenduojama N = 10–20
N_TIKRINIMUI = [5, 10, 15, 20]  # pakankamumas: ar ARI vidurkis nusistovi
DALIS = 0.8
# (min_samples, eps daugiklis): pasirinktas variantas (pirmas) ir keli palyginimui.
# eps = kiekvienos imties kelio eps × daugiklis (projekcija perskaičiuojama, todėl visų duomenų
# eps tiesiogiai netinka).
VARIANTAI = {
    "originali": [PASIRINKTI["originali"], (5, 0.8), (10, 1.0), (32, 0.6)],
    "pca5": [PASIRINKTI["pca5"], (5, 1.0), (16, 1.2), (32, 0.6)],
    "pca2": [PASIRINKTI["pca2"], (4, 1.0), (25, 1.2), (50, 0.8)],
    "umap5": [PASIRINKTI["umap5"], (10, 1.0), (5, 1.5), (50, 0.8)],
    "umap2": [PASIRINKTI["umap2"], (10, 1.0), (25, 0.6), (50, 0.8)],
}
ISORINIAI = ["ARI", "NMI", "Jaccard"]
VIDINIAI = ["eps", "klasteriai", "triukšmas %", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]

X_nenorm, _, _ = ikelti(FAILAS_NENORMUOTA)
imciu_sarasas = imtys(len(X_nenorm), N, DALIS)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
print(f"Stabilumas: {N} imčių po {DALIS:.0%} (kiekviena normuojama ir projektuojama atskirai, "
      f"random_state = 0…{N - 1}), {N * (N - 1) // 2} poros kiekvienam variantui")


def vid_sn(x):
    return f"{x.mean():.3f} ± {x.std():.3f}"


pakankamumas = []
for erdve in ERDVES:
    rez = rez_katalogas(erdve)
    poros, santrauka, klasteriu_sk = [], [], []
    for (ms, dg), (p, im) in stabilumas(X_nenorm, erdve, VARIANTAI[erdve], imciu_sarasas).items():
        pav = f"ms={ms}, eps={dg}×kelio"
        poros.append(p.assign(variantas=pav))
        klasteriu_sk += [{"variantas": pav, "klasteriai": x} for x in im["klasteriai"]]
        santrauka.append({"variantas": pav,
                          **{f"{m} vid ± SN": vid_sn(p[m]) for m in ISORINIAI},
                          **{f"{m} min": p[m].min() for m in ISORINIAI},
                          **{f"{m} vid ± SN (imtys)": vid_sn(im[m]) for m in VIDINIAI}})
        for n in N_TIKRINIMUI:
            pn = p[(p["i"] < n) & (p["j"] < n)]
            pakankamumas.append({"erdvė": ERDVIU_PAVADINIMAI[erdve], "variantas": pav,
                                 "pasirinktas": (ms, dg) == PASIRINKTI[erdve], "N": n,
                                 "ARI vid": pn["ARI"].mean(), "ARI SN": pn["ARI"].std()})
    poros = pd.concat(poros)
    santrauka = pd.DataFrame(santrauka).set_index("variantas")
    santrauka.to_csv(f"{rez}/stabilumas.csv")
    print(f"\n{ERDVIU_PAVADINIMAI[erdve]}:\n{santrauka.round(3).T.to_string()}")
    print(f"\nKlasterių skaičius {N} imčių (kiek imčių davė tiek klasterių):")
    print(pd.DataFrame(klasteriu_sk).groupby("variantas", sort=False)["klasteriai"]
          .value_counts().unstack(fill_value=0).sort_index(axis=1))

    _, asys = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, matas in zip(asys, ISORINIAI):
        grupes = [g[matas].values for _, g in poros.groupby("variantas", sort=False)]
        ax.boxplot(grupes, tick_labels=poros["variantas"].unique(), orientation="horizontal")
        ax.set_title(matas)
        ax.grid(alpha=0.3, axis="x")
    plt.suptitle(f"{ERDVIU_PAVADINIMAI[erdve]}: DBSCAN stabilumas "
                 f"({N} × {DALIS:.0%} imtys, kiekviena normuojama atskirai; lyginami bendri objektai)")
    plt.tight_layout()
    plt.savefig(f"{rez}/stabilumas.png", dpi=150)
    plt.close()

# Imčių skaičiaus pakankamumas: pasirinkto varianto ARI vidurkis didinant N
pakankamumas = pd.DataFrame(pakankamumas)
rez = rez_katalogas("palyginimas")
pakankamumas.to_csv(f"{rez}/imciu_skaiciaus_pakankamumas.csv", index=False)
pasirinkti = pakankamumas[pakankamumas["pasirinktas"]]
print("\nImčių skaičiaus pakankamumas (pasirinktas variantas, ARI vidurkis):")
print(pasirinkti.pivot(index="erdvė", columns="N", values="ARI vid")
      .reindex([ERDVIU_PAVADINIMAI[e] for e in ERDVES]).round(3).to_string())

_, ax = plt.subplots(figsize=(7, 4.5))
for erdve in ERDVES:
    d = pasirinkti[pasirinkti["erdvė"] == ERDVIU_PAVADINIMAI[erdve]]
    ax.errorbar(d["N"], d["ARI vid"], yerr=d["ARI SN"], marker="o", capsize=3,
                label=f"{ERDVIU_PAVADINIMAI[erdve]} ({d['variantas'].iloc[0]})")
ax.set(xlabel="Imčių skaičius N", ylabel="ARI vidurkis ± SN", xticks=N_TIKRINIMUI,
       title="Imčių skaičiaus pakankamumas (pasirinkti variantai)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{rez}/imciu_skaiciaus_pakankamumas.png", dpi=150)
plt.close()
