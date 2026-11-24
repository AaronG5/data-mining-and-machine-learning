
import itertools
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

from bendra import (ERDVIU_PAVADINIMAI, IMTIES_DALIS, IMTYS, ISORINIAI, N_TIKRINIMUI, PASIRINKTI, RANDOM_STATE, REZ,
                    UMAP_PARAMETRAI, VIDINIAI, dydziai, ikelti, isoriniai, issaugoti, klasteriai, vardas, vid_sn,
                    vidiniai, visos_erdves, visu_erdviu_klasteriai)
from stabilumas import imciu_pakankamumas, imtys, palyginti_imciu_poras, projektuoti_imtis

pd.set_option("display.width", 250)


def stabilumas(X, erdve, imciu_sarasas):
    projekcijos = projektuoti_imtis(X, erdve, imciu_sarasas)
    zymes = [klasteriai(Z, *PASIRINKTI[erdve]) for Z in projekcijos]
    poros = palyginti_imciu_poras(imciu_sarasas, zymes)
    imciu_kokybe = pd.DataFrame([vidiniai(Z, z) for Z, z in zip(projekcijos, zymes)])
    pakankamumas = pd.DataFrame(imciu_pakankamumas(poros, N_TIKRINIMUI)).set_index("N")["ARI vid"]
    santrauka = {**{f"stab. {m}": vid_sn(poros[m]) for m in ISORINIAI}, "stab. ARI min": poros["ARI"].min(),
                 **{f"stab. {m} imtyse": vid_sn(imciu_kokybe[m]) for m in VIDINIAI},
                 **{f"stab. ARI vid, N={n}": a for n, a in pakankamumas.items()}}
    return santrauka, poros["ARI"]


def isoriniai_eil(a, b):
    rez = isoriniai(a, b)
    return [rez[m] for m in ISORINIAI]


def eksperimentas(erdve, Z, z, atskaita, kodai, stab):
    jungimas, atstumas, k = PASIRINKTI[erdve]
    umap = erdve.startswith("umap")
    return {"data": date.today().isoformat(), "erdvė": ERDVIU_PAVADINIMAI[erdve], "jungimas": jungimas,
            "atstumas": atstumas, "k": k, "dydžiai": dydziai(z),
            "UMAP": f"n_neighbors={UMAP_PARAMETRAI['n_neighbors']}, min_dist={UMAP_PARAMETRAI['min_dist']}" if umap
            else "", "random_state": RANDOM_STATE if umap else "",
            **vidiniai(Z, z), "neigiamas Silhouette": int((silhouette_samples(Z, z) < 0).sum()),
            **dict(zip([f"{m} su atskaita" for m in ISORINIAI],
                       ["atskaita"] * 3 if erdve == "originali" else np.round(isoriniai_eil(atskaita, z), 3))),
            "sutampa su atskaita": int((z == atskaita).sum()),
            **dict(zip([f"{m} su klasėmis" for m in ISORINIAI], isoriniai_eil(kodai, z))),
            "stab. imtys": f"{IMTYS} × {IMTIES_DALIS:.0%}, random_state 0–{IMTYS - 1}", **stab}


def skirtingi_suskirstymai(zymes):
    """Erdvės, duodančios identiškas žymes, sujungiamos į vieną suskirstymą."""
    grupes = {}
    for erdve, z in zymes.items():
        raktas = next((r for r in grupes if np.array_equal(zymes[grupes[r][0]], z)), erdve)
        grupes.setdefault(raktas, []).append(erdve)
    return {", ".join(ERDVIU_PAVADINIMAI[e] for e in erdvs): zymes[erdvs[0]] for erdvs in grupes.values()}


def indelis(V, z, pozymiai):
    """Požymio dalis tarpklasterinėje sklaidoje (normuotoje originalioje erdvėje)."""
    tarp = pd.Series({p: sum((z == c).sum() * (V[z == c, i].mean() - V[:, i].mean()) ** 2 for c in np.unique(z))
                      for i, p in enumerate(pozymiai)})
    return tarp / tarp.sum()


def profiliai(X, V, suskirstymai):
    stulpeliai = {("visi", f"({len(X)})", s): f(X) for s, f in
                  (("mediana", pd.DataFrame.median), ("vidurkis", pd.DataFrame.mean), ("SN", pd.DataFrame.std))}
    for pavadinimas, z in suskirstymai.items():
        for c in np.unique(z):
            dalis = X[z == c]
            stulpeliai |= {(pavadinimas, f"{c} ({len(dalis)})", "mediana"): dalis.median(),
                           (pavadinimas, f"{c} ({len(dalis)})", "vidurkis"): dalis.mean(),
                           (pavadinimas, f"{c} ({len(dalis)})", "SN"): dalis.std()}
        stulpeliai[(pavadinimas, "indėlis", "dalis")] = indelis(V, z, X.columns)
    return pd.DataFrame(stulpeliai).rename_axis(["suskirstymas", "klasteris", "statistika"], axis=1)


def sutapimai(suskirstymai, atskaita, klases):
    return pd.concat({p: pd.concat([pd.crosstab(z, atskaita).add_prefix("atskaitos klasteris "),
                                    pd.crosstab(z, klases)], axis=1).rename_axis(index="klasteris", columns=None)
                      for p, z in suskirstymai.items()}, names=["suskirstymas"])


def piesti(asis, Z, grupes, antraste):
    for g in sorted(set(grupes)):
        asis.scatter(*Z[grupes == g].T, s=6, label=g)
    asis.set_title(antraste, fontsize=10)
    asis.legend(fontsize=7, markerscale=2)


def piesti_klasterius(erdves, zymes, klases):
    """Klasteriai ir tikrosios klasės: originali ir PCA erdvės – PCA 2D, UMAP erdvės – UMAP 2D projekcijoje."""
    for failas, projekcija, erdviu_sarasas in (("klasteriai_originali.png", "pca2", ["originali"]),
                                               ("klasteriai_pca.png", "pca2", ["pca5", "pca2"]),
                                               ("klasteriai_umap.png", "umap2", ["umap5", "umap2"])):
        fig, asys = plt.subplots(1, len(erdviu_sarasas) + 1, figsize=(5 * (len(erdviu_sarasas) + 1), 4.8))
        for asis, erdve in zip(asys, erdviu_sarasas):
            piesti(asis, erdves[projekcija], zymes[erdve],
                   f"{ERDVIU_PAVADINIMAI[erdve]}\n{vardas(*PASIRINKTI[erdve])}")
        piesti(asys[-1], erdves[projekcija], klases, f"Tikrosios klasės\n({ERDVIU_PAVADINIMAI[projekcija]} projekcija)")
        issaugoti(failas)


def piesti_stabiluma(ari):
    plt.figure(figsize=(7, 4))
    plt.boxplot(list(ari.values()), vert=False, showmeans=True, meanline=True,
                meanprops={"color": "C1", "linestyle": "-"}, medianprops={"visible": False})
    plt.yticks(range(1, len(ari) + 1), list(ari))
    plt.gca().invert_yaxis()
    plt.title(f"Stabilumas: {IMTYS} imčių po {IMTIES_DALIS:.0%}")
    plt.xlabel("ARI tarp imčių porų (bendri objektai)")
    issaugoti("stabilumas.png")


X, kodai, pavadinimai = ikelti()
klases = np.asarray(pavadinimai[kodai])
erdves = visos_erdves(X)
atskaita, zymes = visu_erdviu_klasteriai(erdves)

imciu_sarasas = imtys(len(X), IMTYS, IMTIES_DALIS)
bendri = np.mean([len(np.intersect1d(a, b)) for a, b in itertools.combinations(imciu_sarasas, 2)])
print(f"Vidutiniškai bendrų objektų imčių poroje: {bendri:.0f} ({bendri / len(X):.0%})")

eilutes, ari = [], {}
for erdve, Z in erdves.items():
    print(f"Stabilumas: {ERDVIU_PAVADINIMAI[erdve]}")
    stab, ari[ERDVIU_PAVADINIMAI[erdve]] = stabilumas(X, erdve, imciu_sarasas)
    eilutes.append(eksperimentas(erdve, Z, zymes[erdve], atskaita, kodai, stab))
lentele = pd.DataFrame(eilutes)
lentele.round(3).to_csv(REZ / "rezultatai.csv", index=False)
print(lentele.round(3).set_index("erdvė").T.to_string())

suskirstymai = skirtingi_suskirstymai(zymes)
sut = sutapimai(suskirstymai, atskaita, klases)
sut.to_csv(REZ / "sutapimai.csv")
print("\n", sut.to_string())

prof = profiliai(X, erdves["originali"], suskirstymai)
prof.round(3).to_csv(REZ / "profiliai.csv")
print("\n", prof.xs("mediana", axis=1, level="statistika").round(2).to_string())

piesti_klasterius(erdves, zymes, klases)
piesti_stabiluma(ari)
