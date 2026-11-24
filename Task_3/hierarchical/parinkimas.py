"""1. Parametrų parinkimas visose erdvėse.
res/variantai.csv        – visi jungimo ir atstumo deriniai, kiekvienas su savo k (Silhouette maksimumas)
res/k_pasirinkimas.csv   – pasirinkto derinio vidiniai matai ir dendrogramos šuoliai, k = 2–10
res/k_pasirinkimas.png, res/dendrograma.png (originali), res/dendrogramos.png (visos erdvės)"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

from bendra import (ERDVIU_PAVADINIMAI, K_RIBOS, MAZIAUSIAS_KLASTERIS, PASIRINKTI, REZ, VIDINIAI, dydziai, ikelti,
                    issaugoti, jungimo_variantai, k_kreive, klasteriai, vardas, vidiniai, visos_erdves,
                    visu_erdviu_klasteriai)

pd.set_option("display.width", 250)


def variantai(Z, erdve):
    eilutes = []
    for jungimas, atstumas in jungimo_variantai():
        z = max((klasteriai(Z, jungimas, atstumas, k) for k in K_RIBOS), key=lambda z: vidiniai(Z, z)["Silhouette"])
        eilutes.append({"erdvė": ERDVIU_PAVADINIMAI[erdve], "jungimas": jungimas, "atstumas": atstumas,
                        "k": z.max() + 1, **vidiniai(Z, z), "dydžiai": dydziai(z),
                        "neišsigimęs": np.bincount(z).min() >= MAZIAUSIAS_KLASTERIS * len(z),
                        "pasirinktas": (jungimas, atstumas) == PASIRINKTI[erdve][:2]})
    return pd.DataFrame(eilutes)


def nuspalvintas_medis(Z, zymes, jungimas, atstumas):
    """Šaka nuspalvinama klasterio spalva, jei visi jos objektai tame klasteryje, kitaip pilka."""
    medis = linkage(Z, method=jungimas, metric=atstumas)
    spalvos = {i: f"C{z % 10}" for i, z in enumerate(zymes)}
    for i, (a, b, _, _) in enumerate(medis):
        spalvos[len(Z) + i] = spalvos[int(a)] if spalvos[int(a)] == spalvos[int(b)] else "grey"
    return medis, spalvos


def piesti_dendrograma(asis, Z, zymes, erdve, lapai, etiketes):
    jungimas, atstumas, k = PASIRINKTI[erdve]
    medis, spalvos = nuspalvintas_medis(Z, zymes, jungimas, atstumas)
    dendrogram(medis, truncate_mode="lastp", p=lapai, link_color_func=lambda x: spalvos[x],
               no_labels=not etiketes, ax=asis)
    asis.set_title(f"{ERDVIU_PAVADINIMAI[erdve]}\n{vardas(jungimas, atstumas, k)}", fontsize=10)


X, _, _ = ikelti()
erdves = visos_erdves(X)
_, zymes = visu_erdviu_klasteriai(erdves)

visi_variantai = pd.concat([variantai(Z, e) for e, Z in erdves.items()], ignore_index=True)
visi_variantai.to_csv(REZ / "variantai.csv", index=False)
print(visi_variantai.round(3).to_string(index=False))

kreives = pd.concat({ERDVIU_PAVADINIMAI[e]: k_kreive(Z, *PASIRINKTI[e][:2]) for e, Z in erdves.items()},
                    names=["erdvė"])
kreives.to_csv(REZ / "k_pasirinkimas.csv")
print("\n", kreives.round(3).to_string())

fig, asys = plt.subplots(len(erdves), 3, figsize=(12, 2.6 * len(erdves)))
for eilute, erdve in zip(asys, erdves):
    kreive, (jungimas, atstumas, k) = kreives.loc[ERDVIU_PAVADINIMAI[erdve]], PASIRINKTI[erdve]
    for asis, matas in zip(eilute, VIDINIAI):
        asis.plot(kreive.index, kreive[matas], "o-")
        asis.axvline(k, ls="--", c="grey")
        asis.set(title=matas, xlabel="k")
    eilute[0].set_ylabel(f"{ERDVIU_PAVADINIMAI[erdve]}\n{vardas(jungimas, atstumas)}", fontsize=9)
issaugoti("k_pasirinkimas.png")

fig, asis = plt.subplots(figsize=(12, 5))
piesti_dendrograma(asis, erdves["originali"], zymes["originali"], "originali", 30, True)
asis.set_ylabel("Sujungimo atstumas")
issaugoti("dendrograma.png")

fig, asys = plt.subplots(1, len(erdves), figsize=(4.2 * len(erdves), 3.6))
for asis, (erdve, Z) in zip(asys, erdves.items()):
    piesti_dendrograma(asis, Z, zymes[erdve], erdve, 20, False)
asys[0].set_ylabel("Sujungimo atstumas")
issaugoti("dendrogramos.png")
