from itertools import combinations

import numpy as np
import pandas as pd

from erdves import normuoti_ir_projektuoti
from matai import isoriniai


def imtys(n_objektu, kiekis=20, dalis=0.8):
    imties_dydis = int(dalis * n_objektu)
    return [np.sort(np.random.default_rng(seed=i).choice(n_objektu, imties_dydis, replace=False))
            for i in range(kiekis)]


#Kiekviena imtis normuojama ir projektuojama atskirai, kitaip stabilumas pervertinamas
def projektuoti_imtis(X_nenorm, erdve, imciu_sarasas):
    return [normuoti_ir_projektuoti(X_nenorm.iloc[imtis], erdve, random_state=i)
            for i, imtis in enumerate(imciu_sarasas)]


def palyginti_bendrus_objektus(imtis_a, zymes_a, imtis_b, zymes_b):
    _, vietos_a, vietos_b = np.intersect1d(imtis_a, imtis_b, return_indices=True)
    return isoriniai(zymes_a[vietos_a], zymes_b[vietos_b])


def palyginti_imciu_poras(imciu_sarasas, klasteriu_zymes):
    return pd.DataFrame([
        {"i": i, "j": j, **palyginti_bendrus_objektus(imciu_sarasas[i], klasteriu_zymes[i],
                                                      imciu_sarasas[j], klasteriu_zymes[j])}
        for i, j in combinations(range(len(imciu_sarasas)), 2)
    ])


#Ar pakanka imčių
def imciu_pakankamumas(imciu_poros, n_reiksmes, matas="ARI"):
    eilutes = []
    for n in n_reiksmes:
        poros_n = imciu_poros[(imciu_poros["i"] < n) & (imciu_poros["j"] < n)]
        eilutes.append({"N": n, f"{matas} vid": poros_n[matas].mean(),
                        f"{matas} SN": poros_n[matas].std()})
    return eilutes
