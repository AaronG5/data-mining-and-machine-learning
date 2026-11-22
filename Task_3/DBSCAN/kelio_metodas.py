"""1. Kelio (knee) metodas eps parinkimui kiekvienoje erdvėje."""
import matplotlib.pyplot as plt
import pandas as pd

from bendra import (ERDVES, ERDVIU_PAVADINIMAI, MIN_SAMPLES, ikelti, kelio_eps,
                    rez_katalogas, visos_erdves)

erdves = visos_erdves(ikelti()[0])

for erdve in ERDVES:
    X, rez, ms_sarasas = erdves[erdve], rez_katalogas(erdve), MIN_SAMPLES[erdve]
    kelio_rez = {}
    _, asys = plt.subplots(1, len(ms_sarasas), figsize=(4.2 * len(ms_sarasas), 4))
    for ax, ms in zip(asys, ms_sarasas):
        atst, idx, eps = kelio_eps(X, ms)
        kelio_rez[ms] = eps
        ax.plot(atst, lw=1.2)
        ax.axhline(eps, ls="--", c="red", lw=0.8)
        ax.scatter([idx], [eps], c="red", zorder=3)
        ax.set_title(f"min_samples = {ms}\nkelio eps = {eps:.3f}")
        ax.set_xlabel("Objektai, surikiuoti pagal atstumą")
        ax.set_ylabel(f"Atstumas iki {ms - 1}-ojo kaimyno")
    plt.suptitle(ERDVIU_PAVADINIMAI[erdve])
    plt.tight_layout()
    plt.savefig(f"{rez}/kelio_metodas.png", dpi=150)
    plt.close()

    lentele = pd.Series(kelio_rez, name="eps").rename_axis("min_samples")
    lentele.to_csv(f"{rez}/kelio_eps.csv")
    print(f"\n{ERDVIU_PAVADINIMAI[erdve]}:\n{lentele.round(3).to_string()}")
