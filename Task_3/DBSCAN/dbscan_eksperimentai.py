"""2. Kaip keičiasi DBSCAN rezultatai keičiant eps ir min_samples (kiekvienoje erdvėje)."""
import matplotlib.pyplot as plt
import pandas as pd

from bendra import (ERDVES, ERDVIU_PAVADINIMAI, MIN_SAMPLES, dbscan_kelio, ikelti, matai,
                    rez_katalogas, visos_erdves)

EPS_DAUGIKLIAI = [0.6, 0.8, 1.0, 1.2, 1.5]  # eps tyrimas aplink kelio tašką
GRAFIKAI = ["klasteriai", "triukšmas %", "Silhouette", "ARI", "NMI"]

X, kodai, _ = ikelti()
erdves = visos_erdves(X)
pd.set_option("display.width", 200)

for erdve in ERDVES:
    Xv, rez = erdves[erdve], rez_katalogas(erdve)
    eksperimentai = []
    for ms in MIN_SAMPLES[erdve]:
        for dg in EPS_DAUGIKLIAI:
            zymes, eps = dbscan_kelio(Xv, ms, dg)
            eksperimentai.append({"min_samples": ms, "eps daugiklis": dg, "eps": eps,
                                  **matai(Xv, zymes, kodai)})

    lentele = pd.DataFrame(eksperimentai)
    lentele.to_csv(f"{rez}/eksperimentai.csv", index=False)
    print(f"\n{ERDVIU_PAVADINIMAI[erdve]}:")
    print(lentele.round(3).to_string(index=False))

    _, asys = plt.subplots(1, len(GRAFIKAI), figsize=(4 * len(GRAFIKAI), 3.8))
    for ax, st in zip(asys, GRAFIKAI):
        for ms, g in lentele.groupby("min_samples"):
            ax.plot(g["eps"], g[st], marker="o", label=f"min_samples={ms}")
        ax.set_xlabel("eps")
        ax.set_title(st)
        ax.grid(alpha=0.3)
    asys[0].legend()
    plt.suptitle(ERDVIU_PAVADINIMAI[erdve])
    plt.tight_layout()
    plt.savefig(f"{rez}/eksperimentai.png", dpi=150)
    plt.close()
