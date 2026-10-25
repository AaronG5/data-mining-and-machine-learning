import numpy as np, pandas as pd
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

KORELIUOTI = ["X_Minimum", "Y_Minimum", "Pixels_Areas", "LogOfAreas", "SigmoidOfAreas", "Log_X_Index", "Log_Y_Index", "Minimum_of_Luminosity"]
pradine = pd.read_csv("A27_normuota_su_isskirtim_robust.csv") # 1059, su išskirtimis
be_iss = pd.read_csv("A27_normuota_robust.csv") # 1019, be išskirčių
X = pradine.drop(columns="class").values #A: pradinė aibė
X_B = pradine.drop(columns=["class"] + KORELIUOTI).values #B: be koreliuotų požymių (1059)
X_C = be_iss.drop(columns="class").values #C: be išskirčių (1019)
bendri = pd.read_csv("../Task_1/A27_be_loginiu_klaidu.csv").merge(
    pd.read_csv("../Task_1/A27_be_virsutiniu_isskirciu.csv"), how="left", indicator=True)["_merge"].eq("both").values

#NH(k) skaičiavimo funkcija
def nh(X, Z, k): 
    kn = lambda A: NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(return_distance=False)[:, 1:]
    return np.mean([len(set(a) & set(b)) / k for a, b in zip(kn(X), kn(Z))])

def metrikos(X, Z):
    r = {"Spearman": spearmanr(pdist(X), pdist(Z))[0]}
    for k in (5, 15, 50):
        r |= {f"T{k}": trustworthiness(X, Z, n_neighbors=k),
              f"C{k}": trustworthiness(Z, X, n_neighbors=k), f"NH{k}": nh(X, Z, k)}
    return r

#Perplexity ir seed testai
rez = pd.DataFrame([{"perplexity": p, **metrikos(X, TSNE(perplexity=p, init="random", random_state=s).fit_transform(X))}
                    for p in (5, 15, 30, 50) for s in range(5)])
print(rez.groupby("perplexity").agg(["mean", "std"]).round(3).T)

# 2. A / B / C (perplexity 30, PCA init)
tsne = TSNE(perplexity=30, init="pca", random_state=0)
Z_A, Z_B, Z_C = tsne.fit_transform(X), tsne.fit_transform(X_B), tsne.fit_transform(X_C)
print(pd.DataFrame({"A": metrikos(X, Z_A), "B": metrikos(X_B, Z_B),
                    "A (bendri)": metrikos(X[bendri], Z_A[bendri]), "C": metrikos(X_C, Z_C)}).round(3))
print("B: rho(D_A, D_B) =", round(spearmanr(pdist(X), pdist(X_B))[0], 3), " kaimynų sutapimas:", round(nh(Z_A, Z_B, 15), 3))
print("C: kaimynų sutapimas (bendri objektai):", round(nh(Z_A[bendri], Z_C, 15), 3))

# diagramos
ribos = (np.vstack([Z_A, Z_B, Z_C]).min() - 2, np.vstack([Z_A, Z_B, Z_C]).max() + 2)  # bendros ašys

def diagrama(Z, klase, pavadinimas, failas):
    y = klase.astype("category")
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=y.cat.codes, s=10, edgecolors=(0, 0, 0, 0.3), linewidths=0.3)
    plt.title(pavadinimas); plt.xlabel("t-SNE1"); plt.ylabel("t-SNE2")
    plt.xlim(ribos); plt.ylim(ribos); plt.gca().set_aspect("equal")
    plt.legend(sc.legend_elements()[0], y.cat.categories)
    plt.savefig(failas, dpi=150); plt.close()

# A: pradinė aibė (su išskirtimis)
diagrama(Z_A, pradine["class"], "t-SNE: pradinė aibė", "tsne_A_pradine.png")

# B: be koreliuotų požymių
diagrama(Z_B, pradine["class"], "t-SNE: be koreliuotų požymių", "tsne_B_be_koreliuotu.png")

# C: be išskirčių
diagrama(Z_C, be_iss["class"], "t-SNE: be išskirčių", "tsne_C_be_isskirciu.png")