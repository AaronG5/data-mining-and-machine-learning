import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import MDS, trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

FAILAS_A = "../A_originali_aibe.csv"
FAILAS_B = "../B_be_koreliuotu.csv"
FAILAS_C = "../C_be_isskirciu.csv"
KAIMYNAI = [5, 15, 50]


def ikelti(failas):
    duomenys = pd.read_csv(failas)
    kodai = duomenys["class"].astype("category").cat.codes.values
    return duomenys.drop(columns="class"), kodai


def mds_projekcija(X):
    return MDS(n_components=2, n_init=1, max_iter=300, init="random", random_state=0).fit_transform(X)


A, kodai_A = ikelti(FAILAS_A)
B, kodai_B = ikelti(FAILAS_B)
C, kodai_C = ikelti(FAILAS_C)

Z_A, Z_B, Z_C = (mds_projekcija(X) for X in (A, B, C))


def kokybe(X, Z, kodai):
    d0, d1 = pdist(X.values), pdist(Z)
    rez = {
        "Spirmano ρ": spearmanr(d0, d1)[0],
        "Normuotas įtempimas": ((d0 - d1) ** 2).sum() / (d0 ** 2).sum(),
    }
    for k in KAIMYNAI:
        rez[f"T{k}"] = trustworthiness(X.values, Z, n_neighbors=k)
    for k in KAIMYNAI:
        rez[f"C{k}"] = trustworthiness(Z, X.values, n_neighbors=k)
    for k in KAIMYNAI:
        b = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
        rez[f"NH{k}"] = np.mean(kodai[b] == kodai[:, None])
    return rez


def piesti(Z, kodai, pavadinimas, failas):
    plt.figure(figsize=(6.5, 5))
    plt.scatter(Z[:, 0], Z[:, 1], c=kodai, s=8, alpha=0.7)
    plt.title(pavadinimas)
    plt.xlabel("MDS 1")
    plt.ylabel("MDS 2")
    plt.tight_layout()
    plt.savefig(failas, dpi=150)
    plt.close()


print("2D projekcijos kokybė:")
print(pd.DataFrame({"A": kokybe(A, Z_A, kodai_A),
                    "B": kokybe(B, Z_B, kodai_B),
                    "C": kokybe(C, Z_C, kodai_C)}).T.round(3).to_string())

tikslumas = cross_val_score(KNeighborsClassifier(5), Z_A, kodai_A,
                            cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
print(f"\nkNN tikslumas (aibė A, 2D): {tikslumas * 100:.1f} %")

piesti(Z_A, kodai_A, "MDS projekcija", "mds_A.png")
piesti(Z_B, kodai_B, "MDS projekcija be koreliuotų požymių", "mds_B.png")
piesti(Z_C, kodai_C, "MDS projekcija be išskirčių", "mds_C.png")