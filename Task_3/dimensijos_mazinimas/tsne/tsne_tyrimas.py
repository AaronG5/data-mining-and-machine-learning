import numpy as np, pandas as pd
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


#aibė be išskirčių ir be koreliuotų požymių, robust normuota (965)
duomenys = pd.read_csv("../steel_plates_3_klases_normuota_robust.csv")
X = duomenys.drop(columns="class").values

#NHk
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

#perplexity 30, PCA init
Z = TSNE(perplexity=30, init="pca", random_state=0).fit_transform(X)
print(pd.Series(metrikos(X, Z)).round(3))

# diagrama
def diagrama(Z, klase, pavadinimas, failas):
    y = klase.astype("category")
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=y.cat.codes, s=10, edgecolors=(0, 0, 0, 0.3), linewidths=0.3)
    plt.title(pavadinimas); plt.xlabel("t-SNE1"); plt.ylabel("t-SNE2")
    plt.gca().set_aspect("equal")
    plt.legend(sc.legend_elements()[0], y.cat.categories)
    plt.savefig(failas, dpi=150); plt.close()

diagrama(Z, duomenys["class"], "t-SNE: be išskirčių ir koreliuotų požymių", "tsne.png")

def stress1(X, Z):
    dX, dZ = pdist(X), pdist(Z)
    b = (dX * dZ).sum() / (dZ ** 2).sum()
    return np.sqrt(((dX - b * dZ) ** 2).sum() / (dX ** 2).sum())

def knn_tikslumas(Z, y):  
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return cross_val_score(KNeighborsClassifier(n_neighbors=5), Z, y, cv=cv).mean()

y = duomenys["class"]
print("PCA init: Stress-1* =", round(stress1(X, Z), 3))
print("PCA init: kNN tikslumas =", round(knn_tikslumas(Z, y), 3), "| pradinėje erdvėje:", round(knn_tikslumas(X, y), 3))
