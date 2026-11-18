import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

FAILAS = "../../apdoroti_duomenys/steel_plates_3_klases_normuota_robust.csv"
KAIMYNAI = [5, 15, 50]


def ikelti(failas):
    duomenys = pd.read_csv(failas)
    klase = duomenys["class"].astype("category")
    return duomenys.drop(columns="class"), klase.cat.codes.values, klase.cat.categories


X, kodai, klases = ikelti(FAILAS)

pca = PCA().fit(X)
Z = PCA(n_components=2).fit_transform(X)


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


def didziausi_svoriai(pca, stulpeliai, n=5):
    svoriai = pd.DataFrame(pca.components_[:2].T, index=stulpeliai, columns=["PC1", "PC2"])
    for pc in svoriai:
        didziausi = svoriai[pc].reindex(svoriai[pc].abs().nlargest(n).index)
        print(f"{pc}:")
        print(didziausi.round(2).to_string())


def piesti(Z, kodai, pavadinimas, failas):
    plt.figure(figsize=(6.5, 5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=kodai, s=8, alpha=0.7)
    plt.legend(sc.legend_elements()[0], klases)
    plt.title(pavadinimas)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(failas, dpi=150)
    plt.close()


print("Kaupiama paaiškinta dispersija (%):")
print(pd.Series(pca.explained_variance_ratio_.cumsum()[:7] * 100,
                index=[f"{d}D" for d in range(1, 8)]).round(1).to_string())

print("\nPaaiškinta dispersija (%) pagal komponentes:")
print(pd.Series(pca.explained_variance_ratio_[:8] * 100,
                index=[f"PC{i}" for i in range(1, 9)]).round(1).to_string())

print("\nDidžiausi požymių svoriai:")
didziausi_svoriai(pca, X.columns)

print("\n2D projekcijos kokybė:")
print(pd.Series(kokybe(X, Z, kodai)).round(3).to_string())

tikslumas = cross_val_score(KNeighborsClassifier(5), Z, kodai,
                            cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
print(f"\nkNN tikslumas (2D): {tikslumas * 100:.1f} %")

piesti(Z, kodai, "PCA projekcija", "pca.png")
