import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

duomenys = pd.read_csv("A27_normuota_robust.csv")
klase = duomenys["class"]
X = duomenys.drop(columns="class")

pca = PCA()
Z = pca.fit_transform(X)

dispersija = pca.explained_variance_ratio_
print("Paaiskinta dispersija, %:", (dispersija * 100).round(1))
print("Kaupiama dispersija, %:", (dispersija.cumsum() * 100).round(1))

plt.scatter(Z[:, 0], Z[:, 1], c=klase.astype("category").cat.codes, s=7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()