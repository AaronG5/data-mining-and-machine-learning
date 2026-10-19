import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

duomenys = pd.read_csv("A27_normuota_robust.csv")
klase = duomenys["class"]
X = duomenys.drop(columns="class")

pca = PCA(n_components=3)
Z = pca.fit_transform(X)

dispersija = pca.explained_variance_ratio_
print("Paaiskinta dispersija, %:", (dispersija * 100).round(1))
print("3 komponentes kartu, %:", round(dispersija.sum() * 100, 1))

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=klase.astype("category").cat.codes, s=7)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()