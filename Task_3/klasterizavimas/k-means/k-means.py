from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

DATA_FILE: Path = Path(__file__).parent.parent.parent / "dimensijos_mazinimas" / "steel_plates_3_klases_normuota_robust.csv"
OUTPUT_ROOT: Path = Path(__file__).parent / "res"
STATE: int = 123

init_data: pd.DataFrame = pd.read_csv(DATA_FILE, skipinitialspace=True)
classes: pd.Series = init_data["class"]

data: np.ndarray = init_data.drop(columns="class").to_numpy()

def findElbow(k_empirical: int, n_init: int=10) -> None:
   tss: np.float64 = ((data - data.mean(axis=0)) ** 2).sum()

   ks: range = range(1, k_empirical + 1)
   wcss: list[int] = []

   for k in ks:
      model: KMeans = KMeans(n_clusters=k, n_init=n_init, random_state=STATE).fit(data)  # pyright: ignore[reportArgumentType]
      wcss.append(model.inertia_)  # pyright: ignore[reportArgumentType]

   explained: list[np.float64] = [100 * (1 - w / tss) for w in wcss]

   plt.plot(ks, explained, marker="o")
   plt.xlabel("k")
   plt.ylabel("Explained variance %")
   plt.savefig(fname=OUTPUT_ROOT / "knee.png", dpi=250)
   plt.close()

def findSilhouette(k_empirical: int, n_init: int=10) -> list[float]:
   scores: list[float] = []
   for k in range(2, k_empirical + 1):
      labels: np.ndarray = KMeans(n_clusters=k, n_init=n_init, random_state=STATE).fit_predict(data) # pyright: ignore[reportArgumentType]
      scores.append(silhouette_score(data, labels))

   return scores

def plotSilhouette(k:int, n_init: int=10) -> None:
   CLUSTER_COLORS: list[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
   labels: np.ndarray = KMeans(n_clusters=k, n_init=n_init, random_state=STATE).fit_predict(data)  # pyright: ignore[reportArgumentType]
   values: np.ndarray = silhouette_samples(data, labels)  # s(i) for every row
   average: float = values.mean()                         # = silhouette_score

   fig, ax = plt.subplots(figsize=(8, 6))
   gap: int = 15            # empty rows between clusters
   y_start: int = 0
   tick_positions: list[float] = []
   tick_labels: list[str] = []

   for cluster in range(k):
      cluster_values: np.ndarray = np.sort(values[labels == cluster])[::-1]  # widest bar on top
      y: np.ndarray = np.arange(y_start, y_start + len(cluster_values))      # one row per point

      ax.fill_betweenx(y, 0, cluster_values, color=CLUSTER_COLORS[cluster], linewidth=0, label=f"Cluster {cluster}")

      tick_positions.append(y_start + len(cluster_values) / 2)
      tick_labels.append(f"Cluster {cluster}\nn = {len(cluster_values)}")
      y_start += len(cluster_values) + gap

   ax.axvline(average, color="black", linestyle="--", linewidth=1)
   ax.text(average + 0.01, -gap, f"average = {average:.3f}", fontsize=9, va="bottom")
   ax.axvline(0, color="#c3c2b7", linewidth=1)

   ax.set_yticks(tick_positions, tick_labels)
   ax.set_ylim(y_start, -gap * 3)                       # flipped, so cluster 0 is on top
   ax.set_xlim(min(values.min(), 0) - 0.05, 1)          # leaves room if any s(i) < 0
   ax.set_title(f"Silhouette plot, k = {k}", loc="left")
   ax.set_xlabel("Silhouette coefficient s(i)")
   ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
   ax.set_axisbelow(True)
   for side in ("top", "right", "left"):
      ax.spines[side].set_visible(False)
   ax.legend(loc="lower right", frameon=False)

   fig.tight_layout()
   fig.savefig(OUTPUT_ROOT / f"silhouette_k{k}.png", dpi=250)
   plt.close(fig)

def main() -> None:
   k_empirical: int = round(np.sqrt(len(data) / 2))

   findElbow(k_empirical)
   silhouette_scores: list[float] = findSilhouette(k_empirical)
   plotSilhouette(int(np.argmax(silhouette_scores)) + 2)

main()
