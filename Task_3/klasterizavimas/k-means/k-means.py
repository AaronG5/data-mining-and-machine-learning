from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
   silhouette_samples,
   silhouette_score,
)

DATA_FILE: Path = (
   Path(__file__).parent.parent.parent / "dimensijos_mazinimas" / "steel_plates_3_klases_normuota_robust.csv"
)
OUTPUT_ROOT: Path = Path(__file__).parent / "res"
STATE: int = 123

init_data: pd.DataFrame = pd.read_csv(DATA_FILE, skipinitialspace=True)
classes: pd.Series = init_data["class"]

data: np.ndarray = init_data.drop(columns="class").to_numpy()
tss: float = ((data - data.mean(axis=0)) ** 2).sum()

K_RANGE: range = range(2, 11)


def fitModels(ks: range, n_init: int = 10) -> dict[int, KMeans]:
   return {
      k: KMeans(n_clusters=k, n_init=n_init, random_state=STATE).fit(data)  # pyright: ignore[reportArgumentType]
      for k in ks
   }


def findElbow(models: dict[int, KMeans]) -> list[float]:
   return [float(model.inertia_) for model in models.values()]  # pyright: ignore[reportArgumentType]


def findSilhouette(models: dict[int, KMeans]) -> list[float]:
   return [float(silhouette_score(data, model.labels_)) for model in models.values()]


def plotElbow(ks: list[int], wcss: list[float]) -> None:
   plt.plot(ks, wcss, marker="o")
   plt.title("Elbow method")
   plt.xlabel("Number of clusters k")
   plt.ylabel("WCSS")
   plt.savefig(fname=OUTPUT_ROOT / "knee.png", dpi=300)
   plt.close()


def evaluateK(ks: range = K_RANGE, n_init: int = 10) -> pd.DataFrame:
   models: dict[int, KMeans] = fitModels(ks, n_init)
   wcss: list[float] = findElbow(models)

   results: pd.DataFrame = pd.DataFrame(
      {
         "k": list(ks),
         "wcss": wcss,
         "explained_variance_pct": [100 * (tss - w) / tss for w in wcss],  # BCSS / TSS, with BCSS = TSS - WCSS
         "silhouette": findSilhouette(models),
      }
   )
   results.to_csv(OUTPUT_ROOT / "k_metrics.csv", index=False)
   plotElbow(list(ks), wcss)

   return results


def plotSilhouette(k: int, n_init: int = 10) -> None:
   CLUSTER_COLORS: list[str] = [
      "#2a78d6",
      "#eb6834",
      "#1baf7a",
      "#eda100",
      "#e87ba4",
      "#008300",
      "#6250d6",
      "#e34948",
   ]
   labels: np.ndarray = KMeans(
      n_clusters=k,
      n_init=n_init,  # pyright: ignore[reportArgumentType]
      random_state=STATE,
   ).fit_predict(data)
   values: np.ndarray = silhouette_samples(data, labels)
   average: float = values.mean()  # = silhouette_score

   fig, ax = plt.subplots(figsize=(8, 6))
   gap: int = 15  # empty rows between clusters
   y_start: int = 0
   tick_positions: list[float] = []
   tick_labels: list[str] = []

   for cluster in range(k):
      cluster_values: np.ndarray = np.sort(values[labels == cluster])[::-1]
      y: np.ndarray = np.arange(y_start, y_start + len(cluster_values))

      ax.fill_betweenx(
         y,
         0,
         cluster_values,
         color=CLUSTER_COLORS[cluster],
         linewidth=0,
         label=f"Cluster {cluster}",
      )

      tick_positions.append(y_start + len(cluster_values) / 2)
      tick_labels.append(f"Cluster {cluster}\nn = {len(cluster_values)}")
      y_start += len(cluster_values) + gap

   ax.axvline(average, color="black", linestyle="--", linewidth=1)
   ax.text(average + 0.01, -gap, f"average = {average:.3f}", fontsize=9, va="bottom")
   ax.axvline(0, color="#c3c2b7", linewidth=1)

   ax.set_yticks(tick_positions, tick_labels)
   ax.set_ylim(y_start, -gap * 3)
   ax.set_xlim(min(values.min(), 0) - 0.05, 1)
   ax.set_title(f"Silhouette plot, k = {k}", loc="left")
   ax.set_xlabel("Silhouette coefficient s(i)")
   ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
   ax.set_axisbelow(True)
   for side in ("top", "right", "left"):
      ax.spines[side].set_visible(False)
   ax.legend(loc="lower right", frameon=False)

   fig.tight_layout()
   fig.savefig(OUTPUT_ROOT / f"silhouette_k{k}.png", dpi=300)
   plt.close(fig)


def main() -> None:
   k_empirical: int = round(np.sqrt(len(data) / 2))
   print(f"Empirical k = sqrt(n / 2) = {k_empirical}")

   results: pd.DataFrame = evaluateK()
   print(results.round(3).to_string(index=False))

   best_k: int = int(results.loc[results["silhouette"].idxmax(), "k"])
   plotSilhouette(best_k)


main()
