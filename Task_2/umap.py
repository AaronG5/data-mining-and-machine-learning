import json
import math
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


# Experiment settings

N_RUNS = 30
DATASET = "C" # "A" originali, "B" be koreliuotu, "C" be isskirciu
ALIGN_TO_FIRST = True
SHOW_PLOTS = False

BASE_SEED = int(datetime.now().timestamp())
SEEDS = [BASE_SEED + i * 47 for i in range(N_RUNS)]
K_VALUES = [5, 15, 50]

# UMAP parameters

n_neighbors = 25
min_dist = 0.05
n_components = 2
metric = "euclidean"

# Graph settings

CLASS_COLORS = {"Bumps": "#440154", "Other_Faults": "#FDE724"}
POINT_SIZE = 16
POINT_ALPHA = 0.75
EDGE_COLOR = "#333333"
EDGE_WIDTH = 0.3
FIG_SIZE = (7, 7)
DPI = 200
PADDING = 0.05 

DATA_FILES = {
   "A": "A_originali_aibe.csv",
   "B": "B_be_koreliuotu.csv",
   "C": "C_be_isskirciu.csv",
}
DATA_FILE = Path(__file__).parent / DATA_FILES[DATASET]

OUTPUT_ROOT = Path(__file__).parent / "res"

umap_params = dict(
   n_neighbors=n_neighbors, n_components=n_components, metric=metric, min_dist=min_dist,
)

warnings.filterwarnings("ignore", message="n_jobs value .* overridden")

data = pd.read_csv(DATA_FILE, skipinitialspace=True)
klase = data["class"]

FEATURES = data.columns.drop("class").tolist()
X = data[FEATURES].to_numpy()

def knn_preservation(X_high, X_low, k):
   idx_high = NearestNeighbors(n_neighbors=k).fit(X_high).kneighbors(return_distance=False)
   idx_low = NearestNeighbors(n_neighbors=k).fit(X_low).kneighbors(return_distance=False)
   overlap = [len(set(h) & set(l)) / k for h, l in zip(idx_high, idx_low)]
   return float(np.mean(overlap))


def projection_metrics(X_high, X_low, d_high):
   rez = {}
   for k in K_VALUES:
      rez[f"T_{k}"] = trustworthiness(X_high, X_low, n_neighbors=k)
   for k in K_VALUES:
      rez[f"C_{k}"] = trustworthiness(X_low, X_high, n_neighbors=k)
   for k in K_VALUES:
      rez[f"kNN_pres_{k}"] = knn_preservation(X_high, X_low, k)
   d_low = pdist(X_low)
   rez["spearman_rho"] = spearmanr(d_high, d_low).statistic
   alpha = d_high @ d_low / (d_low @ d_low)
   rez["stress_1"] = np.sqrt(np.sum((d_high - alpha * d_low) ** 2) / np.sum(d_high ** 2))
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
   rez["knn_accuracy"] = cross_val_score(
      KNeighborsClassifier(n_neighbors=K_VALUES[0]), X_low, klase, cv=cv
   ).mean()
   return rez


def draw(ax, Z, title=""):
   centras = (Z.max(axis=0) + Z.min(axis=0)) / 2
   puse = (Z.max(axis=0) - Z.min(axis=0)).max() * (1 + PADDING) / 2
   for name, color in CLASS_COLORS.items():
      mask = (klase == name).to_numpy()
      ax.scatter(Z[mask, 0], Z[mask, 1], s=POINT_SIZE, alpha=POINT_ALPHA, color=color,
                 label=f"{name} ({mask.sum()})", edgecolors=EDGE_COLOR,
                 linewidths=EDGE_WIDTH)
   ax.set_xlim(centras[0] - puse, centras[0] + puse)
   ax.set_ylim(centras[1] - puse, centras[1] + puse)
   ax.set_xlabel("UMAP 1")
   ax.set_ylabel("UMAP 2")

   ax.set_aspect("equal")
   ax.grid(alpha=0.25)
   ax.set_title(title, fontsize=10)


param_tag = f"{DATASET}_nn{n_neighbors}_md{min_dist}_{metric}_f{len(FEATURES)}"
batch_dir = OUTPUT_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}_{param_tag}"
batch_dir.mkdir(parents=True, exist_ok=True)

with open(batch_dir / "params.json", "w") as f:
   json.dump({"seeds": SEEDS, "align_to_first": ALIGN_TO_FIRST, "dataset": DATASET,
              "data_file": DATA_FILE.name, "features": FEATURES, "umap": umap_params}, f, indent=2, default=str)

# All runs
embeddings = []
for i, seed in enumerate(SEEDS, start=1):
   print(f"[{i}/{len(SEEDS)}] UMAP, seed={seed}")
   Z = umap.UMAP(random_state=seed, **umap_params).fit_transform(X)
   embeddings.append(Z)

# Align so different graph runs look similiar
embeddings = [Z - Z.mean(axis=0) for Z in embeddings]
if ALIGN_TO_FIRST:
   ref = embeddings[0]
   embeddings = [Z @ orthogonal_procrustes(Z, ref)[0] for Z in embeddings]

param_text = f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"

# Separate graphs
d_high = pdist(X)
eilutes = []
for i, (seed, Z) in enumerate(zip(SEEDS, embeddings), start=1):
   pd.DataFrame(Z, columns=[f"UMAP{j + 1}" for j in range(Z.shape[1])]).assign(
      **{"class": klase}).to_csv(batch_dir / f"run_{i:02d}_seed{seed}.csv", index=False)

   fig, ax = plt.subplots(figsize=FIG_SIZE)
   draw(ax, Z, f"UMAP: {param_text}")
   ax.legend(title="Klasė", loc="best", markerscale=2, framealpha=0.9)
   fig.subplots_adjust(left=0.1, right=0.96, bottom=0.08, top=0.94)
   fig.savefig(batch_dir / f"run_{i:02d}_seed{seed}.png", dpi=DPI)
   plt.close(fig)

   print(f"[{i}/{len(SEEDS)}] metrikos")
   eilutes.append({"run": i, "seed": seed, **projection_metrics(X, Z, d_high)})

# All runs overview
cols = min(len(SEEDS), 3)
rows = math.ceil(len(SEEDS) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False)
for ax in axes.flat[len(SEEDS):]:
   ax.remove()
for ax, Z in zip(axes.flat, embeddings):
   draw(ax, Z)
fig_h = fig.get_figheight()
fig.suptitle(f"UMAP: {param_text}", y=1 - 0.1 / fig_h)
fig.tight_layout(rect=(0, 0.65 / fig_h, 1, 1 - 0.45 / fig_h))
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Klasė", loc="lower center", ncol=len(CLASS_COLORS),
            markerscale=2)
fig.savefig(batch_dir / "overview.png", dpi=DPI, bbox_inches="tight")

# Metrics
metrics = pd.DataFrame(eilutes)
metrics.to_csv(batch_dir / "metrics.csv", index=False)
summary = metrics.drop(columns=["run", "seed"]).agg(["mean", "std"]).T
summary.to_csv(batch_dir / "metrics_summary.csv")

print(f"\nIssaugota: {batch_dir}")
if SHOW_PLOTS:
   plt.show()
