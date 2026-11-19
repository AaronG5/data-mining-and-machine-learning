from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

DATA_FILE: Path = Path(__file__).parent.parent.parent / "dimensijos_mazinimas" / "steel_plates_3_klases_normuota_robust.csv"
OUTPUT_ROOT: Path = Path(__file__).parent / "res"
STATE: int = 123

init_data: pd.DataFrame = pd.read_csv(DATA_FILE, skipinitialspace=True)
classes: pd.Series = init_data["class"]

data: np.ndarray = init_data.drop(columns="class").to_numpy()

def findElbow(k_empirical: int) -> None:
   tss: np.float64 = ((data - data.mean(axis=0)) ** 2).sum()

   ks: range = range(1, k_empirical + 1)
   wcss: list[int] = []

   for k in ks:
      model: KMeans = KMeans(n_clusters=k, n_init=10, random_state=STATE).fit(data)  # pyright: ignore[reportArgumentType]
      wcss.append(model.inertia_)  # pyright: ignore[reportArgumentType]

   explained: list[np.float64] = [100 * (1 - w / tss) for w in wcss]

   plt.plot(ks, explained, marker="o")
   plt.xlabel("k")
   plt.ylabel("Explained variance %")
   plt.savefig(fname=OUTPUT_ROOT / "knee.png", dpi=250)
   # plt.show()

def main() -> None:
   k_empirical: int = round(np.sqrt(len(data) / 2))

   findElbow(k_empirical)


main()
