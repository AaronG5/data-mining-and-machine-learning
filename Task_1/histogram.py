import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_distribution(values: pd.Series, title: str, out_dir: str, save_as_png: bool=True, show: bool=False, bins=30) -> None:
   values = pd.to_numeric(values, errors='coerce').dropna()
   mean = values.mean()
   std = values.std()
   q1, median, q3 = values.quantile([0.25, 0.5, 0.75])

   plt.figure(figsize=(9, 5))

   plt.hist(values, bins, density=True, alpha=0.5, color='steelblue', edgecolor='black')

   if std > 0:
      x = np.linspace(values.min(), values.max(), 500)
      pdf = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
      plt.plot(x, pdf, color='black', lw=2)

   plt.axvline(mean, color='red', label=f'Vidurkis = {mean:.2f}')
   plt.axvline(median, color='green', label=f'Mediana = {median:.2f}')
   plt.axvline(q1, color='orange', label=f'1-as kvartilis = {q1:.2f}')
   plt.axvline(q3, color='purple', label=f'3-as kvartilis = {q3:.2f}')

   plt.title(title)
   plt.xlabel('Įgyjamų reikšmių intervalai')
   plt.ylabel('Dažnis')
   plt.legend(fontsize=8)

   if save_as_png:
      file_dest = title + '.png'
      plt.savefig(os.path.join(out_dir, file_dest), dpi=150, bbox_inches='tight')

   if show:
      plt.show()

   plt.close()