import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_distribution(values: pd.Series, title: str, out_dir: str, class_name: str=None, save_as_png: bool=True, show: bool=False, bins=10, x_min: float=None, x_max: float=None) -> None:
   values = pd.to_numeric(values, errors='coerce').dropna()
   mean = values.mean()
   q1, median, q3 = values.quantile([0.25, 0.5, 0.75])

   range_min = x_min if x_min is not None else values.min()
   range_max = x_max if x_max is not None else values.max()

   if isinstance(bins, int):
      bin_edges = np.linspace(range_min, range_max, bins + 1)
   else:
      bin_edges = bins

   plt.figure(figsize=(9, 5))

   _, edges, _ = plt.hist(values, bin_edges, range=(range_min, range_max), density=True, alpha=0.5, color='steelblue', edgecolor='black')

   if title == 'Length_of_Conveyer' or title == 'Steel_Plate_Thickness':
      plt.xticks(edges, [f'{e}' for e in edges])
   else:
      plt.xticks(edges, [f'{e:.4f}' for e in edges])
   plt.xlim((range_min, range_max))

   plt.axvline(mean, color='red', label=f'Vidurkis = {mean:.2f}')
   plt.axvline(median, color='green', label=f'Mediana = {median:.2f}')
   plt.axvline(q1, color='orange', label=f'1-as kvartilis = {q1:.2f}')
   plt.axvline(q3, color='blue', label=f'3-as kvartilis = {q3:.2f}')

   if class_name is None:
      plt.title(f'Bendra \"{title}\" požymio histograma')
   else: 
      plt.title(f'\"{class_name}\" klasės \"{title}\" požymio histograma')
   plt.xlabel('Įgyjamų reikšmių intervalai')
   plt.ylabel('Tankis')
   plt.legend(fontsize=8)

   if save_as_png:
      file_dest = title + '.png'
      plt.savefig(os.path.join(out_dir, file_dest), dpi=300, bbox_inches='tight')

   if show:
      plt.show()

   plt.close()