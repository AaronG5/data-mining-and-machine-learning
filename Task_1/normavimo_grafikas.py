import os
import pandas as pd
import matplotlib.pyplot as plt

# Iš apačios į viršų: nuo ~1 iki ~10^7
FEATURES = ['Edges_Index', 'Luminosity_Index', 'Steel_Plate_Thickness',
            'Pixels_Areas', 'Sum_of_Luminosity', 'Y_Minimum']

pries = pd.read_csv('A27_be_virsutiniu_isskirciu.csv')[FEATURES]
po = pd.read_csv('A27_normuota_robust.csv')[FEATURES]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, df, title, xlabel in [(ax1, pries, 'Prieš normavimą', 'Pradinė reikšmė, mln.'),
                              (ax2, po, 'Po normavimo', '(x − mediana) / IQR')]:
    ax.boxplot(df, vert=False, tick_labels=FEATURES, patch_artist=True, widths=0.56,
               boxprops=dict(facecolor='lightgray', edgecolor='black'),
               medianprops=dict(color='black', linewidth=1.4),
               whiskerprops=dict(linestyle='--', linewidth=0.8),
               capprops=dict(linewidth=0.8),
               flierprops=dict(markerfacecolor=(0.41, 0.41, 0.41, 0.3), markeredgecolor='dimgray',
                               markeredgewidth=0.6, markersize=4))
    ax.set(title=title, xlabel=xlabel)

ax1.xaxis.set_major_formatter(lambda x, _: f'{x / 1e6:g}')
ax2.axvline(0, color='grey', linewidth=0.8)

os.makedirs('grafikai_normavimas', exist_ok=True)
fig.savefig('grafikai_normavimas/pries_po.png', dpi=150, bbox_inches='tight')
