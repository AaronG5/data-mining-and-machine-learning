import os
import pandas as pd
import matplotlib.pyplot as plt

FILE = 'A27_be_loginiu_klaidu.csv'
CLASSES = ['Bumps', 'Other_Faults']

FEATURES = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
            'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
            'Maximum_of_Luminosity', 'Length_of_Conveyer', 'Steel_Plate_Thickness',
            'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
            'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas',
            'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index',
            'SigmoidOfAreas']

MILD_K = 1.5
EXTREME_K = 3.0
VERY_EXTREME_K = 10.0
SKEW_THRESHOLD = 5.0

CHARTS_DIR = 'grafikai_isskirtys'
CHARTS_DIR_SVARUS = 'grafikai_be_isskirciu'

df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()
df['class'] = df['class'].str.strip()


def ribos(series: pd.Series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return {
        'q1': q1, 'q3': q3, 'iqr': iqr,
        'mild_low': q1 - MILD_K * iqr, 'mild_high': q3 + MILD_K * iqr,
        'ext_low': q1 - EXTREME_K * iqr, 'ext_high': q3 + EXTREME_K * iqr,
    }


def rasti_isskirtis(part: pd.DataFrame, col: str):
    r = ribos(part[col])
    is_ext = (part[col] < r['ext_low']) | (part[col] > r['ext_high'])
    is_mild = ((part[col] < r['mild_low']) | (part[col] > r['mild_high'])) & ~is_ext
    tipas = pd.Series('normal', index=part.index)
    tipas[is_mild] = 'mild'
    tipas[is_ext] = 'extreme'
    return tipas, r


def isskirciu_ataskaita(data: pd.DataFrame):
    rows = []
    for col in FEATURES:
        for kl in CLASSES:
            part = data[data['class'] == kl]
            tipas, r = rasti_isskirtis(part, col)
            rows.append({
                'Požymis': col,
                'Klasė': kl,
                'Paprastų_išskirčių': int((tipas == 'mild').sum()),
                'Ekstremalių_išskirčių': int((tipas == 'extreme').sum())
            })
    return pd.DataFrame(rows)


def braizyti_grafika(data: pd.DataFrame, col: str, filename: str):
    fig, ax = plt.subplots(figsize=(9, 1.3 * len(CLASSES) + 0.8))

    for i, kl in enumerate(CLASSES):
        part = data[data['class'] == kl]
        tipas, r = rasti_isskirtis(part, col)
        y = i

        vidiniai = part[col][(part[col] >= r['mild_low']) & (part[col] <= r['mild_high'])]
        whisker_low = vidiniai.min() if len(vidiniai) else part[col].min()
        whisker_high = vidiniai.max() if len(vidiniai) else part[col].max()
        median = part[col].median()

        ax.plot([whisker_low, r['q1']], [y, y], color='black', lw=0.8, linestyle='--', zorder=1)
        ax.plot([r['q3'], whisker_high], [y, y], color='black', lw=0.8, linestyle='--', zorder=1)
        ax.plot([whisker_low, whisker_low], [y - 0.12, y + 0.12], color='black', lw=0.8)
        ax.plot([whisker_high, whisker_high], [y - 0.12, y + 0.12], color='black', lw=0.8)

        ax.add_patch(plt.Rectangle((r['q1'], y - 0.28), r['q3'] - r['q1'], 0.56,
                                    facecolor='lightgray', edgecolor='black', zorder=2))
        ax.plot([median, median], [y - 0.28, y + 0.28], color='black', lw=1.4, zorder=3)

        mild_pts = part[col][tipas == 'mild']
        ext_pts = part[col][tipas == 'extreme']
        ax.scatter(mild_pts, [y] * len(mild_pts), color='orange', s=18, zorder=4)
        ax.scatter(ext_pts, [y] * len(ext_pts), color='darkred', s=18, zorder=4)

    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels([f'{kl}' for i, kl in enumerate(CLASSES)])
    ax.set_title(col)
    ax.set_xlabel(col)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)


def rasti_ilgauodegius_pozymius(data: pd.DataFrame):
    skew_per_feature = {col: data[col].skew() for col in FEATURES}
    ilgauodegiai = [col for col, skew in skew_per_feature.items() if abs(skew) > SKEW_THRESHOLD]
    print('Ilgauodegiai požymiai (skew > {:.0f}): {}'.format(SKEW_THRESHOLD, ilgauodegiai))
    return ilgauodegiai


def virsija_griezta_tvora(part: pd.DataFrame, col: str):
    q1, q3 = part[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    zema, aukšta = q1 - VERY_EXTREME_K * iqr, q3 + VERY_EXTREME_K * iqr
    return (part[col] < zema) | (part[col] > aukšta)


def rasti_virsijancias_eilutes(data: pd.DataFrame, pozymiai: list):
    virsija = pd.Series(False, index=data.index)

    for col in pozymiai:
        for kl in CLASSES:
            part = data[data['class'] == kl]
            mask = virsija_griezta_tvora(part, col)
            virsija.loc[mask[mask].index] = True

    return virsija


def export_table_csv(table, filename):
    table.to_csv(filename + '.csv', index=False, float_format='%.4f')


os.makedirs(CHARTS_DIR, exist_ok=True)

ataskaita = isskirciu_ataskaita(df)
export_table_csv(ataskaita, 'isskirciu_ataskaita')

for col in FEATURES:
    braizyti_grafika(df, col, os.path.join(CHARTS_DIR, f'{col}.png'))

ilgauodegiai = rasti_ilgauodegius_pozymius(df)
virsijancios_eilutes = rasti_virsijancias_eilutes(df, ilgauodegiai)
print(f'Pašalinta eilučių (virš {VERY_EXTREME_K:.0f}xIQR tvoros ilgauodegiuose požymiuose): '
      f'{virsijancios_eilutes.sum()} iš {len(df)}')

df_svarus = df[~virsijancios_eilutes].reset_index(drop=True)
df_svarus.to_csv('A27_be_virsutiniu_isskirciu.csv', index=False)

os.makedirs(CHARTS_DIR_SVARUS, exist_ok=True)
for col in FEATURES:
    braizyti_grafika(df_svarus, col, os.path.join(CHARTS_DIR_SVARUS, f'{col}.png'))