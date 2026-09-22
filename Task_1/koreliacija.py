import os
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(BASE_DIR, 'A27_svarus.csv')
OUT_DIR = os.path.join(BASE_DIR, 'kor_rez')
FEATURES = ['Log_X_Index', 'Log_Y_Index', 'Empty_Index', 'Square_Index', 'Length_of_Conveyer',
            'Steel_Plate_Thickness', 'Edges_Index', 'Orientation_Index', 'LogOfAreas', 'Luminosity_Index']

KLASĖS = {'Bumps': 0, 'Other_Faults': 1}


def main():
   df = pd.read_csv(FILE, skipinitialspace=True)
   df.columns = df.columns.str.strip()
   df['class'] = df['class'].str.strip()

   duomenys = df[FEATURES].copy()
   duomenys['class'] = df['class'].map(KLASĖS)

   os.makedirs(OUT_DIR, exist_ok=True)

   rho = duomenys.corr(method='spearman')
   rho.to_csv(os.path.join(OUT_DIR, 'koreliacija_spearman.csv'), float_format='%.4f')

   p = pd.DataFrame(stats.spearmanr(duomenys)[1], index=rho.index, columns=rho.columns)
   p.to_csv(os.path.join(OUT_DIR, 'koreliacija_p.csv'), float_format='%.3g')


main()
