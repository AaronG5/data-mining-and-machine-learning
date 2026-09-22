import os
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(BASE_DIR, 'A27_svarus.csv')
FEATURES = ['Log_X_Index', 'Log_Y_Index', 'Empty_Index', 'Square_Index', 'Length_of_Conveyer',
            'Steel_Plate_Thickness', 'Edges_Index', 'Orientation_Index', 'LogOfAreas', 'Luminosity_Index']


def main():
   df = pd.read_csv(FILE, skipinitialspace=True)
   df.columns = df.columns.str.strip()

   rho = df[FEATURES].corr(method='spearman')
   rho.to_csv(os.path.join(BASE_DIR, 'koreliacija_spearman.csv'), float_format='%.4f')

   p = pd.DataFrame(stats.spearmanr(df[FEATURES])[1], index=FEATURES, columns=FEATURES)
   p.to_csv(os.path.join(BASE_DIR, 'koreliacija_p.csv'), float_format='%.3g')


main()
