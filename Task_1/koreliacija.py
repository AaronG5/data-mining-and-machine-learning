import pandas as pd

FILE = 'A27_svarus.csv'
FEATURES = ['Log_X_Index', 'Log_Y_Index', 'Empty_Index', 'Square_Index', 'Length_of_Conveyer',
            'Steel_Plate_Thickness', 'Edges_Index', 'Orientation_Index', 'LogOfAreas', 'Luminosity_Index']


def main():
   df = pd.read_csv(FILE, skipinitialspace=True)
   df.columns = df.columns.str.strip()

   rho = df[FEATURES].corr(method='spearman')
   rho.to_csv('koreliacija_spearman.csv', float_format='%.4f')


main()
