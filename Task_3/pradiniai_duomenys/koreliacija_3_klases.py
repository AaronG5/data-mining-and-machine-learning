import os
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(BASE_DIR, '..', 'steel_plates_3_klases.csv')
OUT_DIR = os.path.join(BASE_DIR, 'kor_rez')

KLASĖS = ['K_Scatch', 'Z_Scratch', 'Bumps']


def main():
   df = pd.read_csv(FILE, skipinitialspace=True)
   df.columns = df.columns.str.strip()
   df['class'] = df['class'].str.strip()

   duomenys = df.drop(columns='class').copy()
   duomenys[[f'is_{kl}' for kl in KLASĖS]] = pd.get_dummies(df['class'])[KLASĖS].astype(int)

   os.makedirs(OUT_DIR, exist_ok=True)

   rho = duomenys.corr(method='spearman')
   rho.to_csv(os.path.join(OUT_DIR, 'koreliacija_spearman_3kl.csv'))

   p = pd.DataFrame(stats.spearmanr(duomenys)[1], index=rho.index, columns=rho.columns)
   p.to_csv(os.path.join(OUT_DIR, 'koreliacija_p_3kl.csv'), float_format='%.3g')

   #požymių poros su rho >= 0.9
   pozymiai = df.drop(columns='class').columns
   rows = []
   for i, a in enumerate(pozymiai):
      for b in pozymiai[i + 1:]:
         if abs(rho.loc[a, b]) >= 0.9:
            rows.append({'Požymis_1': a, 'Požymis_2': b, 'rho': rho.loc[a, b]})
   ataskaita = pd.DataFrame(rows).sort_values('rho', key=abs, ascending=False)
   ataskaita.to_csv(os.path.join(OUT_DIR, 'koreliacijos_ataskaita_3kl.csv'), index=False, float_format='%.4f')
   print(ataskaita.to_string(index=False))


main()
