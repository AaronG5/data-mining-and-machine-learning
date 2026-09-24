import pandas as pd

FILE = 'A27_medianos_pagal_klase.csv'
CLASSES = ['Bumps', 'Other_Faults']
FEATURES = ['Log_X_Index', 'Log_Y_Index', 'Empty_Index', 'Square_Index', 'Length_of_Conveyer',
             'Steel_Plate_Thickness', 'Edges_Index', 'Orientation_Index', 'LogOfAreas', 'Luminosity_Index']
ZERO_ONE_FEATURES = ['Empty_Index', 'Square_Index', 'Edges_Index']
POSITIVE_FEATURES = ['Length_of_Conveyer', 'Steel_Plate_Thickness']

df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()
df['class'] = df['class'].str.strip()

def rasti_logines_klaidas(part: pd.DataFrame):
   klaidos = pd.Series(False, index=part.index)

   for col in ZERO_ONE_FEATURES:
      klaidos |= (part[col] < 0) | (part[col] > 1)

   klaidos |= (part['Orientation_Index'] < -1) | (part['Orientation_Index'] > 1)

   for col in POSITIVE_FEATURES:
      klaidos |= part[col] <= 0

   return klaidos

def klaidu_ataskaita(part: pd.DataFrame):
   rows = []

   for col in ZERO_ONE_FEATURES:
      n = ((part[col] < 0) | (part[col] > 1)).sum()
      rows.append({'Požymis': col, 'Taisyklė': '[0, 1]', 'Pažeidimų': int(n)})

   n = ((part['Orientation_Index'] < -1) | (part['Orientation_Index'] > 1)).sum()
   rows.append({'Požymis': 'Orientation_Index', 'Taisyklė': '[-1, 1]', 'Pažeidimų': int(n)})

   for col in POSITIVE_FEATURES:
      n = (part[col] <= 0).sum()
      rows.append({'Požymis': col, 'Taisyklė': '> 0', 'Pažeidimų': int(n)})

   return pd.DataFrame(rows)

def export_table_csv(table, filename):
   table.to_csv(filename + '.csv', index=False, float_format='%.4f')

ataskaita = klaidu_ataskaita(df)
export_table_csv(ataskaita, 'logines_klaidos_ataskaita')
print(ataskaita)

klaidingos_eilutes = rasti_logines_klaidas(df)

df_svarus = df[~klaidingos_eilutes].reset_index(drop=True)
df_svarus.to_csv('A27_be_loginiu_klaidu.csv', index=False)