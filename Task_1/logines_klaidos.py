import pandas as pd

FILE = 'A27_medianos_pagal_klase.csv'
CLASSES = ['Bumps', 'Other_Faults']

FEATURES = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
            'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
            'Maximum_of_Luminosity', 'Length_of_Conveyer', 'Steel_Plate_Thickness',
            'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
            'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas',
            'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index',
            'SigmoidOfAreas']

RANGE_RULES = {
   'X_Minimum': (0, None),
   'X_Maximum': (0, None),
   'Y_Minimum': (0, None),
   'Y_Maximum': (0, None),
   'Pixels_Areas': (0, None),
   'X_Perimeter': (0, None),
   'Y_Perimeter': (0, None),
   'Sum_of_Luminosity': (0, None),
   'Minimum_of_Luminosity': (0, 255),
   'Maximum_of_Luminosity': (0, 255),
   'Length_of_Conveyer': (0, None),
   'Steel_Plate_Thickness': (0, None),
   'Edges_Index': (0, 1),
   'Empty_Index': (0, 1),
   'Square_Index': (0, 1),
   'Outside_X_Index': (0, 1),
   'Edges_X_Index': (0, 1),
   'Edges_Y_Index': (0, 1),
   'Outside_Global_Index': (0, 1),
   'LogOfAreas': (0, None),
   'Log_X_Index': (0, None),
   'Log_Y_Index': (0, None),
   'Orientation_Index': (-1, 1),
   'Luminosity_Index': (-1, 1),
   'SigmoidOfAreas': (0, 1),
}
EXCLUSIVE_LOWER = ['Pixels_Areas', 'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity',
                    'Length_of_Conveyer', 'Steel_Plate_Thickness']

df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()
df['class'] = df['class'].str.strip()

def pazeidimai(part: pd.DataFrame, col: str):
   lower, upper = RANGE_RULES[col]
   klaidos = pd.Series(False, index=part.index)

   if lower is not None:
      klaidos |= (part[col] <= lower) if col in EXCLUSIVE_LOWER else (part[col] < lower)
   if upper is not None:
      klaidos |= part[col] > upper

   return klaidos

def rysio_pazeidimai(part: pd.DataFrame):
   klaidos = pd.Series(False, index=part.index)
   klaidos |= part['X_Minimum'] > part['X_Maximum']
   klaidos |= part['Y_Minimum'] > part['Y_Maximum']
   klaidos |= part['Minimum_of_Luminosity'] > part['Maximum_of_Luminosity']
   return klaidos

def rasti_logines_klaidas(part: pd.DataFrame):
   klaidos = pd.Series(False, index=part.index)

   for col in FEATURES:
      klaidos |= pazeidimai(part, col)

   klaidos |= rysio_pazeidimai(part)

   return klaidos

def klaidu_ataskaita(part: pd.DataFrame):
   rows = []

   for col in FEATURES:
      lower, upper = RANGE_RULES[col]
      taisykle = f"{'>' if col in EXCLUSIVE_LOWER else '>='} {lower}" if upper is None else f"[{lower}, {upper}]"
      n = pazeidimai(part, col).sum()
      rows.append({'Požymis': col, 'Taisyklė': taisykle, 'Pažeidimų': int(n)})

   rows.append({'Požymis': 'X_Minimum <= X_Maximum', 'Taisyklė': 'santykis',
                'Pažeidimų': int((part['X_Minimum'] > part['X_Maximum']).sum())})
   rows.append({'Požymis': 'Y_Minimum <= Y_Maximum', 'Taisyklė': 'santykis',
                'Pažeidimų': int((part['Y_Minimum'] > part['Y_Maximum']).sum())})
   rows.append({'Požymis': 'Min_Luminosity <= Max_Luminosity', 'Taisyklė': 'santykis',
                'Pažeidimų': int((part['Minimum_of_Luminosity'] > part['Maximum_of_Luminosity']).sum())})

   return pd.DataFrame(rows)

def export_table_csv(table, filename):
   table.to_csv(filename + '.csv', index=False, float_format='%.4f')

ataskaita = klaidu_ataskaita(df)
export_table_csv(ataskaita, 'logines_klaidos_ataskaita')
print(ataskaita)

klaidingos_eilutes = rasti_logines_klaidas(df)
print(f'\nEilučių su logine klaida: {klaidingos_eilutes.sum()} iš {len(df)}')

df_svarus = df[~klaidingos_eilutes].reset_index(drop=True)
df_svarus.to_csv('A27_be_loginiu_klaidu.csv', index=False)