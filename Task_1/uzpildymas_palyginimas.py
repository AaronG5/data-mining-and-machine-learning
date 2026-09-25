import pandas as pd

FILE = 'A27/A27.csv'
CLASSES = ['Bumps', 'Other_Faults']
MISSING_MARKERS = ['unknown', 'error', 'not_measured', '?']

df = pd.read_csv(FILE, skipinitialspace=True)
df.columns = df.columns.str.strip()
df['class'] = df['class'].str.strip()

df['Steel_Plate_Thickness'] = pd.to_numeric(
   df['Steel_Plate_Thickness'].astype(str).str.replace('mm', '', regex=False).str.strip(), errors='coerce')

FEATURES = [col for col in df.columns if col != 'class']

for col in FEATURES:
   df[col] = pd.to_numeric(df[col].replace(MISSING_MARKERS, pd.NA), errors='coerce')


FILL_FEATURES = [col for col in FEATURES if df[col].isna().any()]
print('Columns with missing values:', FILL_FEATURES)

def uzpildyti_pagal_klase(part: pd.DataFrame, budas: str):
   filled = part.copy()

   for col in FILL_FEATURES:
      if budas == 'vidurkis':
         filled[col] = filled.groupby('class')[col].transform(lambda s: s.fillna(s.mean()))
      else:
         filled[col] = filled.groupby('class')[col].transform(lambda s: s.fillna(s.median()))

   return filled

def klasiu_statistika(originalas: pd.DataFrame, vidurkiu: pd.DataFrame, mediana: pd.DataFrame):
   rows = []

   for col in FILL_FEATURES:
      for class_name in CLASSES:
         tikra = originalas.loc[originalas['class'] == class_name, col]

         vid_stulpelis = vidurkiu.loc[vidurkiu['class'] == class_name, col]
         med_stulpelis = mediana.loc[mediana['class'] == class_name, col]

         rows.append({
            'Požymis': col,
            'Klasė': class_name,
            'Trūkstamos reikšmės': int(tikra.isna().sum()),
            'Tikras vidurkis (be trūkstamų)': tikra.mean(),
            'Tikra mediana (be trūkstamų)': tikra.median(),
            'Vidurkis po užpildymo vidurkiu': vid_stulpelis.mean(),
            'Vidurkis po užpildymo mediana': med_stulpelis.mean(),
            'Std. po užpildymo vidurkiu': vid_stulpelis.std(ddof=1),
            'Std. po užpildymo mediana': med_stulpelis.std(ddof=1),
            'Asimetrija po užpildymo vidurkiu': vid_stulpelis.skew(),
            'Asimetrija po užpildymo mediana': med_stulpelis.skew(),
         })

   return pd.DataFrame(rows)

def export_table_csv(table, filename):
   table.to_csv(filename + '.csv', index=False, float_format='%.4f')

df_vidurkiu = uzpildyti_pagal_klase(df, 'vidurkis')
df_mediana = uzpildyti_pagal_klase(df, 'mediana')

filename = 'uzpildymas_pagal_klase'
table = klasiu_statistika(df, df_vidurkiu, df_mediana)
export_table_csv(table, filename)

df_vidurkiu.to_csv('A27_vidurkiu_pagal_klase.csv', index=False)
df_mediana.to_csv('A27_medianos_pagal_klase.csv', index=False)