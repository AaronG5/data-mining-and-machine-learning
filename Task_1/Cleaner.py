import pandas as pd
import dataframe_image as dfi

FILE = 'A27/A27.csv'
CLASSES = ['Bumps', 'Other_Faults']
FEATURES = ['Log_X_Index', 'Log_Y_Index', 'Empty_Index', 'Square_Index', 'Length_of_Conveyer',
             'Steel_Plate_Thickness', 'Edges_Index', 'Orientation_Index', 'LogOfAreas', 'Luminosity_Index']

df = pd.read_csv(FILE)
df = pd.read_csv(FILE, skipinitialspace=True)
df.columns = df.columns.str.strip()
df['class'] = df['class'].str.strip()

def aprosomoji_statistika(part: pd.DataFrame):
   classifiers = part[FEATURES]
   rows = []

   for col in classifiers.columns:
      num = pd.to_numeric(classifiers[col].astype(str).str.replace('mm', '', regex=False).str.strip(), errors='coerce')

      rows.append({
         'Požymis': col,
         'Minimumas': num.min(),
         '1-as kvartilis': num.quantile(0.25),
         'Mediana': num.median(),
         'Vidurkis': num.mean(),
         '3-as kvartilis': num.quantile(0.75),
         'Maksimumas': num.max(),
         'Std. nuokrypis': num.std(ddof=1),
         'Neegzistuojančios reikšmės': int(num.isna().sum())
      })

   return pd.DataFrame(rows)

def export_table_png(table, filename):
   styled = table.style.format(precision=4).hide(axis='index')
   dfi.export(styled, filename + '.png')

def export_table_csv(table, filename):
   table.to_csv(filename + '.csv', index=False, float_format='%.4f')

filename = 'bendra_aprasomoji_statistika'
table = aprosomoji_statistika(df)
export_table_csv(table, filename)

for class_name in CLASSES:
   filename = class_name + '_aprasomoji_statistika'
   part = df[df['class'] == class_name]

   table = aprosomoji_statistika(part)

   # export_table_png(table, filename)
   export_table_csv(table, filename)