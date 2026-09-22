import pandas as pd
import os

from histogram import plot_distribution

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'res')
CLASSES = ['Bumps', 'Other_Faults']
FEATURES = ['Log_X_Index', 'Log_Y_Index', 'Empty_Index', 'Square_Index', 'Length_of_Conveyer',
             'Steel_Plate_Thickness', 'Edges_Index', 'Orientation_Index', 'LogOfAreas', 'Luminosity_Index']

def descriptive_statistics(part: pd.DataFrame) -> pd.DataFrame:
   classifiers = part[FEATURES]
   rows = []

   for col in classifiers.columns:
      num = pd.to_numeric(classifiers[col].astype(str).str.strip(), errors='coerce')

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

def export_table_csv(table: pd.DataFrame, filename: str) -> None:
   file_dest = os.path.join(OUT_DIR, filename + '.csv')
   table.to_csv(file_dest, index=False, float_format='%.4f')

def check_for_duplicates(data: pd.DataFrame) -> pd.DataFrame:
   dupes = data.duplicated(keep='first')

   print(f'Total rows: {len(data)}')
   print(f'Duplicate rows: {data.duplicated().sum()}')

   return data[dupes]


def main():
   os.makedirs(OUT_DIR, exist_ok=True)
   FILE = os.path.join(BASE_DIR, 'A27', 'A27.csv') 

   df = pd.read_csv(FILE, skipinitialspace=True)
   df.columns = df.columns.str.strip()
   df['class'] = df['class'].str.strip()
   df['Steel_Plate_Thickness'] = pd.to_numeric(df['Steel_Plate_Thickness']
                                 .astype(str).str.replace('mm', '', regex=False).str.strip(), errors='coerce')
   
   # duplicates = check_for_duplicates(df)
   # print(duplicates)

   df = df.drop_duplicates().reset_index(drop=True)

   filename = 'bendra_aprasomoji_statistika'
   table = descriptive_statistics(df)
   export_table_csv(table, filename)

   for feature in FEATURES:
      directory = os.path.join(OUT_DIR, 'Bendra')
      plot_distribution(df[feature], feature, directory)

   for class_name in CLASSES:
      filename = class_name + '_aprasomoji_statistika'
      part = df[df['class'] == class_name]

      for feature in FEATURES:
         directory = os.path.join(OUT_DIR, class_name)
         os.makedirs(directory, exist_ok=True)
         plot_distribution(part[feature], feature, directory, class_name)

      table = descriptive_statistics(part)

      export_table_csv(table, filename)

main()