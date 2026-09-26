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
         'Std. nuokrypis': num.std(ddof=1)
      })

   return pd.DataFrame(rows)

def export_table_csv(table: pd.DataFrame, filename: str) -> None:
   file_dest = os.path.join(OUT_DIR, filename + '.csv')
   table.to_csv(file_dest, index=False, float_format='%.4f')

def main():
   os.makedirs(OUT_DIR, exist_ok=True)
   # FILE = os.path.join(BASE_DIR, 'A27', 'A27.csv')
   FILE = os.path.join(BASE_DIR, 'A27_be_virsutiniu_isskirciu.csv')

   df = pd.read_csv(FILE, skipinitialspace=True)
   df.columns = df.columns.str.strip()
   df['class'] = df['class'].str.strip()
   df['Steel_Plate_Thickness'] = pd.to_numeric(df['Steel_Plate_Thickness']
                                 .astype(str).str.replace('mm', '', regex=False).str.strip(), errors='coerce')

   # file_path = os.path.join(OUT_DIR, 'bendra_aprasomoji_statistika')
   # table = descriptive_statistics(df)
   # export_table_csv(table, file_path)

   directory = os.path.join(OUT_DIR, 'Bendra')
   os.makedirs(directory, exist_ok=True)

   for feature in FEATURES:
      plot_distribution(df[feature], feature, directory)

   feature_ranges = {
      feature: (
         pd.to_numeric(df[feature], errors='coerce').min(),
         pd.to_numeric(df[feature], errors='coerce').max()
      )
      for feature in FEATURES
   }

   for class_name in CLASSES:
      file_path = class_name + '_aprasomoji_statistika'
      part = df[df['class'] == class_name]
      directory = os.path.join(OUT_DIR, class_name)
      os.makedirs(directory, exist_ok=True)

      for feature in FEATURES:
         x_min, x_max = feature_ranges[feature]
         plot_distribution(part[feature], feature, directory, class_name, x_min=x_min, x_max=x_max)

      # table = descriptive_statistics(part)
      # export_table_csv(table, file_path)

main()