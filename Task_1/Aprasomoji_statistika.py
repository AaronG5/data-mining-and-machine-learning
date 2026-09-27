import pandas as pd
import os

from histogram import plot_distribution

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'res')
CLASSES = ['Bumps', 'Other_Faults']

# Sukuria ir grąžina aprašomąją statistiką
def descriptive_statistics(part: pd.DataFrame, FEATURES: list[str]) -> pd.DataFrame:
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
      })

   return pd.DataFrame(rows)

def export_table_csv(table: pd.DataFrame, file_path: str) -> None:
   file_path = file_path + '.csv'
   table.to_csv(file_path, index=False, float_format='%.4f')

def main():
   os.makedirs(OUT_DIR, exist_ok=True)
   FILE_UNCLEAN = os.path.join(BASE_DIR, 'A27', 'A27.csv') 
   FILE_CLEAN = os.path.join(BASE_DIR, 'A27_be_virsutiniu_isskirciu.csv') 

   df_unclean = pd.read_csv(FILE_UNCLEAN, skipinitialspace=True)
   df_unclean.columns = df_unclean.columns.str.strip()
   df_unclean['class'] = df_unclean['class'].str.strip()
   df_unclean['Steel_Plate_Thickness'] = pd.to_numeric(df_unclean['Steel_Plate_Thickness']
                                 .astype(str).str.replace('mm', '', regex=False).str.strip(), errors='coerce')

   df = pd.read_csv(FILE_CLEAN, skipinitialspace=True)
   df.columns = df.columns.str.strip()
   df['class'] = df['class'].str.strip()

   FEATURES = [col for col in df.columns if col != 'class']
   FEATURES_PICKED = ['Log_X_Index', 'Log_Y_Index', 'Empty_Index', 'Square_Index', 'Length_of_Conveyer',
               'Steel_Plate_Thickness', 'Edges_Index', 'Orientation_Index', 'LogOfAreas', 'Luminosity_Index']
   
   file_path = os.path.join(OUT_DIR, 'bendra_unclean_aprasomoji_statistika')
   table = descriptive_statistics(df_unclean, FEATURES)
   export_table_csv(table, file_path)

   file_path = os.path.join(OUT_DIR, 'bendra_clean_aprasomoji_statistika')
   table = descriptive_statistics(df, FEATURES)
   export_table_csv(table, file_path)

   feature_ranges = {
      feature: (
         pd.to_numeric(df[feature], errors='coerce').min(),
         pd.to_numeric(df[feature], errors='coerce').max()
      )
      for feature in FEATURES_PICKED
   }

   # Atskirai pagal klases generuojami grafikai bei lentelės su atrinktais požymiais
   for class_name in CLASSES:
      file_path = os.path.join(OUT_DIR, class_name + '_aprasomoji_statistika')
      part = df[df['class'] == class_name]

      table = descriptive_statistics(part, FEATURES)
      export_table_csv(table, file_path)

      directory = os.path.join(OUT_DIR, class_name)
      os.makedirs(directory, exist_ok=True)
      for feature in FEATURES_PICKED:
         x_min, x_max = feature_ranges[feature]
         plot_distribution(part[feature], feature, directory, class_name, x_min=x_min, x_max=x_max)

main()