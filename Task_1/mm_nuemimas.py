import pandas as pd

FILE = 'A27_medianos_pagal_klase.csv'

df = pd.read_csv(FILE)
df['Steel_Plate_Thickness'] = pd.to_numeric(
   df['Steel_Plate_Thickness'].astype(str).str.replace('mm', '', regex=False).str.strip(), errors='coerce')

df.to_csv('A27_medianos_pagal_klase.csv', index=False)