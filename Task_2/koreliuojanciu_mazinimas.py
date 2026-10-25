import pandas as pd

FILE = 'A_originali_aibe.csv'
OUT_FILE = 'B_be_koreliuotu.csv'

# pozymiai su |Spearman rho| >= 0.9 (pagal koreliacijos matrica)
CORR_REMOVED = [
   'X_Minimum',
   'Y_Minimum',
   'Pixels_Areas',
   'SigmoidOfAreas',
   'Minimum_of_Luminosity',
   'LogOfAreas',
   'Log_X_Index',
   'Log_Y_Index',
]

df = pd.read_csv(FILE)
df = df.drop(columns=CORR_REMOVED)

df.to_csv(OUT_FILE, index=False)
print(f'Liko {df.shape[1] - 1} pozymiai, {len(df)} eiluciu -> {OUT_FILE}')
