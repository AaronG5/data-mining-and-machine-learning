import pandas as pd

FILE = 'A27_be_virsutiniu_isskirciu.csv'
OUT_FILE = 'A27_normuota_robust.csv'

df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()
df['class'] = df['class'].str.strip()

poz = df.columns.drop(['class', 'Outside_Global_Index'])
iqr = df[poz].quantile(0.75) - df[poz].quantile(0.25)
df[poz] = (df[poz] - df[poz].median()) / iqr.replace(0, 1)

df.to_csv(OUT_FILE, index=False)
