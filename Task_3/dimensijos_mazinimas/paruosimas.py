import pandas as pd

FILE = '../steel_plates_3_klases_su_koreliuotais.csv'
ATASKAITA = '../pradiniai_duomenys/kor_rez/koreliacijos_ataskaita_3kl.csv'
OUT_FILE = 'steel_plates_3_klases_normuota_robust.csv'

df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()
df['class'] = df['class'].str.strip()

# koreliuotų požymių šalinimas: iš kiekvienos poros su |rho| >= 0.9 pašalinamas antras požymis,
# jei abu dar nepašalinti
ataskaita = pd.read_csv(ATASKAITA)
pasalinti = []
for a, b in zip(ataskaita['Požymis_1'], ataskaita['Požymis_2']):
   if a not in pasalinti and b not in pasalinti:
      pasalinti.append(b)

df = df.drop(columns=pasalinti)
print(f'Pašalinti požymiai ({len(pasalinti)}): {pasalinti}')

# robust normavimas (kaip Task_1/normavimas.py)
poz = df.columns.drop(['class', 'Outside_Global_Index'])
iqr = df[poz].quantile(0.75) - df[poz].quantile(0.25)
df[poz] = (df[poz] - df[poz].median()) / iqr.replace(0, 1)

df.to_csv(OUT_FILE, index=False)
print(f'Liko {df.shape[1] - 1} požymių, {len(df)} eilučių -> {OUT_FILE}')
