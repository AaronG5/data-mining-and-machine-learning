import pandas as pd

FILE = 'steel_plates_full_be_virsutiniu_isskirciu.csv'
OUTPUT = '../steel_plates_3_klases.csv'
CLASSES = ['K_Scatch', 'Z_Scratch', 'Bumps']
NEREIKALINGI = ['TypeOfSteel_A300', 'TypeOfSteel_A400']

df = pd.read_csv(FILE)

# palieka tik tiriamas klases
df_klases = df[df['class'].isin(CLASSES)].reset_index(drop=True)

# pašalina nereikalingus požymius
df_klases = df_klases.drop(columns=NEREIKALINGI)

df_klases.to_csv(OUTPUT, index=False)

print(f'Palikta eilučių: {len(df_klases)} iš {len(df)}, požymių: {df_klases.shape[1] - 1}')
print(df_klases['class'].value_counts())
