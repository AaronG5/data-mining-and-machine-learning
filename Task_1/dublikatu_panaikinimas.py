import pandas as pd

df = pd.read_csv("A27/A27.csv")

# Remove exact duplicate rows
df = df.drop_duplicates()

print(f"Rows after removing duplicates: {len(df)}")

df.to_csv("A27/A27.csv", index=False)