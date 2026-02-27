import pandas as pd

df = pd.read_parquet("stats_2026.parquet")
print(df.shape)
print(df.columns.tolist())
print(df.head(3))