import polars as pl

df = pl.read_parquet(r"data\gold\player_rounds_features.parquet")
print(df.shape)
print(df.columns)
print(df.dtypes)