import polars as pl
from pathlib import Path

# df = pl.read_parquet(Path("data/gold/player_rounds_features.parquet"))
# print(df["primary_position"].value_counts().sort("count", descending=True))

df = pl.read_parquet(Path("data/gold/player_rounds_features.parquet"))
print(df.filter(pl.col("primary_position") == "5/8").shape)
print(df["primary_position"].unique().to_list())
