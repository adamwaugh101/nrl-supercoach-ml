import polars as pl
from pathlib import Path

df = pl.read_parquet(Path("data/gold/player_rounds_features.parquet"))

weights = {
    "TR": 17, "TA": 12, "TC": 4, "GO": 4, "FG": 5,
    "TB": 2, "MT": -1, "FD": 6, "OL": 4, "IO": 2,
    "LB": 10, "LA": 8, "IT": 5, "PC": -2, "ER": -2,
    "HG": 1, "HU": -1, "KB": -3, "SS": -8,
}

expr = sum(pl.col(col) * weight for col, weight in weights.items() if col in df.columns)

df = df.with_columns(expr.alias("calculated_score"))

df = df.with_columns(
    (pl.col("score") - pl.col("calculated_score")).alias("score_diff")
)

action_cols = ["TR", "TA", "GO", "FG", "TB", "MT", "FD", "OL", "IO", 
               "LB", "LA", "IT", "PC", "ER", "HG", "HU", "KB", "SS",
               "TS", "LT", "MG", "MF", "H8"]

expr = sum(pl.col(col) for col in action_cols if col in df.columns)

df = df.with_columns(expr.alias("calculated_score"))
df = df.with_columns((pl.col("score") - pl.col("calculated_score")).alias("score_diff"))

print(df.filter(pl.col("calculated_score").is_null())["year"].value_counts().sort("year"))
