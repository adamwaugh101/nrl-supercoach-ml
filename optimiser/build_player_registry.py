# %%
import polars as pl
import pandas as pd
from pathlib import Path
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
GOLD_PATH = Path("data/gold/player_rounds_features.parquet")
BRONZE_2026_PATH = Path("data/bronze/stats_2026.parquet")
OUTPUT_PATH = Path("data/optimiser/player_registry_2026.parquet")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# %%
# Load 2025 Gold layer — source of career stats and form
gold = pl.read_parquet(GOLD_PATH)
logger.info(f"Gold layer loaded: {gold.shape}")

# %%
# Build career stats from Gold — use all history up to end of 2025
career_stats = (
    gold.group_by("player_name")
    .agg([
        pl.col("score").mean().alias("career_avg"),
        pl.col("score").std().alias("career_std"),
        pl.col("mins").mean().alias("career_avg_mins"),
        pl.col("score").count().alias("career_games"),
    ])
)

# %%
# Get last 5 rounds of 2025 form per player
last_round_2025 = (
    gold.filter(pl.col("year") == 2025)
    .sort(["player_name", "round"])
    .group_by("player_name")
    .agg([
        pl.col("score").tail(5).mean().alias("last5_avg"),
        pl.col("score").tail(3).mean().alias("last3_avg"),
        pl.col("mins").tail(3).mean().alias("last3_avg_mins"),
        pl.col("round").max().alias("last_round_played"),
        pl.col("primary_position").last().alias("primary_position"),
    ])
)

# %%
# Combine career stats with recent form
player_stats = career_stats.join(last_round_2025, on="player_name", how="left")
logger.info(f"Player stats built: {player_stats.shape}")

# %%
# Load 2026 Bronze data
if BRONZE_2026_PATH.exists():
    raw_2026 = pl.from_pandas(pd.read_parquet(BRONZE_2026_PATH))

    # Clean up 2026 data — drop filter rows and nulls
    players_2026 = (
        raw_2026
        .filter(pl.col("id").str.contains(r"^\d+$"))
        .filter(pl.col("Name").is_not_null() & (pl.col("Name") != ""))
        .filter(pl.col("Posn").is_not_null() & (pl.col("Posn") != "") & (pl.col("Posn") != "Pos Pos"))
        .select([
            pl.col("Name").alias("player_name"),
            pl.col("Posn").alias("posn_raw"),
            pl.col("Team").alias("team_2026"),
            pl.col("Price").str.replace_all(r"[\$,]", "").cast(pl.Int64).alias("price_2026"),
            pl.col("BE").cast(pl.Int32, strict=False).alias("break_even_2026"),
        ])
        .with_columns([
            pl.col("posn_raw").str.replace("5/8", "5_8"),
        ])
        .with_columns([
            pl.col("posn_raw").str.split(" ").list.get(0).alias("position_2026"),
            pl.col("posn_raw").str.split(" ").list.get(1, null_on_oob=True).alias("secondary_position_2026"),
        ])
    .drop("posn_raw")
)
    logger.info(f"2026 players loaded: {players_2026.shape[0]}")

else:
    # Fallback — use 2025 final round prices if 2026 not yet available
    logger.warning("2026 Bronze not found — falling back to 2025 final round data")
    players_2026 = (
        gold.filter(pl.col("year") == 2025)
        .sort(["player_name", "round"])
        .group_by("player_name")
        .agg([
            pl.col("price").last().alias("price_2026"),
            pl.col("primary_position").last().alias("position_2026"),
            pl.col("team").last().alias("team_2026"),
            pl.col("break_even").last().alias("break_even_2026"),
        ])
    )
    logger.info(f"Fallback 2025 players loaded: {players_2026.shape}")

# %%
# Join 2026 prices with historical stats
registry = (
    players_2026.join(player_stats, on="player_name", how="left")
)


# %%
# Predicted score — use last3_avg if available, fall back to career_avg
registry = registry.with_columns(
    pl.when(pl.col("last3_avg").is_not_null())
      .then(pl.col("career_avg") * 0.6 + pl.col("last3_avg") * 0.4)
      .otherwise(pl.col("career_avg"))
      .alias("predicted_score")
)
# %%
# Flag players likely to play — avg mins >= 40 over last 3 rounds
registry = registry.with_columns(
    pl.when(pl.col("last3_avg_mins").is_not_null() & (pl.col("last3_avg_mins") >= 40))
      .then(True)
      .otherwise(False)
      .alias("likely_to_play")
)

registry = registry.filter(
    (pl.col("career_games") >= 10) | (pl.col("career_avg").is_null())
)
logger.info(f"After minimum games filter: {registry.shape}")

# %%
# Summary
logger.info(f"Registry built: {registry.shape}")
logger.info(f"Players likely to play: {registry.filter(pl.col('likely_to_play')).shape[0]}")
logger.info(f"Players with no history: {registry.filter(pl.col('career_avg').is_null()).shape[0]}")

print(registry.select([
    "player_name", "position_2026", "team_2026", "price_2026",
    "predicted_score", "career_avg", "last3_avg", "likely_to_play"
]).sort("predicted_score", descending=True).head(20))

registry.write_parquet(OUTPUT_PATH)
logger.info(f"Registry saved to {OUTPUT_PATH}")

print(registry.sort("predicted_score", descending=True).to_pandas().to_csv(index=False))