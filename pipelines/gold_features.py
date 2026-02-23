# %%
import sys
import polars as pl
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
SILVER_DIR = Path("data/silver")
GOLD_DIR = Path("data/gold")
GOLD_DIR.mkdir(parents=True, exist_ok=True)

MIN_PRICE = 102400  # SuperCoach minimum player price

# %%
def add_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add lag score and price change features within each player-season."""
    return df.with_columns([
        # Lag scores
        pl.col("score").shift(1).over(["player_name", "year"]).alias("score_lag_1"),
        pl.col("score").shift(2).over(["player_name", "year"]).alias("score_lag_2"),
        pl.col("score").shift(3).over(["player_name", "year"]).alias("score_lag_3"),

        # Lag price changes
        pl.col("round_price_change").shift(1).over(["player_name", "year"]).alias("price_change_lag_1"),
        pl.col("round_price_change").shift(2).over(["player_name", "year"]).alias("price_change_lag_2"),
    ])


# %%
def add_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add rolling average and std features within each player-season."""
    return df.with_columns([
        # Rolling averages
        pl.col("score").rolling_mean(3).over(["player_name", "year"]).alias("rolling_avg_3"),
        pl.col("score").rolling_mean(5).over(["player_name", "year"]).alias("rolling_avg_5"),

        # Rolling std — scoring consistency/volatility
        pl.col("score").rolling_std(3).over(["player_name", "year"]).alias("rolling_std_3"),
        pl.col("score").rolling_std(5).over(["player_name", "year"]).alias("rolling_std_5"),

        # Rolling price change momentum
        pl.col("round_price_change").rolling_mean(3).over(["player_name", "year"]).alias("price_momentum_3"),
    ])


# %%
def add_form_trend(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add form trend — difference between avg of last 3 scores vs avg of 3 scores before that.
    Positive = improving form, negative = declining form.
    """
    return df.with_columns([
        (
            pl.col("score").rolling_mean(3).over(["player_name", "year"]) -
            pl.col("score").shift(3).rolling_mean(3).over(["player_name", "year"])
        ).alias("form_trend")
    ])


# %%
def add_season_avg_to_date(df: pl.DataFrame) -> pl.DataFrame:
    """Add cumulative season average up to (but not including) the current round."""
    return df.with_columns([
        pl.col("score")
            .shift(1)
            .rolling_mean(27, min_samples=1)
            .over(["player_name", "year"])
            .alias("season_avg_to_date")
    ])


# %%
def add_break_even_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add break-even gap features — key signal for price movement."""
    df = df.with_columns([
        # How much did they beat/miss their break-even this round
        (pl.col("score") - pl.col("break_even")).alias("be_gap"),
    ])

    df = df.with_columns([
        # Rolling average of be_gap over last 3 rounds
        pl.col("be_gap").rolling_mean(3).over(["player_name", "year"]).alias("avg_be_gap_3rd"),

        # Lag be_gap
        pl.col("be_gap").shift(1).over(["player_name", "year"]).alias("be_gap_lag_1"),
    ])

    # Consecutive rounds beating break-even
    df = df.with_columns([
        pl.col("be_gap").shift(1).over(["player_name", "year"]).gt(0).cast(pl.Int32).alias("_be_beat_1"),
        pl.col("be_gap").shift(2).over(["player_name", "year"]).gt(0).cast(pl.Int32).alias("_be_beat_2"),
        pl.col("be_gap").shift(3).over(["player_name", "year"]).gt(0).cast(pl.Int32).alias("_be_beat_3"),
    ])

    df = df.with_columns([
        (pl.col("_be_beat_1") + pl.col("_be_beat_2") + pl.col("_be_beat_3")).alias("consecutive_be_beaten")
    ])

    df = df.drop(["_be_beat_1", "_be_beat_2", "_be_beat_3"])

    return df


# %%
def add_price_value_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add price value and rookie/underpriced signals."""
    # Get initial price per player-season (round 1 price)
    initial_prices = (
        df.filter(pl.col("round") == 1)
        .select(["player_name", "year", "price"])
        .rename({"price": "initial_price"})
    )

    df = df.join(initial_prices, on=["player_name", "year"], how="left")

    df = df.with_columns([
        # Price vs initial — how much have they risen/fallen this season
        (pl.col("price") - pl.col("initial_price")).alias("price_vs_initial"),

        # Price per point — value metric (lower = better value)
        (pl.col("price") / pl.col("avg_score").cast(pl.Float64).clip(lower_bound=1)).alias("price_per_point"),

        # Is at or near minimum price — rookie signal
        (pl.col("price") <= MIN_PRICE * 1.1).alias("is_min_price"),
    ])

    return df


# %%
def add_opponent_strength(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add opponent average points allowed per position.
    Calculated as the historical average score conceded by each team to each position.
    """
    opponent_strength = (
        df.group_by(["opponent", "primary_position"])
        .agg(
            pl.col("score").mean().alias("opponent_avg_pts_allowed")
        )
    )

    df = df.join(opponent_strength, on=["opponent", "primary_position"], how="left")

    return df


# %%
def add_career_avg(df: pl.DataFrame) -> pl.DataFrame:
    """Add each player's career average score across all seasons."""
    career_avgs = (
        df.group_by("player_name")
        .agg(pl.col("score").mean().alias("career_avg"))
    )

    df = df.join(career_avgs, on="player_name", how="left")

    return df


# %%
def add_price_percentile(df: pl.DataFrame) -> pl.DataFrame:
    """Add price percentile within position and round — where does this player sit price-wise?"""
    df = df.with_columns([
        pl.col("price")
            .rank(method="average")
            .over(["year", "round", "primary_position"])
            .alias("price_rank_in_position"),

        pl.col("score")
            .rank(method="average")
            .over(["year", "round", "primary_position"])
            .alias("score_rank_in_position"),
    ])

    return df


# %%
def run_gold_transform():
    """Load Silver parquet, engineer features, save as Gold parquet."""
    silver_path = SILVER_DIR / "player_rounds.parquet"

    if not silver_path.exists():
        logger.error(f"Silver parquet not found at {silver_path}")
        return

    logger.info("Loading Silver data")
    df = pl.read_parquet(silver_path)
    logger.info(f"Loaded {len(df)} rows")

    # Sort by player, year, round — critical for window functions
    df = df.sort(["player_name", "year", "round"])

    logger.info("Adding lag features")
    df = add_lag_features(df)

    logger.info("Adding rolling features")
    df = add_rolling_features(df)

    logger.info("Adding form trend")
    df = add_form_trend(df)

    logger.info("Adding season avg to date")
    df = add_season_avg_to_date(df)

    logger.info("Adding break-even features")
    df = add_break_even_features(df)

    logger.info("Adding price value features")
    df = add_price_value_features(df)

    logger.info("Adding opponent strength")
    df = add_opponent_strength(df)

    logger.info("Adding career averages")
    df = add_career_avg(df)

    logger.info("Adding price percentiles")
    df = add_price_percentile(df)

    output_path = GOLD_DIR / "player_rounds_features.parquet"
    df.write_parquet(output_path)
    logger.success(f"Saved {len(df)} rows with {len(df.columns)} features to {output_path}")


# %%
run_gold_transform()