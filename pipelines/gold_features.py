# %%
import sys
import polars as pl
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
GOLD_DIR = Path("data/gold")
GOLD_DIR.mkdir(parents=True, exist_ok=True)

MIN_PRICE = 102400  # SuperCoach minimum player price


# Maps all known team name variants → 3-letter silver codes
TEAM_NAME_MAP = {
    # Full names / S3 matchup string names
    "Sea Eagles":           "MNL",
    "Rabbitohs":            "STH",
    "Roosters":             "SYD",
    "Broncos":              "BRO",
    "Knights":              "NEW",
    "Raiders":              "CBR",
    "Storm":                "MEL",
    "Warriors":             "NZL",
    "Panthers":             "PTH",
    "Bulldogs":             "BUL",
    "Sharks":               "SHA",
    "Tigers":               "WST",
    "Eels":                 "PAR",
    "Titans":               "GCT",
    "Cowboys":              "NQC",
    "Dragons":              "STG",
    "Dolphins":             "DOL",
    # Alternate long-form names that may appear in older S3 data
    "Manly":                "MNL",
    "South Sydney":         "STH",
    "Sydney Roosters":      "SYD",
    "Brisbane":             "BRO",
    "Newcastle":            "NEW",
    "Canberra":             "CBR",
    "Melbourne":            "MEL",
    "New Zealand":          "NZL",
    "Penrith":              "PTH",
    "Canterbury":           "BUL",
    "Cronulla":             "SHA",
    "Wests Tigers":         "WST",
    "Parramatta":           "PAR",
    "Gold Coast":           "GCT",
    "North Queensland":     "NQC",
    "St George Illawarra":  "STG",
    "St. George Illawarra": "STG",
}

ACTION_COLS = [
    "TR", "TS", "LT", "GO", "MG", "FG", "MF", "TA", "MT",
    "TB", "FD", "OL", "IO", "LB", "LA", "FT", "KB", "H8",
    "HU", "HG", "IT", "KD", "PC", "ER", "SS",
    "mins", "bppm", "cv"
]


# %%
def add_match_context(df: pl.DataFrame) -> pl.DataFrame:
    """
    Join match context (ground condition, weather, is_home) onto player rounds.
    Also adds is_bye flag for players whose team has a bye that round.

    Data sources:
      - data/bronze/match_context/nrl_match_context.parquet   (S3, 2015-2024)
      - data/bronze/match_context/nrl_match_context_2025.parquet (Playwright scraped)
    """

    # ── 1. Load and normalise the S3 context (2015-2024) ──────────────────────
    ctx_path_historical = BRONZE_DIR / "match_context/nrl_match_context.parquet"
    ctx_path_2025       = BRONZE_DIR / "match_context/nrl_match_context_2025.parquet"

    frames = []

    if ctx_path_historical.exists():
        ctx_hist = pl.read_parquet(ctx_path_historical).select([
            "year", "round", "team", "opponent", "is_home",
            "ground_condition", "weather_condition",
        ]).with_columns([
            pl.col("year").cast(pl.Int64),
            pl.col("round").cast(pl.Int64, strict=False),
        ])
        frames.append(ctx_hist)

    if ctx_path_2025.exists():
        ctx_2025 = pl.read_parquet(ctx_path_2025).select([
            "year", "round", "team", "opponent", "is_home",
            "ground_condition", "weather_condition",
        ]).with_columns([
            pl.col("year").cast(pl.Int64),
            pl.col("round").cast(pl.Int64, strict=False),
        ])
        frames.append(ctx_2025)
    else:
        logger.warning("2025 match context parquet not found — skipping")

    if not frames:
        logger.warning("No match context data available — skipping join")
        return df

    ctx = pl.concat(frames, how="diagonal")

    print("Ctx row counts by year:")
    print(ctx.group_by("year").agg(pl.len().alias("count")).sort("year"))

    # ── 2. Normalise team names in context → 3-letter silver codes ────────────
    # Build a Polars-compatible replacement expression using when/then chains
    team_expr = pl.col("team")
    opp_expr  = pl.col("opponent")

    for name, code in TEAM_NAME_MAP.items():
        team_expr = pl.when(pl.col("team") == name).then(pl.lit(code)).otherwise(team_expr)
        opp_expr  = pl.when(pl.col("opponent") == name).then(pl.lit(code)).otherwise(opp_expr)

    ctx = ctx.with_columns([
        team_expr.alias("team"),
        opp_expr.alias("opponent"),
        pl.col("year").cast(pl.Int64),
        pl.col("round").cast(pl.Int32, strict=False),
        pl.col("is_home").cast(pl.Boolean),
    ])


    print("Ctx teams BEFORE length filter:")
    print(ctx["team"].unique().sort())

    # Drop any rows where team code wasn't resolved (shouldn't happen, but safety first)
    ctx = ctx.filter(pl.col("team").str.len_chars() == 3)

    # ── 3. Derive bye rounds from context ─────────────────────────────────────
    # Any team that appears in the silver player data for a given year/round
    # but NOT in the match context has a bye that round.
    playing = ctx.select(["year", "round", "team"]).unique()

    all_team_rounds = (
        df.select(["year", "round", "team"])
        .filter(pl.col("team").str.len_chars() == 3)  # filter out empty 2015 rows
        .unique()
    )
    print("Ctx teams in round 1 2025:")
    print(playing.filter((pl.col("year") == 2025) & (pl.col("round") == 1)).sort("team"))

    print("Sample playing teams in ctx (year=2025, round=1):")
    print(playing.filter((pl.col("year") == 2025) & (pl.col("round") == 1)).sort("team"))

    print("\nSilver teams in round 1 2025:")
    print(all_team_rounds.filter((pl.col("year") == 2025) & (pl.col("round") == 1)).sort("team"))

    bye_rounds = (
        all_team_rounds
        .join(playing, on=["year", "round", "team"], how="anti")
        .with_columns(pl.lit(True).alias("is_bye"))
    )

    # Debug
    sample_silver = df.select(["year", "round", "team"]).unique().head(5)
    sample_ctx = ctx.select(["year", "round", "team"]).unique().head(5)
    print("Silver sample:")
    print(sample_silver)
    print("\nCtx sample:")
    print(sample_ctx)

    # Try a manual join on one known row
    test_year = sample_silver["year"][0]
    test_round = sample_silver["round"][0]
    test_team = sample_silver["team"][0]
    print(f"\nLooking for year={test_year}, round={test_round}, team={test_team} in ctx:")
    print(ctx.filter(
        (pl.col("year") == test_year) &
        (pl.col("round") == test_round) &
        (pl.col("team") == test_team)
    ))

    # ── 4. Join match context onto player rounds ───────────────────────────────
    df = df.join(
        ctx.select(["year", "round", "team", "is_home", "ground_condition", "weather_condition"]),
        on=["year", "round", "team"],
        how="left",
    )

    # ── 5. Join bye flags ──────────────────────────────────────────────────────
    df = df.join(
        bye_rounds,
        on=["year", "round", "team"],
        how="left",
    )

    df = df.with_columns(
        pl.col("is_bye").fill_null(False)
    )

    logger.info(
        f"Match context joined — "
        f"is_home nulls: {df['is_home'].null_count()}, "
        f"bye rows: {df['is_bye'].sum()}"
    )

    # Debug — remove after fixing
    print(df.select(["year", "round"]).dtypes)
    print(ctx.select(["year", "round"]).dtypes)
    return df


#%%
def add_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add lag score and price change features within each player-season."""
    return df.with_columns([
        pl.col("score").shift(1).over(["player_name", "year"]).alias("score_lag_1"),
        pl.col("score").shift(2).over(["player_name", "year"]).alias("score_lag_2"),
        pl.col("score").shift(3).over(["player_name", "year"]).alias("score_lag_3"),
        pl.col("round_price_change").shift(1).over(["player_name", "year"]).alias("price_change_lag_1"),
        pl.col("round_price_change").shift(2).over(["player_name", "year"]).alias("price_change_lag_2"),
    ])

def add_action_score_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add calculated action score and its lag/rolling features."""
    action_cols = [
        "TR", "TA", "GO", "FG", "TB", "MT", "FD", "OL", "IO",
        "LB", "LA", "IT", "PC", "ER", "HG", "HU", "KB", "SS",
        "TS", "LT", "MG", "MF", "H8"
    ]

    # Only sum columns that exist in the dataframe
    available = [c for c in action_cols if c in df.columns]
    action_sum = sum(pl.col(c) for c in available)

    return df.with_columns(
        action_sum.alias("action_score")
    ).with_columns([
        pl.col("action_score").shift(1).over(["player_name", "year"]).alias("action_score_lag_1"),
        pl.col("action_score").rolling_mean(3).over(["player_name", "year"]).alias("action_score_rolling_3"),
        pl.col("action_score").rolling_mean(5).over(["player_name", "year"]).alias("action_score_rolling_5"),
    ])
# %%
def add_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add rolling average and std features within each player-season."""
    return df.with_columns([
        pl.col("score").rolling_mean(3).over(["player_name", "year"]).alias("rolling_avg_3"),
        pl.col("score").rolling_mean(5).over(["player_name", "year"]).alias("rolling_avg_5"),
        pl.col("score").rolling_std(3).over(["player_name", "year"]).alias("rolling_std_3"),
        pl.col("score").rolling_std(5).over(["player_name", "year"]).alias("rolling_std_5"),
        pl.col("round_price_change").rolling_mean(3).over(["player_name", "year"]).alias("price_momentum_3"),
    ])


# %%
def add_form_trend(df: pl.DataFrame) -> pl.DataFrame:
    """Form trend — difference between avg of last 3 scores vs 3 before that."""
    return df.with_columns([
        (
            pl.col("score").rolling_mean(3).over(["player_name", "year"]) -
            pl.col("score").shift(3).rolling_mean(3).over(["player_name", "year"])
        ).alias("form_trend")
    ])


# %%
def add_season_avg_to_date(df: pl.DataFrame) -> pl.DataFrame:
    """Cumulative season average up to (but not including) current round."""
    return df.with_columns([
        pl.col("score")
            .shift(1)
            .rolling_mean(27, min_samples=1)
            .over(["player_name", "year"])
            .alias("season_avg_to_date")
    ])


# %%
def add_break_even_features(df: pl.DataFrame) -> pl.DataFrame:
    """Break-even gap features — key signal for price movement."""
    df = df.with_columns([
        (pl.col("score") - pl.col("break_even")).alias("be_gap"),
    ])

    df = df.with_columns([
        pl.col("be_gap").rolling_mean(3).over(["player_name", "year"]).alias("avg_be_gap_3rd"),
        pl.col("be_gap").shift(1).over(["player_name", "year"]).alias("be_gap_lag_1"),
    ])

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
    """Price value and rookie/underpriced signals."""
    initial_prices = (
        df.filter(pl.col("round") == 1)
        .select(["player_name", "year", "price"])
        .rename({"price": "initial_price"})
    )

    df = df.join(initial_prices, on=["player_name", "year"], how="left")

    df = df.with_columns([
        (pl.col("price") - pl.col("initial_price")).alias("price_vs_initial"),
        (pl.col("price") / pl.col("avg_score").cast(pl.Float64).clip(lower_bound=1)).alias("price_per_point"),
        (pl.col("price") <= MIN_PRICE * 1.1).alias("is_min_price"),
    ])

    return df


# %%
def add_opponent_strength(df: pl.DataFrame) -> pl.DataFrame:
    """Opponent average points allowed per position using only historical data (no leakage)."""
    df = df.sort(["year", "round"])

    opponent_strength = (
        df.select(["year", "round", "opponent", "primary_position", "score"])
        .with_columns(
            pl.col("score")
              .shift(1)
              .over(["opponent", "primary_position"])
              .alias("score_lagged")
        )
        .group_by(["opponent", "primary_position"])
        .agg(pl.col("score_lagged").mean().alias("opponent_avg_pts_allowed"))
    )

    df = df.drop("opponent_avg_pts_allowed") if "opponent_avg_pts_allowed" in df.columns else df
    df = df.join(opponent_strength, on=["opponent", "primary_position"], how="left")

    return df


# %%
def add_career_avg(df: pl.DataFrame) -> pl.DataFrame:
    """Player career average score across all seasons."""
    career_avgs = (
        df.group_by("player_name")
        .agg(pl.col("score").mean().alias("career_avg"))
    )

    df = df.join(career_avgs, on="player_name", how="left")

    return df


# %%
def add_price_percentile(df: pl.DataFrame) -> pl.DataFrame:
    """Price and score rank within position and round."""
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
def add_lagged_action_stats(df: pl.DataFrame) -> pl.DataFrame:
    """
    Lag and roll action stats so they're available as prior-round features.
    These are legitimately known before the next round and are
    highly predictive of future performance.
    """
    existing_cols = [c for c in ACTION_COLS if c in df.columns]

    # lag_1 — last round's value
    lag_1 = [
        pl.col(c).shift(1).over(["player_name", "year"]).alias(f"{c}_lag_1")
        for c in existing_cols
    ]

    # rolling avg over 3 rounds
    roll_3 = [
        pl.col(c).shift(1).rolling_mean(3).over(["player_name", "year"]).alias(f"{c}_roll_avg_3")
        for c in existing_cols
    ]

    # rolling avg over 5 rounds
    roll_5 = [
        pl.col(c).shift(1).rolling_mean(5).over(["player_name", "year"]).alias(f"{c}_roll_avg_5")
        for c in existing_cols
    ]

    df = df.with_columns(lag_1 + roll_3 + roll_5)

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

    logger.info("Adding action score features")
    df = add_action_score_features(df)

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

    logger.info("Adding lagged and rolling action stats")
    df = add_lagged_action_stats(df)

    logger.info("Adding match context (home/away, conditions, byes)")
    df = add_match_context(df)

    output_path = GOLD_DIR / "player_rounds_features.parquet"
    df.write_parquet(output_path)
    logger.success(f"Saved {len(df)} rows with {len(df.columns)} features to {output_path}")


# %%
run_gold_transform()