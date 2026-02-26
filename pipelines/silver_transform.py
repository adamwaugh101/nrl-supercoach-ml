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
SILVER_DIR.mkdir(parents=True, exist_ok=True)

# %%
COLS_TO_DROP = ["cb", "Photo", "Namedot", "Team2", "Jersey", "weather", "Name2", "id"]

MONEY_COLS = ["Price", "RoundPriceChange", "SeasonPriceChange"]

FLOAT_COLS = [
    "PPM", "CVRd", "AvgMins", "TwoRdAvg", "ThreeRdAvg", "FiveRdAvg",
    "ThreeRdMins", "FiveRdMins", "Avg1to10", "Avg11to18", "Avg19to26",
    "SixtySixty", "BPPM", "BasePowerPPM", "BasePower", "AvgPC", "AvgER",
    "AvgPCER", "H8percent", "TBPERCENT", "MTPERCENT", "OLILPERCENT",
    "BasePercent", "BaseAvg", "ScoringAvg", "CreateAvg", "EvadeAvg",
    "NegativeAvg", "BasePowerAvg"
]

INT_COLS = [
    "Score", "Mins", "AvgScore", "BE", "Played",
    "Base", "Attack", "Playmaking", "Power", "Negative",
    "TR", "TS", "LT", "GO", "MG", "FG", "MF", "TA", "MT",
    "TB", "FD", "OL", "IO", "LB", "LA", "FT", "KB", "H8",
    "HU", "HG", "IT", "KD", "PC", "ER", "SS"
]

RENAME_MAP = {
    "Name": "player_name",
    "Posn": "position",
    "Team": "team",
    "Price": "price",
    "Score": "score",
    "Mins": "mins",
    "AvgScore": "avg_score",
    "BE": "break_even",
    "CVRd": "cv",
    "AvgMins": "avg_mins",
    "PPM": "ppm",
    "SeasonPriceChange": "season_price_change",
    "Played": "games_played",
    "Rd": "round",
    "RoundPriceChange": "round_price_change",
    "vs": "opponent",
    "TwoRdAvg": "avg_2rd",
    "ThreeRdAvg": "avg_3rd",
    "FiveRdAvg": "avg_5rd",
    "ThreeRdMins": "avg_3rd_mins",
    "FiveRdMins": "avg_5rd_mins",
    "Avg1to10": "avg_1_10",
    "Avg11to18": "avg_11_18",
    "Avg19to26": "avg_19_26",
    "SixtySixty": "sixty_sixty",
    "BPPM": "bppm",
    "BasePowerPPM": "base_power_ppm",
    "BasePower": "base_power",
    "AvgPC": "avg_pc",
    "AvgER": "avg_er",
    "AvgPCER": "avg_pc_er",
    "H8percent": "h8_pct",
    "TBPERCENT": "tb_pct",
    "MTPERCENT": "mt_pct",
    "OLILPERCENT": "ol_il_pct",
    "BasePercent": "base_pct",
    "Base": "base",
    "Attack": "attack",
    "Playmaking": "playmaking",
    "Power": "power",
    "Negative": "negative",
    "BaseAvg": "base_avg",
    "ScoringAvg": "scoring_avg",
    "CreateAvg": "create_avg",
    "EvadeAvg": "evade_avg",
    "NegativeAvg": "negative_avg",
    "BasePowerAvg": "base_power_avg",
}


# %%
def transform_bronze_to_silver(df: pl.DataFrame) -> pl.DataFrame:
    """Apply all Silver transformations to a Bronze DataFrame."""

    # Filter junk rows — keep only rows where id is numeric
    df = df.filter(pl.col("id").str.contains(r"^\d+$"))

    # Drop totals rows
    df = df.filter(pl.col("Rd") != "Totals")

    # Drop unwanted columns
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(cols_to_drop)

# Drop rows with no name or position
    df = df.filter(
        pl.col("Name").is_not_null() & (pl.col("Name") != "") &
        pl.col("Posn").is_not_null() & (pl.col("Posn") != "") &
        (pl.col("Posn") != "Pos Pos")
    )

    # Clean money columns — strip $ and , then cast to int
    for col in MONEY_COLS:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col)
                .str.replace_all(r"[\$,]", "")
                .cast(pl.Int64, strict=False)
                .alias(col)
            )

    # Cast round to int
    df = df.with_columns(
        pl.col("Rd").cast(pl.Int32, strict=False)
    )

    # Cast int columns
    int_cols_existing = [c for c in INT_COLS if c in df.columns]
    df = df.with_columns([
        pl.col(c).cast(pl.Int32, strict=False) for c in int_cols_existing
    ])

    # Cast float columns
    float_cols_existing = [c for c in FLOAT_COLS if c in df.columns]
    df = df.with_columns([
        pl.col(c).cast(pl.Float32, strict=False) for c in float_cols_existing
    ])

    # Trim whitespace from name and position
    df = df.with_columns([
        pl.col("Name").str.strip_chars(),
        pl.col("Posn").str.strip_chars(),
    ])

    # Drop rows with no name or position
    df = df.filter(
        pl.col("Name").is_not_null() & (pl.col("Name") != "") &
        pl.col("Posn").is_not_null() & (pl.col("Posn") != "")
    )

    # Rename to snake_case
    rename_existing = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df = df.rename(rename_existing)

    # Replace "5/8" with "5_8" in position to avoid issues with "/" in column names later
    df = df.with_columns(
    pl.col("position").str.replace("5/8", "5_8")
        )

    # Split position into primary and secondary
    df = df.with_columns([
        pl.col("position").str.split(" ").list.get(0).alias("primary_position"),
        pl.col("position").str.split(" ").list.get(1, null_on_oob=True).alias("secondary_position"),
        
        ])
    
    return df


# %%
def run_silver_transform():
    """Load all Bronze parquets, transform, and save as a single Silver parquet."""
    bronze_files = sorted(BRONZE_DIR.glob("*.parquet"))

    if not bronze_files:
        logger.error(f"No Bronze parquets found in {BRONZE_DIR}")
        return

    all_dfs = []
    for f in bronze_files:
        logger.info(f"Loading {f.name}")
        df = pl.read_parquet(f)
        df_clean = transform_bronze_to_silver(df)
        all_dfs.append(df_clean)
        logger.info(f"{f.name} → {len(df_clean)} clean rows")

    combined = pl.concat(all_dfs, how="diagonal")

    output_path = SILVER_DIR / "player_rounds.parquet"
    combined.write_parquet(output_path)
    logger.success(f"Saved {len(combined)} total rows to {output_path}")


# %%
run_silver_transform()