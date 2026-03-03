# %%
import sys
import argparse
import polars as pl
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
GOLD_PATH         = Path("data/gold/player_rounds_features.parquet")
FIXTURE_PATH      = Path("data/bronze/fixtures/nrl_fixtures_2026.parquet")
REGISTRY_PATH     = Path("data/optimiser/player_registry_2026.parquet")
PLAYER_STATES_PATH = Path("data/gold/player_states.parquet")
OUTPUT_DIR        = Path("data/optimiser/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cluster momentum bonus — based on price_momentum_3 and score variance profiles
# Positive = favour in optimiser, negative = penalise
CLUSTER_BONUS = {
    0:  1,   # Boom-bust mid-tier      (+momentum, high variance)
    1:  0,   # Steady mid-tier         (flat momentum, low variance)
    2: -2,   # Declining mid           (negative momentum)
    3:  3,   # Rising premiums         (strongest positive momentum)
    4: -4,   # Low/declining cheapies  (strongest negative momentum)
    5:  2,   # Consistent premiums     (positive momentum, reliable)
}
ANOMALY_LAMBDA = 0.5  # risk-aversion weight on anomaly score

# Maps full team names (from fixture scraper) → 3-letter registry codes
TEAM_NAME_MAP = {
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
    "Wests Tigers":         "WST",
    "Eels":                 "PAR",
    "Titans":               "GCT",
    "Cowboys":              "NQC",
    "Dragons":              "STG",
    "Dolphins":             "DOL",
}

# %%
def get_round_opponents(fixture: pl.DataFrame, round_num: int) -> dict:
    """
    Returns a dict mapping team_code → opponent_code for a given round.
    Teams on a bye will not appear in the dict.
    """
    round_fixtures = fixture.filter(pl.col("round") == round_num)

    if len(round_fixtures) == 0:
        logger.error(f"No fixtures found for round {round_num}")
        return {}

    opponents = {}
    for row in round_fixtures.iter_rows(named=True):
        home = TEAM_NAME_MAP.get(row["home_team"])
        away = TEAM_NAME_MAP.get(row["away_team"])

        if home is None:
            logger.warning(f"Unknown home team: {row['home_team']}")
        if away is None:
            logger.warning(f"Unknown away team: {row['away_team']}")

        if home and away:
            opponents[home] = away
            opponents[away] = home

    logger.info(f"Round {round_num} matchups: {opponents}")
    return opponents


# %%
def get_latest_player_states() -> pl.DataFrame:
    """
    Load the most recent cluster and anomaly score per player from the
    autoencoder output. Returns one row per player (latest year + round).
    Players with no history get cluster=-1 and anomaly_score=0 (neutral).
    """
    if not PLAYER_STATES_PATH.exists():
        logger.warning("player_states.parquet not found — skipping cluster signal")
        return pl.DataFrame(schema={
            "player_name": pl.String,
            "cluster": pl.Int32,
            "anomaly_score": pl.Float32,
        })

    states = pl.read_parquet(PLAYER_STATES_PATH)
    return (
        states.sort(["player_name", "year", "round"])
        .group_by("player_name")
        .agg([
            pl.col("cluster").last(),
            pl.col("anomaly_score").last(),
        ])
    )


def get_latest_matchup_avgs(gold: pl.DataFrame) -> pl.DataFrame:
    """
    For each player/opponent combination, get the most recent
    matchup_adjusted_avg. This represents their blended historical
    performance against that specific opponent.
    """
    return (
        gold.sort(["player_name", "opponent", "year", "round"])
        .group_by(["player_name", "opponent"])
        .agg(pl.col("matchup_adjusted_avg").last())
    )


# %%
def build_round_predictions(round_num: int) -> pl.DataFrame:
    """
    Build a prediction dataframe for a given round by joining:
      - Player registry (prices, positions, career stats)
      - Round fixtures (who plays who)
      - Matchup averages from gold features
    """

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading fixture data")
    fixture = pl.read_parquet(FIXTURE_PATH)

    logger.info("Loading player registry")
    registry = pl.read_parquet(REGISTRY_PATH)

    logger.info("Loading gold features")
    gold = pl.read_parquet(GOLD_PATH)

    # ── Get opponents for this round ───────────────────────────────────────────
    opponents = get_round_opponents(fixture, round_num)

    if not opponents:
        logger.error(f"Cannot build predictions — no fixture data for round {round_num}")
        return pl.DataFrame()

    # ── Add opponent and bye flag to registry ──────────────────────────────────
    opponent_df = pl.DataFrame({
        "team_2026": list(opponents.keys()),
        "round_opponent": list(opponents.values()),
    })

    registry = registry.join(opponent_df, on="team_2026", how="left")

    registry = registry.with_columns(
        pl.col("round_opponent").is_null().alias("is_bye")
    )

    bye_teams = registry.filter(pl.col("is_bye"))["team_2026"].unique().to_list()
    if bye_teams:
        logger.info(f"Bye teams this round: {bye_teams}")

    # ── Join matchup-adjusted averages ─────────────────────────────────────────
    logger.info("Computing matchup-adjusted averages")
    matchup_avgs = get_latest_matchup_avgs(gold)

    registry = registry.join(
        matchup_avgs.rename({
            "opponent": "round_opponent",
            "matchup_adjusted_avg": "round_matchup_avg"
        }),
        on=["player_name", "round_opponent"],
        how="left"
    )

    # Fall back to career_avg where no matchup history exists
    registry = registry.with_columns(
        pl.when(pl.col("is_bye"))
          .then(pl.lit(0.0))
          .when(pl.col("round_matchup_avg").is_not_null())
          .then(pl.col("round_matchup_avg"))
          .otherwise(pl.col("career_avg"))
          .alias("round_predicted_score")
    )

    registry = registry.with_columns(
        pl.when(pl.col("round_predicted_score").is_null())
          .then(pl.col("career_avg"))
          .otherwise(pl.col("round_predicted_score"))
          .alias("round_predicted_score")
    )

    logger.info(f"Built predictions for {len(registry)} players")
    logger.info(f"  With matchup history: {registry['round_matchup_avg'].is_not_null().sum()}")
    logger.info(f"  Falling back to career avg: {registry['round_matchup_avg'].is_null().sum()}")
    logger.info(f"  On bye (score=0): {registry['is_bye'].sum()}")

    # ── Join autoencoder cluster signal ───────────────────────────────────────
    logger.info("Joining player state clusters")
    player_states = get_latest_player_states()

    if len(player_states) > 0:
        registry = registry.join(player_states, on="player_name", how="left")
        registry = registry.with_columns([
            pl.col("cluster").fill_null(-1).cast(pl.Int32),
            pl.col("anomaly_score").fill_null(0.0),
        ])
    else:
        registry = registry.with_columns([
            pl.lit(-1).cast(pl.Int32).alias("cluster"),
            pl.lit(0.0).alias("anomaly_score"),
        ])

    # Map cluster → momentum bonus (unknown/new players get 0)
    cluster_bonus_map = CLUSTER_BONUS
    registry = registry.with_columns(
        pl.col("cluster")
          .replace(cluster_bonus_map, default=0)
          .cast(pl.Float64)
          .alias("cluster_bonus")
    )

    # adjusted_predicted_score = base score + cluster momentum - anomaly penalty
    registry = registry.with_columns(
        (
            pl.col("round_predicted_score")
            + pl.col("cluster_bonus")
            - (ANOMALY_LAMBDA * pl.col("anomaly_score"))
        ).alias("adjusted_predicted_score")
    )

    matched = (registry["cluster"] >= 0).sum()
    logger.info(f"  Players with cluster signal: {matched} / {len(registry)}")

    return registry


# %%
def run(round_num: int):
    logger.info(f"Building predictions for round {round_num}")

    predictions = build_round_predictions(round_num)

    if len(predictions) == 0:
        return

    output_path = OUTPUT_DIR / f"predictions_round_{round_num}.parquet"
    predictions.write_parquet(output_path)
    logger.success(f"Saved predictions → {output_path}")

    # Preview top predicted players
    preview = (
        predictions
        .filter(pl.col("is_bye") == False)
        .filter(pl.col("likely_to_play") == True)
        .sort("round_predicted_score", descending=True)
        .select([
            "player_name", "team_2026", "round_opponent",
            "position_2026", "price_2026",
            "career_avg", "round_matchup_avg", "round_predicted_score"
        ])
        .head(20)
    )

    print(f"\n===== TOP 20 PREDICTED SCORES — ROUND {round_num} =====")
    print(preview)


# %%
parser = argparse.ArgumentParser()
parser.add_argument("--round", type=int, required=True, help="Round number to predict")
args = parser.parse_args()

run(args.round)