# %%
import polars as pl
import pandas as pd
from pathlib import Path
from loguru import logger
import sys
import argparse
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, value

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--round", type=int, required=True, help="Round number to optimise for")
args = parser.parse_args()

ROUND = args.round

# %%
PREDICTIONS_PATH = Path(f"data/optimiser/predictions/predictions_round_{ROUND}.parquet")
SALARY_CAP = 11_950_000

POSITION_REQUIREMENTS = {
    "HOK": 2,
    "FRF": 4,
    "2RF": 6,
    "HFB": 2,
    "5_8": 2,
    "CTW": 7,
    "FLB": 2,
}
FLEX_COUNT = 1
TOTAL_PLAYERS = sum(POSITION_REQUIREMENTS.values()) + FLEX_COUNT  # 26

# %%
if not PREDICTIONS_PATH.exists():
    logger.error(f"Predictions file not found: {PREDICTIONS_PATH}")
    logger.error(f"Run: uv run python pipelines/predict_round.py --round {ROUND}")
    sys.exit(1)

registry = pl.read_parquet(PREDICTIONS_PATH)
logger.info(f"Predictions loaded: {registry.shape}")

# %%
# Filter to likely starters only
df = registry.filter(pl.col("likely_to_play") == True)
logger.info(f"Players likely to play: {df.shape[0]}")

# %%
# Load team lists and filter out reserves
TEAM_LIST_PATH = Path(f"data/optimiser/team_lists_2026_round_{ROUND}.parquet")

if TEAM_LIST_PATH.exists():
    team_lists = pl.read_parquet(TEAM_LIST_PATH)
    reserves = team_lists.filter(pl.col("status") == "reserve")["player_name"].to_list()
    df = df.to_pandas()
    df["player_name_normalised"] = df["player_name"].str.replace("'", "", regex=False)
    reserves_normalised = [r.replace("'", "") for r in reserves]
    df = df[~df["player_name_normalised"].isin(reserves_normalised)].reset_index(drop=True)
    logger.info(f"Removed reserves — {len(df)} players remaining")
else:
    logger.warning("No team list found — skipping reserve filter")
    df = df.to_pandas()

df = df.reset_index(drop=True)

# %%
# Manual excludes — injured players not yet in team lists
manual_excludes = ["Mulitalo, Ronaldo", "Bostock, Jack", "Bird, Jack"]
df = df[~df["player_name"].isin(manual_excludes)].reset_index(drop=True)
logger.info(f"Removed {len(manual_excludes)} manual excludes — {len(df)} players remaining")

players = df.index.tolist()

# %%
prob = LpProblem("SuperCoach_Team_Selector", LpMaximize)

selected = LpVariable.dicts("selected", players, cat="Binary")
flex     = LpVariable.dicts("flex", players, cat="Binary")

# %%
# Objective — maximise round predicted score (matchup-adjusted)
prob += lpSum(
    df.loc[i, "round_predicted_score"] * selected[i] for i in players
)

# %%
# Assignment variables
assigned = {}
for i in players:
    for pos in POSITION_REQUIREMENTS.keys():
        eligible = (
            df.loc[i, "position_2026"] == pos or
            df.loc[i, "secondary_position_2026"] == pos
        )
        if eligible:
            assigned[(i, pos)] = LpVariable(f"assigned_{i}_{pos}", cat="Binary")

# %%
# Constraints

# Bye exclusion — driven by is_bye flag from predict_round pipeline
for i in players:
    if df.loc[i, "is_bye"]:
        prob += selected[i] == 0

# Total players
prob += lpSum(selected[i] for i in players) == TOTAL_PLAYERS

# Salary cap
prob += lpSum(df.loc[i, "price_2026"] * selected[i] for i in players) <= SALARY_CAP

# Each selected non-flex player assigned to exactly one position
for i in players:
    eligible_positions = [pos for pos in POSITION_REQUIREMENTS.keys() if (i, pos) in assigned]
    if eligible_positions:
        prob += lpSum(assigned[(i, pos)] for pos in eligible_positions) == selected[i] - flex[i]

# Each position filled exactly
for pos, count in POSITION_REQUIREMENTS.items():
    prob += lpSum(
        assigned[(i, pos)] for i in players if (i, pos) in assigned
    ) == count

# Flex constraints
prob += lpSum(flex[i] for i in players) == FLEX_COUNT
for i in players:
    prob += flex[i] <= selected[i]

# Assignment only if selected
for (i, pos), var in assigned.items():
    prob += var <= selected[i]

# %%
prob.solve()
logger.info(f"Status: {LpStatus[prob.status]}")

# %%
# Extract results
selected_players = df[[value(selected[i]) == 1 for i in players]].copy()

assigned_pos_map = {}
for i in players:
    if value(flex[i]) == 1:
        assigned_pos_map[i] = "FLEX"
    else:
        for pos in POSITION_REQUIREMENTS.keys():
            if (i, pos) in assigned and value(assigned[(i, pos)]) == 1:
                assigned_pos_map[i] = pos
                break

selected_players["is_flex"] = selected_players.index.map(lambda i: value(flex[i]) == 1)
selected_players["assigned_position"] = selected_players.index.map(assigned_pos_map)
selected_players = selected_players.sort_values(
    ["assigned_position", "round_predicted_score"], ascending=[True, False]
)

# %%
total_price     = selected_players["price_2026"].sum()
total_predicted = selected_players["round_predicted_score"].sum()
remaining_budget = SALARY_CAP - total_price

print(f"\n===== SELECTED TEAM — ROUND {ROUND} =====")
print(f"{'Player':<30} {'Pos':<6} {'Team':<5} {'Opponent':<10} {'Price':>10} {'Career':>8} {'Matchup':>8} {'Pred':>8} {'Flex':>5}")
print("-" * 95)

for _, row in selected_players.iterrows():
    flex_flag  = "✓" if row["is_flex"] else ""
    matchup    = f"{row['round_matchup_avg']:.1f}" if pd.notna(row.get("round_matchup_avg")) else "—"
    opponent   = row.get("round_opponent", "")
    print(
        f"{row['player_name']:<30} {row['assigned_position']:<6} {row['team_2026']:<5} "
        f"{str(opponent):<10} ${row['price_2026']:>9,} {row['career_avg']:>8.1f} "
        f"{matchup:>8} {row['round_predicted_score']:>8.1f} {flex_flag:>5}"
    )

print("-" * 95)
print(f"{'TOTAL':<30} {'':6} {'':5} {'':10} ${total_price:>9,} {'':>8} {'':>8} {total_predicted:>8.1f}")
print(f"Remaining budget: ${remaining_budget:,}")
print(f"Total players: {len(selected_players)}")

# %%
print("\n===== POSITIONAL SLOT COUNT =====")
for pos, count in POSITION_REQUIREMENTS.items():
    filled = sum(
        1 for i in players
        if (i, pos) in assigned and value(assigned[(i, pos)]) == 1
    )
    print(f"{pos}: {filled} assigned (need {count})")