# %%
import polars as pl
import pandas as pd
from pathlib import Path
from loguru import logger
import sys
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, value

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
REGISTRY_PATH = Path("data/optimiser/player_registry_2026.parquet")
SALARY_CAP = 11_950_000
BYE_TEAMS_R1 = {"WST"}


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
ROUND = 1
# %%
registry = pl.read_parquet(REGISTRY_PATH)
logger.info(f"Registry loaded: {registry.shape}")

# %%
# Filter to likely starters only
df = registry.filter(pl.col("likely_to_play") == True)
logger.info(f"Players likely to play: {df.shape[0]}")

# Convert to pandas for PuLP
df = df.to_pandas()
df = df.reset_index(drop=True)
players = df.index.tolist()


# %%
# Load team lists and filter out reserves

TEAM_LIST_PATH = Path(f"data/optimiser/team_lists_2026_round_{ROUND}.parquet")


if TEAM_LIST_PATH.exists():
    team_lists = pl.read_parquet(TEAM_LIST_PATH)
    reserves = team_lists.filter(pl.col("status") == "reserve")["player_name"].to_list()
    df = df[~df["player_name"].isin(reserves)].reset_index(drop=True)
    players = df.index.tolist()
    logger.info(f"Removed {len(reserves)} reserves — {len(df)} players remaining")
else:
    logger.warning("No team list found — skipping reserve filter")

#%%

#Normalise names for matching — remove apostrophes and trim whitespace
df["player_name_normalised"] = df["player_name"].str.replace("'", "", regex=False)
reserves_normalised = [r.replace("'", "") for r in reserves]
df = df[~df["player_name_normalised"].isin(reserves_normalised)].reset_index(drop=True)
players = df.index.tolist()

# %%
# Manual excludes — injured players not yet in team lists
manual_excludes = ["Mulitalo, Ronaldo", "Bostock, Jack","Bird, Jack"]
df = df[~df["player_name"].isin(manual_excludes)].reset_index(drop=True)
players = df.index.tolist()
logger.info(f"Removed {len(manual_excludes)} manual excludes — {len(df)} players remaining")


# Debug name matching
# registry_names = set(df["player_name"].tolist())
# reserve_names = set(reserves)
# matched = registry_names & reserve_names
# unmatched = reserve_names - registry_names

# logger.info(f"Matched reserves: {matched}")
# logger.info(f"Unmatched reserves: {unmatched}")    
# %%
prob = LpProblem("SuperCoach_Team_Selector", LpMaximize)

# Decision variables — 1 if player selected, 0 if not
selected = LpVariable.dicts("selected", players, cat="Binary")

# Flex slot — 1 if player is selected as flex
flex = LpVariable.dicts("flex", players, cat="Binary")

# %%
# Objective — maximise total predicted score
# Best 17 of 18 score (lowest drops out)
# Approximate by maximising total of all 18 — optimiser will naturally
# prefer higher scorers, the lowest will be the flex insurance
prob += lpSum(
    # df.loc[i, "predicted_score"] * selected[i] for i in players
    # df.loc[i, "career_avg"] * selected[i] for i in players   
    df.loc[i, "matchup_adjusted_avg"] * selected[i] for i in players    
)

# Assignment variables — assign each player to a specific position slot
# A player can only be assigned to a position they're eligible for
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
# %%
# Constraints
for i in players:
    if df.loc[i, "team_2026"] in BYE_TEAMS_R1:
        prob += selected[i] == 0
# Total players = 26
prob += lpSum(selected[i] for i in players) == TOTAL_PLAYERS

# Salary cap
prob += lpSum(df.loc[i, "price_2026"] * selected[i] for i in players) <= SALARY_CAP

# Each selected non-flex player must be assigned to exactly one position
for i in players:
    eligible_positions = [
        pos for pos in POSITION_REQUIREMENTS.keys()
        if (i, pos) in assigned
    ]
    if eligible_positions:
        prob += lpSum(assigned[(i, pos)] for pos in eligible_positions) == selected[i] - flex[i]

# Each position must have exactly the required number of players assigned
for pos, count in POSITION_REQUIREMENTS.items():
    prob += lpSum(
        assigned[(i, pos)] 
        for i in players 
        if (i, pos) in assigned
    ) == count

# Flex slot — exactly 1 flex player
prob += lpSum(flex[i] for i in players) == FLEX_COUNT

# Flex player must be selected
for i in players:
    prob += flex[i] <= selected[i]

# A player can only be assigned if selected
for (i, pos), var in assigned.items():
    prob += var <= selected[i]

# %%
# Solve
prob.solve()
logger.info(f"Status: {LpStatus[prob.status]}")

# %%
# Extract results
selected_players = df[[value(selected[i]) == 1 for i in players]].copy()
flex_players = df[[value(flex[i]) == 1 for i in players]].copy()

selected_players["is_flex"] = selected_players.index.isin(flex_players.index)

selected_players["assigned_position"] = selected_players.index.map(
    lambda i: "FLEX" if value(flex[i]) == 1 else next(
        (pos for pos in POSITION_REQUIREMENTS.keys() 
         if (i, pos) in assigned and value(assigned[(i, pos)]) == 1), ""
    )
)

# Build assigned position map
assigned_pos_map = {}
for i in players:
    if value(flex[i]) == 1:
        assigned_pos_map[i] = "FLEX"
    else:
        for pos in POSITION_REQUIREMENTS.keys():
            if (i, pos) in assigned and value(assigned[(i, pos)]) == 1:
                assigned_pos_map[i] = pos
                break

selected_players["assigned_position"] = selected_players.index.map(assigned_pos_map)

selected_players = selected_players.sort_values(
    ["assigned_position", "predicted_score"], ascending=[True, False]
)

# %%
total_price = selected_players["price_2026"].sum()
total_predicted = selected_players["predicted_score"].sum()
remaining_budget = SALARY_CAP - total_price

print("\n===== SELECTED TEAM =====")
print(f"{'Player':<30} {'Pos':<6} {'Team':<5} {'Price':>10} {'Pred Score':>12} {'Flex':>6}")
print("-" * 75)

for _, row in selected_players.iterrows():
    flex_flag = "✓" if row["is_flex"] else ""
    print(
        f"{row['player_name']:<30} {row['position_2026']:<6} {row['team_2026']:<5} "
        f"${row['price_2026']:>9,} {row['predicted_score']:>12.1f} {flex_flag:>6}"
    )

print("-" * 75)
print(f"{'TOTAL':<30} {'':6} {'':5} ${total_price:>9,} {total_predicted:>12.1f}")
print(f"Remaining budget: ${remaining_budget:,}")
print(f"Total players: {len(selected_players)}")

# %%
# Show dual position players and which slot they're filling
# %%
print("\n===== POSITION BREAKDOWN =====")
print(f"{'Player':<30} {'Assigned To':<10} {'Primary':<8} {'Secondary':<10} {'Flex':<6}")
print("-" * 68)

for _, row in selected_players.iterrows():
    i = row.name
    secondary = row["secondary_position_2026"] if pd.notna(row.get("secondary_position_2026")) else ""
    dual = f"({secondary})" if secondary else ""
    flex_flag = "✓" if row["is_flex"] else ""

    assigned_pos = "FLEX" if row["is_flex"] else ""
    if not row["is_flex"]:
        for pos in POSITION_REQUIREMENTS.keys():
            if (i, pos) in assigned and value(assigned[(i, pos)]) == 1:
                assigned_pos = pos
                break

    print(
    f"{row['player_name']:<30} {row['assigned_position']:<6} {row['team_2026']:<5} "
    f"${row['price_2026']:>9,} {row['predicted_score']:>12.1f} {flex_flag:>6}"
)

print("\n===== POSITIONAL SLOT COUNT =====")
for pos, count in POSITION_REQUIREMENTS.items():
    filled = sum(
        1 for i in players
        if (i, pos) in assigned and value(assigned[(i, pos)]) == 1
    )
    print(f"{pos}: {filled} assigned (need {count})")