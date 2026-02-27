# %%
import httpx
from bs4 import BeautifulSoup
import polars as pl
import re
from pathlib import Path
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
ROUND = 1
URL = f"https://leagueunlimited.com/news/leagueunlimited-nrl-teams-2026-round-{ROUND}/"
OUTPUT_PATH = Path(f"data/optimiser/team_lists_2026_round_{ROUND}.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# %%
def normalise_name(name: str) -> str:
    """Convert 'Firstname Surname' to 'Surname, Firstname'."""
    parts = name.strip().split()
    if len(parts) < 2:
        return name.strip()
    return f"{parts[-1]}, {' '.join(parts[:-1])}"

def classify_jersey(number: int) -> str:
    if number <= 13:
        return "starter"
    elif number <= 17:
        return "bench"
    else:
        return "reserve"

# %%
logger.info(f"Fetching round {ROUND} team lists from LeagueUnlimited")
response = httpx.get(URL, follow_redirects=True, timeout=15)
soup = BeautifulSoup(response.text, "html.parser")

# %%
# Find all bold tags — jersey numbers and player names are in <strong> tags
# Pattern: **1.** Kalyn Ponga **2.** Dominic Young ...
content = soup.get_text()

# Extract all team blocks — each team is under an img tag with team name
# Parse the raw text for numbered player lists
player_pattern = re.compile(r'\*\*(\d+)\.\*\*\s+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)+)')

# %%
# Better approach — parse the HTML bold tags directly
rows = []
current_team = None

for tag in soup.find_all(["img", "strong"]):
    # Team headers are img tags with alt text containing team name
    if tag.name == "img" and tag.get("alt") and any(
        keyword in tag.get("alt", "") 
        for keyword in ["Knights", "Cowboys", "Bulldogs", "Dragons", "Storm", 
                       "Eels", "Warriors", "Roosters", "Broncos", "Panthers",
                       "Sharks", "Titans", "Sea Eagles", "Raiders", "Dolphins", "Rabbitohs"]
    ):
        current_team = tag.get("alt").replace(" ", "_").upper()

    # Player entries are bold tags with pattern "N."
    if tag.name == "strong" and current_team:
        text = tag.get_text(strip=True)
        match = re.match(r'^(\d+)\.$', text)
        if match:
            jersey = int(match.group(1))
            # Player name is the next sibling text
            next_text = tag.next_sibling
            if next_text:
                player_name = str(next_text).strip()
                if player_name:
                    rows.append({
                        "team": current_team,
                        "jersey": jersey,
                        "player_name_raw": player_name,
                        "player_name": normalise_name(player_name),
                        "status": classify_jersey(jersey),
                    })

# %%
df = pl.DataFrame(rows)
logger.info(f"Parsed {len(df)} player entries across {df['team'].n_unique()} teams")
print(df.filter(pl.col("status") == "reserve").select(["team", "jersey", "player_name", "status"]))

df.write_parquet(OUTPUT_PATH)
logger.info(f"Saved to {OUTPUT_PATH}")