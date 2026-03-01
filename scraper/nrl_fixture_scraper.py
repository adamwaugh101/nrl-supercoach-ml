# %%
import httpx
import json
import time
import sys
import polars as pl
from bs4 import BeautifulSoup
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
BRONZE_DIR = Path("data/bronze/fixtures")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

COMPETITION_ID = "111"  # NRL Premiership
BASE_URL = "https://www.nrl.com/draw/?competition={competition}&round={round}&season={year}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
}

MAX_ROUNDS = 30
YEARS = [2025]

# %%
def fetch_round(year: int, round_num: int, client: httpx.Client) -> list[dict] | None:
    """
    Fetch fixture data for a single round.
    Returns list of match dicts, empty list if no matches, None on error.
    """
    url = BASE_URL.format(competition=COMPETITION_ID, round=round_num, year=year)

    try:
        response = client.get(url, headers=HEADERS, timeout=20, follow_redirects=True)
    except Exception as e:
        logger.warning(f"  Round {round_num}: Request failed — {e}")
        return None

    if response.status_code != 200:
        logger.warning(f"  Round {round_num}: HTTP {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    vue_div = soup.find("div", {"id": "vue-draw"})
    if not vue_div:
        logger.debug(f"  Round {round_num}: No #vue-draw div — likely past end of season")
        return []

    try:
        raw_json = vue_div["q-data"].replace("&quot;", '"')
        data = json.loads(raw_json)
    except (KeyError, json.JSONDecodeError) as e:
        logger.warning(f"  Round {round_num}: Failed to parse q-data — {e}")
        return None

    fixtures = data.get("fixtures", [])
    matches = []

    for fixture in fixtures:
        if fixture.get("type") != "Match":
            continue

        home  = fixture.get("homeTeam", {})
        away  = fixture.get("awayTeam", {})
        clock = fixture.get("clock", {})

        # Store everything as strings — cast to correct types after DataFrame
        # creation to avoid Polars schema inference conflicts across rounds
        matches.append({
            "year":             str(year),
            "round":            str(round_num),
            "round_title":      str(fixture.get("roundTitle") or ""),
            "home_team":        str(home.get("nickName") or ""),
            "home_score":       str(home.get("score") or ""),
            "away_team":        str(away.get("nickName") or ""),
            "away_score":       str(away.get("score") or ""),
            "venue":            str(fixture.get("venue") or ""),
            "match_date":       str(clock.get("kickOffTimeLong") or ""),
            "match_state":      str(fixture.get("matchState") or ""),
            "match_centre_url": "https://www.nrl.com" + str(fixture.get("matchCentreUrl") or ""),
        })

    return matches

# %%
def scrape_year(year: int, client: httpx.Client) -> list[dict]:
    """Scrape all rounds for a year, stopping after 2 consecutive empty rounds."""
    all_matches = []
    empty_rounds = 0

    for round_num in range(1, MAX_ROUNDS + 1):
        logger.info(f"  Fetching round {round_num}...")
        matches = fetch_round(year, round_num, client)

        if matches is None:
            empty_rounds += 1
        elif len(matches) == 0:
            empty_rounds += 1
            logger.info(f"  Round {round_num}: No matches — {empty_rounds} empty round(s)")
        else:
            empty_rounds = 0
            all_matches.extend(matches)
            logger.info(f"  Round {round_num}: {len(matches)} matches")

        if empty_rounds >= 2:
            break

        time.sleep(0.5)

    return all_matches

# %%
def run(years: list[int] = YEARS):
    all_rows = []

    with httpx.Client() as client:
        for year in years:
            logger.info(f"Scraping {year}...")
            matches = scrape_year(year, client)
            logger.info(f"  {year}: {len(matches)} total matches")
            all_rows.extend(matches)

    if not all_rows:
        logger.error("No data collected")
        return

    # All values stored as strings — now cast to correct types
    df = pl.DataFrame(all_rows)
    df = df.with_columns([
        pl.col("year").cast(pl.Int32),
        pl.col("round").cast(pl.Int32),
        pl.col("home_score").cast(pl.Int32, strict=False),
        pl.col("away_score").cast(pl.Int32, strict=False),
    ])

    output_path = BRONZE_DIR / f"nrl_fixtures_{'_'.join(str(y) for y in years)}.parquet"
    df.write_parquet(output_path)

    logger.success(f"Saved {len(df)} matches → {output_path}")
    logger.info(f"\n{df.select(['year', 'round', 'home_team', 'away_team', 'venue', 'match_state']).head(10)}")

# %%
run()