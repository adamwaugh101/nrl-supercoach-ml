# %%
import time
import sys
import polars as pl
from pathlib import Path
from playwright.sync_api import sync_playwright
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
BRONZE_DIR    = Path("data/bronze")
FIXTURES_PATH = BRONZE_DIR / "fixtures/nrl_fixtures_2025.parquet"
OUTPUT_PATH   = BRONZE_DIR / "match_context/nrl_match_context_2025.parquet"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

PAGE_DELAY = 2.0  # seconds between pages — be polite

# %%
def scrape_match_weather(page, url: str) -> dict:
    """
    Navigate to a match centre URL and extract ground condition and weather.
    Returns a dict with 'ground_condition' and 'weather_condition' (both may be None).
    """
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
    except Exception as e:
        logger.warning(f"  Failed to load {url} — {e}")
        return {"ground_condition": None, "weather_condition": None}

    result = {"ground_condition": None, "weather_condition": None}

    try:
        # Each <p class="match-weather__text"> contains a label and a <span> with the value
        weather_paras = page.query_selector_all("p.match-weather__text")

        for para in weather_paras:
            text = para.inner_text().strip()
            span = para.query_selector("span")
            value = span.inner_text().strip() if span else None

            if "Ground" in text:
                result["ground_condition"] = value
            elif "Weather" in text:
                result["weather_condition"] = value

    except Exception as e:
        logger.warning(f"  Failed to parse weather from {url} — {e}")

    return result

# %%
def run():
    # Load fixtures — only scrape completed matches
    df_fixtures = pl.read_parquet(FIXTURES_PATH)
    completed = df_fixtures.filter(pl.col("match_state") == "FullTime")

    logger.info(f"Found {len(completed)} completed matches to scrape")

    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for i, row in enumerate(completed.iter_rows(named=True)):
            url = row["match_centre_url"]
            logger.info(f"  [{i+1}/{len(completed)}] Round {row['round']}: {row['home_team']} v {row['away_team']}")

            weather = scrape_match_weather(page, url)

            logger.info(f"    Ground: {weather['ground_condition']} | Weather: {weather['weather_condition']}")

            base = {
                "year":              str(row["year"]),
                "round":             str(row["round"]),
                "home_team":         str(row["home_team"]),
                "away_team":         str(row["away_team"]),
                "ground_condition":  str(weather["ground_condition"] or ""),
                "weather_condition": str(weather["weather_condition"] or ""),
            }

            # Two rows per match — one per team — mirrors the S3 match context schema
            rows.append({**base, "team": str(row["home_team"]), "opponent": str(row["away_team"]), "is_home": "True"})
            rows.append({**base, "team": str(row["away_team"]), "opponent": str(row["home_team"]), "is_home": "False"})

            time.sleep(PAGE_DELAY)

        browser.close()

    if not rows:
        logger.error("No data collected")
        return

    df = pl.DataFrame(rows)
    df = df.with_columns([
        pl.col("year").cast(pl.Int32),
        pl.col("round").cast(pl.Int32),
        pl.col("is_home").map_elements(lambda x: x == "True", return_dtype=pl.Boolean),
    ])

    df.write_parquet(OUTPUT_PATH)
    logger.success(f"Saved {len(df)} rows → {OUTPUT_PATH}")
    logger.info(f"\n{df.head(10)}")

# %%
run()