# %%
import httpx
import polars as pl
import json
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
BRONZE_DIR = Path("data/bronze/match_context")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://geo145327-staging.s3.ap-southeast-2.amazonaws.com/public/NRL/{year}/NRL_detailed_match_data_{year}.json"
YEARS = range(2015, 2026)  # 2025 attempted, skipped if missing

# %%
# In earlier years (approx 2015-2019), the fields are shifted/scrambled.
# The scraper stored values in wrong keys. We detect this by checking whether
# "Completion Rate" looks like a time string (e.g. "3.49s") which means
# the block is shifted. We remap accordingly.
SHIFTED_FIELD_MAP = {
    "Completion Rate":         "Average_Play_Ball_Speed",
    "Average_Play_Ball_Speed": "Kick_Defusal",
    "Kick_Defusal":            "Effective_Tackle",
    "Effective_Tackle":        "Completion Rate",
}

def is_shifted(team_data: dict) -> bool:
    """Detect if fields are in the older scrambled layout."""
    val = team_data.get("Completion Rate", "")
    # Completion rate should look like "78%" not "3.49s"
    return isinstance(val, str) and val.endswith("s")

def fix_shifted_fields(team_data: dict) -> dict:
    """Remap scrambled fields to their correct keys."""
    fixed = dict(team_data)
    for wrong_key, correct_key in SHIFTED_FIELD_MAP.items():
        fixed[correct_key] = team_data.get(wrong_key, -1)
    return fixed

# %%
def safe_float(val) -> float | None:
    """Convert messy strings like '78%', '3.49s', '1,870' to float."""
    if val is None or val == -1:
        return None
    s = str(val).replace("%", "").replace("s", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None

def parse_team_stats(team_data: dict, is_home_shifted: bool) -> dict:
    """Extract and clean a flat dict of team stats."""
    if is_home_shifted:
        team_data = fix_shifted_fields(team_data)

    return {
        "time_in_possession":    team_data.get("time_in_possession"),
        "all_runs":              safe_float(team_data.get("all_runs")),
        "all_run_metres":        safe_float(team_data.get("all_run_metres")),
        "post_contact_metres":   safe_float(team_data.get("post_contact_metres")),
        "line_breaks":           safe_float(team_data.get("line_breaks")),
        "tackle_breaks":         safe_float(team_data.get("tackle_breaks")),
        "offloads":              safe_float(team_data.get("offloads")),
        "kicks":                 safe_float(team_data.get("kicks")),
        "kicking_metres":        safe_float(team_data.get("kicking_metres")),
        "tackles_made":          safe_float(team_data.get("tackles_made")),
        "missed_tackles":        safe_float(team_data.get("missed_tackles")),
        "errors":                safe_float(team_data.get("errors")),
        "penalties_conceded":    safe_float(team_data.get("penalties_conceded")),
        "interchanges_used":     safe_float(team_data.get("interchanges_used")),
        "completion_rate_pct":   safe_float(team_data.get("Completion Rate")),
        "avg_play_ball_speed_s": safe_float(team_data.get("Average_Play_Ball_Speed")),
        "kick_defusal_pct":      safe_float(team_data.get("Kick_Defusal")),
        "effective_tackle_pct":  safe_float(team_data.get("Effective_Tackle")),
        "tries":                 safe_float(team_data.get("tries")),
    }

# %%
def parse_year(year: int, raw: dict) -> list[dict]:
    """
    Parse one year's JSON into a list of flat match-team rows.
    Each match produces TWO rows: one for the home team, one for the away team.
    This makes joining onto player-round data straightforward.
    """
    rows = []
    # Each item in the NRL list is one round: [{"1": [...]}, {"2": [...]}, ...]
    nrl_list = raw.get("NRL", [])

    for round_obj in nrl_list:
     for round_key, matches in round_obj.items():
        # round_key is "1", "2", ... or sometimes "Finals Week 1" etc
        try:
            round_num = int(round_key)
            round_type = "regular"
        except ValueError:
            round_num = None
            round_type = round_key  # e.g. "Finals Week 1"

        for match_dict in matches:
            for matchup, data in match_dict.items():
                # matchup looks like "Broncos v Rabbitohs"
                parts = matchup.split(" v ")
                if len(parts) != 2:
                    continue
                home_team, away_team = parts[0].strip(), parts[1].strip()

                match_meta = data.get("match", {})
                home_raw   = data.get("home", {})
                away_raw   = data.get("away", {})

                # Detect field shift on home side (away side shifts too if home does)
                shifted = is_shifted(home_raw)

                home_stats = parse_team_stats(home_raw, shifted)
                away_stats = parse_team_stats(away_raw, shifted)

                base = {
                    "year":              year,          # int
                    "round":             round_num,     # int or None
                    "round_type":        round_type,
                    "home_team":         home_team,
                    "away_team":         away_team,
                    "ground_condition":  match_meta.get("ground_condition"),
                    "weather_condition": match_meta.get("weather_condition"),
                    "main_ref":          match_meta.get("main_ref"),
                    "fields_shifted":    shifted,       # bool
                }

                # Home team row
                home_row = {**base, "team": home_team, "opponent": away_team, "is_home": True}
                home_row.update({f"team_{k}": v for k, v in home_stats.items()})
                home_row.update({f"opp_{k}": v for k, v in away_stats.items()})
                rows.append(home_row)

                away_row = {**base, "team": away_team, "opponent": home_team, "is_home": False}
                away_row.update({f"team_{k}": v for k, v in away_stats.items()})
                away_row.update({f"opp_{k}": v for k, v in home_stats.items()})
                rows.append(away_row)

    return rows

# %%
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-AU,en;q=0.9",
    "Referer": "https://nrlpredictions.net/",
}

def download_year(year: int) -> dict | None:
    """Download JSON for one year from S3. Returns None if not found."""
    url = BASE_URL.format(year=year)
    try:
        response = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"{year}: HTTP {response.status_code} — skipping")
            return None
    except Exception as e:
        logger.warning(f"{year}: Failed to download — {e}")
        return None

# %%
def run():
    all_rows = []

    for year in YEARS:
        logger.info(f"Downloading {year}...")
        raw = download_year(year)
        if raw is None:
            continue

        rows = parse_year(year, raw)
        logger.info(f"  {year}: {len(rows) // 2} matches → {len(rows)} team-rows")
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No data collected — check URLs or network")
        return

    df = pl.DataFrame(all_rows, infer_schema_length=None)

    non_numeric = {
        "year", "round", "round_type", "home_team", "away_team",
        "ground_condition", "weather_condition", "main_ref",
        "team", "opponent", "fields_shifted", "is_home",
        "team_time_in_possession", "opp_time_in_possession",
    }
    numeric_cols = [c for c in df.columns if c not in non_numeric]

    df = df.with_columns([
        pl.col("year").cast(pl.Int32),
        pl.col("round").cast(pl.Int32, strict=False),
        pl.col("fields_shifted") == "True",
        pl.col("is_home") == "True",
        *[pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols],
    ])
    output_path = BRONZE_DIR / "nrl_match_context.parquet"
    df.write_parquet(output_path)
    logger.success(f"Saved {len(df)} rows, {len(df.columns)} columns → {output_path}")
    logger.info(f"Years covered: {sorted(df['year'].unique().to_list())}")
    logger.info(f"Sample columns: {df.columns[:10]}")

# %%
run()