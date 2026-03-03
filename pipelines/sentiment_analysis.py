"""
sentiment_analysis.py
---------------------
Reads scraped commentary JSON files, sends each article to Claude for structured
player sentiment extraction, then saves a sentiment_weights.parquet file for use
in the player registry and optimiser.

Output schema (per player, aggregated across articles):
    player_name       str
    team              str
    position          str   (HOK | FRF | 2RF | HFB | 5_8 | CTW | FLB)
    sentiment_score   f32   (-1.0 very negative → 1.0 very positive)
    confidence        f32   (0.0 → 1.0, author conviction)
    recommendation    str   (buy | hold | monitor | avoid)
    key_insight       str
    sources           str   (comma-joined URLs)
"""

import json
import os
from pathlib import Path

import anthropic
import polars as pl
from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COMMENTARY_DIR = Path("data/raw/commentary")
OUTPUT_PATH = Path("data/optimiser/sentiment_weights.parquet")

# ---------------------------------------------------------------------------
# Claude prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert NRL SuperCoach fantasy sports analyst. "
    "Extract player-specific insights from SuperCoach commentary articles. "
    "Always respond with valid JSON only — no markdown fences, no explanation."
)


def _build_prompt(title: str, body: str) -> str:
    return f"""Analyse this NRL SuperCoach article and extract all player mentions that contain fantasy-relevant insight.

For each such player return a JSON object with exactly these fields:
  "player_name"     — Full name, Firstname Lastname format
  "team"            — NRL team name (e.g. "Brisbane Broncos", "Melbourne Storm")
  "position"        — SuperCoach position: one of HOK, FRF, 2RF, HFB, 5_8, CTW, FLB
  "sentiment_score" — Float −1.0 (strongly avoid) to 1.0 (must buy)
  "confidence"      — Float 0.0 to 1.0 reflecting how certain the author seems
  "recommendation"  — One of: buy, hold, monitor, avoid
  "key_insight"     — Single sentence capturing the core fantasy takeaway

Return a JSON array of these objects. Skip players mentioned only in passing with no fantasy context.

Article title: {title}

Article body:
{body}"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def _analyse_article(client: anthropic.Anthropic, article: dict) -> list[dict]:
    """Send one article to Claude and return a list of player dicts."""
    prompt = _build_prompt(article["title"], article["body"])

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip accidental markdown fences if Claude adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    players = json.loads(raw)

    for p in players:
        p["source_url"] = article["url"]

    return players


def _aggregate(records: list[dict]) -> pl.DataFrame:
    """Deduplicate across articles by averaging numeric scores."""
    df = pl.DataFrame(
        records,
        schema={
            "player_name": pl.Utf8,
            "team": pl.Utf8,
            "position": pl.Utf8,
            "sentiment_score": pl.Float32,
            "confidence": pl.Float32,
            "recommendation": pl.Utf8,
            "key_insight": pl.Utf8,
            "source_url": pl.Utf8,
        },
        infer_schema_length=None,
    )

    aggregated = df.group_by("player_name").agg([
        pl.col("team").last(),
        pl.col("position").last(),
        pl.col("sentiment_score").mean().round(3),
        pl.col("confidence").mean().round(3),
        # Pick the most bullish recommendation if multiple articles disagree
        pl.col("recommendation").sort_by(
            pl.col("sentiment_score")
        ).last(),
        pl.col("key_insight").last(),
        pl.col("source_url").str.concat(", ").alias("sources"),
    ])

    return aggregated.sort("sentiment_score", descending=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run():
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to your .env file."
        )

    json_files = sorted(COMMENTARY_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No commentary JSON files found in {COMMENTARY_DIR}. "
            "Run scraper/commentary_scraper.py first."
        )

    client = anthropic.Anthropic(api_key=api_key)
    logger.info(f"Analysing {len(json_files)} article(s) with Claude...")

    all_records: list[dict] = []

    for path in json_files:
        article = json.loads(path.read_text(encoding="utf-8"))

        if not article.get("body"):
            logger.warning(f"  Skipping {path.name} — no body content")
            continue

        logger.info(f"  → {path.name}")
        try:
            players = _analyse_article(client, article)
            logger.info(f"     {len(players)} player(s) extracted")
            all_records.extend(players)
        except json.JSONDecodeError as exc:
            logger.error(f"     JSON parse error: {exc}")
        except Exception as exc:
            logger.error(f"     Error: {exc}")

    if not all_records:
        logger.warning("No player data extracted — nothing saved.")
        return

    df = _aggregate(all_records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT_PATH)
    logger.success(f"Saved {len(df)} players to {OUTPUT_PATH}")

    print(df.select([
        "player_name", "team", "position",
        "sentiment_score", "confidence", "recommendation", "key_insight"
    ]))


if __name__ == "__main__":
    run()
