# %%
import sys
import argparse
import subprocess
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
parser = argparse.ArgumentParser(description="Run the full weekly SuperCoach pipeline")
parser.add_argument("--round", type=int, required=True, help="Round number to run")
parser.add_argument("--skip-refresh", action="store_true", help="Skip data refresh (steps 1-4) and run selection only")
parser.add_argument("--skip-sentiment", action="store_true", help="Skip commentary scraping and sentiment scoring (use when Playwright is unavailable)")
args = parser.parse_args()

ROUND = args.round

# %%
REFRESH_STEPS = [
    ("Scraping 2026 stats",       ["uv", "run", "python", "scraper/stats_scraper.py"]),
    ("Running silver transform",  ["uv", "run", "python", "pipelines/silver_transform.py"]),
    ("Running gold features",     ["uv", "run", "python", "pipelines/gold_features.py"]),
    ("Building player registry",  ["uv", "run", "python", "optimiser/build_player_registry.py"]),
]

SENTIMENT_STEPS = [
    ("Scraping commentary",          ["uv", "run", "python", "scraper/commentary_scraper.py", "--round", str(ROUND)]),
    ("Scoring commentary sentiment", ["uv", "run", "python", "pipelines/sentiment_analysis.py"]),
]

SELECTION_STEPS = [
    ("Scraping team lists",       ["uv", "run", "python", "scraper/scrape_team_lists.py",  "--round", str(ROUND)]),
    ("Building predictions",      ["uv", "run", "python", "pipelines/predict_round.py",    "--round", str(ROUND)]),
    ("Running optimiser",         ["uv", "run", "python", "optimiser/optimiser.py",        "--round", str(ROUND)]),
]

# %%
def run_step(step_name: str, cmd: list[str]):
    logger.info(f"\n>>> {step_name}...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"Step failed: {step_name}")
        logger.error("Pipeline halted — fix the error above and rerun")
        sys.exit(1)
    logger.info(f"✓ {step_name} complete")

# %%
logger.info(f"Starting weekly pipeline — round {ROUND}")
logger.info("=" * 50)

if args.skip_refresh:
    logger.info("Skipping data refresh (--skip-refresh flag set)")
else:
    logger.info("--- DATA REFRESH ---")
    for step_name, cmd in REFRESH_STEPS:
        run_step(step_name, cmd)

if args.skip_sentiment:
    logger.info("Skipping sentiment steps (--skip-sentiment flag set)")
else:
    logger.info("\n--- COMMENTARY & SENTIMENT ---")
    for step_name, cmd in SENTIMENT_STEPS:
        run_step(step_name, cmd)

logger.info("\n--- ROUND SELECTION ---")
for step_name, cmd in SELECTION_STEPS:
    run_step(step_name, cmd)

logger.success(f"\nRound {ROUND} pipeline complete")