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
args = parser.parse_args()

ROUND = args.round

# %%
STEPS = [
    ("Scraping team lists",     ["uv", "run", "python", "scraper/scrape_team_lists.py", "--round", str(ROUND)]),
    ("Building predictions",    ["uv", "run", "python", "pipelines/predict_round.py",   "--round", str(ROUND)]),
    ("Running optimiser",       ["uv", "run", "python", "optimiser/optimiser.py",       "--round", str(ROUND)]),
]

# %%
logger.info(f"Starting weekly pipeline for round {ROUND}")
logger.info("=" * 50)

for step_name, cmd in STEPS:
    logger.info(f"\n>>> {step_name}...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"Step failed: {step_name}")
        logger.error("Pipeline halted — fix the error above and rerun")
        sys.exit(1)

    logger.info(f"✓ {step_name} complete")

logger.success(f"\nRound {ROUND} pipeline complete")