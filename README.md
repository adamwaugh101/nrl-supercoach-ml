# NRL SuperCoach ML

ML-driven NRL SuperCoach team optimisation.

## Project Goal
Predict player scores round-by-round and optimise team selection within SuperCoach budget and positional constraints.

## Stack
- **Scraping:** Playwright (headless browser)
- **Environment:** UV
- **Processing:** PySpark
- **Storage:** Databricks (Medallion architecture — Bronze / Silver / Gold)
- **ML:** scikit-learn, XGBoost, LightGBM (ensemble)
- **Optimisation:** PuLP (constrained linear programming)
- **Experiment Tracking:** MLflow (Databricks managed)

## Structure
```
nrl-supercoach-ml/
├── scraper/          # Playwright scraper for nrlsupercoachstats.com
├── pipelines/        # PySpark jobs for Bronze → Silver → Gold
├── models/           # Training, evaluation, ensemble logic
├── optimiser/        # Team selection optimisation layer
└── notebooks/        # EDA and exploration
```

## Setup

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync

# Install Playwright browsers
uv run playwright install chromium

# Copy and configure environment variables
cp .env.example .env
```

## Usage
```bash
# Run historical scrape
uv run python scraper/stats_scraper.py
```
