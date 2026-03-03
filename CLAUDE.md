# NRL SuperCoach ML Optimiser

## Project Goal
Build a machine learning system that optimises NRL SuperCoach team selection — both initial squad building (26-player roster) and weekly lineup decisions — to outperform baseline approaches and experienced human players.

SuperCoach is an Australian rugby league fantasy sports game where players select teams within budget constraints and positional requirements to maximise points each round.

---

## Tech Stack

| Purpose | Tool |
|---|---|
| Environment & dependency management | UV |
| Language | Python |
| Data processing | Polars (chosen over PySpark to avoid Java deps) |
| ML models | XGBoost / LightGBM |
| Optimisation | PuLP (Binary Integer Linear Programming) |
| Experiment tracking & model registry | Azure ML + MLflow |
| Web scraping | Playwright |
| Version control | GitHub |

> **Note on Azure ML:** Originally planned Databricks but pivoted away due to free tier compute restrictions. Azure ML is the chosen middle ground — MLflow integration without the overhead.

---

## Architecture: Medallion Pipeline

```
Bronze  →  Silver  →  Gold
(raw)      (clean)    (features)
```

- **Bronze:** Raw scraped HTML/data from nrlsupercoachstats.com
- **Silver:** Cleaned and normalised player-round records
- **Gold:** Feature-engineered dataset ready for modelling

---

## Data Source

- **URL:** https://www.nrlsupercoachstats.com/stats.php
- **Coverage:** 11 years of historical data (2015–2025)
- **Volume:** ~80,000 player-round records
- **Scraper:** Playwright-based

---

## What Has Been Built

### Data Pipeline ✅
- Playwright scraper pulling 11 years of historical player stats
- Full Bronze → Silver → Gold pipeline complete
- ~80,000 player-round records processed

### Feature Engineering ✅
99 engineered features including:
- Lag scores (previous rounds)
- Rolling averages
- Break-even gaps
- Price momentum indicators
- Opponent strength metrics
- Match context features (home/away, ground, weather) — scraped via Playwright

### ML Model ✅
- XGBoost model trained and validated
- **MAE: 16.71** (~8 points better than baseline)
- Data leakage resolved — current-round stats excluded, lagged versions preserved

### Optimiser ✅
- PuLP-based Binary Integer Linear Programming
- Constraints: squad size, salary cap, positional requirements
- Successfully generates team selections

---

## Known Issues / Gaps

- **Match context features** (home/away, weather) showed minimal predictive impact — player-specific factors dominate
- **Optimiser limitations:**
  - Doesn't account for minimum-priced player selection (cash generation strategy)
  - No concentration risk management
  - Treats all selected players equally (no floor/ceiling variance modelling)
- **Two distinct problems** not yet fully separated: squad building vs weekly lineup selection
- **Azure ML workspace** setup was in progress at last session

---

## Critical Lessons Learned

**Data leakage:** Initial models showed suspiciously perfect performance because current-round statistics leaked the target variable. Always exclude current-round stats; lagged versions are fine.

**Optimisation nuance:** Experienced SuperCoach players follow established strategies (minimum-priced selections for cash generation, avoiding concentration risk) that a naive score-maximising optimiser won't replicate.

---

## What We're Doing Next

1. **Azure ML workspace** — complete setup for MLflow experiment tracking and model registry
2. **2026 player registry** — build dataset for the current season
3. **Bye-round awareness** — integrate bye schedules into the optimiser
4. **Configurable optimiser** — parameters for player counts, positions, budget
5. **Weekly prediction pipeline** — connect predictions to the optimiser end-to-end
6. **Matchup analysis** — career averages as baseline + opponent defensive weakness multipliers
7. **Separate the two optimisation problems** cleanly:
   - Squad building (season-long value, price rises, trading flexibility)
   - Weekly lineup (bye management, matchup-specific performance)

---

## Running the Project

```bash
# Install dependencies
uv sync

# Scrape data
# Run scraper scripts from src/scraping/

# Run pipeline
# Bronze → Silver → Gold processing scripts

# Train model
# Run training script, logs to Azure ML / MLflow

# Generate team selection
# Run optimiser with current predictions
```

> Update this section as the project structure is built out.

---

