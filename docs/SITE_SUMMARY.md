> **AI Onboarding Guide** — See also `.github/copilot-instructions.md` for full coding conventions.

# MLB Betting Analytics (baseball-predictions) — Site Summary

## What This App Does

MLB betting analytics platform that generates daily wagering recommendations across three markets (moneyline, run line, totals) using XGBoost and LightGBM models. Provides Kelly-criterion bet sizing, a Streamlit dashboard with 7 pages, and a DraftKings Pick 6 player prop calculator.

## Quick Start

```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS/Linux

# 2. Run the app
streamlit run predictions.py
```

Data is pre-warmed from Parquet files on first load. GitHub Actions handles nightly data refresh and model retraining.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (multi-page, 7 pages) |
| ML | XGBoost, LightGBM, scikit-learn |
| Data storage | Parquet (primary), CSV (raw downloads) |
| Scheduling | APScheduler |
| Bankroll | Kelly Criterion (`src/bankroll/kelly.py`) |
| API (optional) | FastAPI + Uvicorn |
| Visualization | Plotly, Altair |

## Key Files

| File | Purpose |
|---|---|
| `predictions.py` | Streamlit entry point — home page with per-game recommendations (moneyline, run line, O/U) |
| `streamlit_app/pages/` | 7 pages: Today, Stats, Matchup Analysis, Models, Performance, Pick 6, About |
| `src/picks/daily_pipeline.py` | Main daily orchestration pipeline — pick generation + settlement |
| `src/models/` | Feature engineering + XGBoost/LightGBM training |
| `src/ingestion/` | Data fetching, Parquet consolidation |
| `src/bankroll/kelly.py` | Kelly Criterion bet sizing |
| `src/data/queries.py` | **All** data access functions — reads Parquet, returns DataFrames |
| `src/data/cache.py` | `@st.cache_data` wrappers around query functions |

## Data Flow

1. **Ingestion**: MLB Stats API + odds APIs → CSV files in `data_files/raw/`
2. **Consolidation**: CSVs → Parquet in `data_files/processed/`
3. **Feature engineering**: `src/models/` builds feature matrices from Parquet
4. **Prediction**: XGBoost/LightGBM → output probabilities per bet type
5. **Pick generation**: Edge vs implied odds → filter by confidence tier → `picks_history.parquet`
6. **Dashboard**: Streamlit reads Parquet via `src/data/queries.py` → displays picks with badge signals

## Confidence Tiers

| Tier | Threshold | Sizing |
|---|---|---|
| HIGH | Edge > 6%, strong model agreement | Half-Kelly |
| MEDIUM | Edge 3–6%, single model | Quarter-Kelly |
| LOW | Edge 1–3% | Tracked, minimal sizing |

## Badge Signals on Home Page

| Badge | Meaning |
|---|---|
| ✅ BET | HIGH confidence |
| ➡ LEAN | MEDIUM confidence |
| ⛔ PASS | LOW confidence |

## Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `ODDS_API_KEY` | The Odds API — live MLB odds | Required |
| PostgreSQL connection vars | Optional database backend | Optional |

## Critical Conventions

- **Primary storage is Parquet** — never use PostgreSQL as the default
- Use `pyarrow` schemas defined in `src/data/schemas.py` when writing Parquet
- Business logic lives in `src/` — never in Streamlit pages
- Pre-warm all large datasets with `@st.cache_data` at startup; tabs reference pre-loaded variables
- Use `pathlib.Path` for all file paths — never string concatenation
- The `load_parquet_to_postgres` helper exists but is unused; it may be removed

## Common Gotchas

- Tabs must reference pre-loaded session-state variables, not call data functions themselves (avoids per-tab reload delays)
- Three separate XGBoost models for three bet types (underdog ML, run line, totals) — each has its own feature set
- `src/ingestion/fg_park.py` and `src/ingestion/fg_guts.py` handle FanGraphs park/guts constants — check these exist before running the pipeline
