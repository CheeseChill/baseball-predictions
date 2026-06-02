# MLB Betting Analytics — Next 5 Features to Implement

> **Based on:** Codebase gap analysis as of July 2025

---

## Feature 1: Umpire Pitch-Call Heat Map

**Why:** The game context factors already include `umpire runs/g` as a single number, but MLB bettors know that individual umpires have measurably different strike zones that affect total runs. A visual strike zone heat map per umpire would be a flagship UI feature no mainstream betting tool offers.

**How:**
1. Extend `src/ingestion/` with a `ump_pitchcast.py` fetcher that pulls umpire-level pitch call data from Baseball Savant's ump-scorecards endpoint (public API)
2. Store per-umpire strike probability by pitch location zone in `data_files/processed/ump_zones.parquet`
3. Add a Plotly heatmap visualization to `streamlit_app/pages/1_Today.py` under each game's umpire section
4. Add `ump_runs_g` rolling stat to the moneyline feature set in `src/models/`

**Complexity:** Medium

---

## Feature 2: Pitcher Fatigue / REST Modeling

**Why:** Pitcher rest days, days since last outing, and pitch count in last start directly affect performance. This data is available from the MLB Stats API and is one of the most reliable pre-game signals for totals prediction.

**How:**
1. Add `rest_days`, `pitches_last_start`, and `innings_last_start` features to `src/models/` feature engineering
2. Fetch from MLB Stats API `/api/v1/people/{pitcherId}/stats?group=pitching&type=gameLog&season={year}`
3. Add these to the totals model feature set (they predict run scoring, not just win/loss)
4. Display on the Today page cards: "SP well-rested" or "SP on 4 days rest" contextual badge

**Complexity:** Medium

---

## Feature 3: Best Bets JSON Export for Sports-Picks-Grid

**Why:** Baseball-predictions is listed in the sports-picks-grid aggregator REPOS mapping, but `data_files/best_bets_today.json` may not be written consistently. Ensuring this file is always generated with the correct schema would integrate the app into the unified dashboard.

**How:**
1. Add `scripts/export_best_bets.py` that reads the latest picks from `data_files/processed/picks_history.parquet`
2. Filter to today's picks with confidence tier ≥ MEDIUM
3. Write `data_files/best_bets_today.json` per the unified schema (`meta` + `bets` array)
4. Add this export step to the GitHub Actions nightly pipeline after pick generation
5. Validate the schema matches `docs/02-unified-schema.md` in sports-picks-grid

**Complexity:** Low

---

## Feature 4: Park Factor-Adjusted Totals Model

**Why:** The game context already shows park factor but the totals model does not incorporate it as a feature. Coors Field games are significantly different from Petco Park games — adjusting expected runs by park factor would improve the totals model's edge.

**How:**
1. `src/ingestion/fg_park.py` already fetches FanGraphs park factors — verify it runs and check its output schema
2. Merge park factors into the totals feature matrix in `src/models/`
3. Add `park_factor`, `park_hr_factor`, and `park_run_factor` as numeric features
4. Retrain the totals XGBoost model and compare edge with/without park features using cross-validation

**Complexity:** Low

---

## Feature 5: Series Context Feature

**Why:** MLB teams play 3-4 game series against the same opponent. Teams that lost the previous game in a series often perform differently (lineup changes, bullpen depletion, manager adjustments). A binary `is_series_opener` and `series_games_count` feature could improve prediction for games 2+ of a series.

**How:**
1. In `src/picks/daily_pipeline.py`, add logic to detect when today's game is part of an ongoing series
2. Compute: `is_series_opener` (binary), `series_game_number` (1–4), `series_wins_home`, `series_wins_away` from the schedule
3. Add these to the run-line feature matrix (series context matters most for ATS)
4. Display "Series: Game 2 of 3" badge on Today page cards

**Complexity:** Low
