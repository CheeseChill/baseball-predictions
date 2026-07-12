"""Quick end-to-end smoke test: today's real schedule/odds/weather ->
build_todays_features() -> the 3 real trained models.

Run from the repo root:
    python test_today_pipeline.py
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from src.ingestion.mlb_stats import fetch_todays_probable_pitchers
from src.ingestion.odds import fetch_current_odds, get_consensus_line
from src.ingestion.weather import fetch_weather_for_games
from src.picks.daily_pipeline import _pivot_odds
from src.models.today_features import build_todays_features
from src.models.underdog_model import predict_moneyline
from src.models.spread_model import predict_spread
from src.models.totals_model import predict_totals

print("1. Fetching today's schedule + probable pitchers...")
schedule = fetch_todays_probable_pitchers()
print(f"   {len(schedule)} games found")
if schedule.empty:
    print("   No games today - nothing to test. Try again on a game day.")
    raise SystemExit(0)
print(schedule[["away_team", "home_team", "away_probable_pitcher", "home_probable_pitcher"]].to_string())

print("\n2. Fetching odds...")
odds_raw = fetch_current_odds()
consensus = get_consensus_line(odds_raw)
game_odds = _pivot_odds(consensus)
print(f"   {len(game_odds)} games with odds")

print("\n3. Fetching weather...")
weather = fetch_weather_for_games(schedule)
print(f"   {len(weather)} games with weather")

print("\n4. Building feature matrix (this calls live pitcher-stat + rest-day "
      "lookups per game, may take a bit)...")
feat = build_todays_features(schedule, game_odds, weather)
print(f"   shape: {feat.shape}")

from src.models.features import ALL_FEATURE_COLS
missing = [c for c in ALL_FEATURE_COLS if c not in feat.columns]
print(f"   missing ALL_FEATURE_COLS: {missing}")
nan_share = feat[ALL_FEATURE_COLS].isnull().mean().sort_values(ascending=False)
print("   top-10 columns by % missing:")
print(nan_share.head(10))

print("\n5. Running real models...")
ml = predict_moneyline(
    model_or_path="models/moneyline_xgb_v1.joblib",
    game_features=feat, home_ml_col="home_moneyline", away_ml_col="away_moneyline",
)
sp = predict_spread(
    model_or_path="models/spread_xgb_v1.joblib",
    game_features=feat, spread_price_col="home_spread_price",
)
ou = predict_totals(
    model_or_path="models/totals_xgb_v1.joblib",
    game_features=feat, over_price_col="over_price", under_price_col="under_price",
)

print("\n=== MONEYLINE ===")
cols = [c for c in ["hometeam", "visteam", "pred_home_win_prob", "edge", "edge_home", "edge_away"] if c in ml.columns]
print(ml[cols].to_string() if cols else ml.to_string())

print("\n=== SPREAD ===")
print(sp.to_string())

print("\n=== TOTALS ===")
print(ou.to_string())
