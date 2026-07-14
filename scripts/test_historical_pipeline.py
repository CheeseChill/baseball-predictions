"""Smoke-test the live prediction pipeline against a REAL, already-played
game day, instead of waiting for the next live MLB slate.

Runs the exact same code path predictions.py's _load_model_predictions()
uses (fetch_todays_probable_pitchers -> build_todays_features -> 3 trained
models), but pointed at a past date, then cross-checks each prediction
against what actually happened (from Retrosheet) so you can eyeball
whether the numbers look sane after any pipeline/model change.

Usage:
    python scripts/test_historical_pipeline.py --date 2026-06-15
    python scripts/test_historical_pipeline.py            # defaults to
                                                            # the most recent
                                                            # date with games
                                                            # in Retrosheet

Notes:
  - Needs network access (MLB Stats API + Open-Meteo) — run this on your
    own machine, not in a restricted sandbox.
  - Weather lookups may silently return nothing for older dates
    (Open-Meteo's live forecast endpoint only covers a recent window);
    build_todays_features() already falls back gracefully when weather
    is missing, same as it does live.
  - Probable-pitcher data from statsapi.schedule() for old dates reflects
    who *actually* started (not a pregame guess), which is fine for this
    smoke test — we're checking "does the pipeline run and produce sane
    numbers", not re-litigating that day's pregame uncertainty.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.mlb_stats import fetch_todays_probable_pitchers
from src.ingestion.weather import fetch_weather_for_games
from src.models.today_features import build_todays_features, team_short_name
from src.models.underdog_model import predict_moneyline
from src.models.spread_model import predict_spread
from src.models.totals_model import predict_totals
from retrosheet import load_gameinfo

MODEL_DIR = ROOT / "models"


def _default_date() -> str:
    gi = load_gameinfo(2020, 2026)
    return gi["date"].max().strftime("%Y-%m-%d")


def main(target_date: str) -> None:
    print(f"Testing pipeline against real game day: {target_date}\n")

    schedule = fetch_todays_probable_pitchers(target_date)
    if schedule.empty:
        print("No games found for this date (off day / All-Star break / bad date). Try another date.")
        return
    print(f"Schedule: {len(schedule)} games")

    odds = pd.DataFrame(columns=["away_team", "home_team"])
    weather = fetch_weather_for_games(schedule)
    print(f"Weather rows fetched: {len(weather)} (0 is OK — falls back to neutral defaults)")

    features = build_todays_features(schedule, odds, weather)
    if features.empty:
        print("build_todays_features() returned empty — something upstream failed. Check the NOTE/WARN lines above.")
        return
    print(f"Feature matrix: {features.shape}\n")

    ml_preds = predict_moneyline(MODEL_DIR / "moneyline_xgb_v1.joblib", features)
    rl_preds = predict_spread(MODEL_DIR / "spread_xgb_v1.joblib", features)
    ou_preds = predict_totals(MODEL_DIR / "totals_xgb_v1.joblib", features)

    out = features[["hometeam", "visteam"]].copy()
    out["pred_home_win_prob"]   = ml_preds["pred_home_win_prob"].values
    out["pred_home_cover_prob"] = rl_preds["pred_cover_prob"].values
    out["pred_over_prob"]       = ou_preds["pred_prob_over"].values
    out["model_exp_total"]      = features["exp_total"].values

    # Cross-check against what actually happened that day
    gi = load_gameinfo(2020, 2026)
    actual = gi[gi["date"].dt.strftime("%Y-%m-%d") == target_date][
        ["hometeam", "visteam", "hruns", "vruns", "total_runs", "wteam"]
    ]

    merged = out.merge(actual, on=["hometeam", "visteam"], how="left")
    merged["actual_home_win"] = (merged["wteam"] == merged["hometeam"])
    merged["actual_home_cover"] = (merged["hruns"] - merged["vruns"]) >= 2
    merged["actual_over"] = merged["total_runs"] > merged["model_exp_total"]

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(merged[[
        "hometeam", "visteam",
        "pred_home_win_prob", "actual_home_win",
        "pred_home_cover_prob", "actual_home_cover",
        "pred_over_prob", "model_exp_total", "total_runs", "actual_over",
    ]].round(3).to_string(index=False))

    print("\nSanity checks:")
    print(f"  - Any NaN predictions? {merged[['pred_home_win_prob','pred_home_cover_prob','pred_over_prob']].isna().any().any()}")
    print(f"  - All probs in [0,1]? {((out[['pred_home_win_prob','pred_home_cover_prob','pred_over_prob']] >= 0) & (out[['pred_home_win_prob','pred_home_cover_prob','pred_over_prob']] <= 1)).all().all()}")
    print(f"  - Home win predicted-vs-actual agreement rate: {(merged['pred_home_win_prob'].round().astype(int) == merged['actual_home_win'].astype(int)).mean():.1%}")
    print("    (This is ONE day, not a real accuracy measurement — just a not-obviously-broken check.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, a real past game day")
    args = parser.parse_args()
    main(args.date or _default_date())
