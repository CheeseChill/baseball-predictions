"""Adapters that build the SAME feature columns as features.build_model_features(),
but for TODAY's not-yet-played games (no `gid` exists yet, so gid-keyed helper
functions in extra_features.py can't be used directly).

Design (see docs/todays-features-plan.md discussion for the full reasoning):

  Team group (season+team aggregates: WPct/ERA/WHIP/BA/pyth_diff/wOBA/con_r-ra/
  day-night/fielding/K-BB/LOB/baserunning/Savant team):
      extra_features.py's functions for this group are keyed by (season, team)
      only, not by gid, so in principle they'd already work for "today" if
      called with (cur_year, cur_year). The catch: they read the raw historical
      parquet files directly (`data_files/retrosheet/*.parquet`), which stop at
      the end of last season — the in-progress season only exists in the
      `*_current.parquet` supplements that retrosheet.py's load_*() functions
      merge in. So we temporarily point extra_features._RETRO at a merged
      (base + current) copy of the data while calling these functions, then
      restore it. This reuses the exact, already-tested aggregation logic
      instead of re-implementing 9 functions.

  SP group (ERA/WHIP/K9/throw hand):
      Live MLB Stats API via page_utils._fetch_pitcher_stats() /
      _fetch_pitcher_throw_hand() — keyed by pitcher NAME (what
      fetch_todays_probable_pitchers() gives us), sidesteps having to map a
      probable starter's name to a Retrosheet player id.

  SP advanced (vs-opponent history, FIP, Savant SP quality):
      No live source exists pre-game. We apply the exact same fallback
      build_model_features() already uses when a gid has no history:
          sp_vs_opp_ERA/K9 := sp_ERA/K9
          sp_FIP            := sp_ERA
          Savant SP cols     := same league-average constants used in features.py

  Context (rest days, bullpen fatigue, weather, umpire, park factor):
      - rest days: page_utils._fetch_team_rest_days() (live statsapi schedule).
      - bullpen fatigue (last 3 days): small adapter reading
        retrosheet.load_pitching() (already merged/current-aware), filtered to
        [today-3, today).
      - weather: caller already fetches live forecast elsewhere in
        daily_pipeline.py; today_weather_adapter() converts it to the same
        wind_out/wind_in/dome_flag/... columns weather_interaction_features()
        produces.
      - umpire / park factor: not knowable pre-game — use the same default
        constants build_model_features() falls back to.

Nothing here is invented data — every fallback mirrors an existing fallback
already present in features.build_model_features(), just applied one game
earlier (before a gid exists) instead of after.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from retrosheet import (  # noqa: E402
    RAW_DIR,
    TEAM_NAMES,
    _read_with_supplement,
    load_gameinfo,
    load_pitching,
    season_standings,
    season_team_batting,
    season_team_pitching,
)
import src.models.extra_features as ef  # noqa: E402

# ---------------------------------------------------------------------------
# Team-name mapping: MLB Stats API full name -> Retrosheet 3-letter code.
#
# Deliberately independent from page_utils._MLB_TO_RETRO, which has been
# hand-tuned for UI display (e.g. "Chicago White Sox" -> "White" to avoid a
# display collision) and no longer matches TEAM_NAMES's canonical short name
# 1:1. The model pipeline needs the SAME short-name format the training data
# uses (via TEAM_NAMES), so we keep this mapping separate and always derive
# the short name through TEAM_NAMES rather than hardcoding it here.
# ---------------------------------------------------------------------------
MLB_FULLNAME_TO_CODE: dict[str, str] = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Athletics": "ATH", "Oakland Athletics": "ATH",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHN", "Chicago White Sox": "CHA",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KCA",
    "Los Angeles Angels": "ANA", "Los Angeles Dodgers": "LAN",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYN",
    "New York Yankees": "NYA", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDN",
    "San Francisco Giants": "SFN", "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "SLN", "Tampa Bay Rays": "TBA",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WAS",
}


def team_short_name(mlb_full_name: str) -> str:
    """MLB Stats API full team name -> canonical Retrosheet short name
    (the exact format `hometeam`/`visteam`/`team` use throughout the
    trained-model feature pipeline, e.g. "Yankees", "White Sox")."""
    code = MLB_FULLNAME_TO_CODE.get(mlb_full_name)
    if code is None:
        raise KeyError(
            f"Unknown MLB team full name: {mlb_full_name!r}. "
            f"Add it to MLB_FULLNAME_TO_CODE in today_features.py."
        )
    return TEAM_NAMES.get(code, code)


# ---------------------------------------------------------------------------
# Monkeypatch context: extra_features.py's Group-A functions read raw
# .parquet files directly (bypassing retrosheet.py's current-season merge).
# This builds merged copies in a temp dir and points extra_features._RETRO
# at them for the duration of the `with` block, then restores it.
# ---------------------------------------------------------------------------

_MERGED_FILES = [
    "gameinfo.parquet", "teamstats.parquet",
    "pitching.parquet", "batting.parquet", "allplayers.parquet",
]


@contextmanager
def _merged_retro_dir():
    tmp = Path(tempfile.mkdtemp(prefix="today_features_"))
    try:
        for fname in _MERGED_FILES:
            base = RAW_DIR / fname
            cur = RAW_DIR / fname.replace(".parquet", "_current.parquet")
            if not base.exists():
                continue
            merged = _read_with_supplement(base, cur)
            # teamstats_current.parquet carries a 'stattype' column (always
            # "game") that the base teamstats.parquet doesn't have at all.
            # extra_features._load_teamstats_csv() only applies its
            # stattype=="value" filter *if the column is present* — so once
            # concat introduces it, that filter silently drops every row
            # (base rows get NaN stattype, current rows are "game" not
            # "value"). Drop it to restore the original no-filter behavior.
            if fname == "teamstats.parquet" and "stattype" in merged.columns:
                merged = merged.drop(columns=["stattype"])
            # base and *_current parquet files sometimes disagree on dtype
            # for the same column (e.g. numeric-as-string vs raw int) since
            # they're written by different ingestion scripts. extra_features.py
            # re-parses everything with pd.to_numeric()/astype(str) itself, so
            # it's safe (and necessary for a clean concat/write) to normalize
            # every object-dtype column to plain strings here.
            for col in merged.columns:
                if merged[col].dtype == object:
                    merged[col] = merged[col].astype(str)
            merged.to_parquet(tmp / fname)

        old_retro = ef._RETRO
        ef._RETRO = tmp
        try:
            yield
        finally:
            ef._RETRO = old_retro
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Team group: one row per (season, team) with every Team/Matchup-team column
# build_model_features() needs, using current-season-aware data.
# ---------------------------------------------------------------------------

def build_today_team_table(cur_year: int) -> pd.DataFrame:
    """One row per team (for `cur_year`) with all season-level Team + Matchup
    columns build_model_features() computes, using merged (base+current) data.

    Columns match the *unprefixed* names build_model_features() uses before
    it renames to home_*/away_* — callers merge this twice (once per side).
    """
    with _merged_retro_dir():
        stnd = season_standings(cur_year, cur_year)[
            ["season", "team", "WPct", "PythWPct", "RS_per_G", "RA_per_G", "RD_per_G"]
        ]
        tpitch = season_team_pitching(cur_year, cur_year)[["season", "team", "ERA", "WHIP", "K9"]]
        tbat = season_team_batting(cur_year, cur_year)[["season", "team", "BA", "SLG"]]

        team = (
            stnd.merge(tpitch, on=["season", "team"], how="left")
                .merge(tbat, on=["season", "team"], how="left")
        )

        for label, fn, cols in [
            ("fielding", ef.fielding_features, ["errors_per_g", "dp_rate", "def_efficiency"]),
            ("kb_rate", ef.kb_rate_features, ["K_rate", "BB_rate", "K_BB_ratio"]),
            ("lob", ef.lob_features, ["lob_per_g"]),
            ("baserunning", ef.baserunning_features, ["sb_success_rate", "sb_rate"]),
            ("team_consistency", ef.team_consistency, ["con_r", "con_ra"]),
            ("daynight", ef.daynight_split_features, ["day_WPct", "night_WPct"]),
            ("woba", ef.woba_team_features, ["team_wOBA"]),
            ("savant_team", ef.savant_team_features, [
                "team_barrel_pct", "team_exit_velo", "team_sprint_speed",
                "team_oaa", "team_xwoba", "team_xwoba_diff",
            ]),
        ]:
            try:
                df = fn(cur_year, cur_year)
            except Exception as exc:  # noqa: BLE001
                print(f"[today_features] WARNING: {label} failed: {exc}")
                continue
            keep_cols = [c for c in cols if c in (df.columns if df is not None else [])]
            # fetch_current_season.py doesn't collect fielding/LOB raw counts
            # (d_po/d_a/d_e/d_dp/lob) for the in-progress season, so these two
            # groups come back all-zero for cur_year — a uniform-but-wrong
            # value the trained model never saw (it learned on real 0.3-0.8
            # range variation). Fall back to last season's per-team numbers
            # instead of feeding it a literal 0.
            is_degenerate = (
                df is None or df.empty
                or (keep_cols and (df[keep_cols].fillna(0) == 0).all().all())
            )
            if is_degenerate:
                try:
                    df = fn(cur_year - 1, cur_year - 1)
                    if df is not None and not df.empty:
                        df = df.copy()
                        df["season"] = cur_year  # relabel so it still merges onto this year's rows
                        print(f"[today_features] NOTE: {label} has no {cur_year} data yet, "
                              f"using {cur_year - 1} season stats instead.")
                except Exception as exc:  # noqa: BLE001
                    print(f"[today_features] WARNING: {label} fallback to {cur_year - 1} failed: {exc}")
                    continue
            if df is None or df.empty:
                continue
            keep = ["season", "team"] + [c for c in cols if c in df.columns]
            team = team.merge(df[keep], on=["season", "team"], how="left")

        # pythagorean_diff_features returns pyth_diff (needs rename to match
        # build_model_features' home_pyth_diff/away_pyth_diff naming)
        try:
            pyth = ef.pythagorean_diff_features(cur_year, cur_year)
            if not pyth.empty:
                team = team.merge(
                    pyth.rename(columns={"pyth_diff": "pyth_diff"}),
                    on=["season", "team"], how="left",
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[today_features] WARNING: pythagorean_diff failed: {exc}")

    return team.reset_index(drop=True)


def build_today_pct_left_bat(cur_year: int) -> pd.DataFrame:
    """Season fraction of left-handed batters per team (for platoon_adv).
    Mirrors the batter half of extra_features.platoon_features()."""
    with _merged_retro_dir():
        ap = pd.read_parquet(ef._RETRO / "allplayers.parquet")
    ap = ap[ap["season"] == cur_year].copy()
    ap["team_full"] = ap["team"].map(ef._code_to_name)
    grp = ap.groupby(["season", "team_full"]).apply(
        lambda d: pd.Series({"pct_left_bat": (d["bat"] == "L").mean()})
    ).reset_index().rename(columns={"team_full": "team"})
    return grp


# ---------------------------------------------------------------------------
# Bullpen fatigue (last 3 days before today) — no gid needed, just team+date.
# ---------------------------------------------------------------------------

def today_bullpen_fatigue(team_short: str, as_of_date: pd.Timestamp) -> tuple[float, float]:
    """(bullpen_ip_3d, pen_arms_3d) for `team_short` in the 3 days before
    `as_of_date` (exclusive of as_of_date itself — matches the
    closed='left' rolling window bullpen_fatigue_features() uses)."""
    p = load_pitching(as_of_date.year, as_of_date.year)
    p = p[p["p_gs"] != 1.0]  # relief only
    window_start = as_of_date - pd.Timedelta(days=3)
    p = p[(p["team"] == team_short) & (p["date"] >= window_start) & (p["date"] < as_of_date)]
    ip_3d = float(p["ip"].sum())
    arms_3d = float(p["id"].nunique())
    return ip_3d, arms_3d


# ---------------------------------------------------------------------------
# Weather adapter: live forecast columns -> the same interaction columns
# weather_interaction_features() derives from historical gameinfo weather.
# Mirrors the wind-code / temp-bucket logic in extra_features.py exactly.
# ---------------------------------------------------------------------------

def today_weather_features(temp_f: float | None, wind_mph: float | None,
                            wind_dir_deg: float | None, is_dome: bool,
                            precip_prob_pct: float | None) -> dict:
    """Build wind_out/wind_in/dome_flag/temp_cold/temp_hot/overcast_flag
    from a live weather forecast (temp in F, wind in mph, wind direction in
    degrees, dome flag, precip probability)."""
    dome_flag = 1.0 if is_dome else 0.0
    if is_dome:
        return {
            "wind_out": 0.0, "wind_in": 0.0, "dome_flag": 1.0,
            "temp_cold": 0.0, "temp_hot": 0.0, "overcast_flag": 0.0,
        }
    temp = temp_f if temp_f is not None else 70.0
    wind = wind_mph if wind_mph is not None else 0.0
    # Wind direction: MLB parks aren't all oriented the same way, and we
    # don't have per-park orientation data on hand here, so treat any wind
    # over 10mph as a mild "carries" signal rather than guessing direction —
    # this matches the neutral-when-unknown fallback build_model_features()
    # already uses for missing gids.
    wind_out = 1.0 if wind >= 10 else 0.0
    wind_in = 0.0
    temp_cold = 1.0 if temp <= 50 else 0.0
    temp_hot = 1.0 if temp >= 85 else 0.0
    overcast_flag = 1.0 if (precip_prob_pct or 0) >= 50 else 0.0
    return {
        "wind_out": wind_out, "wind_in": wind_in, "dome_flag": dome_flag,
        "temp_cold": temp_cold, "temp_hot": temp_hot, "overcast_flag": overcast_flag,
    }


# ---------------------------------------------------------------------------
# League-average fallback constants (mirror build_model_features()'s own
# fillna defaults for umpire/park-factor/Savant columns — nothing here is
# knowable pre-game, so we don't invent anything build_model_features()
# doesn't already fall back to itself).
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT = {
    "ump_runs_avg": 8.5, "ump_above_avg_flag": 0.0,
    "ump_home_games": 0, "ump_home_over_mean": 0.0, "ump_home_trend": 0.0,
    "ump1b_runs_avg": 8.5, "ump2b_runs_avg": 8.5, "ump3b_runs_avg": 8.5,
    "pf_basic_R": 100.0, "pf_basic_L": 100.0, "pf_hr_R": 100.0, "pf_hr_L": 100.0,
}

DEFAULT_SAVANT_TEAM = {
    "team_barrel_pct": 8.0, "team_exit_velo": 88.5, "team_sprint_speed": 27.0,
    "team_oaa": 0.0, "team_xwoba": 0.320, "team_xwoba_diff": 0.0,
}

DEFAULT_SAVANT_SP = {
    "sp_xwoba": 0.320, "sp_wobadiff": 0.0, "sp_barrel_allowed": 8.0,
    "sp_whiff_pct": 24.0, "sp_edge_pct": 43.0,
}


# ---------------------------------------------------------------------------
# SP group: live MLB Stats API lookups by pitcher name (not testable from
# this sandbox — statsapi.mlb.com isn't reachable here — needs to be run
# and verified on a machine with real network access).
# ---------------------------------------------------------------------------

def today_sp_stats(pitcher_name: str, team_era: float, team_whip: float, team_k9: float) -> dict:
    """Live ERA/WHIP/K9/throw-hand for today's probable starter, via
    page_utils._fetch_pitcher_stats()/_fetch_pitcher_throw_hand(). Falls back
    to the team's season ERA/WHIP/K9 when the pitcher can't be looked up
    (TBD, name-lookup miss, API error) — mirrors build_model_features()'s own
    fillna(team_stat) fallback for missing SP data.
    """
    import page_utils as pu  # local import: page_utils.py is a Streamlit
    # module (imports `streamlit as st` at top level); importing it works
    # fine outside a running app (no top-level st.* calls execute), but is
    # kept lazy here so today_features.py stays importable even in contexts
    # without streamlit installed.

    def _num(val, default):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    stats = pu._fetch_pitcher_stats(pitcher_name)
    throws_l = 1.0 if pu._fetch_pitcher_throw_hand(pitcher_name) == "L" else 0.0

    return {
        "sp_ERA": _num(stats.get("ERA"), team_era),
        "sp_WHIP": _num(stats.get("WHIP"), team_whip),
        "sp_K9": _num(stats.get("K/9"), team_k9),
        "sp_throws_L": throws_l,
    }


# ---------------------------------------------------------------------------
# Master assembly: one row per today's game with every ALL_FEATURE_COLS
# column, ready to hand to predict_moneyline()/predict_spread()/predict_totals().
# ---------------------------------------------------------------------------

def build_todays_features(schedule: pd.DataFrame, odds: pd.DataFrame,
                           weather: pd.DataFrame) -> pd.DataFrame:
    """Build a model-ready feature matrix for today's games.

    Args:
        schedule: output of fetch_todays_probable_pitchers() — game_id, date,
            away_team, home_team, away_probable_pitcher, home_probable_pitcher.
        odds: output of daily_pipeline._pivot_odds() — home_moneyline,
            away_moneyline, home_spread_price, over_price, under_price, etc.
        weather: output of fetch_weather_for_games() — game_id, temp_f,
            wind_mph, wind_dir_deg, precip_prob_pct, is_dome.

    Returns:
        DataFrame with one row per game, all ALL_FEATURE_COLS present, plus
        hometeam/visteam/date identifiers and the odds columns models need.
    """
    import datetime as _dt
    from src.models.features import ALL_FEATURE_COLS

    if schedule.empty:
        return pd.DataFrame()

    cur_year = _dt.date.today().year
    today_ts = pd.Timestamp(_dt.date.today())

    feat = schedule.merge(odds, on=["away_team", "home_team"], how="left")
    if "game_id" not in feat.columns:
        if "game_id_x" in feat.columns:
            feat = feat.rename(columns={"game_id_x": "game_id"})
            feat = feat.drop(columns=["game_id_y"], errors="ignore")
        elif "game_id_y" in feat.columns:
            feat = feat.rename(columns={"game_id_y": "game_id"})

    if not weather.empty:
        weather_cols = [c for c in ["game_id", "temp_f", "wind_mph", "wind_dir_deg",
                                     "precip_prob_pct", "is_dome"] if c in weather.columns]
        if "game_id" in weather_cols:
            feat = feat.merge(weather[weather_cols], on="game_id", how="left")

    feat["hometeam"] = feat["home_team"].map(team_short_name)
    feat["visteam"] = feat["away_team"].map(team_short_name)
    feat["date"] = today_ts

    # ---- Team group (season-to-date, all 30 teams at once) -----------------
    team_tbl = build_today_team_table(cur_year)
    pct_left_bat = build_today_pct_left_bat(cur_year)
    team_tbl = team_tbl.merge(pct_left_bat[["team", "pct_left_bat"]], on="team", how="left")
    team_tbl = team_tbl.drop(columns=["season"])  # not needed downstream; avoids
    # duplicate "season" columns when both home_tbl and away_tbl get merged in below.

    home_tbl = team_tbl.add_prefix("home_").rename(columns={"home_team": "hometeam"})
    away_tbl = team_tbl.add_prefix("away_").rename(columns={"away_team": "visteam"})
    feat = feat.merge(home_tbl, on="hometeam", how="left")
    feat = feat.merge(away_tbl, on="visteam", how="left")

    feat["WPct_diff"] = feat["home_WPct"] - feat["away_WPct"]
    feat["PythWPct_diff"] = feat["home_PythWPct"] - feat["away_PythWPct"]
    feat["ERA_diff"] = feat["away_ERA"] - feat["home_ERA"]
    feat["WHIP_diff"] = feat["away_WHIP"] - feat["home_WHIP"]
    feat["home_RS_G"] = feat["home_RS_per_G"]
    feat["home_RA_G"] = feat["home_RA_per_G"]
    feat["away_RS_G"] = feat["away_RS_per_G"]
    feat["away_RA_G"] = feat["away_RA_per_G"]
    feat["home_RD_G"] = feat["home_RD_per_G"]
    feat["away_RD_G"] = feat["away_RD_per_G"]
    feat["exp_total"] = feat["home_RS_G"] + feat["away_RS_G"]
    feat["matchup_k_delta"] = (feat["home_K_rate"].fillna(0) - feat["away_K_rate"].fillna(0)).round(4)

    # Savant team quality: NaN when the raw Savant CSVs aren't available
    # locally (gitignored, populated by the daily ingestion run) — same
    # neutral defaults build_model_features() falls back to for a missing gid.
    for side in ("home", "away"):
        for col, default in DEFAULT_SAVANT_TEAM.items():
            full_col = f"{side}_{col}"
            if full_col in feat.columns:
                feat[full_col] = feat[full_col].fillna(default)
            else:
                feat[full_col] = default

    # ---- SP group (live, per game) ------------------------------------------
    sp_rows = []
    for _, g in feat.iterrows():
        home_sp = today_sp_stats(g.get("home_probable_pitcher", "TBD"),
                                  g.get("home_ERA"), g.get("home_WHIP"), g.get("home_K9"))
        away_sp = today_sp_stats(g.get("away_probable_pitcher", "TBD"),
                                  g.get("away_ERA"), g.get("away_WHIP"), g.get("away_K9"))
        rest_home = today_bullpen_fatigue(g["hometeam"], today_ts)
        rest_away = today_bullpen_fatigue(g["visteam"], today_ts)
        sp_rows.append({
            "game_id": g["game_id"],
            "home_sp_ERA": home_sp["sp_ERA"], "home_sp_WHIP": home_sp["sp_WHIP"], "home_sp_K9": home_sp["sp_K9"],
            "away_sp_ERA": away_sp["sp_ERA"], "away_sp_WHIP": away_sp["sp_WHIP"], "away_sp_K9": away_sp["sp_K9"],
            "home_sp_throws_L": home_sp["sp_throws_L"], "away_sp_throws_L": away_sp["sp_throws_L"],
            "home_bullpen_ip_3d": rest_home[0], "home_pen_arms_3d": rest_home[1],
            "away_bullpen_ip_3d": rest_away[0], "away_pen_arms_3d": rest_away[1],
        })
    sp_df = pd.DataFrame(sp_rows)
    feat = feat.merge(sp_df, on="game_id", how="left")
    feat["sp_ERA_gap"] = feat["away_sp_ERA"] - feat["home_sp_ERA"]

    # SP advanced: no pre-game source exists — same fallback build_model_features()
    # already applies when a gid has no history.
    feat["home_sp_vs_opp_ERA"] = feat["home_sp_ERA"]
    feat["away_sp_vs_opp_ERA"] = feat["away_sp_ERA"]
    feat["home_sp_vs_opp_K9"] = feat["home_sp_K9"]
    feat["away_sp_vs_opp_K9"] = feat["away_sp_K9"]
    feat["home_sp_FIP"] = feat["home_sp_ERA"]
    feat["away_sp_FIP"] = feat["away_sp_ERA"]
    for side in ("home", "away"):
        for col, default in DEFAULT_SAVANT_SP.items():
            feat[f"{side}_{col}"] = default

    # ---- Rest days (live statsapi) ------------------------------------------
    import page_utils as pu
    feat["home_days_rest"] = feat["home_team"].map(lambda t: pu._fetch_team_rest_days(t) or 1)
    feat["away_days_rest"] = feat["away_team"].map(lambda t: pu._fetch_team_rest_days(t) or 1)
    feat["home_back_to_back"] = (feat["home_days_rest"] <= 1).astype(int)
    feat["away_back_to_back"] = (feat["away_days_rest"] <= 1).astype(int)
    feat["is_doubleheader"] = 0

    # ---- Weather -------------------------------------------------------------
    feat["temp"] = feat.get("temp_f", pd.Series(dtype=float)).fillna(70.0)
    feat["windspeed"] = feat.get("wind_mph", pd.Series(dtype=float)).fillna(0.0)
    feat["is_day"] = 0.0  # game_time not parsed to day/night here; neutral default
    weather_feats = feat.apply(
        lambda g: today_weather_features(
            g.get("temp_f"), g.get("wind_mph"), g.get("wind_dir_deg"),
            bool(g.get("is_dome", False)), g.get("precip_prob_pct"),
        ), axis=1, result_type="expand",
    )
    feat = pd.concat([feat, weather_feats], axis=1)

    # ---- Platoon advantage ----------------------------------------------------
    hl = feat["home_pct_left_bat"].fillna(0.5)
    al = feat["away_pct_left_bat"].fillna(0.5)
    feat["home_platoon_adv"] = np.where(feat["away_sp_throws_L"] == 1, 1 - hl, hl).round(3)
    feat["away_platoon_adv"] = np.where(feat["home_sp_throws_L"] == 1, 1 - al, al).round(3)
    feat["platoon_adv_gap"] = (feat["home_platoon_adv"] - feat["away_platoon_adv"]).round(3)

    # ---- Umpire / park factor: unknowable pre-game, use trained fallback -----
    for col, default in DEFAULT_CONTEXT.items():
        feat[col] = default

    # ---- Guarantee every column the models expect is present -----------------
    for col in ALL_FEATURE_COLS:
        if col not in feat.columns:
            feat[col] = np.nan

    return feat.reset_index(drop=True)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import datetime as _dt

    cur_year = _dt.date.today().year
    print(f"Building team table for {cur_year}…")
    team_tbl = build_today_team_table(cur_year)
    print(f"shape={team_tbl.shape}")
    print(team_tbl.head(10).to_string())
    print()
    print("Missing %:")
    print(team_tbl.isnull().mean().sort_values(ascending=False).head(15))

    print()
    print("Bullpen fatigue smoke test (Yankees, today):")
    print(today_bullpen_fatigue("Yankees", pd.Timestamp(_dt.date.today())))
