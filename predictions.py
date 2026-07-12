"""
Entry point for the Betting Cleanup MLB dashboard.

  - st.set_page_config()  called exactly once here
  - home_page()           landing page with per-game betting recommendations
  - st.navigation()       6-page sidebar navigation
"""

import sys
import datetime
import math
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from page_utils import (
    ROOT,
    MLB_BLUE,
    MLB_RED,
    _fetch_todays_schedule,
    _fetch_team_standings,
    _fetch_espn_odds,
    _load_precomputed,
    _load_model_results,
    _estimate_win_prob,
    _american_to_implied_prob,
    _prob_bar_html,
    init_session_state,
    add_betting_oracle_footer,
)

from src.models.today_features import build_todays_features, team_short_name
from src.models.underdog_model import predict_moneyline
from src.models.spread_model import predict_spread
from src.models.totals_model import predict_totals
from src.ingestion.mlb_stats import fetch_todays_probable_pitchers
from src.ingestion.weather import fetch_weather_for_games

logger = logging.getLogger(__name__)

MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="Betting Cleanup - MLB Predictions",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f9fafb; color: #111827; }
    section[data-testid="stSidebar"] { background-color: #001f4d; }
    section[data-testid="stSidebar"] * { color: #e5e7eb !important; }
    h1, h2, h3 { color: #002D72; }
    </style>
    """,
    unsafe_allow_html=True,
)

def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height

# ---------------------------------------------------------------------------
# Model predictions (Phase 3: real XGBoost models, replaces hand-coded
# heuristics). Built once per page load and looked up per game below.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _load_model_predictions() -> pd.DataFrame:
    """Build today's feature matrix and run all 3 trained models.

    Returns a DataFrame with one row per game: hometeam, visteam,
    pred_home_win_prob, pred_home_cover_prob, pred_over_prob, model_exp_total.
    Returns an empty DataFrame if the schedule/features/models are
    unavailable — callers must fall back to the heuristics in that case.
    """
    try:
        schedule = fetch_todays_probable_pitchers()
        if schedule.empty:
            return pd.DataFrame()

        # No live sportsbook odds are wired into the feature matrix here —
        # ESPN odds (used for edge/UI below) are fetched separately and
        # don't affect the models' probability outputs.
        odds = pd.DataFrame(columns=["away_team", "home_team"])
        weather = fetch_weather_for_games(schedule)

        features = build_todays_features(schedule, odds, weather)
        if features.empty:
            return pd.DataFrame()

        ml_preds = predict_moneyline(MODEL_DIR / "moneyline_xgb_v1.joblib", features)
        rl_preds = predict_spread(MODEL_DIR / "spread_xgb_v1.joblib", features)
        ou_preds = predict_totals(MODEL_DIR / "totals_xgb_v1.joblib", features)

        out = features[["hometeam", "visteam"]].copy()
        out["pred_home_win_prob"]   = ml_preds["pred_home_win_prob"].values
        out["pred_home_cover_prob"] = rl_preds["pred_cover_prob"].values
        out["pred_over_prob"]       = ou_preds["pred_prob_over"].values
        out["model_exp_total"]      = features["exp_total"].values
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model prediction pipeline failed, falling back to heuristics: %s", exc)
        return pd.DataFrame()


def _get_model_row(model_preds: pd.DataFrame, home_full: str, away_full: str) -> dict | None:
    """Look up a game's model predictions by MLB Stats API full team names."""
    if model_preds.empty:
        return None
    try:
        home_short = team_short_name(home_full)
        away_short = team_short_name(away_full)
    except KeyError:
        return None
    match = model_preds[
        (model_preds["hometeam"] == home_short) & (model_preds["visteam"] == away_short)
    ]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _short(full_name: str) -> str:
    """Last word of a team name, e.g. 'New York Yankees' -> 'Yankees'."""
    return full_name.split()[-1] if full_name else ""


def _get_rs_g(team_full: str, hist_stnd: pd.DataFrame) -> float:
    """Team RS/G from most recent Retrosheet season. Defaults to 4.5."""
    try:
        last = team_full.split()[-1]
        sub  = hist_stnd[hist_stnd["team"].str.contains(last, case=False, na=False)]
        if not sub.empty:
            return float(sub.sort_values("season").iloc[-1]["RS_per_G"])
    except Exception:
        pass
    return 4.5


def _build_game_recs(
    g: dict,
    espn_game: dict | None,
    standings: dict,
    hist_stnd: pd.DataFrame,
    model_row: dict | None = None,
) -> dict:
    """
    Build moneyline / run-line / over-under recommendations for one game.
    Returns dict with optional keys 'ml', 'rl', 'ou'.

    model_row, when present, carries pred_home_win_prob / pred_home_cover_prob /
    pred_over_prob from the trained XGBoost models (src/models/*_model.py).
    Falls back to the hand-coded heuristics (_estimate_win_prob, **1.4,
    RS/G vs posted total) whenever the model row is unavailable for this game.
    """
    home_full = g.get("home_name", "")
    away_full = g.get("away_name", "")
    if model_row is not None:
        home_prob = float(model_row["pred_home_win_prob"])
    else:
        home_prob = _estimate_win_prob(home_full, away_full, standings)
    away_prob = 1.0 - home_prob
    recs: dict = {}

    if not espn_game:
        return recs

    # -- Moneyline --
    ml_h_raw = espn_game.get("ml_home")
    ml_a_raw = espn_game.get("ml_away")
    if ml_h_raw and ml_a_raw:
        try:
            ml_h   = int(ml_h_raw)
            ml_a   = int(ml_a_raw)
            impl_h = _american_to_implied_prob(ml_h)
            impl_a = _american_to_implied_prob(ml_a)
            recs["ml"] = {
                "home": {
                    "team":     home_full,
                    "odds_str": f"+{ml_h}" if ml_h >= 0 else str(ml_h),
                    "impl":     impl_h,
                    "est_prob": home_prob,
                    "edge":     home_prob - impl_h,
                },
                "away": {
                    "team":     away_full,
                    "odds_str": f"+{ml_a}" if ml_a >= 0 else str(ml_a),
                    "impl":     impl_a,
                    "est_prob": away_prob,
                    "edge":     away_prob - impl_a,
                },
                "best": "home" if (home_prob - impl_h) >= (away_prob - impl_a) else "away",
            }
        except (TypeError, ValueError):
            pass

    # -- Run Line (+-1.5): favorite covers -1.5; underdog covers +1.5 --
    spread_h_raw = espn_game.get("spread_home")
    spread_a_raw = espn_game.get("spread_away", "—")

    def _parse_american(raw) -> int | None:
        try:
            return int(str(raw).replace("+", ""))
        except (ValueError, TypeError):
            return None

    spread_h_val = _parse_american(spread_h_raw)
    spread_a_val = _parse_american(spread_a_raw)

    # Use ESPN's actual run-line odds to determine who is the -1.5 favorite
    # (most reliable — this is the real market data for this specific market).
    # Fall back to moneyline heuristic only if ESPN spread odds are missing:
    # lower (more negative) ML odds = stronger favorite = they give -1.5.
    ml_h_val = _parse_american(espn_game.get("ml_home"))
    ml_a_val = _parse_american(espn_game.get("ml_away"))

    if spread_h_val is not None and spread_a_val is not None:
        # Positive spread odds → that team is the -1.5 side
        home_favorite = spread_h_val > 0 and spread_a_val <= 0
    elif ml_h_val is not None and ml_a_val is not None:
        home_favorite = ml_h_val < ml_a_val
    else:
        home_favorite = False

    # home_cover_prob = model's P(home wins by 2+ runs). This directly answers
    # "does home cover −1.5" but NOT "does away cover −1.5" (that would need
    # P(away wins by 2+), a different, unmodeled event — 1 - home_cover_prob
    # also includes 1-run home wins/losses, which don't cover either side).
    # So the model is only used in the home-favorite case; the heuristic
    # covers the away-favorite case until a symmetric model exists.
    home_cover_prob = float(model_row["pred_home_cover_prob"]) if model_row is not None else None

    if home_favorite:
        home_rl = home_cover_prob if home_cover_prob is not None else home_prob ** 1.4
        away_rl = 1.0 - home_rl
        home_pick = f"{_short(home_full)} −1.5"
        away_pick = f"{_short(away_full)} +1.5"
    else:
        away_rl = away_prob ** 1.4
        home_rl = 1.0 - away_rl
        home_pick = f"{_short(home_full)} +1.5"
        away_pick = f"{_short(away_full)} −1.5"

    if spread_h_raw and str(spread_h_raw) not in ("—", "", "None"):
        try:
            sho    = _parse_american(spread_h_raw)
            if sho is None:
                raise ValueError
            impl_h = _american_to_implied_prob(sho)
            if spread_a_raw and str(spread_a_raw) not in ("—", "", "None"):
                sao = _parse_american(spread_a_raw)
                if sao is None:
                    raise ValueError
                impl_a = _american_to_implied_prob(sao)
                away_odds_str = f"+{sao}" if sao >= 0 else str(sao)
            else:
                impl_a = 1.0 - impl_h
                away_odds_str = "—"

            recs["rl"] = {
                "home": {
                    "pick":     home_pick,
                    "odds_str": f"+{sho}" if sho >= 0 else str(sho),
                    "impl":     impl_h,
                    "est_prob": home_rl,
                    "edge":     home_rl - impl_h,
                },
                "away": {
                    "pick":     away_pick,
                    "odds_str": away_odds_str,
                    "impl":     impl_a,
                    "est_prob": away_rl,
                    "edge":     away_rl - impl_a,
                },
                "best": "home" if (home_rl - impl_h) >= (away_rl - impl_a) else "away",
            }
        except (TypeError, ValueError):
            pass

    # -- Over / Under --
    ou_raw   = espn_game.get("over_under")
    ov_raw   = espn_game.get("over_odds")
    un_raw   = espn_game.get("under_odds")
    if ou_raw and ov_raw and un_raw:
        try:
            posted = float(ou_raw)
            if model_row is not None:
                exp_total  = float(model_row["model_exp_total"])
                over_prob  = float(model_row["pred_over_prob"])
            else:
                exp_total  = _get_rs_g(home_full, hist_stnd) + _get_rs_g(away_full, hist_stnd)
                diff       = exp_total - posted
                over_prob  = max(0.20, min(0.80, 0.50 + diff * 0.06))
            under_prob = 1.0 - over_prob

            def _parse(raw) -> int | None:
                try:
                    return int(str(raw).replace("+", ""))
                except (ValueError, TypeError):
                    return None

            def _fmt(raw, i) -> str:
                if i is None:
                    return "—"
                return f"+{i}" if i >= 0 else str(i)

            ov_int  = _parse(ov_raw)
            un_int  = _parse(un_raw)
            impl_ov = _american_to_implied_prob(ov_int)  if ov_int  else 0.5
            impl_un = _american_to_implied_prob(un_int) if un_int else 0.5

            recs["ou"] = {
                "posted":    posted,
                "exp_total": exp_total,
                "over": {
                    "pick":     f"Over  {posted}",
                    "odds_str": _fmt(ov_raw, ov_int),
                    "impl":     impl_ov,
                    "est_prob": over_prob,
                    "edge":     over_prob - impl_ov,
                },
                "under": {
                    "pick":     f"Under {posted}",
                    "odds_str": _fmt(un_raw, un_int),
                    "impl":     impl_un,
                    "est_prob": under_prob,
                    "edge":     under_prob - impl_un,
                },
                "best": "over" if (over_prob - impl_ov) >= (under_prob - impl_un) else "under",
            }
        except (TypeError, ValueError):
            pass

    return recs


def _rec_card_html(label: str, side: dict, exp_info: str) -> str:
    """Render one market recommendation as an HTML block."""
    edge_pct = side["edge"] * 100
    if edge_pct > 3:
        color, badge = "#16a34a", "✅ BET"
    elif edge_pct > 0:
        color, badge = "#d97706", "➡ LEAN"
    else:
        color, badge = "#dc2626", "⛔ PASS"

    if side.get("team"):
        pick_text = _short(side["team"])   # e.g. "New York Yankees" → "Yankees"
    else:
        pick_text = side.get("pick", "—")  # e.g. "Nationals +1.5" or "Over 8.5" — keep as-is
    return (
        f'<div style="background:{color}18;border-left:4px solid {color};'
        f'padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:4px">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<b style="font-size:0.88rem">{pick_text}</b>'
        f'<span style="background:{color};color:white;border-radius:6px;padding:1px 8px;'
        f'font-size:0.7rem;font-weight:700">{badge}</span></div>'
        f'<div style="font-size:0.78rem;color:#555;margin-top:2px">'
        f'Odds: <b>{side["odds_str"]}</b>'
        f' &nbsp;|&nbsp; Edge: <b style="color:{color}">{edge_pct:+.1f}%</b></div>'
        f'<div style="font-size:0.73rem;color:#888">{exp_info}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

def home_page() -> None:
    """Landing page: hero metrics + per-game ML/Spread/O-U recommendations."""

    # Header
    hdr_left, hdr_right = st.columns([1, 5])
    with hdr_left:
        _logo = ROOT / "data_files" / "logo.png"
        if _logo.exists():
            st.image(str(_logo), width=110)
    with hdr_right:
        st.markdown(
            f"<h1 style='margin-bottom:0;color:#002D72'>⚾ Betting Cleanup</h1>"
            f"<p style='color:#6b7280;margin-top:2px'>MLB Predictions &nbsp;·&nbsp; "
            f"{datetime.date.today().strftime('%A, %B %d, %Y')}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Cached data
    games_today   = _fetch_todays_schedule()
    standings     = _fetch_team_standings()
    espn_odds     = _fetch_espn_odds()
    model_results = _load_model_results()
    model_preds   = _load_model_predictions()
    _pre          = _load_precomputed()
    hist_stnd     = _pre["standings"]
    init_session_state()

    # Hero metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    total_games  = len(games_today)
    games_w_odds = sum(
        1 for g in games_today
        if any(g.get("home_name", "").split()[-1].lower() in eo.get("home_team", "").lower()
               for eo in espn_odds)
    )
    accuracy = model_results["moneyline"]["metrics"].get("accuracy") if model_results else None
    roc_auc  = model_results["moneyline"]["metrics"].get("roc_auc")  if model_results else None
    m1.metric("Today's Games",   total_games)
    m2.metric("Games with Odds", games_w_odds)
    m3.metric("ML Model AUC",    f"{roc_auc:.4f}" if roc_auc  else "—",
              help="Moneyline XGBoost ROC-AUC on held-out test set.")
    m4.metric("Model Accuracy",  f"{accuracy:.1%}" if accuracy else "—")
    m5.metric("Odds Source",     "ESPN" if espn_odds else "Unavailable")

    st.markdown("---")

    if not games_today:
        st.info("No MLB games scheduled today, or the MLB Stats API is unreachable.")
    else:
        st.markdown(f"### 🎯 Today's Games & Betting Recommendations")
        _model_caption = (
            "trained XGBoost models" if not model_preds.empty
            else "current-season heuristics (model features unavailable today)"
        )
        st.caption(
            f"Win probability, run line &amp; O/U: {_model_caption}. "
            "✅ BET = edge > 3% &nbsp;·&nbsp; ➡ LEAN = 0–3% &nbsp;·&nbsp; ⛔ PASS = negative edge."
        )

        _status_labels = {
            "Final": "🏁 Final", "Game Over": "🏁 Final",
            "In Progress": "🔴 LIVE", "Scheduled": "🕐 Scheduled",
            "Pre-Game": "⏳ Pre-Game", "Warmup": "⏳ Warmup",
            "Delayed": "⚠️ Delayed", "Postponed": "🚫 Postponed",
        }

        for idx, g in enumerate(games_today):
            away_full = g.get("away_name", "Away")
            home_full = g.get("home_name", "Home")
            away_sp   = g.get("away_probable_pitcher", "TBD") or "TBD"
            home_sp   = g.get("home_probable_pitcher", "TBD") or "TBD"
            status    = g.get("status", "Scheduled")
            venue     = g.get("venue_name", "—")

            gtime_raw = g.get("game_datetime", "")
            if gtime_raw:
                try:
                    dt_utc    = datetime.datetime.fromisoformat(gtime_raw.replace("Z", "+00:00"))
                    gtime_str = (dt_utc - datetime.timedelta(hours=4)).strftime("%I:%M %p ET")
                except Exception:
                    gtime_str = "TBD"
            else:
                gtime_str = "TBD"

            score_str = ""
            if str(status).lower() in ("final", "game over", "in progress", "live"):
                if g.get("away_score") is not None and g.get("home_score") is not None:
                    score_str = f" &nbsp;·&nbsp; **{g['away_score']}–{g['home_score']}**"

            hk = home_full.split()[-1].lower()
            espn_game = next((eo for eo in espn_odds if hk in eo.get("home_team", "").lower()), None)
            model_row = _get_model_row(model_preds, home_full, away_full)
            recs      = _build_game_recs(g, espn_game, standings, hist_stnd, model_row)
            home_prob = (
                float(model_row["pred_home_win_prob"]) if model_row is not None
                else _estimate_win_prob(home_full, away_full, standings)
            )

            with st.container(border=True):
                # ── Game header ──
                hdr_c1, hdr_c2 = st.columns([3, 2])
                with hdr_c1:
                    st.markdown(
                        f"#### {away_full} @ {home_full}"
                        + (score_str if score_str else ""),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<small>🏟️ {venue} &nbsp;·&nbsp; "
                        f"{_status_labels.get(status, status)} &nbsp;·&nbsp; "
                        f"🕐 {gtime_str}</small><br>"
                        f"<small>SP: <b>{away_sp}</b> (away) &nbsp;/&nbsp; <b>{home_sp}</b> (home)</small>",
                        unsafe_allow_html=True,
                    )
                with hdr_c2:
                    st.markdown(_prob_bar_html(home_prob, home_full, away_full), unsafe_allow_html=True)

                if not recs:
                    st.caption("⏳ Odds not yet available for this game.")
                    continue

                st.divider()

                # ── Three bet markets ──
                col_ml, col_rl, col_ou = st.columns(3)

                with col_ml:
                    st.markdown("##### 💵 Moneyline")
                    if "ml" in recs:
                        ml   = recs["ml"]
                        best = ml["best"]
                        side = ml[best]
                        other = ml["away" if best == "home" else "home"]
                        exp   = f"Est: {side['est_prob']:.0%} · Impl: {side['impl']:.0%}"
                        st.markdown(_rec_card_html("ML", side, exp), unsafe_allow_html=True)
                        st.caption(
                            f"Other side: {_short(other['team'])} {other['odds_str']} "
                            f"(edge {other['edge']*100:+.1f}%)"
                        )
                    else:
                        st.caption("— odds unavailable —")

                with col_rl:
                    st.markdown("##### 📏 Run Line (±1.5)")
                    if "rl" in recs:
                        rl   = recs["rl"]
                        best = rl["best"]
                        side = rl[best]
                        other = rl["away" if best == "home" else "home"]
                        exp   = f"Est cover: {side['est_prob']:.0%} · Impl: {side['impl']:.0%}"
                        st.markdown(_rec_card_html("RL", side, exp), unsafe_allow_html=True)
                        st.caption(
                            f"Other side: {other['pick']} "
                            f"(edge {other['edge']*100:+.1f}%)"
                        )
                    else:
                        st.caption("— odds unavailable —")

                with col_ou:
                    st.markdown("##### 📊 Over/Under")
                    if "ou" in recs:
                        ou   = recs["ou"]
                        best = ou["best"]
                        side = ou[best]
                        other = ou["under" if best == "over" else "over"]
                        exp   = (
                            f"Exp total: {ou['exp_total']:.1f} · "
                            f"Posted: {ou['posted']} · "
                            f"Impl: {side['impl']:.0%}"
                        )
                        st.markdown(_rec_card_html("OU", side, exp), unsafe_allow_html=True)
                        st.caption(
                            f"Other side: {other['pick'].strip()} {other['odds_str']} "
                            f"(edge {other['edge']*100:+.1f}%)"
                        )
                    else:
                        st.caption("— odds unavailable —")

                # ── Deep-dive link ──
                st.markdown("")
                if st.button(
                    "🔍 View Full Game Details →",
                    key=f"home_detail_{idx}",
                    width='stretch',
                ):
                    st.session_state["schedule_selected_game"] = g
                    st.switch_page("pages/1_Today.py")

    st.markdown("---")

    # Navigation tiles
    st.markdown("### Explore")
    tc = st.columns(3)
    tiles = [
        ("📅", "Today",            "Full schedule with detailed game drill-down", "pages/1_Today.py"),
        ("📊", "Stats",            "Standings · Batting · Pitching · Leaders",   "pages/2_Stats.py"),
        ("🆚", "Matchup Analysis", "H2H history · Rolling win-rate charts",       "pages/3_Matchup_Analysis.py"),
    ]
    for col, (icon, title, desc, path) in zip(tc, tiles):
        with col:
            with st.container(border=True):
                st.markdown(f'<div style="text-align:center;font-size:1.8rem;padding-top:4px">{icon}</div>',
                            unsafe_allow_html=True)
                st.page_link(path, label=f"**{title}**")
                st.caption(desc)

    tc2 = st.columns(3)
    tiles2 = [
        ("🤖", "Models",      "XGBoost features · Evaluation · Savant research",  "pages/4_Models.py"),
        ("📈", "Performance", "Pick history · Model P&L · Kelly bankroll",         "pages/5_Performance.py"),
        ("ℹ️", "About",       "Methodology, data sources & tech stack",            "pages/7_Info.py"),
    ]
    for col, (icon, title, desc, path) in zip(tc2, tiles2):
        with col:
            with st.container(border=True):
                st.markdown(f'<div style="text-align:center;font-size:1.8rem;padding-top:4px">{icon}</div>',
                            unsafe_allow_html=True)
                st.page_link(path, label=f"**{title}**")
                st.caption(desc)

    add_betting_oracle_footer()


# ---------------------------------------------------------------------------
# Navigation (8 pages: Home + 7)
# ---------------------------------------------------------------------------
pg = st.navigation(
    {
        "": [
            st.Page(home_page, title="Home", icon="🏠", default=True),
            st.Page("pages/1_Today.py",            title="Today",            icon="📅"),
            st.Page("pages/2_Stats.py",            title="Stats",            icon="📊"),
            st.Page("pages/3_Matchup_Analysis.py", title="Matchup Analysis", icon="🆚"),
            st.Page("pages/4_Models.py",           title="Models",           icon="🤖"),
            st.Page("pages/5_Performance.py",      title="Performance",      icon="📈"),
            st.Page("pages/6_Pick_6.py",           title="Pick 6",           icon="🎯"),
            st.Page("pages/7_Info.py",             title="About",            icon="ℹ️"),
        ],
    }
)
pg.run()
