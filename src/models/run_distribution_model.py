"""Unified run-distribution model (Hướng C).

Thay vì train 3 XGBClassifier độc lập (home_win / home_cover / went_over)
trên cùng feature set nhưng target khác nhau — không có ràng buộc nào ép
chúng nhất quán với nhau — model này học TRỰC TIẾP kỳ vọng số run mỗi đội
ghi được (mu_home, mu_away) qua 2 XGBRegressor (objective="count:poisson"),
rồi suy toán cả 3 market từ đúng 1 nguồn:

    diff  = home_runs - away_runs        ~ Skellam(mu_home, mu_away)
    total = home_runs + away_runs        ~ Poisson(mu_home + mu_away)
            (tổng 2 Poisson độc lập vẫn là Poisson)

    P(home win)          = P(diff > 0)   = 1 - skellam.cdf(0,  mu_home, mu_away)
    P(home covers -1.5)  = P(diff >= 2)  = 1 - skellam.cdf(1,  mu_home, mu_away)
    P(over line)         = P(total > L)  = poisson.sf(floor(L), mu_home + mu_away)

Vì cả 3 con số cùng suy ra từ (mu_home, mu_away), chúng KHÔNG THỂ tự mâu
thuẫn nhau — P(cover) <= P(win) luôn đúng theo cấu trúc toán học, không cần
IsotonicRegression + runtime clamp như 3-model cũ nữa (calibration vẫn có
thể áp dụng cho mu_home/mu_away riêng nếu cần, nhưng không bắt buộc để đảm
bảo tính nhất quán — điều đó đã có sẵn).

Target thô: hruns, vruns (số run thực tế mỗi đội — không phải nhị phân).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import model_io
from .features import TOTALS_FEATURES, calculate_edge

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)

HOME_MODEL_PATH = MODEL_DIR / "run_dist_home_xgb_v1"
AWAY_MODEL_PATH = MODEL_DIR / "run_dist_away_xgb_v1"

# Cùng bộ feature với totals model cũ (superset), trừ exp_total — vốn chỉ là
# home_RS_G + away_RS_G đã có sẵn trong tập, đưa thêm vào không sai nhưng dư
# thừa và không cần thiết cho một model học trực tiếp mu_home/mu_away.
RUN_DIST_FEATURES: list[str] = [c for c in TOTALS_FEATURES if c != "exp_total"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_regressor() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(
            objective="count:poisson",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
        )),
    ])


def train_run_distribution_model(
    features_df: pd.DataFrame,
    feature_cols: list[str] = RUN_DIST_FEATURES,
    test_size: float = 0.2,
) -> dict:
    """Train mu_home / mu_away regressors và đánh giá cả 3 market suy ra từ đó.

    Dùng chronological split giống 3 model cũ để so sánh công bằng.

    Returns:
        dict: model_home, model_away, metrics (per-market, so sánh trực
        tiếp với AUC/Brier của 3 model cũ), consistency_violations,
        importances_home/away, test_df (mu_home, mu_away, 3 xác suất suy
        ra, và target thật của cả 3 market).
    """
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    required = feature_cols + ["hruns", "vruns", "home_win", "home_cover", "went_over", "exp_total"]
    df = features_df.sort_values("date").dropna(subset=required)

    X = df[feature_cols].values
    y_home = df["hruns"].astype(float).values
    y_away = df["vruns"].astype(float).values

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_home_train, y_home_test = y_home[:split_idx], y_home[split_idx:]
    y_away_train, y_away_test = y_away[:split_idx], y_away[split_idx:]

    model_home = _make_regressor()
    model_home.fit(X_train, y_home_train)
    model_away = _make_regressor()
    model_away.fit(X_train, y_away_train)

    mu_home = np.clip(model_home.predict(X_test), 1e-3, None)
    mu_away = np.clip(model_away.predict(X_test), 1e-3, None)

    reg_metrics = {
        "mae_home_runs": float(mean_absolute_error(y_home_test, mu_home)),
        "mae_away_runs": float(mean_absolute_error(y_away_test, mu_away)),
    }

    # --- Suy ra cả 3 market từ (mu_home, mu_away) ---------------------------
    p_home_win   = 1 - skellam.cdf(0, mu_home, mu_away)
    p_home_cover = 1 - skellam.cdf(1, mu_home, mu_away)  # diff >= 2

    exp_total_test = df["exp_total"].values[split_idx:]
    mu_total = mu_home + mu_away
    p_over = poisson.sf(np.floor(exp_total_test), mu_total)

    test_df = df.iloc[split_idx:][
        ["date", "hometeam", "visteam", "hruns", "vruns", "exp_total",
         "home_win", "home_cover", "went_over"]
    ].copy().reset_index(drop=True)
    test_df["mu_home"] = mu_home
    test_df["mu_away"] = mu_away
    test_df["pred_home_win_prob"] = p_home_win
    test_df["pred_home_cover_prob"] = p_home_cover
    test_df["pred_over_prob"] = p_over

    # --- Kiểm chứng tính nhất quán (phải luôn = 0 theo cấu trúc toán học) --
    consistency_violations = int((test_df["pred_home_cover_prob"] > test_df["pred_home_win_prob"] + 1e-9).sum())

    def _clf_metrics(y_true, y_prob, label):
        y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
        return {
            f"brier_score_{label}": float(brier_score_loss(y_true, y_prob)),
            f"log_loss_{label}":    float(log_loss(y_true, y_prob)),
            f"roc_auc_{label}":     float(roc_auc_score(y_true, y_prob)),
        }

    metrics = {**reg_metrics, "consistency_violations": consistency_violations}
    metrics.update(_clf_metrics(test_df["home_win"], p_home_win, "moneyline"))
    metrics.update(_clf_metrics(test_df["home_cover"], p_home_cover, "spread"))
    metrics.update(_clf_metrics(test_df["went_over"], p_over, "totals"))

    importances_home = pd.DataFrame({
        "feature": feature_cols,
        "importance": model_home.named_steps["xgb"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importances_away = pd.DataFrame({
        "feature": feature_cols,
        "importance": model_away.named_steps["xgb"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    model_io.save_regressor_pipeline(model_home, HOME_MODEL_PATH, feature_cols=feature_cols)
    model_io.save_regressor_pipeline(model_away, AWAY_MODEL_PATH, feature_cols=feature_cols)

    return {
        "model_home": model_home,
        "model_away": model_away,
        "metrics": metrics,
        "importances_home": importances_home,
        "importances_away": importances_away,
        "feature_cols": feature_cols,
        "test_df": test_df,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_game(
    game_features: pd.DataFrame,
    feature_cols: "list[str] | None" = None,
    total_line_col: str = "exp_total",
    spread_line: float = 1.5,
    home_ml_col: str | None = None,
    away_ml_col: str | None = None,
    home_spread_price_col: str | None = None,
    away_spread_price_col: str | None = None,
    over_price_col: str | None = None,
    under_price_col: str | None = None,
) -> pd.DataFrame:
    """Suy ra cả 3 market (moneyline/run line/total) từ 1 model phân phối duy nhất.

    Thay thế predict_moneyline + predict_spread + predict_totals cũ.
    Xử lý TỰ NHIÊN cả 2 chiều home/away favorite cho run line (không cần
    heuristic fallback riêng cho away-favorite như kiến trúc cũ) — vì
    mu_home/mu_away vốn đã bất đối xứng theo đúng thực lực từng đội.

    Args:
        game_features: DataFrame chứa feature_cols + (tuỳ chọn) cột line/odds.
        feature_cols: Mặc định dùng feature_cols đã lưu cùng model.
        total_line_col: Cột chứa mốc tổng điểm (mặc định exp_total).
        spread_line: Biên độ run line (mặc định 1.5, chuẩn MLB).
        home_ml_col, away_ml_col: Cột odds Mỹ (moneyline) — nếu có, tính
            edge_home/edge_away = P(model) - implied_probability(odds).
        home_spread_price_col, away_spread_price_col: Cột odds cho run line
            phía home/away (giá của bên -1.5/+1.5 tương ứng) — nếu có, tính
            edge_home_cover/edge_away_cover.
        over_price_col, under_price_col: Cột odds cho Over/Under — nếu có,
            tính edge_over/edge_under.

    Returns:
        DataFrame: hometeam, visteam, mu_home, mu_away,
        pred_home_win_prob, pred_away_win_prob,
        pred_home_cover_prob, pred_away_cover_prob,
        pred_over_prob, pred_under_prob, pick_moneyline, pick_runline, pick_total,
        [edge_home, edge_away, edge_home_cover, edge_away_cover, edge_over, edge_under]
        cho các market có odds tương ứng.
    """
    model_home = model_io.load_regressor_pipeline(HOME_MODEL_PATH)
    model_away = model_io.load_regressor_pipeline(AWAY_MODEL_PATH)
    trained_cols = model_io.load_feature_cols(HOME_MODEL_PATH)

    cols = feature_cols or trained_cols or RUN_DIST_FEATURES
    X = game_features[cols].fillna(0).values

    mu_home = np.clip(model_home.predict(X), 1e-3, None)
    mu_away = np.clip(model_away.predict(X), 1e-3, None)

    half_wins = int(np.floor(spread_line))  # 1.5 -> diff >= 2 covers
    p_home_win = 1 - skellam.cdf(0, mu_home, mu_away)
    p_home_cover = 1 - skellam.cdf(half_wins, mu_home, mu_away)
    p_away_cover = skellam.cdf(-half_wins - 1, mu_home, mu_away)

    mu_total = mu_home + mu_away
    if total_line_col in game_features.columns:
        line = game_features[total_line_col].values
    else:
        line = mu_total
    p_over = poisson.sf(np.floor(line), mu_total)

    id_cols = [c for c in ("date", "hometeam", "visteam") if c in game_features.columns]
    results = game_features[id_cols].copy().reset_index(drop=True)
    results["mu_home"] = mu_home.round(3)
    results["mu_away"] = mu_away.round(3)
    results["pred_home_win_prob"] = p_home_win.round(4)
    results["pred_away_win_prob"] = (1 - p_home_win).round(4)
    results["pred_home_cover_prob"] = p_home_cover.round(4)
    results["pred_away_cover_prob"] = p_away_cover.round(4)
    results["pred_over_prob"] = np.round(p_over, 4)
    results["pred_under_prob"] = np.round(1 - p_over, 4)
    results["pick_moneyline"] = np.where(p_home_win >= 0.5, "Home", "Away")
    results["pick_runline"] = np.where(
        p_home_cover >= p_away_cover, f"Home -{spread_line}", f"Away +{spread_line}"
    )
    results["pick_total"] = np.where(p_over >= 0.5, "Over", "Under")

    def _edge(prob, odds_col):
        if odds_col is None or odds_col not in game_features.columns:
            return None
        odds = game_features[odds_col].values
        return [
            calculate_edge(float(p), o) if pd.notna(o) else np.nan
            for p, o in zip(prob, odds)
        ]

    if home_ml_col or away_ml_col:
        e_home = _edge(p_home_win, home_ml_col)
        e_away = _edge(1 - p_home_win, away_ml_col)
        if e_home is not None:
            results["edge_home"] = e_home
        if e_away is not None:
            results["edge_away"] = e_away

    if home_spread_price_col or away_spread_price_col:
        e_hc = _edge(p_home_cover, home_spread_price_col)
        e_ac = _edge(p_away_cover, away_spread_price_col)
        if e_hc is not None:
            results["edge_home_cover"] = e_hc
        if e_ac is not None:
            results["edge_away_cover"] = e_ac

    if over_price_col or under_price_col:
        e_ov = _edge(p_over, over_price_col)
        e_un = _edge(1 - p_over, under_price_col)
        if e_ov is not None:
            results["edge_over"] = e_ov
        if e_un is not None:
            results["edge_under"] = e_un

    return results
