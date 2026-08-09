# src/evaluation/distribution_backtest.py
"""Walk-forward backtest cho run_distribution_model (Hướng C).

Khác với src/evaluation/backtester.py (chạy trên 3 XGBClassifier cũ), module
này backtest đúng model đang chạy thật trên Today page: 2 XGBRegressor
Poisson học (mu_home, mu_away), suy ra cả 3 market (moneyline / run line /
totals) từ CÙNG một cặp mu qua Skellam/Poisson — giống hệt predict_game().

QUAN TRỌNG — về ROI:
Repo hiện KHÔNG có dữ liệu giá cược lịch sử (data_files/raw/odds/ rỗng,
model_features.parquet không có cột odds/ml/line nào). Vì vậy:

  - Moneyline: KHÔNG bịa ra ROI/edge, vì không có giá thị trường lịch sử
    nào để so sánh. Thay vào đó báo cáo calibration + hit-rate theo từng
    khoảng xác suất, và một bảng "breakeven odds" — để khi có giá thật
    (vd từ TheRundown lưu lại theo thời gian) thì so sánh được ngay.
  - Run line / Totals: ROI được tính ở vig CHUẨN NGÀNH giả định -110 hai
    bên — đây là XẤP XỈ, không phải giá thị trường thật (giá run line
    thực tế lệch xa -110 tùy favorite/underdog, xem bug run-line vừa
    fix). Con số ROI ở đây chỉ mang tính tham khảo tương đối, không phải
    kết luận cuối về lời/lỗ thật.

Walk-forward: train trên cửa sổ [start:train_end], test trên
[train_end:train_end+test_window], trượt tới, lặp lại — không leak dữ liệu
tương lai vào lúc train (đúng tinh thần backtester.py cũ).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.models.run_distribution_model import RUN_DIST_FEATURES, _make_regressor

STANDARD_VIG_ODDS = -110  # giả định chuẩn ngành cho spread/totals; KHÔNG phải giá thật


# ---------------------------------------------------------------------------
# Kết quả từng cược (để tính ROI xấp xỉ ở vig chuẩn, hoặc None nếu không có
# cơ sở giá — trường hợp moneyline)
# ---------------------------------------------------------------------------

@dataclass
class DistBetResult:
    game_id: object
    date: object
    market: str          # 'moneyline', 'runline', 'totals'
    pick: str             # 'home'/'away' hoặc 'over'/'under'
    predicted_prob: float       # xác suất của PHÍA ĐÃ CHỌN (luôn >= 0.5, dùng cho bảng hit-rate theo bucket)
    raw_prob_home: float        # P(home win) GỐC, chưa argmax — dùng để tính AUC/Brier so sánh công bằng với train-time metrics
    actual_home: int            # outcome thực tế phía home (1/0) — home_win / home_cover / went_over tùy market
    edge_vs_vig: float | None   # None nếu không có cơ sở giá (moneyline)
    result: str            # 'win' / 'loss' / 'push'
    profit_units: float | None  # None nếu không có cơ sở giá (moneyline)


@dataclass
class DistBacktestResult:
    period: str
    bets: list[DistBetResult] = field(default_factory=list)

    def market_df(self, market: str) -> pd.DataFrame:
        rows = [b for b in self.bets if b.market == market]
        return pd.DataFrame([b.__dict__ for b in rows])


def _profit_at_odds(odds: int, result: str) -> float:
    if result == "push":
        return 0.0
    if result != "win":
        return -1.0
    return odds / 100 if odds > 0 else 100 / abs(odds)


def _implied_prob(odds: int) -> float:
    return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)


# ---------------------------------------------------------------------------
# Walk-forward backtest chính
# ---------------------------------------------------------------------------

def walk_forward_backtest_distribution(
    features_df: pd.DataFrame,
    feature_cols: list[str] = RUN_DIST_FEATURES,
    spread_line: float = 1.5,
    train_window_games: int = 1200,
    test_window_games: int = 200,
    step_size: int = 100,
    min_edge_for_bet: float = 0.02,  # chỉ áp dụng cho runline/totals (có vig chuẩn)
) -> DistBacktestResult:
    """Backtest walk-forward run_distribution_model trên cả 3 market cùng lúc.

    Mỗi cửa sổ: train lại 2 regressor (mu_home, mu_away) CHỈ trên dữ liệu quá
    khứ, dự đoán trên cửa sổ test kế tiếp — không có leakage tương lai.
    """
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    required = feature_cols + ["hruns", "vruns", "home_win", "home_cover", "went_over", "exp_total"]
    df = features_df.sort_values("date").dropna(subset=required).reset_index(drop=True)

    half_wins = int(np.floor(spread_line))
    vig_implied = _implied_prob(STANDARD_VIG_ODDS)

    all_bets: list[DistBetResult] = []

    start = 0
    while start + train_window_games + test_window_games <= len(df):
        train_end = start + train_window_games
        test_end = train_end + test_window_games

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train = train_df[feature_cols].values
        y_home_train = train_df["hruns"].astype(float).values
        y_away_train = train_df["vruns"].astype(float).values

        model_home = _make_regressor()
        model_home.fit(X_train, y_home_train)
        model_away = _make_regressor()
        model_away.fit(X_train, y_away_train)

        X_test = test_df[feature_cols].values
        mu_home = np.clip(model_home.predict(X_test), 1e-3, None)
        mu_away = np.clip(model_away.predict(X_test), 1e-3, None)

        p_home_win = 1 - skellam.cdf(0, mu_home, mu_away)
        p_home_cover = 1 - skellam.cdf(half_wins, mu_home, mu_away)
        p_away_cover = skellam.cdf(-half_wins - 1, mu_home, mu_away)

        mu_total = mu_home + mu_away
        line = test_df["exp_total"].values
        p_over = poisson.sf(np.floor(line), mu_total)

        for i, (_, row) in enumerate(test_df.iterrows()):
            gid = row.get("gid", row.name)
            gdate = row.get("date")

            # -- Moneyline: không có giá thị trường lịch sử -> không tính ROI --
            pw = float(p_home_win[i])
            pick_ml = "home" if pw >= 0.5 else "away"
            actual_home_win = int(row["home_win"])
            correct_ml = (pick_ml == "home" and actual_home_win == 1) or (
                pick_ml == "away" and actual_home_win == 0
            )
            all_bets.append(DistBetResult(
                game_id=gid, date=gdate, market="moneyline", pick=pick_ml,
                predicted_prob=pw if pick_ml == "home" else 1 - pw,
                raw_prob_home=pw,
                actual_home=actual_home_win,
                edge_vs_vig=None,
                result="win" if correct_ml else "loss",
                profit_units=None,
            ))

            # -- Run line: có vig chuẩn -110 giả định, lọc theo min_edge --
            phc, pac = float(p_home_cover[i]), float(p_away_cover[i])
            pick_rl, p_rl = ("home", phc) if phc >= pac else ("away", pac)
            edge_rl = p_rl - vig_implied
            if edge_rl >= min_edge_for_bet:
                actual_cover = int(row["home_cover"])
                correct_rl = (pick_rl == "home" and actual_cover == 1) or (
                    pick_rl == "away" and actual_cover == 0
                )
                all_bets.append(DistBetResult(
                    game_id=gid, date=gdate, market="runline", pick=pick_rl,
                    predicted_prob=p_rl, raw_prob_home=phc, actual_home=actual_cover,
                    edge_vs_vig=edge_rl,
                    result="win" if correct_rl else "loss",
                    profit_units=_profit_at_odds(
                        STANDARD_VIG_ODDS, "win" if correct_rl else "loss"
                    ),
                ))

            # -- Totals: có vig chuẩn -110 giả định, lọc theo min_edge --
            po = float(p_over[i])
            pick_t, p_t = ("over", po) if po >= 0.5 else ("under", 1 - po)
            edge_t = p_t - vig_implied
            if edge_t >= min_edge_for_bet:
                actual_over = int(row["went_over"])
                correct_t = (pick_t == "over" and actual_over == 1) or (
                    pick_t == "under" and actual_over == 0
                )
                all_bets.append(DistBetResult(
                    game_id=gid, date=gdate, market="totals", pick=pick_t,
                    predicted_prob=p_t, raw_prob_home=po, actual_home=actual_over,
                    edge_vs_vig=edge_t,
                    result="win" if correct_t else "loss",
                    profit_units=_profit_at_odds(
                        STANDARD_VIG_ODDS, "win" if correct_t else "loss"
                    ),
                ))

        start += step_size

    period = f"{df['date'].min()} to {df['date'].max()}"
    return DistBacktestResult(period=period, bets=all_bets)


# ---------------------------------------------------------------------------
# Báo cáo
# ---------------------------------------------------------------------------

def moneyline_report(bt: DistBacktestResult) -> dict:
    """Calibration + hit-rate theo bucket xác suất + bảng breakeven odds.

    Không có ROI vì không có giá thị trường lịch sử — xem docstring module.
    """
    ml = bt.market_df("moneyline")
    if ml.empty:
        return {}

    y_true = ml["actual_home"].astype(int).values
    y_prob = ml["raw_prob_home"].values

    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)))
    auc = float(roc_auc_score(y_true, y_prob))

    bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-80%", "80%+"]
    ml = ml.copy()
    ml["bucket"] = pd.cut(ml["predicted_prob"].values, bins=bins, labels=labels, right=False)

    bucket_rows = []
    for lbl in labels:
        sub = ml[ml["bucket"] == lbl]
        if sub.empty:
            continue
        n = len(sub)
        wins = (sub["result"] == "win").sum()
        win_rate = wins / n
        # Breakeven American odds cần có để win_rate này hòa vốn.
        # win_rate > 0.5 -> cần lấy được giá dương (đội dưới cửa) hoặc giá
        # âm không quá sâu; công thức: breakeven_decimal = 1 / win_rate
        breakeven_decimal = 1 / win_rate if win_rate > 0 else float("inf")
        if breakeven_decimal >= 2:
            breakeven_american = (breakeven_decimal - 1) * 100
        else:
            breakeven_american = -100 / (breakeven_decimal - 1)
        bucket_rows.append({
            "bucket": lbl, "n_bets": n, "win_rate": round(win_rate, 4),
            "breakeven_odds_needed": round(breakeven_american),
        })

    return {
        "n_bets": len(ml),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "auc": round(auc, 4),
        "by_probability_bucket": pd.DataFrame(bucket_rows),
    }


def priced_market_report(bt: DistBacktestResult, market: str) -> dict:
    """ROI xấp xỉ (vig -110 giả định) cho runline hoặc totals."""
    sub = bt.market_df(market)
    if sub.empty:
        return {}
    n = len(sub)
    wins = (sub["result"] == "win").sum()
    losses = (sub["result"] == "loss").sum()
    total_units = sub["profit_units"].sum()
    roi = total_units / n if n else 0.0
    return {
        "market": market,
        "n_bets": n,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(wins / n, 4) if n else 0.0,
        "total_units": round(float(total_units), 2),
        "roi_approx_at_vig_110": round(float(roi), 4),
        "note": "ROI xấp xỉ ở vig chuẩn -110 hai bên, KHÔNG phải giá thị trường thật",
    }
