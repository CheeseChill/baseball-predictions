# scripts/run_distribution_backtest.py
"""Backtest walk-forward cho run_distribution_model (Hướng C) — model đang
chạy thật trên Today page, khác với scripts/run_evaluation.py (chạy 3-model
XGBClassifier cũ).

Usage:
    python scripts/run_distribution_backtest.py

Xem docstring src/evaluation/distribution_backtest.py để biết vì sao
moneyline không có ROI (thiếu dữ liệu giá lịch sử), còn runline/totals có
ROI xấp xỉ ở vig chuẩn -110 (không phải giá thị trường thật).
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from src.models.features import build_model_features
from src.evaluation.distribution_backtest import (
    walk_forward_backtest_distribution,
    moneyline_report,
    priced_market_report,
)


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    _section("BUILDING FEATURE MATRIX")
    features_df = build_model_features(2020, 2025)
    print(f"Feature matrix: {features_df.shape[0]:,} games x {features_df.shape[1]} columns")

    _section("WALK-FORWARD BACKTEST — run_distribution_model (Hướng C)")
    print("train_window=1200 games, test_window=200 games, step=100 games")
    print("(mỗi cửa sổ train lại model CHỈ trên dữ liệu quá khứ — không leak tương lai)")
    bt = walk_forward_backtest_distribution(features_df)
    print(f"Period: {bt.period}")

    _section("MONEYLINE — không có ROI thật (không có giá lịch sử), chỉ calibration + hit-rate")
    ml_rep = moneyline_report(bt)
    if ml_rep:
        print(f"  N bets:      {ml_rep['n_bets']:,}")
        print(f"  AUC:         {ml_rep['auc']}")
        print(f"  Brier score: {ml_rep['brier_score']}  (càng thấp càng tốt, 0.25 = đoán random)")
        print(f"  Log loss:    {ml_rep['log_loss']}")
        print("\n  Hit-rate theo khoảng xác suất model dự đoán + giá breakeven cần có:")
        print(ml_rep["by_probability_bucket"].to_string(index=False))
        print(
            "\n  Đọc bảng: nếu bucket 60-65% có win_rate thực tế ~62% và "
            "breakeven_odds_needed = -163, nghĩa là bạn cần bắt được giá "
            "TỐT HƠN -163 (vd -150, -140...) trên thị trường thật thì cửa "
            "này mới thật sự có lời — không phải cứ đúng hướng là lời."
        )
    else:
        print("  Không có dữ liệu.")

    for market, label in [("runline", "RUN LINE"), ("totals", "TOTALS (OVER/UNDER)")]:
        _section(f"{label} — ROI xấp xỉ ở vig chuẩn -110 (KHÔNG phải giá thật)")
        rep = priced_market_report(bt, market)
        if rep:
            print(f"  N bets (đã lọc edge >= 2% so với vig -110): {rep['n_bets']:,}")
            print(f"  Win rate:  {rep['win_rate']:.4f}")
            print(f"  ROI xấp xỉ: {rep['roi_approx_at_vig_110']:.4f}")
            print(f"  Total units: {rep['total_units']}")
            print(f"  ⚠️  {rep['note']}")
        else:
            print("  Không có dữ liệu.")

    _section("LƯU KẾT QUẢ")
    ml_df = bt.market_df("moneyline")
    rl_df = bt.market_df("runline")
    tot_df = bt.market_df("totals")
    out_dir = root / "data_files" / "processed"

    ml_df.to_parquet(out_dir / "dist_backtest_moneyline_bets.parquet", index=False)
    rl_df.to_parquet(out_dir / "dist_backtest_runline_bets.parquet", index=False)
    tot_df.to_parquet(out_dir / "dist_backtest_totals_bets.parquet", index=False)

    summary_rows = []
    if ml_rep:
        summary_rows.append({
            "market": "moneyline", "n_bets": ml_rep["n_bets"], "auc": ml_rep["auc"],
            "brier_score": ml_rep["brier_score"], "roi_approx": None,
            "note": "Không có ROI — thiếu dữ liệu giá lịch sử",
        })
    for market in ("runline", "totals"):
        rep = priced_market_report(bt, market)
        if rep:
            summary_rows.append({
                "market": market, "n_bets": rep["n_bets"], "auc": None,
                "brier_score": None, "roi_approx": rep["roi_approx_at_vig_110"],
                "note": rep["note"],
            })
    pd.DataFrame(summary_rows).to_parquet(out_dir / "dist_backtest_summary.parquet", index=False)
    print(f"  Đã lưu vào {out_dir}/dist_backtest_*.parquet")
    print("\nDone.")


if __name__ == "__main__":
    main()
