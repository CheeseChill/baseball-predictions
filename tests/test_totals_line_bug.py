import sys
import types

import numpy as np
import pandas as pd

sys.modules.setdefault("statsapi", types.ModuleType("statsapi"))

from src.models.run_distribution_model import (
    RUN_DIST_FEATURES,
    predict_game,
    over_prob_at_line,
)


def _fake_game_features(n=3) -> pd.DataFrame:
    """Feature matrix tối thiểu để chạy predict_game() với model đã train
    sẵn trong models/ — giá trị feature không quan trọng cho test này, chỉ
    cần khác 0 cho vài cột chính để mu_home/mu_away không suy biến.
    """
    rng = np.random.default_rng(42)
    data = {c: rng.normal(0, 1, n) for c in RUN_DIST_FEATURES}
    df = pd.DataFrame(data)
    df["date"] = pd.Timestamp("2026-08-08")
    df["hometeam"] = [f"H{i}" for i in range(n)]
    df["visteam"] = [f"V{i}" for i in range(n)]
    return df


def test_over_prob_at_line_matches_poisson_survival():
    # Sanity check thuần toán học, không cần model: P(over) tại 1 mốc thấp
    # hơn hẳn mu_total phải cao hơn P(over) tại mốc cao hơn hẳn mu_total.
    mu_home, mu_away = np.array([4.5]), np.array([4.0])  # mu_total = 8.5
    p_low_line = over_prob_at_line(mu_home, mu_away, 6.5)
    p_high_line = over_prob_at_line(mu_home, mu_away, 10.5)
    assert p_low_line[0] > 0.5
    assert p_high_line[0] < 0.5
    assert p_low_line[0] > p_high_line[0]


def test_predict_game_over_prob_differs_by_total_line_col():
    # Regression test: pred_over_prob PHẢI đổi theo total_line_col được
    # truyền vào — nếu ai lỡ hard-code lại "exp_total" ở nơi gọi predict_game()
    # (bug đã fix trong daily_pipeline.py/afternoon_refresh.py), test này sẽ
    # phát hiện ngay vì việc dùng đúng cột kèo thật (posted_total) mà không
    # đổi ra p_over khác thì có gì đó sai.
    feats = _fake_game_features()
    feats["exp_total"] = 9.0
    feats["posted_total"] = 6.0  # cố tình lệch xa exp_total

    preds_exp = predict_game(feats, total_line_col="exp_total")
    preds_posted = predict_game(feats, total_line_col="posted_total")

    # mu_home/mu_away không đổi (cùng feature input) nhưng pred_over_prob
    # phải khác nhau rõ rệt vì mốc so sánh khác nhau.
    assert np.allclose(preds_exp["mu_home"], preds_posted["mu_home"])
    assert not np.allclose(
        preds_exp["pred_over_prob"].values, preds_posted["pred_over_prob"].values
    )
    # Mốc thấp hơn (posted_total=6.0 < exp_total=9.0) -> P(over) phải cao hơn.
    assert (preds_posted["pred_over_prob"].values > preds_exp["pred_over_prob"].values).all()


def test_predict_game_falls_back_to_exp_total_when_posted_missing_per_row():
    # posted_total NaN ở 1 vài trận (chưa mở kèo) không được làm p_over
    # của CHÍNH trận đó thành NaN -> phải fallback về exp_total cho riêng
    # dòng đó, các dòng có posted_total vẫn dùng giá trị thật.
    feats = _fake_game_features(n=3)
    feats["exp_total"] = [8.0, 8.5, 9.0]
    feats["posted_total"] = [7.0, np.nan, 9.5]

    preds = predict_game(feats, total_line_col="posted_total")
    assert preds["pred_over_prob"].notna().all()

    preds_exp_only = predict_game(feats, total_line_col="exp_total")
    # Dòng index 1 (posted_total NaN) phải fallback về đúng công thức dùng
    # exp_total[1] -> khớp với chạy hoàn toàn bằng exp_total.
    assert np.isclose(
        preds["pred_over_prob"].values[1], preds_exp_only["pred_over_prob"].values[1]
    )
