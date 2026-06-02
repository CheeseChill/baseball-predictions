# Baseball Predictions — Model Suggested Enhancements

## Priority 1: Calibration & Accuracy

### Calibration Curve Analysis
- Add `sklearn.calibration.calibration_curve` to the Model Performance page to visualise how well predicted probabilities match actual outcomes (e.g., games predicted at 70% should win ~70% of the time).
- Apply **Platt scaling** or **isotonic regression** as a post-processing step to each model output.

### Ensemble Stacking
- Currently three independent models. Add a lightweight **stacking meta-learner** (logistic regression) trained on hold-out fold predictions from the three base models.
- Expected calibration error improvement: ~2–4%.

### Run Line Model Improvements
- Add **pitcher matchup differential** (starter ERA gap) as an interaction feature.
- Add **bullpen strength** (ERA last 7 days for relievers) — current model lacks pen depth.
- Add **park-adjusted OPS** differences to reduce variance from park effects.

## Priority 2: Feature Engineering

### Pitcher Fatigue
- Add `days_rest` for starting pitcher (0, 1, 2, 3, 4+).
- Add rolling `pitcher_ip_last_7` to detect overused starters.

### Weather Context
- Wind direction and speed (balls-in-play heavy parks like Wrigley/Coors amplified by wind).
- Temperature below 45°F historically suppresses run scoring; encode as binary flag.

### Umpire Strike Zone
- ESPN / Baseball Savant track umpire zone tendencies. Add `ump_k_rate_diff` (above/below league mean).

## Priority 3: Model Infrastructure

### Automated Retraining Trigger
- Current retraining is schedule-based. Add a **performance drift check** — retrain automatically if rolling 30-game accuracy drops more than 3% from baseline.

### Prediction Confidence Intervals
- Bootstrap prediction intervals (1000 resamples) on moneyline edge to show uncertainty bands in the dashboard.

### Feature Importance Drift
- Log top-10 feature importances per model weekly. Alert if the #1 feature changes (indicates distribution shift).

## Priority 4: Advanced Metrics

### SHAP Values
- Add SHAP game-level explanations to the Today page: "Model favours NYY because SP ERA gap (+2.1), bullpen advantage (+1.4)."

### Closing Line Value (CLV)
- Track CLV weekly: compare prediction odds at open vs. closing line. Positive CLV confirms model has genuine edge vs. the market.

### Kelly Fraction Optimisation
- Current half-Kelly. Backtest fractional Kelly values (0.25x, 0.33x, 0.5x) over 3 seasons and surface optimal fraction per bet type.
