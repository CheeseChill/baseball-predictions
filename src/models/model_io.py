"""Portable save/load for sklearn Pipelines wrapping an XGBClassifier.

Why this exists
----------------
``joblib.dump()`` on a ``Pipeline`` that contains an ``XGBClassifier``
pickles XGBoost's internal C buffer. That buffer is NOT guaranteed to load
back correctly across different Python / numpy / OS environments, even when
the xgboost package version matches exactly on both sides. In practice this
showed up as:

    xgboost.core.XGBoostError: input stream corrupted

when a model trained under one Python version (e.g. 3.12) was loaded under
another (e.g. 3.14).

The fix
-------
Split the pipeline in two and save each half in a format that IS portable:

  - Every step *before* the classifier (e.g. StandardScaler) is a plain
    numpy/sklearn object -> pickled with joblib as usual.
  - The XGBoost step is saved with ``XGBClassifier.save_model()``, which
    writes XGBoost's own native JSON format. This is explicitly documented
    by XGBoost as the portable, cross-version/cross-platform way to persist
    a model (unlike pickling the estimator).

Given a stem like ``models/moneyline_xgb_v1`` (no extension), this writes:

  - ``models/moneyline_xgb_v1.scaler.joblib``  (pre-classifier steps)
  - ``models/moneyline_xgb_v1.xgb.json``       (the XGBoost model)
  - ``models/moneyline_xgb_v1.meta.json``      (small bit of bookkeeping)

and ``load_pipeline()`` reassembles a working ``Pipeline`` from those three
files.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def _scaler_path(stem: "str | Path") -> Path:
    return Path(str(stem) + ".scaler.joblib")


def _xgb_path(stem: "str | Path") -> Path:
    return Path(str(stem) + ".xgb.json")


def _meta_path(stem: "str | Path") -> Path:
    return Path(str(stem) + ".meta.json")


def is_portable_model(stem: "str | Path") -> bool:
    """True if a portable (model_io) model exists at this stem."""
    return _xgb_path(stem).exists()


def save_pipeline(
    pipeline: Pipeline,
    stem: "str | Path",
    feature_cols: "list[str] | None" = None,
) -> None:
    """Save a Pipeline whose LAST step is an XGBClassifier, portably.

    Args:
        pipeline: A fitted sklearn Pipeline, e.g.
            Pipeline([("scaler", StandardScaler()), ("xgb", XGBClassifier())]).
        stem: Path without extension, e.g. MODEL_DIR / "moneyline_xgb_v1".
            Three files are written: "<stem>.scaler.joblib",
            "<stem>.xgb.json", "<stem>.meta.json".
        feature_cols: The exact, ordered list of feature columns the
            pipeline was fit on. Training filters the "full" feature list
            down to whatever columns actually exist in that run's data
            (e.g. 99 out of 121 when some Savant columns are missing), so
            this is NOT always the same as a module-level FEATURES
            constant. Saving it here lets predict-time code rebuild the
            exact same column set instead of guessing — a mismatch here
            raises "X has N features, but StandardScaler is expecting M
            features as input".
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    *pre_steps, (clf_name, clf) = pipeline.steps

    if not isinstance(clf, XGBClassifier):
        raise TypeError(
            "save_pipeline() expects the last pipeline step to be an "
            f"XGBClassifier, got {type(clf).__name__}. For non-XGBoost "
            "classifiers, fall back to joblib.dump() directly."
        )

    pre_pipeline = Pipeline(pre_steps) if pre_steps else None
    joblib.dump(pre_pipeline, _scaler_path(stem))

    clf.save_model(str(_xgb_path(stem)))

    with open(_meta_path(stem), "w") as f:
        json.dump({"clf_step_name": clf_name, "feature_cols": feature_cols}, f)


def _calibrator_path(stem: "str | Path") -> Path:
    return Path(str(stem) + ".calibrator.joblib")


def save_calibrator(calibrator, stem: "str | Path") -> None:
    """Save a probability calibrator (e.g. sklearn IsotonicRegression) alongside a model.

    Unlike the XGBoost step, an IsotonicRegression is a plain sklearn/numpy
    object with no C-buffer portability issue, so a regular joblib.dump is
    fine here.
    """
    joblib.dump(calibrator, _calibrator_path(stem))


def load_calibrator(stem: "str | Path"):
    """Load a saved calibrator, or None if this model has no calibrator file.

    None is a valid, expected return for models trained before calibration
    was added — callers should treat it as "use the raw model probability".
    """
    path = _calibrator_path(stem)
    if not path.exists():
        return None
    return joblib.load(path)


def load_feature_cols(stem: "str | Path") -> "list[str] | None":
    """Return the feature_cols saved alongside this model, if any."""
    meta_path = _meta_path(stem)
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f).get("feature_cols")


def load_pipeline(stem: "str | Path") -> Pipeline:
    """Load a Pipeline saved with save_pipeline().

    Raises FileNotFoundError if no portable model exists at this stem
    (i.e. "<stem>.xgb.json" is missing).
    """
    stem = Path(stem)
    xgb_path = _xgb_path(stem)
    scaler_path = _scaler_path(stem)
    meta_path = _meta_path(stem)

    if not xgb_path.exists():
        raise FileNotFoundError(
            f"No portable model found at '{xgb_path}'. Expected files "
            "created by model_io.save_pipeline()."
        )

    clf_step_name = "xgb"
    if meta_path.exists():
        with open(meta_path) as f:
            clf_step_name = json.load(f).get("clf_step_name", "xgb")

    clf = XGBClassifier()
    clf.load_model(str(xgb_path))

    steps = []
    if scaler_path.exists():
        pre_pipeline = joblib.load(scaler_path)
        if pre_pipeline is not None:
            steps.extend(pre_pipeline.steps)
    steps.append((clf_step_name, clf))

    return Pipeline(steps)
