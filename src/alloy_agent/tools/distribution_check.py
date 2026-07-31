"""Check whether a prediction falls inside the training data distribution.

Every XGBoost prediction is technically valid, but predictions for inputs that
lie outside the training distribution are extrapolations — the model's behaviour
there is undefined. This tool flags such cases so the agent's report can warn
the user.

Heuristics:
  - "in":          value lies strictly between training [min, max]
  - "near_*":      value is within 5% of the training bound (likely fine, worth flagging)
  - "out_*":       value is outside the training bound (extrapolation, definitely flag)

Output schema (dict):
  {
    "target": str,
    "ood_features":     [(name, value, bound, ratio), ...],
    "near_bound_features": [(name, value, bound, ratio), ...],
    "in_range_features": [(name, value), ...],
    "nearest_neighbors":   [row, ...],   # training rows closest to this alloy (K rows)
    "warning": str,  # human-readable summary
  }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from alloy_agent.schemas import AlloyInput
from alloy_agent.tools._model_utils import FIXTURE_TO_MODEL_KEY


_NEARNESS_FRACTION = 0.05  # within 5% of bound → "near"
_NUMERIC_KINDS = (int, float, np.integer, np.floating)


# Path to training CSVs — relative to the project root.
def _csv_path(target: str) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    if target == "yield_strength":
        return project_root / "Agent-acta" / "YSA-All-元素-微观-增强 -特征筛选-12 - 副本.csv"
    if target == "oxidation_mass_gain":
        return project_root / "Agent-acta" / "OXIDATION.csv"
    raise ValueError(f"unknown target: {target!r}")


def _csv_features(target: str) -> list[str]:
    """Column order in the training CSV (excluding the target column)."""
    df = pd.read_csv(_csv_path(target), nrows=0)
    target_col = "YS" if target == "yield_strength" else "Mass gain"
    return [c for c in df.columns if c != target_col]


def _numeric(value: object) -> float:
    if isinstance(value, _NUMERIC_KINDS):
        return float(value)
    return 0.0


def _classify(value: float, lo: float, hi: float) -> str:
    """Where does `value` sit relative to [lo, hi]?"""
    span = max(hi - lo, 1e-9)
    margin = span * _NEARNESS_FRACTION
    if value < lo - margin:
        return "out_lo"
    if value > hi + margin:
        return "out_hi"
    if value < lo + margin:
        return "near_lo"
    if value > hi - margin:
        return "near_hi"
    return "in"


def _bound_for_classification(
    value: float,
    lo: float,
    hi: float,
    classification: str,
) -> tuple[float, float]:
    """Return the actual bound involved in a near/out classification."""
    bound = lo if classification.endswith("_lo") else hi
    ratio = value / bound if bound != 0 else 0.0
    return bound, ratio


def _alloy_to_feature_row(alloy: AlloyInput, features: list[str]) -> dict[str, float]:
    """Map an AlloyInput into a single training-csv-style row.

    Uses `FIXTURE_TO_MODEL_KEY` to translate human-readable keys
    (`oxidation_temperature`, `strength_test_temperature`, ...) into the
    abbreviated names the model was trained on (`Toxidation`, `Ttest`, ...).
    """
    row: dict[str, float] = {f: 0.0 for f in features}
    for k, v in alloy.composition.items():
        model_key = FIXTURE_TO_MODEL_KEY.get(k, k)
        if model_key in row:
            row[model_key] = _numeric(v)
    for k, v in alloy.processing.items():
        model_key = FIXTURE_TO_MODEL_KEY.get(k, k)
        if model_key in row:
            row[model_key] = _numeric(v)
    for k, v in alloy.test_conditions.items():
        model_key = FIXTURE_TO_MODEL_KEY.get(k, k)
        if model_key in row:
            row[model_key] = _numeric(v)
    for k, v in alloy.microstructure.items():
        model_key = FIXTURE_TO_MODEL_KEY.get(k, k)
        if model_key in row:
            row[model_key] = _numeric(v)
    return row


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Standardised Euclidean distance — every column scaled to unit variance
    so features with large natural range (e.g. Toxidation ~1000) don't dominate."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    std = a.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return np.sqrt(((a - b) / std) ** 2).sum(axis=1)


def _nearest_neighbours(
    pool: pd.DataFrame,
    query: np.ndarray,
    k: int = 5,
) -> list[dict[str, Any]]:
    """K rows from `pool` whose feature vector is closest to `query`."""
    feature_cols = [c for c in pool.columns if c not in ("YS", "Mass gain")]
    pool_x = pool[feature_cols].values
    if pool_x.size == 0:
        return []
    dists = _distance(pool_x, query)
    idx = np.argsort(dists)[:k]
    return [{c: pool.iloc[int(i)][c] for c in pool.columns} for i in idx]


def check_distribution(
    alloy: AlloyInput,
    target: str,
    k: int = 5,
) -> dict[str, Any]:
    """Classify a prediction as in / near / out relative to training data."""
    csv_path = _csv_path(target)
    if not csv_path.exists():
        return {
            "target": target,
            "ood_features": [],
            "near_bound_features": [],
            "in_range_features": [],
            "nearest_neighbors": [],
            "warning": f"训练 CSV 未找到: {csv_path}",
        }

    df = pd.read_csv(csv_path)
    features = _csv_features(target)
    query = _alloy_to_feature_row(alloy, features)
    query_arr = np.array([query[f] for f in features], dtype=float)

    ood: list[tuple[str, float, float, float]] = []
    near: list[tuple[str, float, float, float]] = []
    in_range: list[tuple[str, float]] = []
    warning_parts: list[str] = []

    for f in features:
        value = query_arr[features.index(f)]
        lo, hi = float(df[f].min()), float(df[f].max())
        classification = _classify(value, lo, hi)
        if classification == "in":
            in_range.append((f, value))
        elif classification.startswith("near"):
            bound, ratio = _bound_for_classification(value, lo, hi, classification)
            near.append((f, value, bound, ratio))
            direction = "下" if classification.endswith("_lo") else "上"
            warning_parts.append(
                f"{f}={value:.3f} 接近训练集{direction}界 {bound:.3f}(占比 {ratio:.1%}),预测值在边界"
            )
        else:
            bound, ratio = _bound_for_classification(value, lo, hi, classification)
            ood.append((f, value, bound, ratio))
            direction = "下界" if classification.endswith("_lo") else "上界"
            warning_parts.append(
                f"{f}={value:.3f} 超出训练集{direction} {bound:.3f},可能是外推"
            )

    neighbours = _nearest_neighbours(df, query_arr, k=k)
    target_col = "YS" if target == "yield_strength" else "Mass gain"

    return {
        "target": target,
        "ood_features": ood,
        "near_bound_features": near,
        "in_range_features": in_range,
        "nearest_neighbors": neighbours,
        "warning": " · ".join(warning_parts),
        "nearest_neighbor_target_value": (
            float(neighbours[0][target_col]) if neighbours else None
        ),
    }


if __name__ == "__main__":
    from alloy_agent.fixtures import make_default_alloy_input

    alloy = make_default_alloy_input()
    for tgt in ("yield_strength", "oxidation_mass_gain"):
        r = check_distribution(alloy, tgt)
        print(f"\n--- {tgt} ---")
        print(f"warning: {r['warning'] or '✅ all features in range'}")
        print(f"ood: {r['ood_features']}")
        print(f"near: {r['near_bound_features']}")
        print(
            f"closest training sample: target={r['nearest_neighbor_target_value']:.3f}"
            if r["nearest_neighbor_target_value"] is not None
            else ""
        )
