"""Shared helpers for the XGBoost-backed tools.

Centralises the boilerplate that used to be duplicated across
`strength_model.py`, `oxidation_model.py`, `shap_explainer.py`, and
`nsga2_optimizer.py`:
  - resolving the on-disk model directory (env var override or default)
  - the fixture-to-model feature name translation
  - numeric coercion and XGBoost loaders
  - extracting the 12-dim YS / Oxidation feature vector from an AlloyInput
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


# Human-readable fixture keys → the abbreviated names the trained XGBoost
# models expect. Identical to what each tool used to define locally.
FIXTURE_TO_MODEL_KEY: dict[str, str] = {
    "oxidation_temperature": "Toxidation",
    "oxidation_time": "toxidation",
    "aging_temperature": "Tage",
    "aging_time": "tage",
    "strength_test_temperature": "Ttest",
    # The training CSVs use "Vol" (no Unicode), the fixture uses "Vγ′" (γ-prime
    # symbol). Both names refer to γ' volume fraction.
    "Vγ′": "Vol",
}


# File name under model_dir() for each model the agent can load.
_MODEL_FILES: dict[str, str] = {
    "ys": "xgboost_ys_元素_微观_增强_特征筛选_12_model.pkl",
    "oxidation": "xgboost_mass_gain_model.pkl",
}

# Feature orderings (read from the training CSVs).
_YS_FEATURE_ORDER = ["Co", "Al", "Ta", "Ti", "Ni", "Cr", "V", "Mo",
                     "Tage", "tage", "Ttest", "Vol"]
_OX_FEATURE_ORDER = ["Co", "Al", "W", "Ta", "Ti", "Nb", "Ni", "Cr", "V", "Mo",
                     "Toxidation", "toxidation"]


def model_dir() -> Path:
    """Resolve the directory containing the collaborator's .pkl files.

    Override with `ALLOY_AGENT_MODEL_DIR=/path/to/models`; otherwise look in
    `<project-root>/Agent-acta/`.
    """
    override = os.environ.get("ALLOY_AGENT_MODEL_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "Agent-acta"


def numeric(value: object) -> float:
    """Coerce any value to float; non-numeric values become 0.0."""
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def load_xgboost(name: str) -> Optional[Any]:
    """Lazy load a XGBoost model by short name. Returns None if missing/broken.

    Args:
        name: one of "ys" or "oxidation".
    """
    pkl_path = model_dir() / _MODEL_FILES[name]
    if not pkl_path.exists():
        return None
    try:
        import joblib
        return joblib.load(pkl_path)
    except Exception:
        return None


def _build_sources(alloy) -> dict[str, float]:
    """Translate AlloyInput's three named buckets + microstructure into a flat
    `model_feature_name -> value` dict, applying the fixture-key translation.
    """
    sources: dict[str, float] = {}
    for k, v in alloy.composition.items():
        sources[FIXTURE_TO_MODEL_KEY.get(k, k)] = numeric(v)
    for k, v in alloy.processing.items():
        sources[FIXTURE_TO_MODEL_KEY.get(k, k)] = numeric(v)
    for k, v in alloy.test_conditions.items():
        sources[FIXTURE_TO_MODEL_KEY.get(k, k)] = numeric(v)
    for k, v in alloy.microstructure.items():
        sources[FIXTURE_TO_MODEL_KEY.get(k, k)] = numeric(v)
    return sources


def extract_features_ys(alloy) -> list[float]:
    """Build the 12-dim YS feature vector in the order the model was trained on."""
    sources = _build_sources(alloy)
    return [numeric(sources.get(name)) for name in _YS_FEATURE_ORDER]


def extract_features_oxidation(alloy) -> list[float]:
    """Build the 12-dim Oxidation feature vector in the order the model was trained on."""
    sources = _build_sources(alloy)
    return [numeric(sources.get(name)) for name in _OX_FEATURE_ORDER]