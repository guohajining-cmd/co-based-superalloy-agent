"""Real XGBoost yield-strength model.

Loads the .pkl the collaborating student trained; falls back to a placeholder
formula when the .pkl is absent or unloadable, so the rest of the agent keeps
working during development.

Feature extraction lives in `alloy_agent.tools._model_utils`.
"""

from __future__ import annotations

from alloy_agent.schemas import AlloyInput, StrengthPrediction
from alloy_agent.tools._model_utils import extract_features_ys, load_xgboost, numeric


# Placeholder constants used when the real model is not loadable.
_GAMMA_PRIME_ELEMENTS = ("Al", "Ti", "Ta")
_STRENGTH_BASE = 600.0
_STRENGTH_PER_UNIT = 30.0


# Lazy load: import never fails even if the .pkl is missing / broken.
_MODEL = load_xgboost("ys")


def predict_yield_strength(alloy: AlloyInput) -> StrengthPrediction:
    if _MODEL is None:
        gamma_prime = sum(numeric(alloy.composition.get(el)) for el in _GAMMA_PRIME_ELEMENTS)
        value = _STRENGTH_BASE + _STRENGTH_PER_UNIT * gamma_prime
        return StrengthPrediction(
            value=round(value, 2),
            unit="MPa",
            source="mock",
            note="未加载到真 XGBoost 模型,使用占位公式 600 + 30·(Al+Ti+Ta)。",
        )

    import numpy as np
    features = np.array([extract_features_ys(alloy)], dtype=float)
    value = float(_MODEL.predict(features)[0])
    return StrengthPrediction(
        value=round(value, 2),
        unit="MPa",
        source="real_model",
        note="XGBoost 模型(12 维特征),特征顺序见 _model_utils._YS_FEATURE_ORDER。",
    )