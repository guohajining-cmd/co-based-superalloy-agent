"""Real XGBoost oxidation-mass-gain model.

The collaborating student's `xgboost_mass_gain_model.pkl` was saved without a
fitted booster (joblib.dump fired on an empty XGBRegressor config). She said
"you can retrain it from OXIDATION.csv" — we did, using her exact
hyperparameters (`max_depth=7, learning_rate=0.0478, …`) and `random_state=42`,
then dumped it back to the same path. The booster is now loadable.

Falls back to a placeholder formula if the .pkl is missing / unloadable.
Feature extraction lives in `alloy_agent.tools._model_utils`.
"""

from __future__ import annotations

from alloy_agent.schemas import AlloyInput, OxidationPrediction
from alloy_agent.tools._model_utils import extract_features_oxidation, load_xgboost, numeric


# Placeholder constants used when the real model is not loadable.
_OXIDATION_BASE = 5.0
_OXIDATION_PER_CR = 0.3
_OXIDATION_FLOOR = 0.5


_MODEL = load_xgboost("oxidation")


def predict_oxidation_mass_gain(alloy: AlloyInput) -> OxidationPrediction:
    if _MODEL is None:
        cr = numeric(alloy.composition.get("Cr"))
        value = max(_OXIDATION_FLOOR, _OXIDATION_BASE - _OXIDATION_PER_CR * cr)
        return OxidationPrediction(
            value=round(value, 2),
            unit="mg/cm2",
            source="mock",
            note="未加载到真 XGBoost 模型,使用占位公式 max(0.5, 5.0 − 0.3·Cr)。",
        )

    import numpy as np
    features = np.array([extract_features_oxidation(alloy)], dtype=float)
    value = float(_MODEL.predict(features)[0])
    return OxidationPrediction(
        value=round(value, 2),
        unit="mg/cm2",
        source="real_model",
        note=(
            "XGBoost 模型(12 维特征),由合作方授权下用其 OXIDATION.csv + 脚本超参重训,"
            "特征顺序见 _model_utils._OX_FEATURE_ORDER。"
        ),
    )