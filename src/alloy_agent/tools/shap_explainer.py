"""SHAP explanation for the real XGBoost models.

We use `shap.TreeExplainer`, which is fast for tree models (no background
sample needed) and works directly on the loaded .pkl — no need for the
collaborator to save and resend the explainer object. The explainer is
lazily built on first call and cached at module level.

Falls back to a placeholder when the model is not loadable.
"""

from __future__ import annotations

from typing import Any

from alloy_agent.schemas import AlloyInput, ShapExplanation
from alloy_agent.tools._model_utils import (
    extract_features_oxidation,
    extract_features_ys,
    load_xgboost,
)


# Lazy + cached explainers: built on first use, kept for the lifetime of the module.
_explainers: dict[str, Any] = {}


def _get_explainer(model_key: str, model):
    if model_key in _explainers:
        return _explainers[model_key]
    import shap
    _explainers[model_key] = shap.TreeExplainer(model)
    return _explainers[model_key]


def _strength_placeholder_features(alloy: AlloyInput) -> list[dict]:
    """Placeholder SHAP-style features for fallback when the model is missing."""
    features: list[dict] = []
    ta = alloy.composition.get("Ta", 0)
    ti = alloy.composition.get("Ti", 0)
    al = alloy.composition.get("Al", 0)
    cr = alloy.composition.get("Cr", 0)
    if isinstance(ta, (int, float)) and ta > 0:
        features.append({"feature": "Ta", "effect": "positive", "description": f"Ta 含量 {ta},对 γ' 强化有正向贡献。"})
    if isinstance(ti, (int, float)) and ti > 0:
        features.append({"feature": "Ti", "effect": "positive", "description": f"Ti 含量 {ti},促进 γ' 相析出。"})
    if isinstance(al, (int, float)) and al > 0:
        features.append({"feature": "Al", "effect": "positive", "description": f"Al 含量 {al},是 γ' 相主成分。"})
    if isinstance(cr, (int, float)) and 0 < cr < 5:
        features.append({"feature": "Cr", "effect": "risk", "description": f"Cr 含量 {cr} 偏低,可能不足以形成致密氧化层。"})
    return features[:4]


def _oxidation_placeholder_features(alloy: AlloyInput) -> list[dict]:
    features: list[dict] = []
    cr = alloy.composition.get("Cr", 0)
    v = alloy.composition.get("V", 0)
    al = alloy.composition.get("Al", 0)
    if isinstance(cr, (int, float)) and cr > 0:
        effect = "positive" if cr >= 5 else "risk"
        desc = (f"Cr 含量 {cr},有利于形成保护性 Al2O3/Cr2O3 氧化层。" if cr >= 5
                else f"Cr 含量 {cr} 偏低,抗氧化保护可能不足。")
        features.append({"feature": "Cr", "effect": effect, "description": desc})
    if isinstance(v, (int, float)) and v > 0:
        features.append({"feature": "V", "effect": "risk", "description": f"V 含量 {v},可能生成挥发性氧化物,降低抗氧化性能。"})
    if isinstance(al, (int, float)) and al > 0:
        features.append({"feature": "Al", "effect": "positive", "description": f"Al 含量 {al},在高温下形成保护性 Al2O3 层。"})
    return features[:4]


def explain_with_shap(alloy: AlloyInput, target: str) -> ShapExplanation:
    if target == "yield_strength":
        model = load_xgboost("ys")
        feature_vector = extract_features_ys
        placeholder = _strength_placeholder_features(alloy)
        if model is None:
            return ShapExplanation(
                target=target, top_features=placeholder, source="mock",
                summary="未加载到真 XGBoost 模型,使用占位 SHAP 解释。",
            )
        summary = "真 SHAP:对当前合金,各特征对 YS 预测的贡献值(正=提升强度,负=降低强度)。"
    elif target == "oxidation_mass_gain":
        model = load_xgboost("oxidation")
        feature_vector = extract_features_oxidation
        placeholder = _oxidation_placeholder_features(alloy)
        if model is None:
            return ShapExplanation(
                target=target, top_features=placeholder, source="mock",
                summary="未加载到真 XGBoost 模型,使用占位 SHAP 解释。",
            )
        summary = "真 SHAP:对当前合金,各特征对 Mass Gain 预测的贡献值(正=增加氧化,负=抑制氧化)。"
    else:
        return ShapExplanation(
            target=target, top_features=[], source="mock",
            summary=f"未识别的 SHAP 目标 {target!r}。",
        )

    import numpy as np
    explainer = _get_explainer(target, model)
    features = np.array([feature_vector(alloy)], dtype=float)
    shap_values = explainer.shap_values(features, check_additivity=False)
    base_value = float(explainer.expected_value)
    # feature order comes from the same _model_utils constants the extraction used
    from alloy_agent.tools._model_utils import _YS_FEATURE_ORDER, _OX_FEATURE_ORDER
    feature_names = _YS_FEATURE_ORDER if target == "yield_strength" else _OX_FEATURE_ORDER

    rows = []
    for name, sv in zip(feature_names, shap_values[0]):
        effect = "positive" if sv > 0 else "negative" if sv < 0 else "neutral"
        rows.append({
            "feature": name,
            "effect": effect,
            "value": round(float(sv), 3),
            "description": f"{name} 的 SHAP 贡献 = {sv:+.3f}。",
        })
    rows.sort(key=lambda r: abs(r["value"]), reverse=True)
    return ShapExplanation(
        target=target,
        top_features=rows[:6],
        summary=summary + f" 基线值 = {base_value:.3f}。",
        source="real_model",
    )