"""Validation helpers for the rule-based agent loop."""

from __future__ import annotations

from alloy_agent.state import AgentState


def validate_required_inputs(state: AgentState) -> list[str]:
    """Return required fields that are missing or only filled by defaults."""
    missing: list[str] = []
    if state.alloy_input is None:
        return ["alloy_input"]

    if any("没有识别到合金成分" in warning for warning in state.warnings):
        missing.append("composition")

    composition = state.alloy_input.composition
    if not composition:
        missing.append("composition")

    required_test_conditions = (
        "strength_test_temperature",
        "oxidation_temperature",
        "oxidation_time",
    )
    for field in required_test_conditions:
        if field not in state.alloy_input.test_conditions:
            missing.append(f"test_conditions.{field}")

    return sorted(set(missing))


def validate_agent_result(state: AgentState) -> list[str]:
    """Return warnings for suspicious but non-fatal tool results."""
    if not state.result:
        return ["没有生成 Agent 结果。"]

    warnings: list[str] = []
    payload = state.result.get("result", {})

    evaluation = payload.get("evaluation", payload)
    oxidation = evaluation.get("oxidation", {})
    oxidation_value = oxidation.get("value")
    if isinstance(oxidation_value, (int, float)) and oxidation_value < 0:
        warnings.append("氧化增重预测为负值，需要人工复核。")

    optimization = payload.get("optimization")
    if optimization is not None:
        candidates = optimization.get("candidates", [])
        if not candidates:
            warnings.append("NSGA-II 未返回候选合金。")
        for candidate in candidates:
            ox = candidate.get("predicted_oxidation")
            if isinstance(ox, (int, float)) and ox < 0:
                warnings.append(
                    f"Rank {candidate.get('rank')} 候选氧化增重为负值，需要人工复核。"
                )

    return warnings
