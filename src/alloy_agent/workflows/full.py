"""Full analysis: evaluate input alloy + run NSGA-II for better candidates.

The "full" mode is the headline workflow — one input, three sequential steps,
one comprehensive report:
  1. evaluate the input alloy (YS + Oxidation + SHAP for each)
  2. derive composition_bounds from the input and run NSGA-II to find Pareto
     candidates that improve on it
  3. assemble a structured FullResult and a human-readable report
"""

from __future__ import annotations

from alloy_agent.design_constraints import DEFAULT_CONSTRAINTS
from alloy_agent.schemas import AlloyInput, FullResult, OptimizationRequest
from alloy_agent.tool_trace import tool_call
from alloy_agent.workflows.evaluate import (
    bounds_from_composition,
    run_evaluation_workflow,
)


def _build_optimization_request(
    alloy: AlloyInput,
    extra_bounds: dict | None = None,
    search_space: str = "local",
) -> OptimizationRequest:
    """Build an OptimizationRequest derived from the input alloy.

    The bounds come from:
      1. any extra_bounds the caller passed (highest priority)
      2. bounds derived from alloy.composition (±delta per element)
    If a key is in both, extra_bounds wins.
    """
    derived = bounds_from_composition(alloy.composition, profile=search_space)
    vol = alloy.microstructure.get("Vol", alloy.microstructure.get("Vγ′"))
    if search_space == "local" and isinstance(vol, (int, float)):
        derived["Vol"] = [float(vol), float(vol)]
    if extra_bounds:
        derived.update(extra_bounds)

    return OptimizationRequest(
        objectives={
            "maximize": ["yield_strength"],
            "minimize": ["oxidation_mass_gain"],
        },
        constraints=dict(DEFAULT_CONSTRAINTS),
        composition_bounds=derived,
        processing=alloy.processing,
        test_conditions=alloy.test_conditions,
    )


def run_full_workflow(
    alloy: AlloyInput,
    include_optimization: bool = True,
    extra_bounds: dict | None = None,
    search_space: str = "local",
) -> FullResult:
    """Evaluate the alloy, then optionally optimize around it.

    Args:
        alloy: the input alloy composition / processing / conditions.
        include_optimization: if False, skip the NSGA-II step (only evaluate).
        extra_bounds: optional override / extension of derived composition_bounds.
        search_space: "local" for Agent-side nearby search, "script" for
            original collaborator-script bounds.
    """
    evaluation = run_evaluation_workflow(alloy)
    tool_trace = list(evaluation.get("tool_trace", []))

    optimization: dict | None = None
    if include_optimization:
        from alloy_agent.workflows.optimize import run_optimization_workflow
        tool_trace.append(
            tool_call(
                "优化请求构建",
                "_build_optimization_request",
                "根据当前合金自动生成 NSGA-II 成分搜索范围",
            )
        )
        request = _build_optimization_request(alloy, extra_bounds, search_space=search_space)
        optimization = run_optimization_workflow(request)
        tool_trace.extend(optimization.get("tool_trace", []))

    summary = _build_summary(evaluation, optimization)
    tool_trace.append(
        tool_call("完整报告", "generate_full_report", "合并评估、解释和优化结果")
    )
    return FullResult(
        evaluation=evaluation,
        optimization=optimization,
        summary=summary,
        tool_trace=tool_trace,
    )


def _build_summary(evaluation: dict, optimization: dict | None) -> str:
    strength = evaluation["strength"]["value"]
    oxidation = evaluation["oxidation"]["value"]
    s_unit = evaluation["strength"]["unit"]
    o_unit = evaluation["oxidation"]["unit"]
    head = (
        f"当前合金预测:屈服强度 {strength} {s_unit},"
        f"氧化增重 {oxidation} {o_unit}。"
    )
    if not optimization or not optimization.get("candidates"):
        return head + " (未运行优化)"
    best = optimization["candidates"][0]
    return (
        head
        + f" NSGA-II 找到 {len(optimization['candidates'])} 个 Pareto 候选,"
        + f" 最佳 YS={best['predicted_strength']} MPa,"
        + f" Oxidation={best['predicted_oxidation']} mg/cm²。"
    )
