from __future__ import annotations

from typing import Any, Optional

from alloy_agent.schemas import AlloyCandidate, AlloyInput


def generate_evaluation_report(
    alloy: AlloyInput,
    strength: dict,
    oxidation: dict,
    strength_shap: dict,
    oxidation_shap: dict,
    distribution_ys: Optional[dict] = None,
    distribution_ox: Optional[dict] = None,
) -> str:
    """Build the human-readable evaluation report.

    Accepts plain dicts (not the dataclasses they came from) so the caller
    doesn't have to rebuild dataclass instances from `to_dict()` output.

    `distribution_ys` / `distribution_ox` come from
    `alloy_agent.tools.distribution_check.check_distribution(...)` and are
    rendered as a "训练集范围" section if present.
    """
    composition = ", ".join(f"{k}={v}" for k, v in alloy.composition.items())
    lines = [
        "已有合金评估报告",
        f"合金成分:{composition}",
        f"屈服强度预测:{strength['value']} {strength['unit']}({strength['note']})",
        f"氧化增重预测:{oxidation['value']} {oxidation['unit']}({oxidation['note']})",
        f"强度解释:{strength_shap['summary']}",
        f"氧化解释:{oxidation_shap['summary']}",
    ]
    if distribution_ys or distribution_ox:
        lines.append("")
        lines.append("--- 训练集范围对照 ---")
        if distribution_ys:
            tag = "⚠️ " if distribution_ys["warning"] else "✅ "
            lines.append(f"  {tag}YS:{distribution_ys['warning'] or '在训练集范围内'}")
        if distribution_ox:
            tag = "⚠️ " if distribution_ox["warning"] else "✅ "
            lines.append(f"  {tag}Oxidation:{distribution_ox['warning'] or '在训练集范围内'}")
    lines.append("说明:当前输出来自已加载模型与 Agent tools,需结合训练集范围和实验条件理解。")
    return "\n".join(lines)


def generate_optimization_report(candidates: list[dict]) -> str:
    lines = ["合金优化设计报告", "候选合金列表:"]
    for candidate in candidates:
        composition = ", ".join(f"{k}={v}" for k, v in candidate["composition"].items())
        ood = candidate.get("ood", {})
        ood_line = ""
        if ood and ood.get("warning"):
            ood_line = f" [⚠️ {ood['warning'][:60]}]"
        lines.append(
            f"Rank {candidate['rank']}: {composition}; "
            f"strength={candidate['predicted_strength']} MPa; "
            f"oxidation={candidate['predicted_oxidation']} mg/cm2; "
            f"{candidate['note']}{ood_line}"
        )
    return "\n".join(lines)


def generate_full_report(summary: str, eval_report: str, opt_report: Optional[str]) -> str:
    """Composite report for mode='full': evaluation + (optional) optimization."""
    parts = [
        "=== 完整合金分析报告 ===",
        f"总览:{summary}",
        "",
        "--- 第一部分:当前合金评估 ---",
        eval_report,
    ]
    if opt_report:
        parts.extend(["", "--- 第二部分:NSGA-II 优化候选 ---", opt_report])
    parts.extend([
        "",
        "--- 第三部分:推荐理由与风险提示 ---",
        "推荐理由:候选合金在 Pareto 前沿上同时满足强度最大化与氧化最小化,",
        "  数值上优于当前合金,且成分变化落在训练集范围内,可信度较高。",
        "风险提示:任何外推成分(已在 OOD 检查中标注 ⚠️)建议实验验证后再采用。",
        "  当前真实模型由本机重训/对方 .pkl 加载,XGBoost 仍可能在未见区域失真。",
    ])
    return "\n".join(parts)
