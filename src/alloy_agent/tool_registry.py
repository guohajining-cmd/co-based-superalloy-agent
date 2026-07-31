"""Central registry of every tool the agent can call.

A `ToolDefinition` records the callable plus a static description of its input
and output schemas. Workflows and the agent dispatcher consume this registry
to build a tool trace, validate inputs, and surface capabilities in the UI
without hard-coding names in three different files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from alloy_agent.schemas import (
    AlloyInput,
    OptimizationRequest,
    StrengthPrediction,
    OxidationPrediction,
    ShapExplanation,
    AlloyCandidate,
)


@dataclass(frozen=True)
class ToolDefinition:
    """Static metadata about one tool the agent can invoke.

    Attributes
    ----------
    name : str
        Symbolic name (also the registry key). Matches the value used in
        `tool_trace` entries.
    step_label : str
        Human-readable step name shown in the UI / report.
    purpose : str
        One-line explanation of what this tool does.
    inputs : list[str]
        Names of the inputs the tool expects (kept loose: dict keys for tools
        that take AlloyInput, scalar arg names for the rest).
    outputs : list[str]
        Names of the outputs the tool produces.
    impl : Callable[..., Any]
        The actual function. Registered here so dispatch can be data-driven.
    """

    name: str
    step_label: str
    purpose: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    impl: Optional[Callable[..., Any]] = None
    workflow_origin: list[str] = field(default_factory=list)
    """Which workflow(s) use this tool, e.g. ['evaluate', 'full'].
    Lets the UI / report ask "what does mode=evaluate actually run?"."""
    category: str = ""
    """One of {'prediction', 'explanation', 'optimization', 'report',
    'workflow', 'meta'}. Used for grouping in the registry dump."""

    def trace_entry(self, target: Optional[str] = None) -> dict[str, Any]:
        """Build a tool_trace entry for this tool.

        We don't try to call the function here — just record what *would* be
        called and why. The actual call goes elsewhere.
        """
        entry: dict[str, Any] = {
            "step": self.step_label,
            "tool": self.name,
            "purpose": self.purpose,
        }
        if target is not None:
            entry["target"] = target
        return entry


def _register_all() -> dict[str, ToolDefinition]:
    """Build the registry. Importing is deferred to keep this module
    importable in environments where the underlying tools aren't loaded yet
    (e.g. when generating documentation).
    """
    from alloy_agent.agent_loop import run_agent_loop
    from alloy_agent.llm_tool_agent import run_llm_tool_agent
    from alloy_agent.natural_language import parse_user_request
    from alloy_agent.tools.strength_model import predict_yield_strength
    from alloy_agent.tools.oxidation_model import predict_oxidation_mass_gain
    from alloy_agent.tools.shap_explainer import explain_with_shap
    from alloy_agent.tools.distribution_check import check_distribution
    from alloy_agent.tools.nsga2_optimizer import run_nsga2_optimization
    from alloy_agent.tools.report_generator import (
        generate_evaluation_report,
        generate_optimization_report,
        generate_full_report,
    )
    from alloy_agent.workflows.evaluate import run_evaluation_workflow
    from alloy_agent.workflows.optimize import run_optimization_workflow
    from alloy_agent.workflows.full import run_full_workflow
    from alloy_agent.validators import validate_agent_result, validate_required_inputs

    defs = [
        ToolDefinition(
            name="parse_user_request",
            step_label="自然语言解析",
            purpose="自然语言转标准 AlloyInput 和运行模式",
            inputs=["text: str"],
            outputs=["ParsedUserRequest"],
            impl=parse_user_request,
            workflow_origin=["run_agent_loop", "web_streamlit"],
            category="meta",
        ),
        ToolDefinition(
            name="validate_required_inputs",
            step_label="输入完整性检查",
            purpose="判断调用模型前是否缺少必要输入",
            inputs=["state: AgentState"],
            outputs=["missing_fields"],
            impl=validate_required_inputs,
            workflow_origin=["run_agent_loop"],
            category="validation",
        ),
        ToolDefinition(
            name="predict_yield_strength",
            step_label="强度预测",
            purpose="调用 XGBoost 屈服强度模型",
            inputs=["alloy: AlloyInput"],
            outputs=[f"{f} ({t})" for f, t in StrengthPrediction.__dataclass_fields__.items()],
            impl=predict_yield_strength,
            workflow_origin=["run_evaluation_workflow", "run_full_workflow"],
            category="prediction",
        ),
        ToolDefinition(
            name="predict_oxidation_mass_gain",
            step_label="氧化预测",
            purpose="调用 XGBoost 氧化增重模型",
            inputs=["alloy: AlloyInput"],
            outputs=[f"{f} ({t})" for f, t in OxidationPrediction.__dataclass_fields__.items()],
            impl=predict_oxidation_mass_gain,
            workflow_origin=["run_evaluation_workflow", "run_full_workflow"],
            category="prediction",
        ),
        ToolDefinition(
            name="explain_with_shap",
            step_label="SHAP 解释",
            purpose="用 SHAP TreeExplainer 解释单样本预测",
            inputs=["alloy: AlloyInput", "target: str"],
            outputs=[f"{f} ({t})" for f, t in ShapExplanation.__dataclass_fields__.items()],
            impl=explain_with_shap,
            workflow_origin=["run_evaluation_workflow", "run_full_workflow"],
            category="explanation",
        ),
        ToolDefinition(
            name="check_distribution",
            step_label="训练集范围对照",
            purpose="判断输入是否落在训练集分布内,标记外推点",
            inputs=["alloy: AlloyInput", "target: str"],
            outputs=["ood_features", "near_bound_features", "warning", "nearest_neighbors"],
            impl=check_distribution,
            workflow_origin=["run_evaluation_workflow", "run_full_workflow"],
            category="explanation",
        ),
        ToolDefinition(
            name="run_nsga2_optimization",
            step_label="多目标搜索",
            purpose="运行 NSGA-II 生成 Pareto 候选合金",
            inputs=["request: OptimizationRequest"],
            outputs=[f"{f} ({t})" for f, t in AlloyCandidate.__dataclass_fields__.items()],
            impl=run_nsga2_optimization,
            workflow_origin=["run_optimization_workflow", "run_full_workflow"],
            category="optimization",
        ),
        ToolDefinition(
            name="generate_evaluation_report",
            step_label="评估报告",
            purpose="汇总已有合金预测与解释",
            inputs=["strength, oxidation, strength_shap, oxidation_shap, distribution (opt)"],
            outputs=["text report"],
            impl=generate_evaluation_report,
            workflow_origin=[],
            category="report",
        ),
        ToolDefinition(
            name="generate_optimization_report",
            step_label="优化报告",
            purpose="汇总 Pareto 候选合金",
            inputs=["candidates", "distribution (opt)"],
            outputs=["text report"],
            impl=generate_optimization_report,
        ),
        ToolDefinition(
            name="generate_full_report",
            step_label="完整报告",
            purpose="合并评估 + 优化两段报告",
            inputs=["eval_report, opt_report, summary"],
            outputs=["text report"],
            impl=generate_full_report,
            category="report",
        ),
        ToolDefinition(
            name="validate_agent_result",
            step_label="结果自检",
            purpose="检查预测和候选结果是否存在明显风险",
            inputs=["state: AgentState"],
            outputs=["warnings"],
            impl=validate_agent_result,
            workflow_origin=["run_agent_loop"],
            category="validation",
        ),
        ToolDefinition(
            name="run_evaluation_workflow",
            step_label="评估工作流",
            purpose="一次跑完 4 个工具:强度 + 氧化 + 2 份 SHAP",
            inputs=["alloy: AlloyInput"],
            outputs=["{strength, oxidation, strength_shap, oxidation_shap, tool_trace, report}"],
            impl=run_evaluation_workflow,
            category="workflow",
        ),
        ToolDefinition(
            name="run_optimization_workflow",
            step_label="优化工作流",
            purpose="NSGA-II + 报告生成",
            inputs=["request: OptimizationRequest"],
            outputs=["{candidates, report, tool_trace}"],
            impl=run_optimization_workflow,
            category="workflow",
        ),
        ToolDefinition(
            name="run_full_workflow",
            step_label="完整工作流",
            purpose="评估 + 优化 + 总结",
            inputs=["alloy: AlloyInput", "include_optimization: bool"],
            outputs=["{evaluation, optimization, summary, tool_trace}"],
            impl=run_full_workflow,
            category="workflow",
        ),
        ToolDefinition(
            name="run_agent_loop",
            step_label="Agent Loop",
            purpose="解析、输入检查、工具选择、结果自检的规则型循环控制器",
            inputs=["user_text: str", "search_space: str"],
            outputs=["AgentState"],
            impl=run_agent_loop,
            category="workflow",
        ),
        ToolDefinition(
            name="run_llm_tool_agent",
            step_label="LLM Tool-Calling Agent",
            purpose="由 LLM 在白名单 tools 中选择材料设计工具,未配置时回退到规则 Agent Loop",
            inputs=["user_text: str", "client/api_key (optional)", "search_space: str"],
            outputs=["AgentState"],
            impl=run_llm_tool_agent,
            category="workflow",
        ),
    ]
    return {d.name: d for d in defs}


TOOL_REGISTRY: dict[str, ToolDefinition] = _register_all()


def get_tool(name: str) -> ToolDefinition:
    """Look up a tool by name. Raises if not registered."""
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Tool {name!r} not in registry. Known: {list(TOOL_REGISTRY)}")
    return TOOL_REGISTRY[name]
