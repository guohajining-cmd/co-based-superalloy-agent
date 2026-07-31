from __future__ import annotations

from alloy_agent.schemas import (
    AgentRequest,
    AgentResponse,
    AlloyInput,
    OptimizationRequest,
)
from alloy_agent.tools.report_generator import (
    generate_full_report,
    generate_optimization_report,
)
from alloy_agent.workflows.evaluate import run_evaluation_workflow
from alloy_agent.workflows.optimize import run_optimization_workflow
from alloy_agent.tool_trace import tool_call


def _run_evaluate(req: AgentRequest) -> AgentResponse:
    if req.alloy_input is None:
        raise ValueError("mode='evaluate' requires alloy_input")
    alloy = req.alloy_input
    dispatch_trace = tool_call(
        step="评估调度",
        tool="run_agent:evaluate",
        purpose="评估已有合金,先 evaluate workflow 后 generate_evaluation_report",
    )
    result = run_evaluation_workflow(alloy)
    eval_tool_trace = result.get("tool_trace", [])
    result["tool_trace"] = [dispatch_trace, *eval_tool_trace]
    report = result["report"]
    return AgentResponse(mode="evaluate", result=result, report=report)


def _run_optimize(req: AgentRequest) -> AgentResponse:
    if req.optimization_request is None:
        raise ValueError("mode='optimize' requires optimization_request")
    dispatch_trace = tool_call(
        step="优化调度",
        tool="run_agent:optimize",
        purpose="多目标搜索候选合金,先 optimize workflow 后 generate_optimization_report",
    )
    result = run_optimization_workflow(req.optimization_request)
    opt_tool_trace = result.get("tool_trace", [])
    result["tool_trace"] = [dispatch_trace, *opt_tool_trace]
    report = generate_optimization_report(result["candidates"])
    return AgentResponse(mode="optimize", result=result, report=report)


def _run_full(req: AgentRequest) -> AgentResponse:
    if req.alloy_input is None:
        raise ValueError("mode='full' requires alloy_input")
    alloy = req.alloy_input
    dispatch_trace = tool_call(
        step="完整分析调度",
        tool="run_agent:full",
        purpose="评估当前合金 + 跑 NSGA-II 找 Pareto,综合后输出",
    )
    from alloy_agent.workflows.full import run_full_workflow
    full = run_full_workflow(
        alloy,
        include_optimization=req.include_optimization,
        search_space=req.search_space,
    )

    eval_dict = full.evaluation
    eval_report = eval_dict["report"]
    opt_report = None
    if full.optimization:
        opt_report = generate_optimization_report(full.optimization["candidates"])

    report = generate_full_report(
        summary=full.summary,
        eval_report=eval_report,
        opt_report=opt_report,
    )
    result = full.to_dict()
    inner_trace = result.get("tool_trace", [])
    result["tool_trace"] = [dispatch_trace, *inner_trace]
    return AgentResponse(mode="full", result=result, report=report)


_DISPATCH = {
    "evaluate": _run_evaluate,
    "optimize": _run_optimize,
    "full": _run_full,
}


def run_agent(request: AgentRequest) -> AgentResponse:
    handler = _DISPATCH.get(request.mode)
    if handler is None:
        raise ValueError(f"Unsupported mode: {request.mode}")
    return handler(request)
