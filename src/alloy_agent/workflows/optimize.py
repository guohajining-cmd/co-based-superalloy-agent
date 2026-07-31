from alloy_agent.schemas import OptimizationRequest
from alloy_agent.tool_trace import tool_call
from alloy_agent.tools.nsga2_optimizer import run_nsga2_optimization
from alloy_agent.tools.report_generator import generate_optimization_report


def run_optimization_workflow(request: OptimizationRequest) -> dict:
    tool_trace = [
        tool_call("多目标搜索", "run_nsga2_optimization", "运行 NSGA-II 生成 Pareto 候选合金"),
        tool_call("候选强度评分", "predict_yield_strength", "作为 NSGA-II 目标函数反复评估候选强度", "yield_strength"),
        tool_call("候选氧化评分", "predict_oxidation_mass_gain", "作为 NSGA-II 目标函数反复评估候选氧化增重", "oxidation_mass_gain"),
        tool_call("优化报告", "generate_optimization_report", "汇总 Pareto 候选合金"),
    ]
    candidates = run_nsga2_optimization(request)
    report = generate_optimization_report([c.to_dict() for c in candidates])

    return {
        "candidates": [candidate.to_dict() for candidate in candidates],
        "report": report,
        "tool_trace": tool_trace,
    }
