from alloy_agent.design_constraints import (
    bounds_from_composition,
)
from alloy_agent.schemas import AlloyInput
from alloy_agent.tools.distribution_check import check_distribution
from alloy_agent.tools.oxidation_model import predict_oxidation_mass_gain
from alloy_agent.tools.report_generator import generate_evaluation_report
from alloy_agent.tools.shap_explainer import explain_with_shap
from alloy_agent.tools.strength_model import predict_yield_strength
from alloy_agent.tool_trace import tool_call


def run_evaluation_workflow(alloy: AlloyInput) -> dict:
    tool_trace = [
        tool_call("强度预测", "predict_yield_strength", "调用 XGBoost 屈服强度模型", "yield_strength"),
        tool_call("氧化预测", "predict_oxidation_mass_gain", "调用 XGBoost 氧化增重模型", "oxidation_mass_gain"),
        tool_call("强度解释", "explain_with_shap", "用 SHAP 解释强度预测", "yield_strength"),
        tool_call("氧化解释", "explain_with_shap", "用 SHAP 解释氧化预测", "oxidation_mass_gain"),
        tool_call("强度分布检查", "check_distribution", "检查强度模型输入是否接近或超出训练集范围", "yield_strength"),
        tool_call("氧化分布检查", "check_distribution", "检查氧化模型输入是否接近或超出训练集范围", "oxidation_mass_gain"),
        tool_call("评估报告", "generate_evaluation_report", "汇总已有合金预测与解释"),
    ]
    strength = predict_yield_strength(alloy)
    oxidation = predict_oxidation_mass_gain(alloy)
    strength_shap = explain_with_shap(alloy, target="yield_strength")
    oxidation_shap = explain_with_shap(alloy, target="oxidation_mass_gain")

    distribution_ys = check_distribution(alloy, "yield_strength")
    distribution_ox = check_distribution(alloy, "oxidation_mass_gain")

    report = generate_evaluation_report(
        alloy=alloy,
        strength=strength.to_dict(),
        oxidation=oxidation.to_dict(),
        strength_shap=strength_shap.to_dict(),
        oxidation_shap=oxidation_shap.to_dict(),
        distribution_ys=distribution_ys,
        distribution_ox=distribution_ox,
    )

    return {
        "strength": strength.to_dict(),
        "oxidation": oxidation.to_dict(),
        "strength_shap": strength_shap.to_dict(),
        "oxidation_shap": oxidation_shap.to_dict(),
        "distribution_ys": distribution_ys,
        "distribution_ox": distribution_ox,
        "report": report,
        "tool_trace": tool_trace,
    }
