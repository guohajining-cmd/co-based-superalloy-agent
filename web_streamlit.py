"""Streamlit UI for the cobalt-based superalloy design agent.

Run with:
    streamlit run web_streamlit.py

Sidebar: edit alloy composition / processing / test conditions.
Main pane: YS + Oxidation big numbers, SHAP bar charts, Pareto scatter,
optimization candidate table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

# Make sure we can import the agent package whether streamlit was launched
# from the project root or from somewhere else.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Force a clean matplotlib style and make Chinese-readable fonts available
# so axis labels and titles render correctly. Falls back silently if the
# specific fonts aren't installed.
matplotlib.rcParams["axes.unicode_minus"] = False
for font_name in ("Arial Unicode MS", "Heiti TC", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei"):
    try:
        matplotlib.rcParams["font.sans-serif"] = [font_name]
        break
    except Exception:
        continue

from alloy_agent.agent_loop import run_agent_loop
from alloy_agent.fixtures import make_default_alloy_input
from alloy_agent.llm_tool_agent import run_llm_tool_agent
from alloy_agent.plotting import padded_axis_limits
from alloy_agent.schemas import AlloyInput
from alloy_agent.workflows.full import run_full_workflow


_ELEMENTS = ["Ni", "Al", "Cr", "Ta", "Ti", "W", "V", "Nb", "Mo"]


def _build_composition(values: dict[str, float]) -> dict[str, float]:
    """Co is derived as 100 - sum(others) so the real XGBoost model can read it."""
    other_sum = sum(values[e] for e in _ELEMENTS)
    return {**values, "Co": max(0.0, 100.0 - other_sum)}


def _display_tool_trace(tool_trace: list[dict]) -> None:
    rows = []
    for index, item in enumerate(tool_trace, start=1):
        rows.append(
            {
                "序号": index,
                "步骤": item.get("step", ""),
                "Tool": item.get("tool", ""),
                "Target": item.get("target", ""),
                "作用": item.get("purpose", ""),
            }
        )
    st.subheader("本次调用的 Tools")
    st.table(rows)


def _display_agent_decisions(decision_trace: list[dict]) -> None:
    rows = []
    for index, item in enumerate(decision_trace, start=1):
        rows.append(
            {
                "序号": index,
                "Action": item.get("action", ""),
                "判断依据": item.get("reason", ""),
            }
        )
    st.subheader("Agent 决策过程")
    st.table(rows)


def _evaluation_summary(eval_d: dict) -> str:
    strength = eval_d["strength"]["value"]
    oxidation = eval_d["oxidation"]["value"]
    s_unit = eval_d["strength"]["unit"]
    o_unit = eval_d["oxidation"]["unit"]
    return f"当前合金预测:屈服强度 {strength} {s_unit},氧化增重 {oxidation} {o_unit}。"


def _fallback_workflow_tool_trace(include_optimization: bool) -> list[dict]:
    trace = [
        {
            "step": "强度预测",
            "tool": "predict_yield_strength",
            "purpose": "调用 XGBoost 屈服强度模型",
            "target": "yield_strength",
        },
        {
            "step": "氧化预测",
            "tool": "predict_oxidation_mass_gain",
            "purpose": "调用 XGBoost 氧化增重模型",
            "target": "oxidation_mass_gain",
        },
        {
            "step": "强度解释",
            "tool": "explain_with_shap",
            "purpose": "用 SHAP 解释强度预测",
            "target": "yield_strength",
        },
        {
            "step": "氧化解释",
            "tool": "explain_with_shap",
            "purpose": "用 SHAP 解释氧化预测",
            "target": "oxidation_mass_gain",
        },
        {
            "step": "评估报告",
            "tool": "generate_evaluation_report",
            "purpose": "汇总已有合金预测与解释",
        },
    ]
    if include_optimization:
        trace.extend(
            [
                {
                    "step": "优化请求构建",
                    "tool": "_build_optimization_request",
                    "purpose": "根据当前合金自动生成 NSGA-II 成分搜索范围",
                },
                {
                    "step": "多目标搜索",
                    "tool": "run_nsga2_optimization",
                    "purpose": "运行 NSGA-II 生成 Pareto 候选合金",
                },
                {
                    "step": "候选强度评分",
                    "tool": "predict_yield_strength",
                    "purpose": "作为 NSGA-II 目标函数反复评估候选强度",
                    "target": "yield_strength",
                },
                {
                    "step": "候选氧化评分",
                    "tool": "predict_oxidation_mass_gain",
                    "purpose": "作为 NSGA-II 目标函数反复评估候选氧化增重",
                    "target": "oxidation_mass_gain",
                },
                {
                    "step": "优化报告",
                    "tool": "generate_optimization_report",
                    "purpose": "汇总 Pareto 候选合金",
                },
            ]
        )
    trace.append(
        {
            "step": "完整报告",
            "tool": "generate_full_report",
            "purpose": "合并评估、解释和优化结果",
        }
    )
    return trace


st.set_page_config(page_title="钴基合金设计 Agent", layout="wide")
st.title("钴基高温合金设计 Agent")
st.caption("XGBoost 屈服强度 + 氧化增重预测 · SHAP 解释 · NSGA-II 多目标优化")

default_alloy = make_default_alloy_input()

st.subheader("自然语言输入")
natural_language_text = st.text_area(
    "用一句话描述要评估或优化的合金",
    value=(
        "帮我评估 Co-30Ni-9Al-7Cr-4Ta-3Ti-2W-1V-1Nb-0.5Mo 这个合金，"
        "750度测试屈服强度，1000度氧化100小时。"
    ),
    height=96,
    placeholder="例如：帮我基于 Co-30Ni-9Al-7Cr-4Ta-3Ti-2W-1V-1Nb-0.5Mo 做NSGA-II优化，推荐候选合金。",
)
natural_language_run = st.button("解析自然语言并运行", type="primary", width="stretch")

# --- Sidebar: input ---
with st.sidebar:
    st.header("合金成分")
    st.caption("Co 自动 = 100 − 其它元素之和")
    raw_values: dict[str, float] = {}
    cols = st.columns(3)
    for i, element in enumerate(_ELEMENTS):
        with cols[i % 3]:
            raw_values[element] = st.number_input(
                element,
                min_value=0.0,
                max_value=100.0,
                value=float(default_alloy.composition.get(element, 0.0)),
                step=0.1,
                key=f"elem_{element}",
            )
    composition = _build_composition(raw_values)
    st.metric("Co (auto)", f"{composition['Co']:.2f}")

    with st.expander("处理 / 测试条件", expanded=False):
        processing: dict[str, float] = {}
        for k, v in default_alloy.processing.items():
            processing[k] = st.number_input(
                f"processing · {k}", value=float(v), key=f"proc_{k}"
            )
        test_conditions: dict[str, float] = {}
        for k, v in default_alloy.test_conditions.items():
            test_conditions[k] = st.number_input(
                f"test · {k}", value=float(v), key=f"test_{k}"
            )

    with st.expander("高级选项", expanded=False):
        microstructure: dict[str, float] = dict(default_alloy.microstructure)
        for k, v in default_alloy.microstructure.items():
            microstructure[k] = st.number_input(
                f"microstructure · {k}", value=float(v), key=f"micro_{k}"
            )
        manual_include_optimization = st.checkbox("运行 NSGA-II 优化", value=True)
        natural_language_agent_mode = st.selectbox(
            "自然语言 Agent 控制层",
            options=["rule", "llm"],
            format_func=lambda value: {
                "rule": "规则 Agent Loop",
                "llm": "LLM tool-calling（需 OPENAI_API_KEY）",
            }[value],
            index=0,
        )
        manual_search_space = st.selectbox(
            "NSGA-II 搜索空间",
            options=["local", "script"],
            format_func=lambda value: {
                "local": "局部搜索：围绕当前合金小范围调整",
                "script": "原始脚本范围：使用合作方 NSGA-II 上下限",
            }[value],
            index=0,
        )
        manual_show_shap = st.checkbox("显示 SHAP 解释", value=True)

    manual_run = st.button("运行完整分析", type="secondary", width="stretch")


# --- Main: results ---
loop_state = None
if natural_language_run:
    try:
        with st.spinner("Agent loop 判断并调用工具中…"):
            if natural_language_agent_mode == "llm":
                loop_state = run_llm_tool_agent(
                    natural_language_text,
                    search_space=manual_search_space,
                )
            else:
                loop_state = run_agent_loop(
                    natural_language_text,
                    search_space=manual_search_space,
                )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

if loop_state is not None:
    _display_agent_decisions(loop_state.decision_trace)
    for warning in loop_state.warnings:
        st.warning(warning)
    if loop_state.pending_question:
        st.warning(loop_state.pending_question)
    if loop_state.status == "waiting_for_input":
        st.info("Agent 已暂停：请补充上面的缺失信息后重新运行。")
        st.stop()
    if loop_state.status != "completed" or loop_state.result is None:
        st.error("Agent loop 未完成，请查看上方 warning。")
        st.stop()
    if loop_state.alloy_input is None:
        st.error("Agent loop 没有生成标准 AlloyInput。")
        st.stop()

    alloy_input = loop_state.alloy_input
    include_optimization = loop_state.include_optimization
    search_space = loop_state.search_space
    show_shap = True
    run_requested = True
    st.info(
        f"Agent 已完成 `{loop_state.intent}` 任务；"
        f"NSGA-II 优化：{'开启' if include_optimization else '关闭'}；"
        f"搜索空间：`{search_space}`。"
    )
    with st.expander("查看 Agent 标准输入", expanded=False):
        st.json(
            {
                "composition": alloy_input.composition,
                "processing": alloy_input.processing,
                "test_conditions": alloy_input.test_conditions,
                "microstructure": alloy_input.microstructure,
            }
        )
else:
    alloy_input = AlloyInput(
        composition=composition,
        processing=processing,
        test_conditions=test_conditions,
        microstructure=microstructure,
    )
    include_optimization = manual_include_optimization
    search_space = manual_search_space
    show_shap = manual_show_shap
    run_requested = manual_run

if not run_requested:
    st.info("上方可以输入自然语言直接运行；也可以在左边手动输入合金成分后运行。")
    st.stop()

if loop_state is not None:
    response_payload = loop_state.result["result"]
    if loop_state.result["mode"] == "full":
        eval_d = response_payload["evaluation"]
        opt_d = response_payload["optimization"]
        run_summary = response_payload["summary"]
    else:
        eval_d = response_payload
        opt_d = None
        run_summary = _evaluation_summary(eval_d)
    tool_trace = list(loop_state.tool_trace)
else:
    with st.spinner("评估 + 优化中…"):
        full = run_full_workflow(
            alloy_input,
            include_optimization=include_optimization,
            search_space=search_space,
        )
    eval_d = full.evaluation
    opt_d = full.optimization
    run_summary = full.summary
    tool_trace = list(getattr(full, "tool_trace", []))
    if not tool_trace:
        tool_trace = _fallback_workflow_tool_trace(include_optimization)

# Distribution check: warn when input is outside training data range.
from alloy_agent.tools.distribution_check import check_distribution
eval_d["ood_ys"] = eval_d.get("distribution_ys") or check_distribution(alloy_input, "yield_strength")
eval_d["ood_ox"] = eval_d.get("distribution_ox") or check_distribution(alloy_input, "oxidation_mass_gain")
declared_distribution_targets = {
    item.get("target") for item in tool_trace if item.get("tool") == "check_distribution"
}
if "yield_strength" not in declared_distribution_targets:
    tool_trace.append(
        {
            "step": "强度分布检查",
            "tool": "check_distribution",
            "purpose": "检查强度模型输入是否接近或超出训练集范围",
            "target": "yield_strength",
        }
    )
if "oxidation_mass_gain" not in declared_distribution_targets:
    tool_trace.append(
        {
            "step": "氧化分布检查",
            "tool": "check_distribution",
            "purpose": "检查氧化模型输入是否接近或超出训练集范围",
            "target": "oxidation_mass_gain",
        }
    )
for c in opt_d.get("candidates", []) if opt_d else []:
    c["ood"] = check_distribution(alloy_input, "oxidation_mass_gain")

_display_tool_trace(tool_trace)

# Big numbers
st.subheader("当前合金预测")
metric_cols = st.columns(2)
with metric_cols[0]:
    st.metric(
        "屈服强度 (YS)",
        f"{eval_d['strength']['value']} {eval_d['strength']['unit']}",
        delta=None,
    )
with metric_cols[1]:
    st.metric(
        "氧化增重 (Mass gain)",
        f"{eval_d['oxidation']['value']} {eval_d['oxidation']['unit']}",
        delta=None,
    )

# One-sentence summary
st.success(run_summary)

# Distribution warnings (emoji-coded: 🟢 in-range / 🟡 near boundary / 🔴 out-of-range)
def _status_icon(check: dict) -> str:
    if check.get("ood_features"):
        return "🔴"
    if check.get("near_bound_features"):
        return "🟡"
    return "🟢"


if eval_d.get("ood_ys"):
    icon = _status_icon(eval_d["ood_ys"])
    msg = eval_d["ood_ys"]["warning"] or "全部特征在训练集范围内"
    st.warning(f"{icon} YS 训练集范围提示:{msg}")
if eval_d.get("ood_ox"):
    icon = _status_icon(eval_d["ood_ox"])
    msg = eval_d["ood_ox"]["warning"] or "全部特征在训练集范围内"
    st.warning(f"{icon} Oxidation 训练集范围提示:{msg}")

# SHAP bar charts
if show_shap:
    st.subheader("SHAP 解释 (top features)")
    shap_cols = st.columns(2)
    with shap_cols[0]:
        st.markdown("**YS 解释**")
        ys_features = eval_d["strength_shap"]["top_features"]
        if ys_features:
            ys_df = {
                "feature": [f["feature"] for f in ys_features],
                "value": [f.get("value", 0) for f in ys_features],
            }
            st.bar_chart(data=ys_df, x="feature", y="value", height=300)
        else:
            st.caption("无 SHAP 数据")
    with shap_cols[1]:
        st.markdown("**Oxidation 解释**")
        ox_features = eval_d["oxidation_shap"]["top_features"]
        if ox_features:
            ox_df = {
                "feature": [f["feature"] for f in ox_features],
                "value": [f.get("value", 0) for f in ox_features],
            }
            st.bar_chart(data=ox_df, x="feature", y="value", height=300)
        else:
            st.caption("无 SHAP 数据")

# Pareto front + candidates
if opt_d and opt_d.get("candidates"):
    st.subheader("Pareto Front (NSGA-II)")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cands_x = [c["predicted_oxidation"] for c in opt_d["candidates"]]
    cands_y = [c["predicted_strength"] for c in opt_d["candidates"]]
    cur_x = eval_d["oxidation"]["value"]
    cur_y = eval_d["strength"]["value"]
    ax.scatter(cands_x, cands_y, c="#4a90c2", s=80, label="Pareto 候选", edgecolors="white")
    ax.scatter([cur_x], [cur_y], c="#c0392b", s=280, marker="*",
               label="当前合金", edgecolors="black", linewidth=0.5, zorder=5)
    ax.set_xlim(padded_axis_limits([*cands_x, cur_x], min_span=4.0, nonnegative=True))
    ax.set_ylim(padded_axis_limits([*cands_y, cur_y], min_span=500.0))
    ax.set_xlabel("Oxidation mass gain (mg/cm²)")
    ax.set_ylabel("Yield strength (MPa)")
    ax.set_title(f"NSGA-II Pareto front ({len(opt_d['candidates'])} 候选)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    st.caption("注：坐标轴已按当前合金与 Pareto 候选自动扩展，仅改变显示比例，不改变预测值。")
    st.pyplot(fig)

    st.subheader("候选详情")
    rows = []
    candidate_columns = ["Co", "Ni", "Al", "Cr", "Ta", "Ti", "W", "V", "Nb", "Mo", "Vol"]
    for c in opt_d["candidates"]:
        row = {
            "Rank": c["rank"],
            "YS (MPa)": c["predicted_strength"],
            "YS gain": round(c["predicted_strength"] - eval_d["strength"]["value"], 2),
            "Oxidation (mg/cm²)": c["predicted_oxidation"],
            "Oxidation reduction": round(eval_d["oxidation"]["value"] - c["predicted_oxidation"], 3),
        }
        row.update({element: c["composition"].get(element, "") for element in candidate_columns})
        rows.append(row)
    st.table(rows)
else:
    st.caption("未运行 NSGA-II 优化(include_optimization=False)。")
