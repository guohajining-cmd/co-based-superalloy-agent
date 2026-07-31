"""Rule-based Agent Loop built around the existing alloy workflows."""

from __future__ import annotations

from alloy_agent.agent import run_agent
from alloy_agent.natural_language import parse_user_request
from alloy_agent.planner import decide_next_action
from alloy_agent.schemas import AgentRequest
from alloy_agent.state import AgentState
from alloy_agent.tool_trace import tool_call
from alloy_agent.validators import validate_agent_result, validate_required_inputs


def run_agent_loop(
    user_text: str,
    *,
    search_space: str = "local",
    max_steps: int = 10,
) -> AgentState:
    """Run the first rule-based loop: parse, validate, call tools, self-check."""
    state = AgentState(user_text=user_text, search_space=search_space)

    for _ in range(max_steps):
        action = decide_next_action(state)
        state.record_decision(action.action, action.reason)

        if action.action == "parse_user_request":
            parsed = parse_user_request(state.user_text)
            state.intent = parsed.mode
            state.alloy_input = parsed.alloy_input
            state.include_optimization = parsed.include_optimization
            state.warnings.extend(parsed.warnings)
            state.tool_trace.append(
                tool_call(
                    "自然语言解析",
                    "parse_user_request",
                    "自然语言转标准 AlloyInput 和运行模式",
                )
            )
            continue

        if action.action == "validate_required_inputs":
            state.missing_fields = validate_required_inputs(state)
            state.input_validated = True
            state.tool_trace.append(
                tool_call(
                    "输入完整性检查",
                    "validate_required_inputs",
                    "判断调用模型前是否缺少必要输入",
                )
            )
            continue

        if action.action == "ask_user":
            state.status = "waiting_for_input"
            state.pending_question = _build_pending_question(state.missing_fields)
            break

        if action.action.startswith("run_agent:"):
            if state.alloy_input is None or state.intent is None:
                state.status = "failed"
                state.warnings.append("Agent 内部状态缺少 alloy_input 或 intent。")
                break
            response = run_agent(
                AgentRequest(
                    mode=state.intent,
                    alloy_input=state.alloy_input,
                    include_optimization=state.include_optimization,
                    search_space=state.search_space,  # type: ignore[arg-type]
                )
            )
            state.result = {
                "mode": response.mode,
                "result": response.result,
                "report": response.report,
            }
            state.report = response.report
            state.tool_trace.extend(response.result.get("tool_trace", []))
            continue

        if action.action == "validate_agent_result":
            state.warnings.extend(validate_agent_result(state))
            state.result_validated = True
            state.tool_trace.append(
                tool_call(
                    "结果自检",
                    "validate_agent_result",
                    "检查预测和候选结果是否存在明显风险",
                )
            )
            continue

        if action.action == "finalize":
            state.status = "completed"
            break

    else:
        state.status = "failed"
        state.warnings.append("Agent loop 超过最大步数，已停止。")

    return state


def _build_pending_question(missing_fields: list[str]) -> str:
    if missing_fields == ["composition"]:
        return "请补充合金成分，例如 Co-30Ni-9Al-7Cr-4Ta-3Ti-2W-1V-1Nb-0.5Mo。"
    return "请补充必要输入：" + "、".join(missing_fields)
