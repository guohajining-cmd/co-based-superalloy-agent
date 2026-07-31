"""Rule planner for the first agent-loop implementation."""

from __future__ import annotations

from dataclasses import dataclass

from alloy_agent.state import AgentState


@dataclass(frozen=True)
class AgentAction:
    action: str
    reason: str


def decide_next_action(state: AgentState) -> AgentAction:
    """Choose the next loop action from observable state."""
    if state.intent is None:
        return AgentAction(
            "parse_user_request",
            "需要先把自然语言转成标准输入和任务模式",
        )
    if not state.input_validated:
        return AgentAction(
            "validate_required_inputs",
            "调用工具前需要确认输入是否完整",
        )
    if state.missing_fields:
        return AgentAction(
            "ask_user",
            "缺少必要输入，不能继续调用模型工具",
        )
    if state.result is None:
        return AgentAction(
            f"run_agent:{state.intent}",
            "输入完整，可以调用已有 workflow tools",
        )
    if not state.result_validated:
        return AgentAction(
            "validate_agent_result",
            "工具返回后需要检查结果是否存在明显风险",
        )
    return AgentAction(
        "finalize",
        "工具调用和结果自检均已完成",
    )
