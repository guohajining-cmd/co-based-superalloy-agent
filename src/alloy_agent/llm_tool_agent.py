"""Optional LLM tool-calling layer for the alloy design agent.

This module is deliberately additive: if no OpenAI client/API key is available,
it falls back to the rule-based agent loop so the demo remains runnable.
"""

from __future__ import annotations

import json
import os
from typing import Any

from alloy_agent.agent import run_agent
from alloy_agent.agent_loop import run_agent_loop
from alloy_agent.fixtures import make_default_alloy_input
from alloy_agent.schemas import AgentRequest, AlloyInput
from alloy_agent.state import AgentState
from alloy_agent.tool_trace import tool_call


DEFAULT_LLM_MODEL = "gpt-5.5"


LLM_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "evaluate_alloy",
        "description": (
            "Evaluate one existing cobalt-based superalloy. Use when the user asks "
            "for performance prediction, evaluation, explanation, SHAP, or reliability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "composition": {"type": "object", "additionalProperties": {"type": "number"}},
                "processing": {"type": "object", "additionalProperties": {"type": "number"}},
                "test_conditions": {"type": "object", "additionalProperties": {"type": "number"}},
                "microstructure": {"type": "object", "additionalProperties": {"type": "number"}},
            },
            "required": ["composition"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "full_alloy_analysis",
        "description": (
            "Evaluate the current alloy and run NSGA-II optimization. Use when the "
            "user asks for design, recommendation, optimization, candidates, or Pareto solutions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "composition": {"type": "object", "additionalProperties": {"type": "number"}},
                "processing": {"type": "object", "additionalProperties": {"type": "number"}},
                "test_conditions": {"type": "object", "additionalProperties": {"type": "number"}},
                "microstructure": {"type": "object", "additionalProperties": {"type": "number"}},
                "include_optimization": {"type": "boolean"},
                "search_space": {"type": "string", "enum": ["local", "script"]},
            },
            "required": ["composition"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "ask_for_missing_input",
        "description": (
            "Pause the agent and ask the user for missing required information. "
            "Use this when alloy composition is missing or the request is too incomplete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "missing_fields": {"type": "array", "items": {"type": "string"}},
                "question": {"type": "string"},
            },
            "required": ["missing_fields", "question"],
            "additionalProperties": False,
        },
    },
]


def run_llm_tool_agent(
    user_text: str,
    *,
    client: Any | None = None,
    model: str | None = None,
    api_key: str | None = None,
    search_space: str = "local",
    max_tool_rounds: int = 2,
) -> AgentState:
    """Run the LLM tool-calling layer, falling back to the rule loop if needed."""
    selected_model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_LLM_MODEL
    llm_client, unavailable_reason = _resolve_client(client=client, api_key=api_key)
    if llm_client is None:
        fallback = run_agent_loop(user_text, search_space=search_space)
        fallback.record_decision(
            "llm_fallback_to_rule_loop",
            "LLM tool-calling 未配置，自动使用规则 Agent Loop",
        )
        fallback.warnings.insert(0, f"LLM tool-calling 未配置：{unavailable_reason}")
        return fallback

    state = AgentState(user_text=user_text, search_space=search_space)
    state.record_decision("llm_request", "把用户任务交给 LLM，由模型选择合适的工具")
    state.tool_trace.append(
        tool_call(
            "LLM 工具规划",
            "llm_tool_calling",
            "使用 LLM 根据自然语言选择材料设计工具",
        )
    )

    first_response = llm_client.responses.create(
        model=selected_model,
        input=[
            {"role": "developer", "content": _developer_prompt(search_space)},
            {"role": "user", "content": user_text},
        ],
        tools=LLM_TOOL_DEFINITIONS,
        tool_choice="auto",
    )

    response = first_response
    for _ in range(max_tool_rounds):
        calls = _function_calls(response)
        if not calls:
            state.report = _response_text(response)
            state.status = "completed"
            state.record_decision("llm_final_response", "LLM 未再请求工具，输出最终回复")
            return state

        call = calls[0]
        state.record_decision(
            f"llm_tool_call:{call['name']}",
            "LLM 选择了一个白名单工具",
        )
        state.tool_trace.append(
            tool_call(
                "LLM 选择工具",
                call["name"],
                "LLM 根据用户任务选择本地材料设计 tool",
            )
        )
        tool_output = _execute_llm_tool(call["name"], call["arguments"], state, search_space)
        if state.status == "waiting_for_input":
            return state

        state.record_decision(
            f"execute_tool:{call['name']}",
            "执行 LLM 选择的本地工具并把结果返回给模型",
        )
        response_id = _field(response, "id")
        response = llm_client.responses.create(
            model=selected_model,
            previous_response_id=response_id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps(tool_output, ensure_ascii=False),
                }
            ],
        )

    state.status = "failed"
    state.warnings.append("LLM tool-calling 超过最大工具轮数，已停止。")
    return state


def _developer_prompt(search_space: str) -> str:
    return (
        "你是钴基高温合金设计 Agent 的工具调度器。"
        "只能通过提供的 tools 处理任务，不要虚构模型结果。"
        "如果用户缺少合金成分，调用 ask_for_missing_input。"
        "如果用户要评估、预测、解释或可靠性分析，调用 evaluate_alloy。"
        "如果用户要优化、推荐候选合金、设计新合金或 NSGA-II/Pareto，调用 full_alloy_analysis。"
        f"默认 NSGA-II search_space 为 {search_space}。"
    )


def _resolve_client(client: Any | None, api_key: str | None) -> tuple[Any | None, str]:
    if client is not None:
        return client, ""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None, "缺少 OPENAI_API_KEY"
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        return None, f"未安装 openai Python 包 ({exc})"
    return OpenAI(api_key=key), ""


def _execute_llm_tool(
    name: str,
    arguments: dict[str, Any],
    state: AgentState,
    default_search_space: str,
) -> dict[str, Any]:
    if name == "ask_for_missing_input":
        state.status = "waiting_for_input"
        state.missing_fields = list(arguments.get("missing_fields") or [])
        state.pending_question = str(arguments.get("question") or "请补充必要输入。")
        return {
            "status": state.status,
            "missing_fields": state.missing_fields,
            "question": state.pending_question,
        }

    if name == "evaluate_alloy":
        alloy = _alloy_from_arguments(arguments)
        response = run_agent(AgentRequest(mode="evaluate", alloy_input=alloy))
        state.intent = "evaluate"
        state.alloy_input = alloy
        state.result = {"mode": response.mode, "result": response.result, "report": response.report}
        state.report = response.report
        state.tool_trace.extend(response.result.get("tool_trace", []))
        return state.result

    if name == "full_alloy_analysis":
        alloy = _alloy_from_arguments(arguments)
        requested_search_space = arguments.get("search_space") or default_search_space
        search_space = requested_search_space if requested_search_space in {"local", "script"} else "local"
        include_optimization = bool(arguments.get("include_optimization", True))
        response = run_agent(
            AgentRequest(
                mode="full",
                alloy_input=alloy,
                include_optimization=include_optimization,
                search_space=search_space,
            )
        )
        state.intent = "full"
        state.alloy_input = alloy
        state.include_optimization = include_optimization
        state.search_space = search_space
        state.result = {"mode": response.mode, "result": response.result, "report": response.report}
        state.report = response.report
        state.tool_trace.extend(response.result.get("tool_trace", []))
        return state.result

    state.status = "failed"
    state.warnings.append(f"LLM 请求了未注册工具: {name}")
    return {"status": "failed", "error": state.warnings[-1]}


def _alloy_from_arguments(arguments: dict[str, Any]) -> AlloyInput:
    default = make_default_alloy_input()
    composition = _balanced_composition(arguments.get("composition") or {})
    processing = dict(default.processing)
    processing.update(_numeric_dict(arguments.get("processing") or {}))
    test_conditions = dict(default.test_conditions)
    test_conditions.update(_numeric_dict(arguments.get("test_conditions") or {}))
    microstructure = dict(default.microstructure)
    microstructure.update(_numeric_dict(arguments.get("microstructure") or {}))
    return AlloyInput(
        composition=composition,
        processing=processing,
        test_conditions=test_conditions,
        microstructure=microstructure,
    )


def _balanced_composition(raw: dict[str, Any]) -> dict[str, float]:
    composition = _numeric_dict(raw)
    non_co_total = sum(value for element, value in composition.items() if element != "Co")
    if "Co" not in composition or composition.get("Co", 0.0) <= 0:
        composition["Co"] = max(0.0, 100.0 - non_co_total)
    return composition


def _numeric_dict(raw: dict[str, Any]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            clean[str(key)] = float(value)
    return clean


def _function_calls(response: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _field(response, "output") or []:
        item_type = _field(item, "type")
        if item_type != "function_call":
            continue
        raw_arguments = _field(item, "arguments") or "{}"
        calls.append(
            {
                "call_id": _field(item, "call_id") or _field(item, "id") or "",
                "name": _field(item, "name") or "",
                "arguments": json.loads(raw_arguments),
            }
        )
    return calls


def _response_text(response: Any) -> str:
    text = _field(response, "output_text")
    if text:
        return str(text)
    parts: list[str] = []
    for item in _field(response, "output") or []:
        if _field(item, "type") != "message":
            continue
        for content in _field(item, "content") or []:
            content_type = _field(content, "type")
            if content_type in {"output_text", "text"}:
                value = _field(content, "text")
                if value:
                    parts.append(str(value))
    return "\n".join(parts)


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
