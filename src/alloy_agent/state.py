"""State container for the rule-based agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from alloy_agent.schemas import AlloyInput


AgentStatus = Literal["running", "waiting_for_input", "completed", "failed"]


@dataclass
class AgentState:
    """Mutable state passed through the rule-based agent loop."""

    user_text: str
    status: AgentStatus = "running"
    intent: str | None = None
    alloy_input: AlloyInput | None = None
    include_optimization: bool = False
    search_space: str = "local"
    input_validated: bool = False
    result_validated: bool = False
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    decision_trace: list[dict[str, Any]] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    report: str | None = None
    pending_question: str | None = None

    def record_decision(self, action: str, reason: str) -> None:
        self.decision_trace.append({"action": action, "reason": reason})

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_text": self.user_text,
            "status": self.status,
            "intent": self.intent,
            "include_optimization": self.include_optimization,
            "search_space": self.search_space,
            "missing_fields": self.missing_fields,
            "warnings": self.warnings,
            "decision_trace": self.decision_trace,
            "tool_trace": self.tool_trace,
            "result": self.result,
            "report": self.report,
            "pending_question": self.pending_question,
        }
