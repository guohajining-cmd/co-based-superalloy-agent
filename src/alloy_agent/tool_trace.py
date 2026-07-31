from __future__ import annotations

from typing import Any


def tool_call(
    step: str,
    tool: str,
    purpose: str,
    target: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "step": step,
        "tool": tool,
        "purpose": purpose,
    }
    if target is not None:
        entry["target"] = target
    return entry
