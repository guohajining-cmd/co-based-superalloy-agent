from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from alloy_agent.fixtures import make_default_alloy_input
from alloy_agent.schemas import AlloyInput


Mode = Literal["evaluate", "full"]

_COMPOSITION_ELEMENTS = {"Co", "Ni", "Al", "Cr", "Ta", "Ti", "W", "V", "Nb", "Mo"}
_OPTIMIZE_WORDS = ("优化", "设计", "推荐", "候选", "NSGA", "Pareto", "帕累托")


@dataclass(frozen=True)
class ParsedUserRequest:
    mode: Mode
    alloy_input: AlloyInput
    include_optimization: bool
    warnings: list[str] = field(default_factory=list)


def parse_user_request(text: str) -> ParsedUserRequest:
    """Parse a Chinese natural-language alloy request into the agent input schema.

    This is intentionally a thin adapter, not a new prediction model. It extracts
    the composition and common experimental conditions, then lets existing tools
    do prediction, explanation, and optimization.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("自然语言输入不能为空")

    default_alloy = make_default_alloy_input()
    warnings: list[str] = []
    composition = _parse_composition(cleaned)
    if composition:
        composition = _with_balanced_co(composition)
    else:
        composition = dict(default_alloy.composition)
        warnings.append("没有识别到合金成分，已使用默认示例成分。")

    processing = dict(default_alloy.processing)
    processing.update(_parse_processing(cleaned))

    test_conditions = dict(default_alloy.test_conditions)
    test_conditions.update(_parse_test_conditions(cleaned))

    microstructure = dict(default_alloy.microstructure)
    microstructure.update(_parse_microstructure(cleaned))

    include_optimization = _looks_like_optimization(cleaned)
    mode: Mode = "full" if include_optimization else "evaluate"

    return ParsedUserRequest(
        mode=mode,
        alloy_input=AlloyInput(
            composition=composition,
            processing=processing,
            test_conditions=test_conditions,
            microstructure=microstructure,
        ),
        include_optimization=include_optimization,
        warnings=warnings,
    )


def _parse_composition(text: str) -> dict[str, float]:
    composition: dict[str, float] = {}
    # Co-30Ni-9Al and "Ni 30, Al 9" are both common in this project.
    for raw_value, element in re.findall(
        r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*[- ]?\s*(Co|Ni|Al|Cr|Ta|Ti|W|V|Nb|Mo)\b",
        text,
        flags=re.IGNORECASE,
    ):
        canonical = _canonical_element(element)
        if canonical in _COMPOSITION_ELEMENTS:
            composition[canonical] = float(raw_value)

    # A leading "Co-" usually means cobalt balance rather than Co=unknown.
    if "Co" in text and composition:
        composition.setdefault("Co", 0.0)
    return composition


def _with_balanced_co(composition: dict[str, float]) -> dict[str, float]:
    non_co_total = sum(value for element, value in composition.items() if element != "Co")
    if composition.get("Co", 0.0) <= 0:
        composition["Co"] = max(0.0, 100.0 - non_co_total)
    return composition


def _parse_test_conditions(text: str) -> dict[str, float]:
    conditions: dict[str, float] = {}

    strength_temp = _find_temperature_near(text, ("屈服", "强度", "测试", "YS"))
    if strength_temp is not None:
        conditions["strength_test_temperature"] = strength_temp

    oxidation_temp = _find_temperature_near(text, ("氧化", "oxidation", "Oxidation"))
    if oxidation_temp is not None:
        conditions["oxidation_temperature"] = oxidation_temp

    oxidation_time = _find_time_near(text, ("氧化", "oxidation", "Oxidation"))
    if oxidation_time is not None:
        conditions["oxidation_time"] = oxidation_time

    return conditions


def _parse_processing(text: str) -> dict[str, float]:
    processing: dict[str, float] = {}

    aging_temp = _find_temperature_near(text, ("时效", "aging", "Aging"))
    if aging_temp is not None:
        processing["aging_temperature"] = aging_temp

    aging_time = _find_time_near(text, ("时效", "aging", "Aging"))
    if aging_time is not None:
        processing["aging_time"] = aging_time

    solution_temp = _find_temperature_near(text, ("固溶", "solution", "Solution"))
    if solution_temp is not None:
        processing["solution_temperature"] = solution_temp

    solution_time = _find_time_near(text, ("固溶", "solution", "Solution"))
    if solution_time is not None:
        processing["solution_time"] = solution_time

    return processing


def _parse_microstructure(text: str) -> dict[str, float]:
    match = re.search(r"(?:Vγ′|Vγ'|Vol|γ′|gamma)\s*[:=为]?\s*(\d+(?:\.\d+)?)\s*%?", text, re.IGNORECASE)
    if not match:
        return {}
    return {"Vγ′": float(match.group(1))}


def _find_temperature_near(text: str, keywords: tuple[str, ...]) -> float | None:
    for keyword in keywords:
        escaped = re.escape(keyword)
        before = re.search(
            rf"(\d+(?:\.\d+)?)\s*(?:°C|℃|度|摄氏度)?[^，。；;\n]{{0,12}}{escaped}",
            text,
            re.IGNORECASE,
        )
        if before:
            return float(before.group(1))
        after = re.search(
            rf"{escaped}[^，。；;\n]{{0,12}}?(\d+(?:\.\d+)?)\s*(?:°C|℃|度|摄氏度)",
            text,
            re.IGNORECASE,
        )
        if after:
            return float(after.group(1))
    return None


def _find_time_near(text: str, keywords: tuple[str, ...]) -> float | None:
    for keyword in keywords:
        escaped = re.escape(keyword)
        before = re.search(
            rf"(\d+(?:\.\d+)?)\s*(?:h|小时)[^，。；;\n]{{0,12}}{escaped}",
            text,
            re.IGNORECASE,
        )
        if before:
            return float(before.group(1))
        after = re.search(
            rf"{escaped}[^，。；;\n]{{0,20}}?(\d+(?:\.\d+)?)\s*(?:h|小时)",
            text,
            re.IGNORECASE,
        )
        if after:
            return float(after.group(1))
    return None


def _looks_like_optimization(text: str) -> bool:
    return any(word.lower() in text.lower() for word in _OPTIMIZE_WORDS)


def _canonical_element(element: str) -> str:
    return element[0].upper() + element[1:].lower()
