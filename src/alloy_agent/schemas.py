from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Union


Number = Union[int, float]
CompositionValue = Union[Number, str]

VALID_OBJECTIVE_KEYS: frozenset[str] = frozenset({"maximize", "minimize"})

ShapEffect = Literal["positive", "negative", "risk"]


@dataclass(frozen=True)
class AlloyInput:
    composition: dict[str, CompositionValue]
    processing: dict[str, Number]
    test_conditions: dict[str, Number]
    # Microstructure features (e.g. {"Vγ′": 70.0}) used by the real YS model.
    # Default empty dict keeps existing call sites that don't supply it working.
    microstructure: dict[str, Number] = field(default_factory=dict)


@dataclass(frozen=True)
class StrengthPrediction:
    value: float | None
    unit: str
    source: Literal["mock", "real_model"]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OxidationPrediction:
    value: float | None
    unit: str
    source: Literal["mock", "real_model"]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShapExplanation:
    target: str
    top_features: list[dict[str, Any]]
    summary: str
    source: Literal["mock", "real_model"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizationRequest:
    objectives: dict[str, list[str]]
    constraints: dict[str, Number]
    composition_bounds: dict[str, list[Number]]
    processing: dict[str, Number]
    test_conditions: dict[str, Number]

    def __post_init__(self) -> None:
        unknown = set(self.objectives.keys()) - VALID_OBJECTIVE_KEYS
        if unknown:
            raise ValueError(
                f"objectives 只能包含 {sorted(VALID_OBJECTIVE_KEYS)},收到未知键 {sorted(unknown)}"
            )
        for feature, bounds in self.composition_bounds.items():
            if not (isinstance(bounds, list) and len(bounds) == 2):
                raise ValueError(
                    f"composition_bounds[{feature}] 必须是 2 元 list,收到 {bounds!r}"
                )
            lo, hi = bounds
            if lo > hi:
                raise ValueError(
                    f"composition_bounds[{feature}] 下界 {lo} 大于上界 {hi}"
                )


@dataclass(frozen=True)
class AlloyCandidate:
    composition: dict[str, CompositionValue]
    predicted_strength: float | None
    predicted_oxidation: float | None
    rank: int
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentRequest:
    mode: Literal["evaluate", "optimize", "full"]
    alloy_input: AlloyInput | None = None
    optimization_request: OptimizationRequest | None = None
    # Used by mode="full": if False, skip the optimize step (only evaluate + SHAP).
    include_optimization: bool = True
    # Used by mode="full": "local" for Agent-side nearby search, "script" for
    # the original collaborator NSGA-II bounds.
    search_space: Literal["local", "script"] = "local"


@dataclass(frozen=True)
class AgentResponse:
    mode: str
    result: dict[str, Any]
    report: str


@dataclass(frozen=True)
class FullResult:
    """Composite result for mode='full': evaluate the input alloy + run NSGA-II
    to find better candidates, then summarize both."""
    evaluation: dict[str, Any]
    optimization: dict[str, Any] | None
    summary: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation": self.evaluation,
            "optimization": self.optimization,
            "summary": self.summary,
            "tool_trace": self.tool_trace,
        }
