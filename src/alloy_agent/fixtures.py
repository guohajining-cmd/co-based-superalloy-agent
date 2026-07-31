"""Default sample inputs shared by web UI, demo script, and tests.

The intent is for these factories to be the single source of truth for the
"what does a typical input look like" question. Callers should not duplicate
these dicts inline; if the shape changes, change it here once.
"""

from __future__ import annotations

from typing import Any, Literal

from alloy_agent.schemas import AlloyInput, OptimizationRequest


Mode = Literal["evaluate", "optimize", "full"]


def make_default_alloy_input() -> AlloyInput:
    """Reference alloy composition, processing, test conditions, and microstructure.

    Co is computed as `100 - sum(other elements)` to match the collaborator's
    NSGA-II `AlloyRepair` mechanism — the real XGBoost model needs Co as a
    numeric feature, not a "Bal." string.
    """
    elements_other_than_co = {
        "Ni": 30,
        "Al": 9,
        "Cr": 7,
        "Ta": 4,
        "Ti": 3,
        "W": 2,
        "V": 1,
        "Nb": 1,
        "Mo": 0.5,
    }
    co_value = max(0.0, 100.0 - sum(elements_other_than_co.values()))
    composition = {"Co": co_value, **elements_other_than_co}

    return AlloyInput(
        composition=composition,
        processing={
            "solution_temperature": 1225,
            "solution_time": 24,
            "aging_temperature": 800,
            "aging_time": 24,
        },
        test_conditions={
            "strength_test_temperature": 750,
            "oxidation_temperature": 1000,
            "oxidation_time": 100,
        },
        # Vγ′ (γ' volume fraction) is the 12th feature of the real YS model.
        # Value 75.0 is a typical Co-based superalloy γ' fraction; verify
        # against the real training set statistics before relying on it.
        microstructure={"Vγ′": 75.0},
    )


def make_default_optimization_request() -> OptimizationRequest:
    """Reference optimization request — maximizes strength, minimizes oxidation."""
    return OptimizationRequest(
        objectives={
            "maximize": ["yield_strength"],
            "minimize": ["oxidation_mass_gain"],
        },
        constraints={
            "yield_strength_min": 800,
            "oxidation_mass_gain_max": 3.0,
        },
        composition_bounds={
            "Ni": [30, 30],
            "Al": [9, 10],
            "Cr": [4, 7],
            "Ta": [1, 4],
            "Ti": [2, 3],
            "W": [0, 2],
            "V": [0, 1.5],
            "Nb": [0, 2],
            "Mo": [0, 2.5],
        },
        processing={
            "aging_temperature": 800,
            "aging_time": 24,
        },
        test_conditions={
            "strength_test_temperature": 750,
            "oxidation_temperature": 1000,
            "oxidation_time": 100,
        },
    )


def make_default_payload(mode: Mode) -> dict[str, Any]:
    """Build the dict shape the web UI textarea displays."""
    if mode == "evaluate":
        alloy = make_default_alloy_input()
        return {
            "mode": "evaluate",
            "alloy_input": {
                "composition": dict(alloy.composition),
                "processing": dict(alloy.processing),
                "test_conditions": dict(alloy.test_conditions),
            },
        }
    if mode == "optimize":
        request = make_default_optimization_request()
        return {
            "mode": "optimize",
            "optimization_request": {
                "objectives": dict(request.objectives),
                "constraints": dict(request.constraints),
                "composition_bounds": {k: list(v) for k, v in request.composition_bounds.items()},
                "processing": dict(request.processing),
                "test_conditions": dict(request.test_conditions),
            },
        }
    if mode == "full":
        alloy = make_default_alloy_input()
        return {
            "mode": "full",
            "include_optimization": True,
            "search_space": "local",
            "alloy_input": {
                "composition": dict(alloy.composition),
                "processing": dict(alloy.processing),
                "test_conditions": dict(alloy.test_conditions),
                "microstructure": dict(alloy.microstructure),
            },
        }
    raise ValueError(f"unsupported mode: {mode!r}")
