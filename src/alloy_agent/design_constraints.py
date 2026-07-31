"""Design-space constraints for NSGA-II and the UI.

Centralised so every component (NSGA-II problem, fixture defaults, UI bounds
derivation) agrees on a single source of truth.
"""

from __future__ import annotations

from typing import Literal, Union

Number = Union[int, float]

# 9 optimisation variables (plus Co derived via balance).
# Delta controls how wide a window each element gets when deriving
# composition_bounds from an input alloy.
DESIGN_ELEMENTS: tuple[str, ...] = (
    "Ni", "Al", "Cr", "Ta", "Ti", "W", "V", "Nb", "Mo",
)

BALANCE_METAL: str = "Co"

SearchSpaceProfile = Literal["local", "script"]

# Local Agent profile: conservative search around the input alloy so the UI can
# demonstrate nearby improvement. This does not alter the collaborator's NSGA-II
# script; it is only an Agent-side option.
LOCAL_BOUNDS_DELTA: dict[str, float] = {
    "Co": 0.0,
    "Ni": 0.0,
    "Al": 1.0,
    "Cr": 1.0,
    "Ta": 1.0,
    "Ti": 1.0,
    "W": 1.0,
    "V": 0.5,
    "Nb": 0.5,
    "Mo": 0.5,
}

# Original collaborator-script profile. Values mirror
# Agent-acta/NSGA-2-双目标 - 副本.py design_variables.
SCRIPT_BOUNDS: dict[str, list[float]] = {
    "Al": [9.0, 10.0],
    "W": [0.0, 2.5],
    "Ta": [1.0, 4.0],
    "Ti": [2.0, 3.0],
    "Nb": [0.0, 2.0],
    "Ni": [30.0, 30.0],
    "Cr": [4.0, 12.0],
    "V": [0.0, 1.5],
    "Mo": [0.0, 2.5],
    "Vol": [70.0, 85.0],
}

# Backwards-compatible name used by tests/docs.
DEFAULT_BOUNDS_DELTA = LOCAL_BOUNDS_DELTA

# Constraints that apply to optimisation. The keys match
# OptimizationRequest.constraints.
DEFAULT_CONSTRAINTS: dict[str, float] = {
    "yield_strength_min": 800.0,
    "oxidation_mass_gain_min": 0.0,
    "oxidation_mass_gain_max": 3.0,
}

# NSGA-II objectives: target -> {"direction": "maximize"|"minimize",
# "description": human-readable, "unit": physical unit}.
OBJECTIVES: dict[str, dict[str, str]] = {
    "yield_strength": {
        "direction": "maximize",
        "description": "Co 基合金在目标温度下的屈服强度(典型 750°C)",
        "unit": "MPa",
    },
    "oxidation_mass_gain": {
        "direction": "minimize",
        "description": "在 1000°C / 100h 等效氧化条件下的单位面积增重",
        "unit": "mg/cm²",
    },
}

# What "good" looks like, in plain numbers. Shown to the user as a sanity
# reference; not enforced by the model.
TARGET_VALUE_HINTS: dict[str, dict[str, str]] = {
    "yield_strength": {"good": "≥ 900 MPa", "marginal": "700-900 MPa"},
    "oxidation_mass_gain": {"good": "≤ 2 mg/cm²", "marginal": "2-5 mg/cm²"},
}


def bounds_from_composition(
    composition: dict[str, Number],
    delta_map: dict[str, float] | None = None,
    profile: SearchSpaceProfile = "local",
) -> dict[str, list[float]]:
    """Derive NSGA-II composition_bounds from an input alloy.

    profile="local": Co/Ni fixed; other elements get [v - delta, v + delta].
    profile="script": use the original collaborator-script bounds.
    """
    if profile == "script":
        co_value = composition.get(BALANCE_METAL)
        out = {k: list(v) for k, v in SCRIPT_BOUNDS.items()}
        if isinstance(co_value, (int, float)):
            out[BALANCE_METAL] = [float(co_value), float(co_value)]
        return out
    if profile != "local":
        raise ValueError(f"unknown search-space profile: {profile!r}")

    delta_map = delta_map or DEFAULT_BOUNDS_DELTA
    out: dict[str, list[float]] = {}
    for key, v in composition.items():
        delta = delta_map.get(key, 0.0)
        if delta == 0.0:
            out[key] = [float(v), float(v)]
        else:
            lo = max(0.0, float(v) - delta)
            hi = min(100.0, float(v) + delta)
            out[key] = [lo, hi]
    return out


def balance_composition(composition: dict) -> dict:
    """Fill in the balance metal (Co by default) so element percentages sum to 100."""
    metal = BALANCE_METAL
    other_sum = sum(
        v for k, v in composition.items()
        if k != metal and isinstance(v, (int, float))
    )
    new = dict(composition)
    new[metal] = max(0.0, 100.0 - other_sum)
    return new
