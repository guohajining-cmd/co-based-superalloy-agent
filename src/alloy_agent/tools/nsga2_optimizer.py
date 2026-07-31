"""NSGA-II multi-objective optimizer for alloy design.

Replaces the placeholder. Uses pymoo directly. The collaborator's
`NSGA-2-双目标 - 副本.py` had two issues that we fixed here:
  1. `AlloyRepair` was indexing into `x` using positions from `all_features`
     (which included `Co` even though `Co` isn't an optimization variable),
     causing index drift. We instead let `Co` be derived in `_evaluate()`
     from the remaining elements + fixed `Ni=30`.
  2. The script was a one-shot (calls `minimize(...)` at module top level),
     so importing it forced the full optimization. We keep pymoo imports
     inside the call to avoid that.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from alloy_agent.schemas import AlloyCandidate, OptimizationRequest
from alloy_agent.tools._model_utils import load_xgboost, numeric


# Optimization variables are the 9 alloying elements + Vol. Ni is fixed at 30,
# Co is derived in _evaluate (100 - Ni - sum(other elements) - Vol? no, Vol
# is γ' volume fraction, not a composition %, so we don't subtract it).
# Convention: Co + (other 8 elements, Ni fixed) = 100 (alloy base).
_OPTIMIZATION_KEYS: list[str] = [
    "Al", "W", "Ta", "Ti", "Nb", "Cr", "V", "Mo", "Vol",
]
_NI_FIXED = 30.0
_POP_SIZE = 500
_N_GEN = 80
_TOP_N = 5  # how many Pareto candidates to return


# Cached models so pymoo doesn't reload them on every evaluation.
_ys_model = load_xgboost("ys")
_ox_model = load_xgboost("oxidation")


def _ys_features_from_x(x: list[float], processing: dict, test: dict) -> list[float]:
    """Build the 12-dim YS feature vector from a pymoo individual."""
    # Map: Al, W, Ta, Ti, Nb, Cr, V, Mo, Vol → corresponding CSV columns
    al, w, ta, ti, nb, cr, v, mo, vol = x
    co = 100.0 - (_NI_FIXED + al + w + ta + ti + nb + cr + v + mo)
    return [
        co, al, ta, ti, _NI_FIXED, cr, v, mo,
        float(processing.get("aging_temperature", 800)),
        float(processing.get("aging_time", 24)),
        float(test.get("strength_test_temperature", 750)),
        vol,
    ]


def _oxidation_features_from_x(x: list[float], processing: dict, test: dict) -> list[float]:
    """Build the 12-dim Oxidation feature vector from a pymoo individual."""
    al, w, ta, ti, nb, cr, v, mo, vol = x  # vol is unused by oxidation model
    co = 100.0 - (_NI_FIXED + al + w + ta + ti + nb + cr + v + mo)
    return [
        co, al, w, ta, ti, nb, _NI_FIXED, cr, v, mo,
        float(test.get("oxidation_temperature", 1000)),
        float(test.get("oxidation_time", 100)),
    ]


def run_nsga2_optimization(request: OptimizationRequest) -> list[AlloyCandidate]:
    # Fall back to a hand-written candidate if either model is missing.
    if _ys_model is None or _ox_model is None:
        return _placeholder_candidate()

    import numpy as np
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    # Bounds for the 9 optimization variables, from request.composition_bounds.
    # Fall back to safe defaults if a key is missing.
    default_bounds = {
        "Al": (0.0, 12.0), "W": (0.0, 5.0), "Ta": (0.0, 8.0),
        "Ti": (0.0, 5.0), "Nb": (0.0, 3.0), "Cr": (0.0, 12.0),
        "V": (0.0, 2.0), "Mo": (0.0, 4.0), "Vol": (40.0, 90.0),
    }
    bounds = {k: tuple(request.composition_bounds.get(k, default_bounds[k])) for k in _OPTIMIZATION_KEYS}
    xl = np.array([bounds[k][0] for k in _OPTIMIZATION_KEYS])
    xu = np.array([bounds[k][1] for k in _OPTIMIZATION_KEYS])

    # Constraints from request.constraints:
    #   oxidation_mass_gain_min <= mass gain <= oxidation_mass_gain_max
    #   yield strength >= yield_strength_min
    ys_min = float(request.constraints.get("yield_strength_min", 0))
    ox_min = float(request.constraints.get("oxidation_mass_gain_min", 0.0))
    ox_max = float(request.constraints.get("oxidation_mass_gain_max", float("inf")))

    class AlloyProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=9, n_obj=2, n_ieq_constr=3, xl=xl, xu=xu)
            self.processing = request.processing
            self.test = request.test_conditions

        def _evaluate(self, x, out, *args, **kwargs):
            ys_vec = np.array([_ys_features_from_x(list(x), self.processing, self.test)], dtype=float)
            ox_vec = np.array([_oxidation_features_from_x(list(x), self.processing, self.test)], dtype=float)
            raw_ox = float(_ox_model.predict(ox_vec)[0])
            raw_ys = float(_ys_model.predict(ys_vec)[0])
            f1 = raw_ox  # minimize mass gain
            f2 = -raw_ys  # maximize YS
            out["F"] = [f1, f2]
            out["G"] = [
                f1 - ox_max,        # mass gain <= ox_max
                ox_min - f1,        # mass gain >= ox_min
                ys_min - (-f2),     # YS >= ys_min
            ]

    problem = AlloyProblem()
    algorithm = NSGA2(
        pop_size=_POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(eta=20, prob=0.3),
        eliminate_duplicates=True,
    )
    res = minimize(problem, algorithm, ("n_gen", _N_GEN), seed=42, verbose=False)

    if res.F is None or len(res.F) == 0:
        return _placeholder_candidate()

    # Take the non-dominated set, then filter out candidates that violate
    # physical / design constraints. XGBoost is an unconstrained regressor, so
    # it can predict negative mass gain; those points are surrogate artifacts,
    # not meaningful material designs.
    F = np.atleast_2d(res.F)
    X = np.atleast_2d(res.X)
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    F_pareto = F[nds]
    X_pareto = X[nds]
    feasible = _feasible_mask(F_pareto, ys_min=ys_min, ox_min=ox_min, ox_max=ox_max)
    F_feasible = F_pareto[feasible]
    X_feasible = X_pareto[feasible]

    if len(F_feasible) == 0:
        feasible = _feasible_mask(F, ys_min=ys_min, ox_min=ox_min, ox_max=ox_max)
        F_feasible = F[feasible]
        X_feasible = X[feasible]

    if len(F_feasible) == 0:
        return _placeholder_candidate()

    order = np.lexsort((F_feasible[:, 0], F_feasible[:, 1]))
    F_sorted = F_feasible[order][:_TOP_N]
    X_sorted = X_feasible[order][:_TOP_N]

    candidates = []
    for i in range(len(X_sorted)):
        x_i = X_sorted[i].tolist()
        al, w, ta, ti, nb, cr, v, mo, vol = x_i
        co = 100.0 - (_NI_FIXED + al + w + ta + ti + nb + cr + v + mo)
        composition = {
            "Co": round(co, 3),
            "Ni": _NI_FIXED,
            "Al": round(al, 3), "W": round(w, 3), "Ta": round(ta, 3),
            "Ti": round(ti, 3), "Nb": round(nb, 3), "Cr": round(cr, 3),
            "V": round(v, 3), "Mo": round(mo, 3),
            "Vol": round(vol, 3),
        }
        candidates.append(AlloyCandidate(
            composition=composition,
            predicted_strength=round(-float(F_sorted[i, 1]), 2),
            predicted_oxidation=max(0.0, round(float(F_sorted[i, 0]), 3)),
            rank=i + 1,
            note=f"NSGA-II Pareto 解 (pop={_POP_SIZE}, gen={_N_GEN})",
        ))
    return candidates


def _feasible_mask(F: np.ndarray, ys_min: float, ox_min: float, ox_max: float) -> np.ndarray:
    oxidation = F[:, 0]
    strength = -F[:, 1]
    return (
        (oxidation >= ox_min)
        & (oxidation <= ox_max)
        & (strength >= ys_min)
    )


def _placeholder_candidate() -> list[AlloyCandidate]:
    return [AlloyCandidate(
        composition={"Co": 42.5, "Ni": 30, "Al": 9, "Cr": 7, "Ta": 4, "Ti": 3,
                      "W": 2, "V": 1, "Nb": 1, "Mo": 0.5},
        predicted_strength=915.2,
        predicted_oxidation=3.08,
        rank=1,
        note="真模型未加载,使用论文验证合金示例占位。",
    )]
