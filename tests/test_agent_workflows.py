import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class AgentWorkflowTests(unittest.TestCase):
    def test_evaluate_workflow_returns_predictions_and_report(self):
        from alloy_agent.agent import run_agent
        from alloy_agent.fixtures import make_default_alloy_input
        from alloy_agent.schemas import AgentRequest

        response = run_agent(AgentRequest(mode="evaluate", alloy_input=make_default_alloy_input()))

        self.assertEqual(response.mode, "evaluate")
        # Both YS and oxidation are on real models now.
        self.assertEqual(response.result["strength"]["source"], "real_model")
        self.assertEqual(response.result["oxidation"]["source"], "real_model")
        self.assertIn("已有合金评估报告", response.report)
        self.assertIn("训练集范围对照", response.report)
        self.assertNotIn("非真实模型预测", response.report)

    def test_evaluate_workflow_declares_called_tools(self):
        from alloy_agent.agent import run_agent
        from alloy_agent.fixtures import make_default_alloy_input
        from alloy_agent.schemas import AgentRequest

        response = run_agent(AgentRequest(mode="evaluate", alloy_input=make_default_alloy_input()))
        trace = response.result["tool_trace"]
        # First entry is the agent dispatch itself; the rest is the workflow.
        workflow_trace = trace[1:]
        tools = [item["tool"] for item in workflow_trace]

        self.assertEqual(tools[0], "predict_yield_strength")
        self.assertIn("explain_with_shap", tools)
        self.assertEqual(tools[-1], "generate_evaluation_report")
        shap_entries = [t for t in workflow_trace if t["tool"] == "explain_with_shap"]
        distribution_entries = [t for t in workflow_trace if t["tool"] == "check_distribution"]
        self.assertEqual(len(shap_entries), 2)
        self.assertEqual(len(distribution_entries), 2)
        self.assertEqual(shap_entries[0]["target"], "yield_strength")
        self.assertEqual(shap_entries[1]["target"], "oxidation_mass_gain")
        self.assertEqual(distribution_entries[0]["target"], "yield_strength")
        self.assertEqual(distribution_entries[1]["target"], "oxidation_mass_gain")

    def test_optimize_workflow_returns_mock_candidates_and_report(self):
        from alloy_agent.agent import run_agent
        from alloy_agent.fixtures import make_default_optimization_request
        from alloy_agent.schemas import AgentRequest

        response = run_agent(
            AgentRequest(mode="optimize", optimization_request=make_default_optimization_request())
        )

        self.assertEqual(response.mode, "optimize")
        self.assertGreaterEqual(len(response.result["candidates"]), 1)
        self.assertEqual(response.result["candidates"][0]["rank"], 1)
        self.assertIn("候选合金", response.report)

    def test_optimize_workflow_declares_called_tools(self):
        from alloy_agent.agent import run_agent
        from alloy_agent.fixtures import make_default_optimization_request
        from alloy_agent.schemas import AgentRequest

        response = run_agent(
            AgentRequest(mode="optimize", optimization_request=make_default_optimization_request())
        )
        tools = [item["tool"] for item in response.result["tool_trace"][1:]]

        self.assertEqual(
            tools,
            [
                "run_nsga2_optimization",
                "predict_yield_strength",
                "predict_oxidation_mass_gain",
                "generate_optimization_report",
            ],
        )

    def test_optimization_request_rejects_malformed_bounds(self):
        from alloy_agent.schemas import OptimizationRequest

        with self.assertRaises(ValueError) as ctx:
            OptimizationRequest(
                objectives={},
                constraints={},
                composition_bounds={"Ni": [30]},
                processing={},
                test_conditions={},
            )
        self.assertIn("Ni", str(ctx.exception))

    def test_optimization_request_rejects_inverted_bounds(self):
        from alloy_agent.schemas import OptimizationRequest

        with self.assertRaises(ValueError) as ctx:
            OptimizationRequest(
                objectives={},
                constraints={},
                composition_bounds={"Ni": [40, 30]},
                processing={},
                test_conditions={},
            )
        self.assertIn("Ni", str(ctx.exception))
        self.assertIn("下界", str(ctx.exception))

    def test_optimization_request_rejects_unknown_objective_key(self):
        from alloy_agent.schemas import OptimizationRequest

        with self.assertRaises(ValueError) as ctx:
            OptimizationRequest(
                objectives={"max": ["yield_strength"]},
                constraints={},
                composition_bounds={},
                processing={},
                test_conditions={},
            )
        self.assertIn("max", str(ctx.exception))

    def test_evaluate_workflow_output_varies_with_input(self):
        from alloy_agent.agent import run_agent
        from alloy_agent.fixtures import make_default_alloy_input
        from alloy_agent.schemas import AgentRequest, AlloyInput

        baseline = run_agent(
            AgentRequest(mode="evaluate", alloy_input=make_default_alloy_input())
        )

        bumped_composition = dict(make_default_alloy_input().composition)
        bumped_composition["Al"] = 15
        bumped = AlloyInput(
            composition=bumped_composition,
            processing=make_default_alloy_input().processing,
            test_conditions=make_default_alloy_input().test_conditions,
        )
        bumped_response = run_agent(AgentRequest(mode="evaluate", alloy_input=bumped))

        # Real XGBoost should react to Al=15 just like the placeholder did.
        self.assertNotEqual(
            baseline.result["strength"]["value"],
            bumped_response.result["strength"]["value"],
            msg="Al 增加 6 个百分点,真 XGBoost 模型的 strength 也应该有可观察的差异",
        )

    def test_real_strength_model_loads_and_predicts_reasonable_value(self):
        """Sanity check: real XGBoost predicts a yield strength in the training
        range (her CV showed MAE ~45 MPa on values typically 600-1200 MPa).
        """
        from alloy_agent.tools.strength_model import _MODEL
        from alloy_agent.tools._model_utils import _YS_FEATURE_ORDER

        if _MODEL is None:
            self.skipTest("YS .pkl 未加载(可能在无 libomp 环境)")

        self.assertEqual(len(_YS_FEATURE_ORDER), 12)
        self.assertIn("Vol", _YS_FEATURE_ORDER)
        self.assertIn("Al", _YS_FEATURE_ORDER)

        from alloy_agent.fixtures import make_default_alloy_input
        from alloy_agent.tools.strength_model import predict_yield_strength

        result = predict_yield_strength(make_default_alloy_input())
        self.assertEqual(result.source, "real_model")
        # Her training set covers Co-based superalloys at 750°C;
        # reasonable YS range is 500-1500 MPa for this alloy class.
        self.assertGreater(result.value, 500)
        self.assertLess(result.value, 1500)

    def test_real_oxidation_model_loads_and_predicts_reasonable_value(self):
        """Sanity check: real XGBoost predicts a mass gain in the training range
        (her CV showed MAE ~0.85 mg/cm² on values typically 0-10).
        """
        from alloy_agent.tools.oxidation_model import _MODEL

        if _MODEL is None:
            self.skipTest("Oxidation .pkl 未加载")

        from alloy_agent.fixtures import make_default_alloy_input
        from alloy_agent.tools.oxidation_model import predict_oxidation_mass_gain

        result = predict_oxidation_mass_gain(make_default_alloy_input())
        self.assertEqual(result.source, "real_model")
        # Reasonable range: 0-15 mg/cm² for Co-based superalloys at 1000°C/100h.
        self.assertGreater(result.value, 0)
        self.assertLess(result.value, 15)

    def test_distribution_warning_reports_the_actual_near_bound(self):
        """Near-bound warnings should report the lower bound when a value is
        near the lower edge, not the larger upper bound."""
        from alloy_agent.fixtures import make_default_alloy_input
        from alloy_agent.tools.distribution_check import check_distribution

        result = check_distribution(make_default_alloy_input(), "yield_strength")
        mo_entries = [item for item in result["near_bound_features"] if item[0] == "Mo"]

        self.assertEqual(len(mo_entries), 1)
        self.assertEqual(mo_entries[0][2], 0.0)
        self.assertIn("Mo=0.500 接近训练集下界 0.000", result["warning"])

    def test_full_workflow_returns_evaluation_and_optimization(self):
        """mode='full' runs evaluate + NSGA-II in one call."""
        from alloy_agent.agent import run_agent
        from alloy_agent.fixtures import make_default_alloy_input
        from alloy_agent.schemas import AgentRequest

        response = run_agent(
            AgentRequest(mode="full", alloy_input=make_default_alloy_input())
        )
        self.assertEqual(response.mode, "full")
        self.assertIn("summary", response.result)
        self.assertIn("evaluation", response.result)
        self.assertIn("optimization", response.result)
        # Evaluation contains the same predictions as evaluate-only mode.
        self.assertIn("strength", response.result["evaluation"])
        self.assertIn("oxidation", response.result["evaluation"])
        # Optimization returns at least 1 Pareto candidate.
        self.assertGreaterEqual(len(response.result["optimization"]["candidates"]), 1)
        # Report combines both sections.
        self.assertIn("完整合金分析报告", response.report)
        self.assertIn("总览", response.report)
        self.assertIn("NSGA-II", response.report)

    def test_full_workflow_declares_evaluation_and_optimization_tools(self):
        """mode='full' should expose the complete tool call chain."""
        from alloy_agent.agent import run_agent
        from alloy_agent.fixtures import make_default_alloy_input
        from alloy_agent.schemas import AgentRequest

        response = run_agent(
            AgentRequest(mode="full", alloy_input=make_default_alloy_input())
        )
        tools = [item["tool"] for item in response.result["tool_trace"]]

        self.assertIn("predict_yield_strength", tools)
        self.assertIn("predict_oxidation_mass_gain", tools)
        self.assertIn("_build_optimization_request", tools)
        self.assertIn("run_nsga2_optimization", tools)
        self.assertEqual(tools[-1], "generate_full_report")

    def test_full_workflow_can_skip_optimization(self):
        """include_optimization=False: only evaluate, no NSGA-II."""
        from alloy_agent.agent import run_agent
        from alloy_agent.fixtures import make_default_alloy_input
        from alloy_agent.schemas import AgentRequest

        response = run_agent(
            AgentRequest(
                mode="full",
                alloy_input=make_default_alloy_input(),
                include_optimization=False,
            )
        )
        self.assertEqual(response.mode, "full")
        self.assertIsNone(response.result["optimization"])
        # Summary should still have the evaluation head.
        self.assertIn("屈服强度", response.result["summary"])
        self.assertNotIn("NSGA-II", response.report)

    def test_full_workflow_derives_bounds_from_composition(self):
        """When no composition_bounds are provided, full mode derives them
        from the input alloy (±delta per element)."""
        from alloy_agent.workflows.full import _build_optimization_request
        from alloy_agent.fixtures import make_default_alloy_input

        alloy = make_default_alloy_input()
        req = _build_optimization_request(alloy)
        # Ni and Co should be fixed (single-point ranges).
        self.assertEqual(req.composition_bounds.get("Ni"), [30.0, 30.0])
        self.assertEqual(req.composition_bounds.get("Co"), [42.5, 42.5])
        # Al should be a range, not single-point.
        al_bounds = req.composition_bounds["Al"]
        self.assertEqual(len(al_bounds), 2)
        self.assertLess(al_bounds[0], al_bounds[1])
        self.assertEqual(req.composition_bounds["Al"], [8.0, 10.0])
        self.assertEqual(req.composition_bounds["Cr"], [6.0, 8.0])
        self.assertEqual(req.composition_bounds["Ta"], [3.0, 5.0])
        self.assertEqual(req.composition_bounds["Ti"], [2.0, 4.0])
        self.assertEqual(req.composition_bounds["W"], [1.0, 3.0])
        self.assertEqual(req.composition_bounds["V"], [0.5, 1.5])
        self.assertEqual(req.composition_bounds["Nb"], [0.5, 1.5])
        self.assertEqual(req.composition_bounds["Mo"], [0.0, 1.0])
        self.assertEqual(req.constraints["yield_strength_min"], 800.0)
        self.assertEqual(req.constraints["oxidation_mass_gain_max"], 3.0)
        self.assertEqual(req.constraints["oxidation_mass_gain_min"], 0.0)
        self.assertEqual(req.composition_bounds["Vol"], [75.0, 75.0])

    def test_full_workflow_can_use_original_script_search_space(self):
        """Agent can select the original NSGA-II script bounds without editing
        the collaborator's script."""
        from alloy_agent.workflows.full import _build_optimization_request
        from alloy_agent.fixtures import make_default_alloy_input

        req = _build_optimization_request(
            make_default_alloy_input(),
            search_space="script",
        )

        self.assertEqual(req.composition_bounds["Al"], [9.0, 10.0])
        self.assertEqual(req.composition_bounds["W"], [0.0, 2.5])
        self.assertEqual(req.composition_bounds["Ta"], [1.0, 4.0])
        self.assertEqual(req.composition_bounds["Ti"], [2.0, 3.0])
        self.assertEqual(req.composition_bounds["Nb"], [0.0, 2.0])
        self.assertEqual(req.composition_bounds["Ni"], [30.0, 30.0])
        self.assertEqual(req.composition_bounds["Cr"], [4.0, 12.0])
        self.assertEqual(req.composition_bounds["V"], [0.0, 1.5])
        self.assertEqual(req.composition_bounds["Mo"], [0.0, 2.5])
        self.assertEqual(req.composition_bounds["Vol"], [70.0, 85.0])

    def test_full_workflow_never_returns_negative_oxidation_candidates(self):
        """Mass gain is physically non-negative; NSGA-II must not expose
        surrogate-model artifacts as recommended candidates."""
        from alloy_agent.agent import run_agent
        from alloy_agent.fixtures import make_default_alloy_input
        from alloy_agent.schemas import AgentRequest

        response = run_agent(
            AgentRequest(mode="full", alloy_input=make_default_alloy_input())
        )

        candidates = response.result["optimization"]["candidates"]
        self.assertGreaterEqual(len(candidates), 1)
        for candidate in candidates:
            self.assertGreaterEqual(candidate["predicted_oxidation"], 0.0)
            self.assertGreaterEqual(candidate["predicted_strength"], 800.0)
            self.assertEqual(candidate["composition"]["Vol"], 75.0)

    def test_fixture_co_is_numeric_and_balances_to_100(self):
        """Lock the Co-as-balance convention so the real XGBoost model gets a
        numeric Co value (model can't read "Bal.").
        """
        from alloy_agent.fixtures import make_default_alloy_input

        alloy = make_default_alloy_input()
        co = alloy.composition["Co"]
        self.assertIsInstance(co, (int, float))
        total = sum(
            v for k, v in alloy.composition.items() if isinstance(v, (int, float))
        )
        # Co + other elements should sum to ~100 (balance).
        self.assertAlmostEqual(total, 100.0, places=2)

    def test_fixtures_produce_valid_inputs(self):
        from alloy_agent.fixtures import (
            make_default_alloy_input,
            make_default_optimization_request,
        )

        # If the fixtures drift out of sync with the schema validation, fail here.
        alloy = make_default_alloy_input()
        request = make_default_optimization_request()
        # Co is now numeric (100 - sum of others) so the real XGBoost model can read it.
        self.assertIsInstance(alloy.composition["Co"], (int, float))
        self.assertIn("maximize", request.objectives)
        for element, bounds in request.composition_bounds.items():
            self.assertEqual(
                len(bounds),
                2,
                msg=f"composition_bounds[{element}] must be a 2-tuple, got {bounds}",
            )
            self.assertLessEqual(
                bounds[0],
                bounds[1],
                msg=f"composition_bounds[{element}] has lower > upper: {bounds}",
            )


if __name__ == "__main__":
    unittest.main()
