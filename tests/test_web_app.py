import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class WebAppTests(unittest.TestCase):
    def test_web_payload_can_run_evaluate_mode(self):
        from alloy_agent.fixtures import make_default_payload
        from alloy_agent.web_app import handle_agent_payload

        response = handle_agent_payload(make_default_payload("evaluate"))

        self.assertEqual(response["mode"], "evaluate")
        self.assertIn("已有合金评估报告", response["report"])
        self.assertEqual(response["result"]["strength"]["source"], "real_model")
        self.assertEqual(response["result"]["oxidation"]["source"], "real_model")

    def test_web_payload_can_run_optimize_mode(self):
        from alloy_agent.fixtures import make_default_payload
        from alloy_agent.web_app import handle_agent_payload

        response = handle_agent_payload(make_default_payload("optimize"))

        self.assertEqual(response["mode"], "optimize")
        self.assertIn("候选合金列表", response["report"])
        self.assertEqual(response["result"]["candidates"][0]["rank"], 1)

    def test_home_page_contains_two_explicit_entrypoints(self):
        from alloy_agent.web_app import render_home_page

        html = render_home_page()

        self.assertIn("评估已有合金", html)
        self.assertIn("优化设计新合金", html)
        self.assertIn("runEvaluate", html)
        self.assertIn("runOptimize", html)
        # Default evaluate payload must be pre-rendered into the page.
        self.assertIn("Co", html)
        self.assertIn("Ni", html)

    def test_web_rejects_unknown_mode(self):
        from alloy_agent.web_app import handle_agent_payload

        with self.assertRaises(ValueError) as ctx:
            handle_agent_payload({"mode": "summarize"})
        self.assertIn("evaluate", str(ctx.exception))

    def test_web_payload_can_run_full_mode(self):
        from alloy_agent.fixtures import make_default_payload
        from alloy_agent.web_app import handle_agent_payload

        response = handle_agent_payload(make_default_payload("full"))

        self.assertEqual(response["mode"], "full")
        self.assertIn("evaluation", response["result"])
        self.assertIn("optimization", response["result"])
        self.assertIn("summary", response["result"])
        self.assertIn("完整合金分析报告", response["report"])

    def test_web_full_mode_can_skip_optimization(self):
        from alloy_agent.web_app import handle_agent_payload

        response = handle_agent_payload({
            "mode": "full",
            "include_optimization": False,
            "alloy_input": {
                "composition": {"Co": 42.5, "Ni": 30, "Al": 9, "Cr": 7, "Ta": 4,
                                 "Ti": 3, "W": 2, "V": 1, "Nb": 1, "Mo": 0.5},
                "processing": {"aging_temperature": 800, "aging_time": 24},
                "test_conditions": {"strength_test_temperature": 750,
                                     "oxidation_temperature": 1000,
                                     "oxidation_time": 100},
            },
        })

        self.assertEqual(response["mode"], "full")
        self.assertIsNone(response["result"]["optimization"])
        self.assertNotIn("NSGA-II", response["report"])

    def test_web_full_mode_rejects_missing_alloy_input(self):
        from alloy_agent.web_app import handle_agent_payload

        with self.assertRaises(ValueError):
            handle_agent_payload({"mode": "full"})

    def test_web_rejects_missing_alloy_input(self):
        from alloy_agent.web_app import handle_agent_payload

        with self.assertRaises(ValueError):
            handle_agent_payload({"mode": "evaluate"})

    def test_web_rejects_malformed_optimization_bounds(self):
        from alloy_agent.web_app import handle_agent_payload

        with self.assertRaises(ValueError):
            handle_agent_payload(
                {
                    "mode": "optimize",
                    "optimization_request": {
                        "objectives": {"maximize": ["yield_strength"]},
                        "constraints": {},
                        "composition_bounds": {"Ni": [40, 30]},
                        "processing": {},
                        "test_conditions": {},
                    },
                }
            )

    def test_evaluate_handler_strength_varies_with_aluminum(self):
        """End-to-end: the same handler that web POSTs hit should react to Al."""
        from alloy_agent.fixtures import make_default_payload
        from alloy_agent.web_app import handle_agent_payload

        baseline = handle_agent_payload(make_default_payload("evaluate"))
        bumped = make_default_payload("evaluate")
        bumped["alloy_input"]["composition"]["Al"] = 15
        bumped_response = handle_agent_payload(bumped)

        self.assertNotEqual(
            baseline["result"]["strength"]["value"],
            bumped_response["result"]["strength"]["value"],
        )

    def test_evaluate_handler_oxidation_varies_with_chromium(self):
        from alloy_agent.fixtures import make_default_payload
        from alloy_agent.web_app import handle_agent_payload

        baseline = handle_agent_payload(make_default_payload("evaluate"))
        bumped = make_default_payload("evaluate")
        bumped["alloy_input"]["composition"]["Cr"] = 2
        bumped_response = handle_agent_payload(bumped)

        self.assertGreater(
            bumped_response["result"]["oxidation"]["value"],
            baseline["result"]["oxidation"]["value"],
        )

    def test_handler_propagates_value_errors_to_caller(self):
        """The handler must raise — the HTTP layer is what turns this into 400.

        If the handler starts swallowing errors, the web layer's 400 path
        silently breaks (returns 200 with garbage). Lock the contract.
        """
        from alloy_agent.web_app import handle_agent_payload

        with self.assertRaises(ValueError):
            handle_agent_payload({"mode": "evaluate"})  # missing alloy_input
        with self.assertRaises(ValueError):
            handle_agent_payload({"mode": "summarize"})


if __name__ == "__main__":
    unittest.main()
