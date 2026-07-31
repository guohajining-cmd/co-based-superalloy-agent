import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class NaturalLanguageParserTests(unittest.TestCase):
    def test_parse_evaluate_request_extracts_alloy_and_test_conditions(self):
        from alloy_agent.natural_language import parse_user_request

        parsed = parse_user_request(
            "帮我评估 Co-30Ni-9Al-7Cr-4Ta-3Ti-2W-1V-1Nb-0.5Mo 这个合金，"
            "750度测试屈服强度，1000度氧化100小时。"
        )

        self.assertEqual(parsed.mode, "evaluate")
        self.assertFalse(parsed.include_optimization)
        self.assertEqual(parsed.alloy_input.composition["Ni"], 30)
        self.assertEqual(parsed.alloy_input.composition["Al"], 9)
        self.assertEqual(parsed.alloy_input.composition["Mo"], 0.5)
        self.assertAlmostEqual(parsed.alloy_input.composition["Co"], 42.5)
        self.assertEqual(
            parsed.alloy_input.test_conditions["strength_test_temperature"], 750
        )
        self.assertEqual(parsed.alloy_input.test_conditions["oxidation_temperature"], 1000)
        self.assertEqual(parsed.alloy_input.test_conditions["oxidation_time"], 100)

    def test_parse_design_request_enables_full_workflow(self):
        from alloy_agent.natural_language import parse_user_request

        parsed = parse_user_request(
            "请基于 Co-30Ni-9Al-7Cr-4Ta-3Ti-2W-1V-1Nb-0.5Mo 做NSGA-II优化，"
            "帮我推荐候选合金。"
        )

        self.assertEqual(parsed.mode, "full")
        self.assertTrue(parsed.include_optimization)

    def test_parsed_request_can_run_agent(self):
        from alloy_agent.agent import run_agent
        from alloy_agent.natural_language import parse_user_request
        from alloy_agent.schemas import AgentRequest

        parsed = parse_user_request(
            "预测 Co-30Ni-9Al-7Cr-4Ta-3Ti-2W-1V-1Nb-0.5Mo 的性能，"
            "750度测试，1000度氧化100小时。"
        )
        response = run_agent(
            AgentRequest(
                mode=parsed.mode,
                alloy_input=parsed.alloy_input,
                include_optimization=parsed.include_optimization,
            )
        )

        self.assertEqual(response.mode, "evaluate")
        self.assertIn("strength", response.result)
        self.assertIn("oxidation", response.result)


if __name__ == "__main__":
    unittest.main()
