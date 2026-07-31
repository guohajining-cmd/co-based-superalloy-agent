import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class AgentLoopTests(unittest.TestCase):
    def test_agent_loop_completes_evaluate_request_and_records_decisions(self):
        from alloy_agent.agent_loop import run_agent_loop

        state = run_agent_loop(
            "帮我评估 Co-30Ni-9Al-7Cr-4Ta-3Ti-2W-1V-1Nb-0.5Mo 这个合金，"
            "750度测试屈服强度，1000度氧化100小时。"
        )

        self.assertEqual(state.status, "completed")
        self.assertEqual(state.intent, "evaluate")
        self.assertIsNotNone(state.result)
        self.assertEqual(state.result["mode"], "evaluate")
        self.assertIn("strength", state.result["result"])

        actions = [step["action"] for step in state.decision_trace]
        self.assertEqual(actions[0], "parse_user_request")
        self.assertIn("validate_required_inputs", actions)
        self.assertIn("run_agent:evaluate", actions)
        self.assertIn("validate_agent_result", actions)

    def test_agent_loop_waits_when_composition_is_missing(self):
        from alloy_agent.agent_loop import run_agent_loop

        state = run_agent_loop("帮我优化一个抗氧化更好的钴基合金")

        self.assertEqual(state.status, "waiting_for_input")
        self.assertIn("composition", state.missing_fields)
        self.assertIsNone(state.result)

        actions = [step["action"] for step in state.decision_trace]
        self.assertIn("parse_user_request", actions)
        self.assertIn("validate_required_inputs", actions)
        self.assertIn("ask_user", actions)
        self.assertNotIn("run_agent:full", actions)

    def test_agent_loop_runs_full_workflow_for_optimization_request(self):
        from alloy_agent.agent_loop import run_agent_loop

        state = run_agent_loop(
            "请基于 Co-30Ni-9Al-7Cr-4Ta-3Ti-2W-1V-1Nb-0.5Mo 做NSGA-II优化，"
            "帮我推荐候选合金。"
        )

        self.assertEqual(state.status, "completed")
        self.assertEqual(state.intent, "full")
        self.assertEqual(state.result["mode"], "full")
        tools = [item["tool"] for item in state.tool_trace]
        self.assertIn("run_nsga2_optimization", tools)


if __name__ == "__main__":
    unittest.main()
