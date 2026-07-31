import json
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class _FakeResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        return {"id": f"resp_{len(self.calls)}", "output": output}


class _FakeClient:
    def __init__(self, outputs):
        self.responses = _FakeResponses(outputs)


class LLMToolAgentTests(unittest.TestCase):
    def test_llm_tool_agent_executes_model_selected_evaluate_tool(self):
        from alloy_agent.llm_tool_agent import run_llm_tool_agent

        client = _FakeClient(
            [
                [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "evaluate_alloy",
                        "arguments": json.dumps(
                            {
                                "composition": {
                                    "Ni": 30,
                                    "Al": 9,
                                    "Cr": 7,
                                    "Ta": 4,
                                    "Ti": 3,
                                    "W": 2,
                                    "V": 1,
                                    "Nb": 1,
                                    "Mo": 0.5,
                                },
                                "test_conditions": {
                                    "strength_test_temperature": 750,
                                    "oxidation_temperature": 1000,
                                    "oxidation_time": 100,
                                },
                            }
                        ),
                    }
                ],
                [{"type": "message", "content": [{"type": "output_text", "text": "评估完成。"}]}],
            ]
        )

        state = run_llm_tool_agent("评估这个合金", client=client, model="fake-model")

        self.assertEqual(state.status, "completed")
        self.assertEqual(state.intent, "evaluate")
        self.assertEqual(state.result["mode"], "evaluate")
        self.assertIn("strength", state.result["result"])
        actions = [item["action"] for item in state.decision_trace]
        self.assertIn("llm_tool_call:evaluate_alloy", actions)
        self.assertIn("execute_tool:evaluate_alloy", actions)
        self.assertEqual(len(client.responses.calls), 2)
        self.assertEqual(client.responses.calls[0]["model"], "fake-model")
        self.assertEqual(client.responses.calls[0]["tools"][0]["type"], "function")
        self.assertEqual(client.responses.calls[1]["previous_response_id"], "resp_1")

    def test_llm_tool_agent_can_pause_for_missing_input(self):
        from alloy_agent.llm_tool_agent import run_llm_tool_agent

        client = _FakeClient(
            [
                [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "ask_for_missing_input",
                        "arguments": json.dumps(
                            {
                                "missing_fields": ["composition"],
                                "question": "请补充合金成分。",
                            }
                        ),
                    }
                ]
            ]
        )

        state = run_llm_tool_agent("帮我优化一个合金", client=client, model="fake-model")

        self.assertEqual(state.status, "waiting_for_input")
        self.assertEqual(state.missing_fields, ["composition"])
        self.assertEqual(state.pending_question, "请补充合金成分。")
        self.assertIsNone(state.result)

    def test_llm_tool_agent_falls_back_to_rule_loop_without_openai_config(self):
        from alloy_agent.llm_tool_agent import run_llm_tool_agent

        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            state = run_llm_tool_agent(
                "帮我优化一个抗氧化更好的钴基合金",
                client=None,
                api_key=None,
            )
        finally:
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key

        self.assertEqual(state.status, "waiting_for_input")
        self.assertIn("composition", state.missing_fields)
        self.assertTrue(any("LLM tool-calling 未配置" in w for w in state.warnings))
        self.assertIn("llm_fallback_to_rule_loop", [d["action"] for d in state.decision_trace])


if __name__ == "__main__":
    unittest.main()
