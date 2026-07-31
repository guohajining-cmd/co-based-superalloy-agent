import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from alloy_agent.agent import run_agent
from alloy_agent.fixtures import make_default_alloy_input, make_default_optimization_request
from alloy_agent.schemas import AgentRequest


def run_evaluate_demo():
    return run_agent(AgentRequest(mode="evaluate", alloy_input=make_default_alloy_input()))


def run_optimize_demo():
    return run_agent(
        AgentRequest(mode="optimize", optimization_request=make_default_optimization_request())
    )


if __name__ == "__main__":
    print("=== Evaluate Demo ===")
    print(run_evaluate_demo().report)
    print()
    print("=== Optimize Demo ===")
    print(run_optimize_demo().report)
