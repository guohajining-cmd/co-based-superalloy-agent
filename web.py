import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from alloy_agent.web_app import run_server


if __name__ == "__main__":
    run_server()
