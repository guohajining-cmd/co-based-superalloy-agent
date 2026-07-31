from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from typing import Any
from urllib.parse import urlparse

from alloy_agent.agent import run_agent
from alloy_agent.fixtures import make_default_payload
from alloy_agent.schemas import AgentRequest, AlloyInput, OptimizationRequest


_TEMPLATE_DIR = Path(__file__).parent / "web" / "templates"
_HOME_TEMPLATE = (_TEMPLATE_DIR / "home.html").read_text(encoding="utf-8")
# CSS / JS contain literal { } which would break str.format. We replaced them
# with sentinels in the template, then restore them here.
_HOME_TEMPLATE = (
    _HOME_TEMPLATE
    .replace("__BRACE_OPEN__", "\x01")
    .replace("__BRACE_CLOSE__", "\x02")
)


def handle_agent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    mode = payload.get("mode")

    if mode == "evaluate":
        alloy_payload = payload.get("alloy_input")
        if not isinstance(alloy_payload, dict):
            raise ValueError("mode='evaluate' requires alloy_input")
        request = AgentRequest(
            mode="evaluate",
            alloy_input=AlloyInput(
                composition=alloy_payload["composition"],
                processing=alloy_payload["processing"],
                test_conditions=alloy_payload["test_conditions"],
            ),
        )
    elif mode == "optimize":
        optimize_payload = payload.get("optimization_request")
        if not isinstance(optimize_payload, dict):
            raise ValueError("mode='optimize' requires optimization_request")
        request = AgentRequest(
            mode="optimize",
            optimization_request=OptimizationRequest(
                objectives=optimize_payload["objectives"],
                constraints=optimize_payload["constraints"],
                composition_bounds=optimize_payload["composition_bounds"],
                processing=optimize_payload["processing"],
                test_conditions=optimize_payload["test_conditions"],
            ),
        )
    elif mode == "full":
        alloy_payload = payload.get("alloy_input")
        if not isinstance(alloy_payload, dict):
            raise ValueError("mode='full' requires alloy_input")
        include_opt = bool(payload.get("include_optimization", True))
        search_space = payload.get("search_space", "local")
        request = AgentRequest(
            mode="full",
            alloy_input=AlloyInput(
                composition=alloy_payload["composition"],
                processing=alloy_payload["processing"],
                test_conditions=alloy_payload["test_conditions"],
                microstructure=alloy_payload.get("microstructure", {}),
            ),
            include_optimization=include_opt,
            search_space=search_space,
        )
    else:
        raise ValueError("mode must be 'evaluate', 'optimize', or 'full'")

    response = run_agent(request)
    return {
        "mode": response.mode,
        "result": response.result,
        "report": response.report,
    }


def render_home_page() -> str:
    evaluate_json = json.dumps(make_default_payload("evaluate"), ensure_ascii=False, indent=2)
    optimize_json = json.dumps(make_default_payload("optimize"), ensure_ascii=False, indent=2)
    full_json = json.dumps(make_default_payload("full"), ensure_ascii=False, indent=2)
    rendered = Template(_HOME_TEMPLATE).safe_substitute(
        evaluate_json=evaluate_json,
        optimize_json=optimize_json,
        full_json=full_json,
    )
    return rendered.replace("\x01", "{").replace("\x02", "}")


class AlloyAgentRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/", "/index.html"}:
            self._send_json({"error": "Not found"}, status=404)
            return
        body = render_home_page().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/agent":
            self._send_json({"error": "Not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = handle_agent_payload(payload)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))
        sys.stderr.flush()

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), AlloyAgentRequestHandler)
    print(f"Alloy Agent UI running at http://{host}:{port}")
    server.serve_forever()
