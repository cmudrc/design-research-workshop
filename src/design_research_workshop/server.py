"""Local HTTP endpoint for the workshop's deterministic model proxy."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from uuid import uuid4

from design_research.agents import OpenAICompatibleHTTPLLMClient

from .bot import WorkshopPyBot

MODEL = "workshop-pybot-v1"
LEGACY_MODEL = "workshop-laptop-proxy-v0"
MAX_REQUEST_BYTES = 65_536


@contextmanager
def workshop_client(
    bot: WorkshopPyBot | None = None,
) -> Iterator[OpenAICompatibleHTTPLLMClient]:
    """Start a loopback endpoint on an available port for one demo."""
    with WorkshopServer(0, bot=bot) as server:
        thread = Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        thread.start()
        try:
            with OpenAICompatibleHTTPLLMClient(
                name="workshop-rule-based",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                default_model=MODEL,
                model_patterns=(MODEL, LEGACY_MODEL),
                api_key_env="DESIGN_RESEARCH_WORKSHOP_PROXY_UNUSED_KEY",
                max_retries=0,
            ) as client:
                yield client
        finally:
            server.shutdown()
            thread.join()


class WorkshopServer(ThreadingHTTPServer):
    def __init__(self, port: int, *, bot: WorkshopPyBot | None = None) -> None:
        self.bot = bot if bot is not None else WorkshopPyBot()
        super().__init__(("127.0.0.1", port), WorkshopHandler)


class WorkshopHandler(BaseHTTPRequestHandler):
    server: WorkshopServer

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        """Keep notebook output focused on the workshop results."""

    def send_json(self, status: HTTPStatus, value: dict[str, object]) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self.send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL,
                            "object": "model",
                            "owned_by": "design-research-workshop",
                        }
                    ],
                },
            )
        else:
            self.send_json(
                HTTPStatus.NOT_FOUND, {"error": {"message": "Unknown endpoint."}}
            )

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json(
                HTTPStatus.NOT_FOUND, {"error": {"message": "Unknown endpoint."}}
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("Missing or oversized request body.")
            request = json.loads(self.rfile.read(length))
            if (
                request["model"] not in (MODEL, LEGACY_MODEL)
                or request.get("stream")
                or request.get("tools")
            ):
                raise ValueError(
                    "Use the workshop model with a non-streaming text request."
                )
            messages = request["messages"]
            reply = self.server.bot.complete(messages)
            if not reply.matched and request["model"] == LEGACY_MODEL:
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": {
                            "message": reply.content["question"],
                            **reply.content,
                        },
                        "workshop": {
                            "reason": reply.reason,
                            "candidates": reply.candidates,
                        },
                    },
                )
                return
            content = json.dumps(reply.content, allow_nan=False)
        except (
            AttributeError,
            KeyError,
            StopIteration,
            TypeError,
            ValueError,
        ) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(error)}})
            return

        self.send_json(
            HTTPStatus.OK,
            {
                "id": f"chatcmpl-{uuid4().hex}",
                "object": "chat.completion",
                "model": request["model"],
                "workshop": {
                    "rule": reply.rule,
                    "reason": reply.reason,
                    "candidates": reply.candidates,
                },
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    with WorkshopServer(args.port) as server:
        print(
            f"{MODEL} listening at http://127.0.0.1:{server.server_port}/v1", flush=True
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
