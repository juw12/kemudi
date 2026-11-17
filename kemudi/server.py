"""Local web app: sliders that move the model while it writes. Standard library only."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .engine import Axis, Engine

WEB = Path(__file__).parent / "web"
LOCAL_HOSTS = ("127.0.0.1", "localhost")
MAX_KNOB = 1.0
MAX_TOKENS = 400
MAX_STORY_CHARACTERS = 4000


class App:
    def __init__(self, engine: Engine, axes: dict[str, Axis], summary: dict, extras: dict | None = None):
        self.engine = engine
        self.axes = axes
        self.summary = summary
        self.extras = extras or {}
        self.stop = threading.Event()
        self.locked = False
        engine.set_gauges(axes)

    def clean_knobs(self, raw: dict) -> dict[str, float]:
        return {
            name: max(-MAX_KNOB, min(MAX_KNOB, float(value)))
            for name, value in (raw or {}).items()
            if name in self.axes
        }

    def info(self) -> dict:
        return {"summary": self.summary, **self.extras}

    def set_live_knobs(self, raw: dict) -> bool:
        """Change the knobs of the running generation. Refused while a locked comparison run is writing."""
        if self.locked:
            return False
        self.engine.set_knobs(self.axes, self.clean_knobs(raw))
        return True

    def generate(self, request: dict):
        """Yield one dictionary per token. A new request stops the one that is running."""
        story = str(request.get("story", ""))[:MAX_STORY_CHARACTERS]
        tokens = max(1, min(MAX_TOKENS, int(request.get("tokens", 120))))
        self.stop.set()
        with self.engine.lock:
            self.stop = stop = threading.Event()
            self.locked = bool(request.get("locked"))
            self.engine.set_knobs(self.axes, self.clean_knobs(request.get("knobs")))
            started = time.monotonic()
            count = 0
            try:
                prompt = self.engine.story_prompt(story)
                for piece, readings in self.engine.stream(
                    prompt,
                    max_new_tokens=tokens,
                    temperature=float(request.get("temperature", 0.8)),
                    seed=int(request.get("seed", 0)),
                    stop=stop,
                ):
                    count += 1
                    yield {"t": piece, "g": {name: round(value, 3) for name, value in readings.items()}}
            finally:
                self.locked = False
            yield {"done": True, "tokens": count, "seconds": round(time.monotonic() - started, 2)}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _local(self) -> bool:
            """Refuse requests that reached this server under another host name (DNS rebinding)."""
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            if host in LOCAL_HOSTS:
                return True
            self.send_error(403, "local use only")
            return False

        def _json(self, payload, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            if not self._local():
                return
            if self.path == "/api/info":
                return self._json(app.info())
            if self.path in ("/", "/index.html"):
                body = (WEB / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            self.send_error(404)

        def do_POST(self):
            if not self._local():
                return
            try:
                request = self._body()
            except ValueError:
                return self.send_error(400, "body must be JSON")
            if self.path == "/api/knobs":
                return self._json({"applied": app.set_live_knobs(request.get("knobs"))})
            if self.path == "/api/stop":
                app.stop.set()
                return self._json({"stopped": True})
            if self.path == "/api/generate":
                return self._stream(request)
            self.send_error(404)

        def _stream(self, request: dict) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            events = app.generate(request)
            try:
                for event in events:
                    self.wfile.write(json.dumps(event).encode() + b"\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionError):
                pass
            finally:
                events.close()

    return Handler


def serve(app: App, port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
