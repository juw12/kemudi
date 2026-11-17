import http.client
import json
import threading

import pytest

from kemudi.server import App, serve


class FakeEngine:
    """Writes the numbers 0, 1, 2... and records the knobs it was given."""

    def __init__(self):
        self.lock = threading.Lock()
        self.knob_calls = []
        self.gate = threading.Event()
        self.gate.set()

    def set_gauges(self, axes):
        self.gauged = list(axes)

    def set_knobs(self, axes, values):
        self.knob_calls.append(dict(values))

    def story_prompt(self, story, controls=""):
        return f"<{story}>"

    def stream(self, prompt, max_new_tokens, temperature, seed, stop):
        for index in range(max_new_tokens):
            self.gate.wait(5)
            if stop.is_set():
                return
            yield f"{index} ", {"mood": 0.123456}


@pytest.fixture()
def running():
    engine = FakeEngine()
    app = App(engine, {"mood": object()}, {"model": "fake", "axes": {}})
    server = serve(app, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield app, engine, server.server_address[1]
    server.shutdown()
    server.server_close()


def call(port, method, path, body=None, host=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/json"}
    if host:
        headers["Host"] = host
    payload = body if isinstance(body, (bytes, type(None))) else json.dumps(body)
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    return response.status, response.read()


def events(raw):
    return [json.loads(line) for line in raw.decode().splitlines()]


def test_info_and_page(running):
    _app, engine, port = running
    status, body = call(port, "GET", "/api/info")
    assert status == 200
    assert json.loads(body) == {"summary": {"model": "fake", "axes": {}}}
    status, page = call(port, "GET", "/")
    assert status == 200 and b"<title>Kemudi</title>" in page
    assert b'id="combo"' in page and b"Each knob was measured alone" in page  # the warning for two or more knobs
    assert call(port, "GET", "/nothing")[0] == 404
    assert engine.gauged == ["mood"]


def test_generate_streams_one_line_per_token_then_a_summary(running):
    _app, engine, port = running
    status, raw = call(port, "POST", "/api/generate", {"story": "Once", "tokens": 3, "knobs": {"mood": 0.2}})
    lines = events(raw)
    assert status == 200
    assert [line["t"] for line in lines[:-1]] == ["0 ", "1 ", "2 "]
    assert lines[0]["g"] == {"mood": 0.123}
    assert lines[-1]["done"] is True and lines[-1]["tokens"] == 3
    assert engine.knob_calls[-1] == {"mood": 0.2}


def test_knobs_are_clamped_and_unknown_axes_dropped(running):
    _app, engine, port = running
    call(port, "POST", "/api/knobs", {"knobs": {"mood": 7, "nonsense": 1}})
    assert engine.knob_calls[-1] == {"mood": 1.0}
    call(port, "POST", "/api/knobs", {"knobs": {"mood": -7}})
    assert engine.knob_calls[-1] == {"mood": -1.0}


def test_token_count_is_capped(running):
    _app, _engine, port = running
    _status, raw = call(port, "POST", "/api/generate", {"story": "x", "tokens": 100000})
    assert events(raw)[-1]["tokens"] == 400


def test_other_host_names_are_refused(running):
    _app, _engine, port = running
    assert call(port, "GET", "/api/info", host="evil.example:80")[0] == 403
    assert call(port, "POST", "/api/knobs", {"knobs": {}}, host="evil.example")[0] == 403
    assert call(port, "GET", "/api/info", host=f"localhost:{port}")[0] == 200


def test_bad_json_is_a_400(running):
    _app, _engine, port = running
    assert call(port, "POST", "/api/knobs", b"{not json")[0] == 400


def test_locked_run_ignores_live_knobs_and_stop_ends_it(running):
    app, engine, port = running
    engine.gate.clear()
    result = {}

    def generate():
        result["raw"] = call(port, "POST", "/api/generate", {"story": "x", "tokens": 50, "locked": True, "knobs": {}})[1]

    worker = threading.Thread(target=generate)
    worker.start()
    while not app.locked:
        pass
    calls_before = len(engine.knob_calls)
    status, body = call(port, "POST", "/api/knobs", {"knobs": {"mood": 0.3}})
    assert json.loads(body) == {"applied": False}
    assert len(engine.knob_calls) == calls_before
    call(port, "POST", "/api/stop", {})
    engine.gate.set()
    worker.join(10)
    lines = events(result["raw"])
    assert lines[-1]["done"] is True and lines[-1]["tokens"] < 50
    assert app.locked is False
    assert json.loads(call(port, "POST", "/api/knobs", {"knobs": {"mood": 0.3}})[1]) == {"applied": True}


def test_info_carries_the_follow_up_measurements_when_they_exist():
    app = App(FakeEngine(), {"mood": object()}, {"model": "fake", "axes": {}}, {"paired": {"mood": []}})
    assert app.info() == {"summary": {"model": "fake", "axes": {}}, "paired": {"mood": []}}
    assert App(FakeEngine(), {}, {"axes": {}}).info() == {"summary": {"axes": {}}}
