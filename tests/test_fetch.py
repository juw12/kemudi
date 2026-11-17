import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kemudi import fetch

PAYLOAD = bytes(range(256)) * 4000  # about 1 MB


class FlakyHandler(BaseHTTPRequestHandler):
    """Serves PAYLOAD with Range support, but cuts the first `cuts` responses short."""

    cuts = 0
    requests = []

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path.startswith("/missing"):
            return self.send_error(404)
        start = int(self.headers.get("Range", "bytes=0-").split("=")[1].rstrip("-"))
        type(self).requests.append(start)
        body = PAYLOAD[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if type(self).cuts > 0:
            type(self).cuts -= 1
            body = body[: len(body) // 3]
        self.wfile.write(body)


@pytest.fixture()
def mirror(monkeypatch, tmp_path):
    FlakyHandler.cuts = 0
    FlakyHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FlakyHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("KEMUDI_HOME", str(tmp_path))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(fetch, "MODEL_FILES", ["weights.bin"])
    monkeypatch.setattr(fetch.time, "sleep", lambda _seconds: None)
    yield base, monkeypatch
    server.shutdown()
    server.server_close()


def test_download_resumes_after_two_cut_connections(mirror):
    base, monkeypatch = mirror
    monkeypatch.setattr(fetch, "MIRRORS", [base + "/{repo}/{name}"])
    FlakyHandler.cuts = 2
    folder = fetch.fetch_model("org/tiny", log=lambda _line: None)
    assert (folder / "weights.bin").read_bytes() == PAYLOAD
    assert not (folder / "weights.bin.part").exists()
    assert len(FlakyHandler.requests) == 3
    assert FlakyHandler.requests[0] == 0 and FlakyHandler.requests[1] > 0 and FlakyHandler.requests[2] > FlakyHandler.requests[1]
    assert fetch.is_complete(folder)


def test_second_mirror_is_used_when_the_first_has_no_file(mirror):
    base, monkeypatch = mirror
    monkeypatch.setattr(fetch, "MIRRORS", [base + "/missing/{repo}/{name}", base + "/{repo}/{name}"])
    folder = fetch.fetch_model("org/tiny", log=lambda _line: None)
    assert (folder / "weights.bin").read_bytes() == PAYLOAD
    assert FlakyHandler.requests == [0]


def test_existing_files_are_not_downloaded_again(mirror):
    base, monkeypatch = mirror
    monkeypatch.setattr(fetch, "MIRRORS", [base + "/{repo}/{name}"])
    fetch.fetch_model("org/tiny", log=lambda _line: None)
    fetch.fetch_model("org/tiny", log=lambda _line: None)
    assert FlakyHandler.requests == [0]


def test_failure_names_the_file(mirror):
    base, monkeypatch = mirror
    monkeypatch.setattr(fetch, "MIRRORS", [base + "/missing/{repo}/{name}"])
    with pytest.raises(RuntimeError, match="weights.bin"):
        fetch.fetch_model("org/tiny", log=lambda _line: None)
