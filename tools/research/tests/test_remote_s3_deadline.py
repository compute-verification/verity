"""A stuck read cannot hang a store command: every S3 request has a hard wall-clock deadline that shuts its socket, and the
request (a whole GET, body included) is retried with backoff.  A local HTTP server plays R2 stalling mid-body and before headers."""
from __future__ import annotations

import http.server
import threading
import time

import pytest

from research.store import remote_s3

BODY = bytes(range(256)) * 16


class _Stall(http.server.BaseHTTPRequestHandler):
    calls: list[str] = []
    stall_first = True

    def log_message(self, *a):
        pass

    def do_GET(self):
        _Stall.calls.append("GET")
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        if len(_Stall.calls) == 1 and _Stall.stall_first:
            self.wfile.write(BODY[:100])
            self.wfile.flush()
            time.sleep(5)                                  # a trickle that stops: recv would block far past the deadline
            return
        self.wfile.write(BODY)

    def do_HEAD(self):
        _Stall.calls.append("HEAD")
        if len(_Stall.calls) == 1 and _Stall.stall_first:
            time.sleep(5)                                  # never answers in time
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.send_header("ETag", '"0123456789abcdef0123456789abcdef"')
        self.end_headers()


@pytest.fixture
def server(monkeypatch):
    _Stall.calls = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stall)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(remote_s3, "REQUEST_DEADLINE_S", 0.5)
    monkeypatch.setattr(remote_s3, "RETRY_BACKOFF_S", 0.01)
    yield remote_s3.S3Remote(f"http://127.0.0.1:{srv.server_address[1]}", "b", "ak", "sk", timeout=30)
    srv.shutdown()


def test_a_get_whose_body_stalls_is_cut_off_and_retried(server, tmp_path):
    t = time.monotonic()
    server.get("k", tmp_path / "out")
    assert (tmp_path / "out").read_bytes() == BODY and _Stall.calls == ["GET", "GET"] and time.monotonic() - t < 4


def test_a_head_that_never_answers_is_cut_off_and_retried(server):
    t = time.monotonic()
    assert server.head("k") == {"bytes": len(BODY), "etag": "0123456789abcdef0123456789abcdef"}
    assert _Stall.calls == ["HEAD", "HEAD"] and time.monotonic() - t < 4


def test_a_request_that_always_stalls_fails_within_its_retries(server, tmp_path, monkeypatch):
    monkeypatch.setattr(_Stall, "stall_first", True)
    monkeypatch.setattr(remote_s3, "RETRIES", 1)
    orig = _Stall.do_GET

    def always(self):                                      # every GET is the first: it always stalls
        _Stall.calls[:] = []
        orig(self)
    monkeypatch.setattr(_Stall, "do_GET", always)
    t = time.monotonic()
    with pytest.raises((TimeoutError, OSError)):
        server.get("k", tmp_path / "out")
    assert time.monotonic() - t < 5 and not (tmp_path / "out").exists()
