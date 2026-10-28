"""A stuck read cannot hang a store command: every S3 request has a hard wall-clock deadline that shuts its socket, and the
request (a whole GET, body included) is retried with backoff.  A local HTTP server plays R2 stalling mid-body and before headers.
The deadline's timer is replaced: a deadline passes when the server stalls, and the stalled handler then waits for the client to
shut its socket, so a request that gets past a stall was cut off by its watchdog, whatever the machine's speed."""
from __future__ import annotations

import http.client
import http.server
import socket
import threading
import urllib.error

import pytest

from research.store import remote_s3

BODY = bytes(range(256)) * 16


class _Deadline:
    """Stands in for the watchdog's threading.Timer: records what was armed; `pass_latest()` makes the newest live one fire."""
    armed: list[_Deadline] = []

    def __init__(self, seconds: float, fire):
        self.seconds, self.fire, self.cancelled, self.fired, self.daemon = seconds, fire, False, False, True

    def start(self) -> None:
        _Deadline.armed.append(self)

    def cancel(self) -> None:
        self.cancelled = True

    @classmethod
    def pass_latest(cls) -> None:
        live = [t for t in cls.armed if not t.cancelled]
        live[-1].fired = True
        live[-1].fire()


class _Stall(http.server.BaseHTTPRequestHandler):
    calls: list[str] = []
    stall_first = True

    def log_message(self, *a):
        pass

    def _stall(self) -> None:
        """The request's deadline passes now; the handler returns only once the client has shut its socket (EOF here)."""
        _Deadline.pass_latest()
        self.rfile.read()

    def do_GET(self):
        _Stall.calls.append("GET")
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        if len(_Stall.calls) == 1 and _Stall.stall_first:
            self.wfile.write(BODY[:100])                   # a trickle that stops
            self.wfile.flush()
            self._stall()
            return
        self.wfile.write(BODY)

    def do_HEAD(self):
        _Stall.calls.append("HEAD")
        if len(_Stall.calls) == 1 and _Stall.stall_first:
            self._stall()                                  # never answers
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.send_header("ETag", '"0123456789abcdef0123456789abcdef"')
        self.end_headers()


@pytest.fixture
def server(monkeypatch):
    _Stall.calls, _Deadline.armed = [], []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stall)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(remote_s3._Watchdog, "timer", _Deadline)
    monkeypatch.setattr(remote_s3, "RETRY_BACKOFF_S", 0.0)
    yield remote_s3.S3Remote(f"http://127.0.0.1:{srv.server_address[1]}", "b", "ak", "sk")
    srv.shutdown()


def _fired() -> list[float]:
    return [t.seconds for t in _Deadline.armed if t.fired]


def test_a_get_whose_body_stalls_is_cut_off_and_retried(server, tmp_path):
    server.get("k", tmp_path / "out")
    assert (tmp_path / "out").read_bytes() == BODY and _Stall.calls == ["GET", "GET"]
    assert _fired() == [remote_s3.REQUEST_DEADLINE_S] and server.deadline_hits[0]["method"] == "GET"


def test_a_large_get_that_trickles_is_cut_off_after_one_slow_window_and_retried(server, tmp_path, monkeypatch):
    # INF-214: a 16 GiB part trickled at 13-170 KB/s for an hour under a 73 h whole-body deadline; a fresh connection got 45-70 MB/s
    monkeypatch.setattr(remote_s3, "GET_STALL_MIN_BYTES", 1000)
    windowed, start = threading.Event(), _Deadline.start

    def arm(self):
        start(self)
        if self.seconds == remote_s3.STALL_WINDOW_S:
            windowed.set()
    monkeypatch.setattr(_Deadline, "start", arm)
    orig = _Stall._stall

    def stall_once_windowed(self):                         # the trickle stops only once the client watches the body's window
        windowed.wait()
        orig(self)
    monkeypatch.setattr(_Stall, "_stall", stall_once_windowed)
    server.get("k", tmp_path / "out")
    assert (tmp_path / "out").read_bytes() == BODY and _Stall.calls == ["GET", "GET"]
    assert _fired() == [remote_s3.STALL_WINDOW_S]
    assert server.deadline_hits == [{"method": "GET", "key": "k", "try": 1, "of": remote_s3.RETRIES + 1, "deadline_s": remote_s3.STALL_WINDOW_S}]


def test_a_head_that_never_answers_is_cut_off_and_retried(server, capsys):
    assert server.head("k") == {"bytes": len(BODY), "etag": "0123456789abcdef0123456789abcdef"}
    assert _Stall.calls == ["HEAD", "HEAD"] and _fired() == [remote_s3.HEAD_DEADLINE_S]
    assert server.deadline_hits == [{"method": "HEAD", "key": "k", "try": 1, "of": 2, "deadline_s": remote_s3.HEAD_DEADLINE_S}] and server.exhausted == []
    assert f"research: HEAD k: no response within {remote_s3.HEAD_DEADLINE_S:.0f}s (try 1 of 2); retrying" in capsys.readouterr().err


def test_a_head_that_never_answers_is_exhausted_after_its_one_retry_and_named(server, monkeypatch):
    orig = _Stall.do_HEAD

    def always(self):
        _Stall.calls[:] = []
        orig(self)
    monkeypatch.setattr(_Stall, "do_HEAD", always)
    with pytest.raises(TimeoutError, match=f"HEAD k2: no response within the {remote_s3.HEAD_DEADLINE_S:.0f}s deadline on all 2 tries"):
        server.head("k2")
    assert [x["key"] for x in server.exhausted] == ["k2"] and len(server.deadline_hits) == 2 and len(_fired()) == 2


def test_a_head_the_service_keeps_failing_ends_in_its_http_error_after_its_one_retry(server, monkeypatch):
    # the 5xx retry counted RETRIES for every method, so a HEAD's last try retried past HEAD_RETRIES: AssertionError("unreachable")
    def do_HEAD(self):
        _Stall.calls.append("HEAD")
        self.send_response(503)
        self.send_header("Content-Length", "0")
        self.end_headers()
    monkeypatch.setattr(_Stall, "do_HEAD", do_HEAD)
    with pytest.raises(urllib.error.HTTPError) as e:
        server.head("k")
    assert e.value.code == 503 and _Stall.calls == ["HEAD"] * (remote_s3.HEAD_RETRIES + 1)


def test_a_retry_after_a_dropped_connection_never_draws_another_idle_one(server, monkeypatch):
    # R2 closes idle keep-alives (a pod's custody push had its sockets in CLOSE-WAIT, then failed with RemoteDisconnected on every try):
    # with a pool full of such connections, the retry must open a new one instead of taking the next dead one
    monkeypatch.setattr(_Stall, "stall_first", False)
    dead = socket.create_server(("127.0.0.1", 0))
    closed = threading.Event()

    def accept_and_close():
        for _ in range(4):
            dead.accept()[0].close()
        closed.set()
    threading.Thread(target=accept_and_close, daemon=True).start()
    for _ in range(4):
        c = http.client.HTTPConnection("127.0.0.1", dead.getsockname()[1])
        c.connect()
        server.pool._idle.append(c)
    closed.wait()
    assert server.head("k") == {"bytes": len(BODY), "etag": "0123456789abcdef0123456789abcdef"}
    assert _Stall.calls == ["HEAD"] and server.pool.idle == 3, "one dead connection tried, then a fresh one; the other three untouched"
    dead.close()


def test_a_request_that_always_stalls_fails_within_its_retries(server, tmp_path, monkeypatch):
    monkeypatch.setattr(_Stall, "stall_first", True)
    monkeypatch.setattr(remote_s3, "RETRIES", 1)
    orig = _Stall.do_GET

    def always(self):                                      # every GET is the first: it always stalls
        _Stall.calls[:] = []
        orig(self)
    monkeypatch.setattr(_Stall, "do_GET", always)
    with pytest.raises((TimeoutError, OSError)):
        server.get("k", tmp_path / "out")
    assert not (tmp_path / "out").exists() and len(_fired()) == 2


def test_a_part_upload_that_stops_moving_is_cut_off_after_one_idle_window_and_retried(server, monkeypatch, tmp_path):
    # nebius hosts: part PUTs sat for over an hour with nothing moving, under a whole-body deadline of hours for a 256 MiB part
    monkeypatch.setattr(remote_s3, "STALL_MIN_BYTES", 1)
    puts: list[int] = []

    def do_PUT(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        puts.append(len(body))
        if len(puts) == 1:
            _Deadline.pass_latest()                        # the window saw the whole body move: re-armed, not fired
            _Deadline.pass_latest()                        # then nothing moves (no response): cut off
            self.rfile.read()
            return
        self.send_response(200)
        self.send_header("ETag", '"' + remote_s3.hashlib.md5(body).hexdigest() + '"')
        self.send_header("Content-Length", "0")
        self.end_headers()
    monkeypatch.setattr(_Stall, "do_PUT", do_PUT, raising=False)
    part = {"n": 1, "offset": 0, "bytes": len(BODY), "sha256": remote_s3.hashlib.sha256(BODY).hexdigest(),
            "md5": remote_s3.hashlib.md5(BODY).hexdigest()}
    path = tmp_path / "f"
    path.write_bytes(BODY)
    assert server._upload_part("k", "u", path, part) == part["md5"]
    assert puts == [len(BODY), len(BODY)]
    assert server.deadline_hits == [{"method": "PUT", "key": "k", "try": 1, "of": remote_s3.RETRIES + 1, "deadline_s": remote_s3.STALL_WINDOW_S}]
    assert all(t.seconds == remote_s3.STALL_WINDOW_S for t in _Deadline.armed)
