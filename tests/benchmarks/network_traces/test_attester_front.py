"""attester_front: the front sends one group of the window circuit's class through the gate's scan rehearsal and keeps
the gate's fetch, witness and all, 0700 under the serve dir's private/certifier, publishing only the bundle less witness/
and its outcome. Its warm-ups go straight to the endpoint, one at a time, each recorded in a closed window of its own
before the gate is asked, and the pick passes over those windows for the group's. At a tick of 4000 ms the front's
rehearsal reaches this tree's gate with the tick in its request and the minutes counted at it, and the network attester
runs the scan's schedule there; a tick the gate refuses (past its range, or on a request that isn't the scan rehearsal)
refuses the front before the gate is asked. Its relay session opens, waits on the gate's status and closes as the module
docstring says, and the piece closes a session nobody closed before it reads the kept run."""
from __future__ import annotations

import argparse
import http.server
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import pytest

HERE = (Path(__file__).resolve().parents[3] / "benchmarks" / "network_traces")
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(HERE), str(Path(__file__).resolve().parent), str(ROOT / "benchmarks" / "one_stage")]
sys.path.append(str(ROOT))  # for benchmarks.network_traces

import attester_front as F  # noqa: E402
import attester_piece as CP  # noqa: E402
import served_verdicts as S  # noqa: E402
from test_served_session import CHALLENGER_HOST, LIVE_GATE, RUN, STUB  # noqa: E402
from test_attester_piece import GATE_HOST, GUARD, K, LATE, OUTPUTS, PASSED, PRIVATE, _write, network_attester  # noqa: E402,F401


PEER = "some-served-node"
#: the piece's close of a relay session nobody closed, as `attester_piece.close_open` runs it
CLOSE = f'{shlex.quote(sys.executable)} {shlex.quote(str(HERE / "attester_front.py"))} close-open "$1"'
#: the front's --gate as the site names the gate, which `challenger_host` resolves
GATE = ("--gate", F.SS.GATE)


@pytest.fixture(autouse=True)
def challenger_host(monkeypatch):
    monkeypatch.setenv("VY_CHALLENGER_1_HOST", CHALLENGER_HOST)


def _stub_gate(tmp_path: Path, monkeypatch) -> None:
    """`test_served_session.STUB` as ssh: the gate it reaches writes its request to tmp_path/request."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ssh").write_text(STUB)
    (tmp_path / "bin" / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.defpath}")
    monkeypatch.setenv("STUB_DIR", str(tmp_path))


def test_the_front_sends_one_group_of_the_class_and_keeps_the_fetch_in_the_serve_dirs_private(tmp_path, monkeypatch):
    _stub_gate(tmp_path, monkeypatch)
    serve, out = tmp_path / "serve", tmp_path / "out" / "certifier-front"
    assert F.main(["front", "18190", str(serve), str(out), *GATE, "--peer", PEER, "--key", str(tmp_path / "key")]) == 0
    assert (tmp_path / "request").read_text() == (
        f"warden-rehearse --peer {PEER} --port 18190 --minutes 7 --requests 4 --height 10 --scan --prompts "
        "--tokens 4 --scan-class w63k4 --window-per-group")
    prompts = json.loads((tmp_path / "prompts.json").read_text())
    assert [len(p) for p in prompts] == [F.PROMPT_IDS] * F.PROMPTS and all(t < 128000 for p in prompts for t in p)
    kept = serve / CP.PRIVATE / RUN / "fetch.tar.gz"
    assert [p.stat().st_mode & 0o777 for p in (serve / "private", serve / CP.PRIVATE, kept)] == [0o700, 0o700, 0o600]
    assert sorted(p.name for p in out.iterdir()) == ["bundle.tgz", "leak-guard.json", "session.json", "status.json"]
    assert (kept.parent / "start.json").stat().st_mode & 0o777 == 0o600
    assert json.loads((out / "session.json").read_text())["keep"] == str(kept)
    assert (kept.parent / CP.SESSION).read_text() == (out / "session.json").read_text()
    assert json.loads((out / "session.json").read_text())["gate"] == LIVE_GATE
    assert F.main(["front", "18190", str(serve), str(out), *GATE, "--peer", PEER, "--tokens", "64"]) == 2
    assert "past one execution phase" in json.loads((out / "session.json").read_text())["refused"]
    assert F.main(["front", "18190", str(serve), str(out), *GATE, "--peer", PEER, "--tokens", "5"]) == 2
    assert "1 to KO = 4 ids" in json.loads((out / "session.json").read_text())["refused"]
    assert F.main(["front", "18190", str(tmp_path / "serve-w63"), str(tmp_path / "out-w63"),
                    *GATE, "--peer", PEER, "--key",
                    str(tmp_path / "key"), "--scan-class", "w63"]) == 0
    assert (tmp_path / "request").read_text().endswith("--tokens 4 --scan-class w63 --window-per-group")


def test_the_front_without_the_challenger_hosts_address_is_refused_by_the_variables_name_before_the_gate(tmp_path,
                                                                                                    monkeypatch):
    """The front's default gate is `SS.GATE`, whose host the environment holds: unset, the front is refused by the
    variable's name and sends nothing; the dry run's ``--gate local`` reads no address (the test below runs it)."""
    _stub_gate(tmp_path, monkeypatch)
    monkeypatch.delenv("VY_CHALLENGER_1_HOST")
    serve, out = tmp_path / "serve", tmp_path / "out"
    assert F.main(["front", "18190", str(serve), str(out), *GATE, "--peer", PEER, "--key", str(tmp_path / "key")]) == 2
    assert json.loads((out / "session.json").read_text())["refused"].startswith("VY_CHALLENGER_1_HOST is unset: the gate is ")
    assert not (tmp_path / "request").exists()


@pytest.mark.parametrize("flags", [GATE, (*GATE, "--session", "--client-port", "8012", "--windows", "20"),
                                   ("--gate", "local")])
def test_the_front_without_its_peer_is_refused_by_the_settings_name_before_the_gate(tmp_path, monkeypatch, flags):
    """The front names no served node of its own: without ``--peer`` the rehearsal, the relay session and the dry run
    are refused, naming the binding `verity run` passes it from, and the gate is never asked."""
    _stub_gate(tmp_path, monkeypatch)
    serve, out = tmp_path / "serve", tmp_path / "out"
    assert F.main(["front", "18190", str(serve), str(out), "--key", str(tmp_path / "key"), *flags]) == 2
    why = json.loads((out / "session.json").read_text())["refused"]
    assert why == F.NO_PEER and "network-attester.network-attester.peer" in why
    assert not (tmp_path / "request").exists() and not (tmp_path / "asked").exists()


@pytest.mark.parametrize("flags", [(), ("--session", "--client-port", "8012", "--windows", "20")])
def test_the_front_without_its_gate_is_refused_by_the_settings_name_before_the_gate(tmp_path, monkeypatch, flags):
    """The front names no gate of its own: without ``--gate`` the rehearsal and the relay session are refused, naming
    the binding `verity run` passes it from and ``local``, and no gate is asked."""
    _stub_gate(tmp_path, monkeypatch)
    serve, out = tmp_path / "serve", tmp_path / "out"
    assert F.main(["front", "18190", str(serve), str(out), "--peer", PEER, *flags]) == 2
    why = json.loads((out / "session.json").read_text())["refused"]
    assert why == F.NO_GATE and "network-attester.network-attester.gate" in why and "or local" in why
    assert not (tmp_path / "request").exists() and not (tmp_path / "asked").exists()


def test_a_fronts_rehearsal_at_4000_ms_reaches_the_gate_with_its_tick_and_the_scans_schedule_there(tmp_path,
                                                                                                   monkeypatch):
    """``--gate local``, this tree's gate in this process: the run's request records the tick and the minutes counted
    at it, and the attester's command is the scan's schedule at it. The tunnel and the rehearsal are stand-ins, the
    rehearsal's check accepting and counting the requests the front asks for."""
    real, ran = F.SS.A.certifier_run, []
    check = json.dumps({"accepted": True, "problems": [], "summary": {"requests": 4}})

    def command(req, work, url):
        ran.append(F.SS.A.network_attester_command(req, work, url))
        return [sys.executable, "-c", "import sys; open(sys.argv[1], 'w').write(sys.argv[2])", str(work / "check.json"),
                check]

    def certifier_run(*args, **kw):
        ran.append(kw["tunnel_argv"]([], 1))
        return real(*args, **kw | {"tunnel_argv": lambda ssh, port: [sys.executable, "-c", ""]}, command=command,
                    up=lambda tunnel, port: True)

    monkeypatch.setattr(F.SS.A, "certifier_run", certifier_run)
    monkeypatch.setattr(F.SS, "pinned", lambda: "unpinned")
    serve, out = tmp_path / "serve", tmp_path / "out"
    assert F.main(["front", "18190", str(serve), str(out), "--peer", PEER, "--gate", "local", "--bucket-ms",
                    "4000"]) == 0
    minutes = F.SS.A.network_attester_group_minutes(1, F.TOKENS, F.HEIGHT, 4000) + F.SS.MARGIN
    assert minutes > F.SS.A.network_attester_group_minutes(1, F.TOKENS, F.HEIGHT) + F.SS.MARGIN
    (req,) = [json.loads(p.read_text()) for p in (serve / CP.PRIVATE).rglob("request.json")]
    assert {k: req.get(k) for k in ("bucket_ms", "scan", "prompts", "window_per_group", "minutes")} == {
        "bucket_ms": 4000, "scan": True, "prompts": {"count": F.PROMPTS, "ids": F.PROMPTS * F.PROMPT_IDS},
        "window_per_group": True, "minutes": minutes}
    relay, network_attester = ran
    assert relay[-5:] == ["relay", "--listen", "1", "--to", "18190"]
    at = network_attester.index("--ids")
    assert network_attester[at:at + 1 + len(F.SS.A.SCAN_SCHEDULE)] == ["--ids", *F.SS.A.scan_schedule(4000, CP.V3C_SCAN_CLASS)]
    assert F.SS.A.scan_schedule(4000) != F.SS.A.SCAN_SCHEDULE
    doc = json.loads((out / "session.json").read_text())
    assert doc["request"] == (f"warden-rehearse --peer {PEER} --port 18190 --minutes {minutes} --requests 4 "
                              "--height 10 --scan --bucket-ms 4000 --prompts --tokens 4 --scan-class w63k4 "
                              "--window-per-group")
    assert (doc["rc"], doc["verdict"]["accepted"]) == (0, True)


@pytest.mark.parametrize("tick, scan, why", [
    ("99", True, "--bucket-ms '99': 100 to 4000"),
    ("4001", True, "--bucket-ms '4001': 100 to 4000"),
    ("4000", False, "--bucket-ms: the scan's tick, with --scan"),
])
def test_a_tick_the_gate_refuses_refuses_the_front_before_the_gate_is_asked(tmp_path, monkeypatch, tick, scan, why):
    """The front's check is this tree's gate's: a tick past the gate's range, or one on a request that isn't the scan
    rehearsal (the front's always is: here its scan flags are dropped), is refused with the gate's reason."""
    _stub_gate(tmp_path, monkeypatch)
    if not scan:
        words = F.SS.words
        monkeypatch.setattr(F.SS, "words", lambda a: [w for w in words(a)
                                                       if w not in ("--scan", "--prompts", "--window-per-group")])
    serve, out = tmp_path / "serve", tmp_path / "out"
    assert F.main(["front", "18190", str(serve), str(out), *GATE, "--peer", PEER, "--key", str(tmp_path / "key"),
                    "--bucket-ms", tick]) == 2
    assert json.loads((out / "session.json").read_text())["refused"] == f"the gate refuses it: {why}"
    assert not (tmp_path / "request").exists()


def _serve_windows(d: Path, windows: list[list[str]]) -> Path:
    """A serve dir as serve-openai leaves it: window k sealed and passed, serving one request a user of ``windows[k]``."""
    d.mkdir(parents=True)
    reqs = [{"engine_id": f"cmpl-{k}{u}-0", "user": u, "windows": [k], "check": {"passed": True}}
            for k, us in enumerate(windows) for u in us]
    _write(d / "requests.jsonl", reqs)
    _write(d / "seals.jsonl", [{"window": k, "requests": [r["engine_id"] for r in reqs if r["windows"] == [k]]}
                               for k in range(len(windows))])
    _write(d / "windows.jsonl", [{"window": k, "passed": True, "commit": {"steps": 4}} for k in range(len(windows))])
    return d


def test_the_pick_passes_over_the_warm_ups_windows_for_the_groups(tmp_path):
    """Each warm-up is a window of one request, before the group's: the pick (the most requests, then the earliest)
    takes the group's window, or its largest part when the engine split it, and a smaller window after it (the load's)
    loses too; only a group split down to single requests ties the warm-ups, whose earlier window then wins."""
    warm, group, load = [["warmup-t-0"], ["warmup-t-1"]], [f"user-{j}" for j in range(4)], ["varied.00000"]
    for name, layout, want in (("whole", [*warm, group, load], 2), ("3+1", [*warm, group[:3], group[3:], load], 2),
                               ("2+2", [*warm, group[:2], group[2:], load], 2),
                               ("1+1+1+1", [*warm, *([u] for u in group), load], 0)):
        got = S.pick(_serve_windows(tmp_path / name, layout), 4)
        assert (got["window"], got["users"]) == (want, sorted(layout[want])), name


class _Endpoint(http.server.BaseHTTPRequestHandler):
    """serve-openai's completions as the warm-up sees them: each answered whole, and recorded in the serve dir's
    request_users.jsonl first (serve-openai records a request there once its window has closed) unless ``record`` is
    off; each request noted with whether the gate had been asked by then."""

    def do_POST(self):
        s = self.server
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        s.seen.append({"path": self.path, "body": body, "headers": {k.lower() for k in self.headers},
                       "gate asked": (s.stub_dir / "request").exists()})
        if s.status != 200:
            self.send_response(s.status)
            self.end_headers()
            return
        if s.record:
            with open(s.serve / "request_users.jsonl", "a") as fh:
                fh.write(json.dumps({"engine_id": f"cmpl-{len(s.seen)}-0", "user": body["user"]}) + "\n")
        out = json.dumps({"id": f"cmpl-{len(s.seen)}", "choices": [{"text": "", "finish_reason": "length"}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    _stub_gate(tmp_path, monkeypatch)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    srv.seen, srv.status, srv.record, srv.serve, srv.stub_dir = [], 200, True, tmp_path / "serve", tmp_path
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_the_front_warms_the_endpoint_up_a_window_at_a_time_before_it_asks_the_gate(tmp_path, endpoint):
    port, serve, out = endpoint.server_address[1], endpoint.serve, tmp_path / "out"
    assert F.main(["front", str(port), str(serve), str(out), *GATE, "--peer", PEER, "--key", str(tmp_path / "key"),
                    "--warmup", "2"]) == 0
    assert [(s["path"], s["gate asked"]) for s in endpoint.seen] == [("/v1/completions", False)] * 2
    assert not any(h.startswith("x-verity-release") for s in endpoint.seen for h in s["headers"])
    bodies = [s["body"] for s in endpoint.seen]
    assert [(len(b["prompt"]), b["max_tokens"], b["temperature"]) for b in bodies] == [(F.PROMPT_IDS, F.TOKENS, 0.0)] * 2
    assert F.PROMPT_IDS <= 32 and all(0 <= t < 128000 for b in bodies for t in b["prompt"])
    users = [b["user"] for b in bodies]
    assert len(set(users)) == 2 and all(u.startswith("warmup-") for u in users)
    assert (tmp_path / "request").read_text() == (
        f"warden-rehearse --peer {PEER} --port {port} --minutes 7 --requests 4 --height 10 --scan --prompts "
        "--tokens 4 --scan-class w63k4 --window-per-group")
    warm = json.loads((out / "warmup.json").read_text())["requests"]
    assert [(w["user"], w["finish_reason"]) for w in warm] == [(u, "length") for u in users]
    assert sorted(p.name for p in out.iterdir()) == ["bundle.tgz", "leak-guard.json", "session.json", "status.json",
                                                     "warmup.json"]


def test_a_warm_up_that_fails_or_isnt_recorded_closed_refuses_the_front_before_the_gate(tmp_path, endpoint):
    port, serve, out = endpoint.server_address[1], endpoint.serve, tmp_path / "out"
    endpoint.status = 503
    assert F.main(["front", str(port), str(serve), str(out), *GATE, "--peer", PEER, "--warmup", "1"]) == 2
    assert json.loads((out / "session.json").read_text())["refused"].startswith("--warmup: request 0: HTTPError")
    assert not (tmp_path / "request").exists() and not (out / "warmup.json").exists()
    endpoint.status, endpoint.record, clock = 200, False, _Clock()
    a = argparse.Namespace(port=port, warmup=1, prompt_ids=F.PROMPT_IDS, vocab=128000, tokens=F.TOKENS)
    with pytest.raises(F.SS.Refused, match="recorded no closed window of .'warmup-"):
        F.warmup(a, serve, clock=clock, sleep=clock.sleep)
    assert clock.t > F.WARMUP_CLOSE_S and not (tmp_path / "request").exists()


class _Clock:
    t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


# ------------------------------------------------------------------------------------------------------ a relay session
# The front's relay session (`front --session`, `close`, and the piece's guard for a session nobody closed), through a
# stub gate that answers a scripted run of statuses, with every nap an injected one: the front sends the gate one
# `relay-session` request and no prompts, returns once the gate's status shows the relay listening and leaves nothing
# in OUT; `close` waits for the session's end, keeps the lane's fetch where the piece reads it and publishes what a
# rehearsal's front does; the piece's steps close a session nobody closed before they read the kept run; and a request
# this tree's gate refuses is refused before the gate is asked.

#: the relay's deployment, as the gate's status gives it once the relay listens and the run's bundle at its end
#: a relay session's deployment as the gate gives it out, and as its relay wrote it in the witness: with where it
#: listened, which the gate gives no one
DEPLOYMENT = {"windows": 20, "origin_unix": 1.5, "params": {"window": 32, "bucket_ms": 100}}
WRITTEN = DEPLOYMENT | {"listen": ["127.0.0.1", 40001], "upstream": ["127.0.0.1", 40002], "origin_raw": 7.5}
#: the gate's status of a relay session before its relay listens, once it does, and at an end its check accepted
WAITING = {"run": RUN, "request": {"device": "warden", "peer": PEER, "client_port": 8012, "windows": 20}, "end": None,
           "active": True, "relay": None}
LISTENING = WAITING | {"relay": {"client": "127.0.0.1:8012", "deployment": DEPLOYMENT}}
ENDED = LISTENING | {"active": False, "end": PASSED}

SESSION_GATE = r'''#!PYTHON
import io, json, os, sys, tarfile
from pathlib import Path
d, cmd = Path(os.environ["STUB_DIR"]), sys.argv[-1]
if any("${" in w for w in sys.argv[1:-1]):
    sys.exit(f"stub ssh: an unfilled destination in {sys.argv[1:-1]!r}")
with open(d / "asked", "a") as f:
    f.write(cmd + "\n")
if cmd.startswith("relay-session "):
    (d / "stdin").write_bytes(sys.stdin.buffer.read())
    if (d / "busy").exists():
        print(json.dumps({"refused": "a rehearsal is running"}))
        sys.exit(3)
    print(json.dumps({"device": "warden", "run": "RUN", "unit": "vy-certifier@RUN.service"}, sort_keys=True))
elif cmd == "status RUN":
    left = json.loads((d / "statuses.json").read_text())
    (d / "statuses.json").write_text(json.dumps(left[1:] or left))
    if left[0] is None:
        sys.exit("stub gate: no answer")
    print(json.dumps(left[0], sort_keys=True))
elif cmd == "fetch RUN" and not (d / "no-fetch").exists():
    status = json.loads((d / "statuses.json").read_text())[0]
    files = {"request.json": json.dumps(status["request"]).encode(), "end.json": json.dumps(status["end"]).encode(),
             "out/deployment.json": json.dumps(DEPLOYMENT).encode(),
             "witness/certified.jsonl": b"a certified line, which only the run's own lane fetches\n"}
    if ((status["end"] or {}).get("verdict") or {}).get("accepted"):   # the gate publishes only an accepted run's
        files["out/certificates.jsonl"] = b""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo("RUN/" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    sys.stdout.buffer.write(buf.getvalue())
else:
    sys.exit(f"stub gate: no verb {cmd!r}")
'''.replace("PYTHON", sys.executable).replace("DEPLOYMENT", repr(DEPLOYMENT)).replace("RUN", RUN)


def _statuses(tmp_path: Path, statuses: list[dict | None]) -> None:
    """What the stub gate answers ``status`` from here on, one a call and the last ever after (None: no answer)."""
    (tmp_path / "statuses.json").write_text(json.dumps(statuses))


def _session_gate(tmp_path: Path, monkeypatch, statuses: list[dict | None]) -> None:
    """`SESSION_GATE` as ssh: each request it gets is a line of tmp_path/asked."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ssh").write_text(SESSION_GATE)
    (tmp_path / "bin" / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.defpath}")
    monkeypatch.setenv("STUB_DIR", str(tmp_path))
    _statuses(tmp_path, statuses)


def _asked(tmp_path: Path) -> list[str]:
    return (tmp_path / "asked").read_text().splitlines() if (tmp_path / "asked").is_file() else []


def _front(tmp_path: Path, *flags: str, serve: str = "serve", gate: str = f"research@{GATE_HOST}") -> list[str]:
    """``front --session``'s words: the endpoint at 18190, SERVE_DIR tmp_path/`serve` and OUT tmp_path/out."""
    return ["front", "18190", str(tmp_path / serve), str(tmp_path / "out"), "--key", str(tmp_path / "key"),
            "--gate", gate, "--peer", PEER, "--session", *flags]


SESSION_FLAGS = ("--client-port", "8012", "--windows", "20")


def test_the_front_opens_a_relay_session_sends_no_group_and_returns_once_its_relay_listens(tmp_path, monkeypatch,
                                                                                           capsys):
    _session_gate(tmp_path, monkeypatch, [LISTENING])
    serve, out = tmp_path / "serve", tmp_path / "out"
    assert F.main(_front(tmp_path, *SESSION_FLAGS)) == 0
    minutes = F.SS.A.session_minutes(20, F.HEIGHT) + F.SS.MARGIN
    assert _asked(tmp_path) == [f"relay-session --peer {PEER} --port 18190 --client-port 8012 --windows 20 "
                                f"--minutes {minutes} --height 10 --scan-class w63k4", f"status {RUN}"]
    assert (tmp_path / "stdin").read_bytes() == b"" and list(out.iterdir()) == []
    keep = serve / CP.PRIVATE
    kept = keep / RUN / CP.RELAY
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"run": RUN, "client": "127.0.0.1:8012",
                                                                    "relay": str(kept)}
    doc = json.loads(kept.read_text())
    assert (doc["run"], doc["gate"], doc["client"], doc["deployment"]) == (RUN, f"research@{GATE_HOST}",
                                                                           "127.0.0.1:8012", DEPLOYMENT)
    assert doc["args"] == {"peer": PEER, "port": 18190, "client_port": 8012, "windows": 20, "minutes": minutes,
                           "height": 10, "bucket_ms": None, "scan_class": "w63k4"}
    assert sorted(p.name for p in (keep / RUN).iterdir()) == [CP.RELAY, "start.json"]
    assert [p.stat().st_mode & 0o777 for p in (serve / "private", keep, keep / RUN, kept, keep / RUN / "start.json")] \
        == [0o700, 0o700, 0o700, 0o600, 0o600]
    assert [d["run"] for d in CP.open_sessions(keep)] == [RUN]
    # a second session in the serve dir is refused while this one is open, before the gate is asked
    assert F.main(_front(tmp_path, "--client-port", "8013", "--windows", "20")) == 2
    assert f"relay session {RUN} is open" in json.loads((out / "session.json").read_text())["refused"]
    assert len(_asked(tmp_path)) == 2
    # the scan's tick and class reach the gate, and the minutes are counted at the tick
    assert F.main(_front(tmp_path, *SESSION_FLAGS, "--bucket-ms", "200", "--scan-class", "w63", serve="serve-200")) == 0
    at_200 = F.SS.A.session_minutes(20, F.HEIGHT, 32 * 200) + F.SS.MARGIN
    assert at_200 > minutes and _asked(tmp_path)[2].endswith(
        f"--windows 20 --minutes {at_200} --height 10 --bucket-ms 200 --scan-class w63")
    # the longest session at the scan's tick: the least height that signs its execution phases, and its minutes at it
    assert F.main(_front(tmp_path, "--client-port", "8012", "--windows", "1119", serve="serve-long")) == 0
    assert (F.SS.A.network_attester_height(1119), F.SS.A.session_minutes(1119, 13) + F.SS.MARGIN) == (13, 66)
    assert _asked(tmp_path)[4].endswith("--windows 1119 --minutes 66 --height 13 --scan-class w63k4")
    assert F.main(_front(tmp_path, "--client-port", "8012", "--windows", "1119", "--bucket-ms", "200",
                          serve="serve-longest")) == 0
    assert _asked(tmp_path)[6].endswith("--windows 1119 --minutes 126 --height 13 --bucket-ms 200 --scan-class w63k4")


@pytest.mark.parametrize("statuses, rc, naps, why", [
    ([WAITING, None, WAITING, LISTENING], 0, 3, None),
    ([WAITING, WAITING | {"active": False, "end": {"run": RUN, "rc": 2, "ended": "tunnel"}}], 1, 1,
     f"relay session {RUN} ended before its relay listened: exit 2, tunnel"),
    ([WAITING | {"active": False}], 1, 1, f"relay session {RUN}: its unit ended and wrote no end.json"),
    ([WAITING], 1, 5 * 60 + 1, f"relay session {RUN}: 5 minutes of asking"),
])
def test_a_sessions_front_waits_on_the_gates_status_and_says_why_one_never_listened(tmp_path, monkeypatch, statuses,
                                                                                    rc, naps, why):
    """The front asks the gate's status every `READY_POLL_S` seconds (a gate that doesn't answer is asked again) until
    the relay listens. A session that ended first, a unit twice seen ended with no end, and a session's own minutes of
    asking each end the front with exit 1 and the reason; the session stays open, so a close keeps its fetch."""
    _session_gate(tmp_path, monkeypatch, statuses)
    if naps > 3:  # a gate that never shows the relay: its answer without the stub's process each time
        monkeypatch.setattr(F, "gate_status", lambda a, keep, run: WAITING)
    slept = []
    a = F.parser().parse_args(_front(tmp_path, *SESSION_FLAGS, "--minutes", "5"))
    assert F.front(a, sleep=slept.append) == rc
    assert slept == [F.READY_POLL_S] * naps
    doc = json.loads((tmp_path / "serve" / CP.PRIVATE / RUN / CP.RELAY).read_text())
    assert [d["run"] for d in CP.open_sessions(tmp_path / "serve" / CP.PRIVATE)] == [RUN]
    if why is None:
        assert doc["client"] == "127.0.0.1:8012" and not (tmp_path / "out" / "session.json").exists()
    else:
        assert "client" not in doc and "deployment" not in doc
        assert json.loads((tmp_path / "out" / "session.json").read_text())["refused"].startswith(why)


@pytest.mark.parametrize("flags, why", [
    ((), "--session: with --client-port PORT"),
    (("--client-port", "8012"), "--session: with --client-port PORT"),
    ((*SESSION_FLAGS, "--warmup", "1"), "--session sends no request of its own"),
    (("--client-port", "18190", "--windows", "20"), "the gate refuses it: --client-port 18190: a port of the peer's"),
    (("--client-port", "7811", "--windows", "20"), "the gate refuses it: --client-port 7811: a port of the peer's"),
    (("--client-port", "8012", "--windows", "1120"), "the gate refuses it: --windows '1120': 1 to 1119"),
    (("--client-port", "8012", "--windows", "1100", "--height", "11"),
     "the gate refuses it: --height 11: the key's 2048 one-time signatures don't sign 1100 execution phases"),
    ((*SESSION_FLAGS, "--minutes", "4"),
     "the gate refuses it: --minutes 4: 20 execution phases of 3200 ms take 5 minutes"),
    ((*SESSION_FLAGS, "--bucket-ms", "50"), "the gate refuses it: --bucket-ms '50': 100 to 4000"),
])
def test_a_session_this_trees_gate_refuses_is_refused_before_the_gate_is_asked(tmp_path, monkeypatch, flags, why):
    _session_gate(tmp_path, monkeypatch, [LISTENING])
    assert F.main(_front(tmp_path, *flags)) == 2
    assert json.loads((tmp_path / "out" / "session.json").read_text())["refused"].startswith(why)
    assert _asked(tmp_path) == [] and list((tmp_path / "serve" / CP.PRIVATE).iterdir()) == []


def test_a_sessions_front_at_the_default_gate_reaches_the_address_the_environment_holds_and_unset_is_refused(
        tmp_path, monkeypatch):
    """``front --session`` at `SS.GATE` fills its host from the environment before the gate is asked, so the kept
    relay, which ``close`` reads its gate from, names the address; unset, the front is refused by the variable's name
    and asks nothing."""
    _session_gate(tmp_path, monkeypatch, [LISTENING])
    monkeypatch.delenv("VY_CHALLENGER_1_HOST")
    assert F.main(_front(tmp_path, *SESSION_FLAGS, gate=F.SS.GATE)) == 2
    assert json.loads((tmp_path / "out" / "session.json").read_text())["refused"].startswith(
        "VY_CHALLENGER_1_HOST is unset: the gate is ")
    assert _asked(tmp_path) == [] and not (tmp_path / "serve" / CP.PRIVATE / RUN).exists()
    monkeypatch.setenv("VY_CHALLENGER_1_HOST", CHALLENGER_HOST)
    assert F.main(_front(tmp_path, *SESSION_FLAGS, gate=F.SS.GATE)) == 0
    assert json.loads((tmp_path / "serve" / CP.PRIVATE / RUN / CP.RELAY).read_text())["gate"] == LIVE_GATE
    assert len(_asked(tmp_path)) == 2


def test_a_session_the_gate_doesnt_start_opens_none(tmp_path, monkeypatch):
    _session_gate(tmp_path, monkeypatch, [LISTENING])
    (tmp_path / "busy").touch()
    assert F.main(_front(tmp_path, *SESSION_FLAGS)) == 2
    refused = json.loads((tmp_path / "out" / "session.json").read_text())["refused"]
    assert refused.startswith("the gate didn't start the session (exit 3)") and "a rehearsal is running" in refused
    assert len(_asked(tmp_path)) == 1 and CP.open_sessions(tmp_path / "serve" / CP.PRIVATE) == []
    assert F.main(["close", str(tmp_path / "serve"), str(tmp_path / "out-close")]) == 2
    assert json.loads((tmp_path / "out-close" / "session.json").read_text())["refused"].startswith(
        "0 relay sessions open")
    assert len(_asked(tmp_path)) == 1


def test_close_waits_for_the_sessions_end_keeps_its_fetch_and_publishes_what_a_rehearsals_front_does(tmp_path,
                                                                                                     monkeypatch):
    _session_gate(tmp_path, monkeypatch, [LISTENING])
    serve, out = tmp_path / "serve", tmp_path / "out-close"
    assert F.main(_front(tmp_path, *SESSION_FLAGS)) == 0
    _statuses(tmp_path, [LISTENING, None, LISTENING, ENDED])
    slept = []
    assert F.close(argparse.Namespace(serve_dir=serve, out=out), sleep=slept.append) == 0
    assert slept == [F.SS.POLL_S] * 3
    assert _asked(tmp_path)[2:] == [f"status {RUN}"] * 4 + [f"fetch {RUN}"]
    kept = serve / CP.PRIVATE / RUN / "fetch.tar.gz"
    assert kept.stat().st_mode & 0o777 == 0o600 and CP.kept_run(serve) == RUN
    assert sorted(p.name for p in out.iterdir()) == ["bundle.tgz", "leak-guard.json", "session.json", "status.json"]
    assert json.loads((out / "status.json").read_text()) == {k: v for k, v in ENDED.items() if k != "relay"}
    with tarfile.open(kept) as full, tarfile.open(out / "bundle.tgz") as shown:
        assert f"{RUN}/witness/certified.jsonl" in full.getnames()
        assert sorted(shown.getnames()) == [f"{RUN}/end.json", f"{RUN}/out/certificates.jsonl",
                                            f"{RUN}/out/deployment.json", f"{RUN}/request.json"]
    doc = json.loads((out / "session.json").read_text())
    assert (doc["run"], doc["rc"], doc["ended"], doc["verdict"]["accepted"], doc["windows"], doc["gate"]) == (
        RUN, 0, "ended", True, 20, f"research@{GATE_HOST}")
    assert doc["request"] == _asked(tmp_path)[0] and doc["keep"] == str(kept)
    assert (kept.parent / CP.SESSION).read_text() == (out / "session.json").read_text()
    assert CP.placement(serve, RUN)["parties"] == {"network_certifier": CP.ATTESTER_MACHINE,
                                                   "prover": CP.PROVER_MACHINE, "verifier": CP.UNNAMED}
    assert CP.open_sessions(serve / CP.PRIVATE) == []
    # with the session closed there is none to close, and the gate isn't asked
    asked = len(_asked(tmp_path))
    assert F.main(["close", str(serve), str(tmp_path / "out-again")]) == 2
    assert len(_asked(tmp_path)) == asked


def test_a_session_the_gates_check_refused_is_closed_and_kept_and_the_piece_doesnt_answer_for_it(tmp_path, monkeypatch):
    """The whole-run rule across a session: an end whose check didn't accept the run (here each connection late) is
    `close`'s exit, its fetch is kept all the same, and the piece's steps refuse on it (`rehearsed`), which leaves the
    network attester refused by that sentence for every window of the served run."""
    _session_gate(tmp_path, monkeypatch, [LISTENING])
    serve = tmp_path / "serve"
    assert F.main(_front(tmp_path, *SESSION_FLAGS)) == 0
    _statuses(tmp_path, [LISTENING | {"active": False, "end": LATE}])
    assert F.main(["close", str(serve), str(tmp_path / "out-close")]) == 1
    doc = json.loads((tmp_path / "out-close" / "session.json").read_text())
    assert (doc["rc"], doc["verdict"]["accepted"]) == (1, False) and CP.kept_run(serve) == RUN
    unpacked, run = CP.unpack(serve / CP.PRIVATE / RUN / "fetch.tar.gz")
    try:
        with pytest.raises(CP.NotRun):
            CP.rehearsed(run)
    finally:
        CP.rm(unpacked)


def test_the_pieces_steps_close_a_relay_session_nobody_closed_before_they_read_the_kept_run(tmp_path, monkeypatch,
                                                                                            capsys):
    _session_gate(tmp_path, monkeypatch, [LISTENING])
    serve, keep = tmp_path / "serve", tmp_path / "serve" / CP.PRIVATE
    assert F.main(_front(tmp_path, *SESSION_FLAGS)) == 0
    _statuses(tmp_path, [LISTENING, ENDED])
    slept = []
    assert F.close_open(serve, sleep=slept.append) == 0
    assert slept == [F.SS.POLL_S] and CP.kept_run(serve) == RUN and CP.open_sessions(keep) == []
    assert sorted(p.name for p in (keep / RUN).iterdir()) == ["fetch.tar.gz", CP.RELAY, CP.SESSION, "start.json"]
    assert (keep / RUN / "fetch.tar.gz").stat().st_mode & 0o777 == 0o600
    assert json.loads((keep / RUN / CP.SESSION).read_text())["gate"] == f"research@{GATE_HOST}"
    # nothing is published and no hold is left: the guard keeps, and a close is what publishes
    assert list((keep / "held").iterdir()) == [] and list((tmp_path / "out").iterdir()) == []
    assert f"relay session {RUN} closed here: no close ran" in capsys.readouterr().out
    asked = len(_asked(tmp_path))
    assert F.close_open(serve) == 0
    CP.close_open(serve, None)
    assert len(_asked(tmp_path)) == asked
    # a step closes it first: by the time it reads the kept run, the session's fetch is the kept one
    again, seen = tmp_path / "serve-step", []
    assert F.main(_front(tmp_path, *SESSION_FLAGS, serve="serve-step")) == 0
    _statuses(tmp_path, [ENDED])

    def step(hold: Path):
        seen.append((CP.kept_run(again), CP.open_sessions(again / CP.PRIVATE)))
        raise CP.NotRun("the step's own")

    with pytest.raises(CP.NotRun, match="the step's own"):
        CP.settle(again, K, tmp_path / "dir", "CERTIFIER_TIMEOUT", step, GUARD, CLOSE)
    assert seen == [(RUN, [])]
    # a session the gate's fetch fails for can't be closed: the step doesn't run, and the session stays open
    third = tmp_path / "serve-no-fetch"
    assert F.main(_front(tmp_path, *SESSION_FLAGS, serve="serve-no-fetch")) == 0
    (tmp_path / "no-fetch").touch()
    with pytest.raises(CP.NotRun, match=f"relay session {RUN}, which no close ended, couldn't be closed here: .*fetch"):
        CP.settle(third, K, tmp_path / "dir", "CERTIFIER_TIMEOUT", step, GUARD, CLOSE)
    assert len(seen) == 1 and [d["run"] for d in CP.open_sessions(third / CP.PRIVATE)] == [RUN]
    # without a close command the step refuses on the open session, naming it
    with pytest.raises(CP.NotRun, match=f"relay session {RUN} is open under .*: no close ran after the load"):
        CP.settle(third, K, tmp_path / "dir", "CERTIFIER_TIMEOUT", step, GUARD)
    assert len(seen) == 1
    assert list((third / CP.PRIVATE / "held").iterdir()) == []


def test_a_local_gates_relay_session_is_this_trees_gate_its_unit_a_relay_each_way(tmp_path, monkeypatch):
    """``--gate local``: this tree's gate takes the front's request on the scratch host under the serve dir's private,
    answers its status and gives its fetch, and the unit the front starts runs `auditor.certifier_run` with a `relay`
    each way in its ssh's place. The unit here is a stand-in that leaves the files a session's does."""
    spent = subprocess.Popen([sys.executable, "-c", ""])
    spent.wait()
    serve, out, started = tmp_path / "serve", tmp_path / "out", []
    keep = serve / CP.PRIVATE

    def unit(pid: int, listens: bool):
        class Unit:
            def __init__(self, argv, **kw):
                started.append(argv)
                self.pid = pid
                if listens:
                    d = F.SS.scratch_host(keep / "local-gate", PEER).attester / argv[4] / F.SS.A.WITNESS
                    d.mkdir()
                    (d / "deployment.json").write_text(json.dumps(WRITTEN))
        return Unit

    monkeypatch.setattr(F.SS, "pinned", lambda: "unpinned")
    with monkeypatch.context() as m:  # a unit that ended at once and wrote no end: the front says so
        m.setattr(CP.subprocess, "Popen", unit(spent.pid, False))
        slept = []
        a = F.parser().parse_args(_front(tmp_path, *SESSION_FLAGS, serve="serve-spent", gate="local"))
        assert F.front(a, sleep=slept.append) == 1 and slept == [F.READY_POLL_S]
    assert "its unit ended and wrote no end.json" in json.loads((out / "session.json").read_text())["refused"]
    with monkeypatch.context() as m:
        m.setattr(CP.subprocess, "Popen", unit(os.getpid(), True))
        assert F.main(_front(tmp_path, *SESSION_FLAGS, gate="local")) == 0
    (doc,) = CP.open_sessions(keep)
    run, host = doc["run"], F.SS.scratch_host(keep / "local-gate", PEER)
    assert started[-1] == [sys.executable, str(HERE / "attester_front.py"), "unit", str(keep), run, PEER, "18190"]
    assert (doc["gate"], doc["unit"], doc["client"], doc["deployment"]) == ("local", os.getpid(), "127.0.0.1:8012",
                                                                            DEPLOYMENT)
    req = json.loads((host.attester / run / "request.json").read_text())
    minutes = F.SS.A.session_minutes(20, F.HEIGHT) + F.SS.MARGIN
    assert {k: req[k] for k in ("device", "peer", "port", "client_port", "windows", "minutes", "height")} == {
        "device": "warden", "peer": PEER, "port": 18190, "client_port": 8012, "windows": 20, "minutes": minutes,
        "height": 10}
    # the unit: certifier_run on that host, its tunnel a relay to the endpoint and one from the clients' port
    ran = []

    def certifier_run(host_, run_, **kw):
        ran.append((host_.root, run_, kw["pinned"], kw["tunnel_argv"]([], 1, (8012, 2))))
        return {"rc": 0}

    with monkeypatch.context() as m:
        m.setattr(F.SS.A, "certifier_run", certifier_run)
        assert F.main(["unit", str(keep), run, PEER, "18190"]) == 0
    assert ran == [(host.root, run, "unpinned", [sys.executable, str(HERE / "attester_front.py"), "relay", "--listen",
                                                 "1", "--to", "18190", "--listen", "8012", "--to", "2"])]
    # the session's end as the unit leaves it, and the close through the same gate
    d = host.attester / run
    (d / "out").mkdir()
    (d / "out" / "deployment.json").write_text(json.dumps(DEPLOYMENT))
    (d / "end.json").write_text(json.dumps(PASSED | {"run": run}))
    assert F.main(["close", str(serve), str(tmp_path / "out-close")]) == 0
    assert CP.kept_run(serve) == run and CP.open_sessions(keep) == []
    shown = json.loads((tmp_path / "out-close" / "session.json").read_text())
    assert (shown["gate"], shown["rc"], shown["verdict"]["accepted"]) == ("local", 0, True)
    assert set(json.loads((tmp_path / "out-close" / "status.json").read_text())) == {"run", "request", "end", "active"}
    with tarfile.open(keep / run / "fetch.tar.gz") as full, tarfile.open(tmp_path / "out-close" / "bundle.tgz") as pub:
        assert f"{run}/{F.SS.A.WITNESS}/deployment.json" in full.getnames()
        assert not any(F.SS.A.WITNESS in n.split("/") for n in pub.getnames())


def test_the_local_tunnel_relays_each_listen_port_to_its_own(monkeypatch):
    """`relay --listen A --to B --listen C --to D`: one relay a pair, the last started once every other listens."""
    order = []

    def relay(listen: int, to: int, *, ready=lambda: None):
        order.append((listen, to, threading.current_thread() is threading.main_thread()))
        ready()

    monkeypatch.setattr(F, "relay", relay)
    assert F.main(["relay", "--listen", "1", "--to", "2"]) == 0
    assert F.main(["relay", "--listen", "3", "--to", "4", "--listen", "5", "--to", "6"]) == 0
    assert order == [(1, 2, True), (3, 4, False), (5, 6, True)]
    with pytest.raises(SystemExit, match="one --to for each --listen"):
        F.main(["relay", "--listen", "1", "--to", "2", "--listen", "3"])


def test_the_served_tables_scan_class_is_v3cs_setup_in_scan_zk():
    """`served_table.SCAN_CLASS` is `V3C_SCAN_CLASS`'s setup (`scan_zk.W63K4`) as served_verdicts.sh's
    SLOTS,PROMPT,OUTPUTS,LANES,EVERY: a slot stride's Program slots over an execution phase's strides, KI, KO, the
    lanes and EVERY."""
    from verity.protocols.resource_accounting.network.prover import scan_zk as SZ

    spec = importlib.util.spec_from_file_location("served_table", ROOT / "tools/verity/cli/verity_cli/served_table.py")
    table = importlib.util.module_from_spec(spec)
    sys.modules["served_table"] = table
    spec.loader.exec_module(table)
    setup = SZ.SETUPS[CP.V3C_SCAN_CLASS]
    c = setup.cls
    assert table.SCAN_CLASS == f"{c.SLOTS * setup.params.window // c.EVERY},{c.KI},{c.KO},{c.lanes},{c.EVERY}"
