"""SPOT's proof server (`verity_flock.proof_server`) on a stand-in for `flock-circuit`'s long-lived modes: each type goes to
one device (the caller's route, else the device with the fewest types), a verifier serves each connection the queue line
written beside its prover's stream line (so interleaved types and devices still meet their own lines), a session is filed
under its own directory, its verdict of record runs once both sides have recorded it, and a process that exits fails its
unfinished sessions without hanging the server's close."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from verity_flock.proof_server import ProofServer, ProofServerError, Statement

#: a stand-in for flock-circuit. ``serve --queue -``: SERVING, then for each queue line one accepted connection, whose
#: prover's label must be the line's; it files the session (``out``, else ``--out``) and prints its SESSION line.
#: ``prove --stream -``: one session a stream line, in order, its LIVE line after the verifier's answer.
FAKE = '''#!{python}
import json, os, pathlib, socket, sys, threading
args = sys.argv[1:]
opt = lambda k: args[args.index(k) + 1]
if args[0] == "serve":
    if os.environ.get("FAKE_SERVE_DIES"):
        sys.exit(3)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    print("SERVING 127.0.0.1:%d" % srv.getsockname()[1], flush=True)
    threads = []
    def one(i, conn, q):
        f = conn.makefile("rw")
        got = json.loads(f.readline())
        ok = got["label"] == q["label"]
        d = pathlib.Path(q.get("out") or opt("--out")) / ("l%04d" % i)
        d.mkdir(parents=True, exist_ok=True)
        (d / "session.json").write_text(json.dumps({{"queue": q, "proved": got, "argv": args, "verdict": {{"accepted": ok}}}}))
        f.write(json.dumps({{"accepted": ok}}) + "\\n")
        f.flush()
        conn.close()
        line = {{"session": "l%04d" % i, "accepted": ok, "label": q["label"]}}
        if "out" in q:
            line["out"] = q["out"]
        print("SESSION\\t" + json.dumps(line), flush=True)
    for i, raw in enumerate(l for l in sys.stdin if l.strip()):
        q = json.loads(raw)
        conn, _ = srv.accept()
        threads.append(threading.Thread(target=one, args=(i, conn, q)))
        threads[-1].start()
    for t in threads:
        t.join()
elif args[0] == "prove":
    die = int(os.environ.get("FAKE_PROVE_DIES_AFTER", "-1"))
    for n, raw in enumerate(sys.stdin):
        if n == die:
            sys.exit(4)
        j = json.loads(raw)
        host, port = j["verifier"].rsplit(":", 1)
        f = socket.create_connection((host, int(port))).makefile("rw")
        f.write(json.dumps({{"label": j["label"], "line": j, "argv": args, "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "verify_ahead": os.environ.get("FC_VERIFY_AHEAD")}}) + "\\n")
        f.flush()
        v = json.loads(f.readline())
        print("LIVE\\t" + json.dumps({{"label": j["label"], "accepted": v["accepted"], "run": n,
                                      "device": os.environ.get("CUDA_VISIBLE_DEVICES")}}), flush=True)
'''


@pytest.fixture
def binary(tmp_path, monkeypatch):
    for k in ("FAKE_SERVE_DIES", "FAKE_PROVE_DIES_AFTER"):
        monkeypatch.delenv(k, raising=False)
    b = tmp_path / "flock-circuit"
    b.write_text(FAKE.format(python=sys.executable))
    b.chmod(0o755)
    return b


def statement(tmp_path: Path, pin: str) -> Statement:
    d = tmp_path / "stage" / pin
    return Statement(pin=pin, circuit=d / "circuit.txt", tables=d / "tables", instances=d / "inst-4.bin",
                     public=d / "pub-4.bin")


def draw(units: list[int]) -> dict:
    return {"law": "subset", "population": 4, "k": len(units), "units": units}


def record(s) -> dict:
    return json.loads((s.dir / "session.json").read_text())


def test_types_routed_across_devices_each_session_on_its_own_line(binary, tmp_path):
    pins = ["aa" * 64, "bb" * 64, "cc" * 64]
    with ProofServer(binary, tmp_path / "ps", devices=[0, 1], verify_ahead=2) as ps:
        sessions = [ps.submit(statement(tmp_path, pins[i % 3]), draw([i % 4]), f"s{i}", verdict=lambda s: record(s)["queue"])
                    for i in range(9)]
        got = [(s.result(), s.verdict()) for s in sessions]
    assert ps.routes == {pins[0]: 0, pins[1]: 1, pins[2]: 0}
    assert set(ps.exits) == {"prover-gpu0", "prover-gpu1", *(f"verifier-{p[:16]}" for p in pins)}
    assert all(rc == 0 for rc in ps.exits.values())
    for i, (s, q) in enumerate(got):
        r = record(s)
        assert s.accepted and s.device == ps.routes[s.pin] and s.live["device"] == str(s.device)
        assert q == r["queue"] and q["label"] == f"s{i}" and q["draw"]["units"] == [i % 4]
        assert q["instances"].endswith("pub-4.bin") and r["proved"]["line"]["instances"].endswith("inst-4.bin")
        assert r["proved"]["verify_ahead"] == "2" and "--gpu" in r["proved"]["argv"] and "--zk" in r["argv"]
        assert s.dir.parent == tmp_path / "ps" / f"verifier-{s.pin[:16]}"


def test_a_caller_route_a_session_dir_and_a_negative_control(binary, tmp_path):
    pin = "dd" * 64
    with ProofServer(binary, tmp_path / "ps", devices=[0, 1], route={pin: 1}) as ps:
        s = ps.submit(statement(tmp_path, pin), draw([0, 1]), "neg", out=tmp_path / "leg" / "sessions", self_check="skip")
        assert s.result().device == 1 and s.verdict() is None
        with pytest.raises(ValueError, match="already submitted"):
            ps.submit(statement(tmp_path, pin), draw([0]), "neg")
    assert s.dir.parent == tmp_path / "leg" / "sessions"
    assert record(s)["proved"]["line"]["self_check"] == "skip"


def test_one_cpu_prover_without_devices(binary, tmp_path):
    with ProofServer(binary, tmp_path / "ps") as ps:
        s = ps.submit(statement(tmp_path, "ee" * 64), draw([3]), "cpu").result()
    assert s.device is None and "--gpu" not in record(s)["proved"]["argv"] and set(ps.exits) >= {"prover-cpu"}


def test_a_dead_prover_fails_its_session_and_the_next_starts_afresh(binary, tmp_path, monkeypatch):
    """The prover exits on reading its second line: that session fails, its verifier (whose queue holds it) is retired,
    and the next session meets a new prover and verifier on their first lines."""
    monkeypatch.setenv("FAKE_PROVE_DIES_AFTER", "1")
    pin = "ff" * 64
    with ProofServer(binary, tmp_path / "ps") as ps:
        first = ps.submit(statement(tmp_path, pin), draw([0]), "s0").result()
        with pytest.raises(ProofServerError, match="prover-cpu exited"):
            ps.submit(statement(tmp_path, pin), draw([1]), "s1").result()
        last = ps.submit(statement(tmp_path, pin), draw([2]), "s2").result()
    assert first.accepted and last.accepted and record(last)["queue"]["label"] == "s2"
    v = f"verifier-{pin[:16]}"
    assert ps.exits == {"prover-cpu": 4, "prover-cpu.2": 0, v: ps.exits[v], f"{v}.2": 0} and ps.exits[v] != 0


def test_a_verifier_that_never_serves_fails_its_session(binary, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SERVE_DIES", "1")
    ps = ProofServer(binary, tmp_path / "ps")
    s = ps.submit(statement(tmp_path, "11" * 64), draw([0]), "s0")
    with pytest.raises(ProofServerError, match="exited"):
        s.result()
    ps.close()
    with pytest.raises(ProofServerError, match="closed"):
        ps.submit(statement(tmp_path, "11" * 64), draw([0]), "s1")


def test_each_verifier_on_its_own_challenger_audited_at_close(binary, tmp_path, monkeypatch):
    """With challengers, each type's verifier is started with its challenger's flags, and `close` hands each challenger
    every session its verifier filed, at once, and gives each session its verifier's result (a stand-in challenger: the
    real one is `challengers`', whose audit `prove_leg`'s FLOCK_CHALLENGER=1 run checks)."""
    from verity_flock import proof_server as PS

    started, ended = [], {}
    monkeypatch.setattr(PS.CH, "start", lambda d: (started.append(d), (object(), ["--tree-from", f"127.0.0.1:{len(started)}",
                                                                              "--tree-secret-file", str(d / "s")]))[1])
    monkeypatch.setattr(PS.CH, "end", lambda d, p, sessions: ended.setdefault(d.name, {"ok": d.name.startswith("verifier-33"),
                                                                                        "sessions": sorted(sessions)}))
    pins = ["33" * 64, "44" * 64]
    with ProofServer(binary, tmp_path / "ps", challenger=True) as ps:
        sessions = [ps.submit(statement(tmp_path, pins[i % 2]), draw([i]), f"s{i}").result() for i in range(4)]
        assert ps.challengers == {}
    names = [f"verifier-{p[:16]}" for p in pins]
    assert [d.name for d in started] == [f"{n}-challenger" for n in names] and set(ps.challengers) == set(names)
    for s in sessions:
        name = f"verifier-{s.pin[:16]}"
        assert record(s)["argv"][-4:-2] == ["--tree-from", f"127.0.0.1:{names.index(name) + 1}"]
        assert s.challenger is ps.challengers[name] and s.challenger["ok"] == (s.pin == pins[0])
        assert s.challenger["sessions"] == sorted(str(x.dir) for x in sessions if x.pin == s.pin)


def test_a_failed_verdict_of_record_is_the_session_verdict_error(binary, tmp_path):
    def boom(s):
        raise RuntimeError("lean refused")

    with ProofServer(binary, tmp_path / "ps") as ps:
        s = ps.submit(statement(tmp_path, "22" * 64), draw([0]), "s0", verdict=boom)
        assert s.result().accepted
        with pytest.raises(RuntimeError, match="lean refused"):
            s.verdict()
