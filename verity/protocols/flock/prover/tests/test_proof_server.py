"""SPOT's proof server (`verity_flock.proof_server`) on a stand-in for `flock-circuit`'s long-lived modes: each type goes to
one device (the caller's route, else the device with the fewest types), a verifier serves each connection the queue line
written beside its prover's stream line (so interleaved types and devices still meet their own lines), a session is filed
under its own directory, its verdict of record runs once both sides have recorded it, and a process that exits fails its
unfinished sessions without hanging the server's close."""
from __future__ import annotations

import json
import os
import sys
import threading
import types
from pathlib import Path

import pytest

from verity_flock import proof_server as PS
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
            sys.exit(int(os.environ.get("FAKE_PROVE_EXIT", "4")))
        j = json.loads(raw)
        host, port = j["verifier"].rsplit(":", 1)
        f = socket.create_connection((host, int(port))).makefile("rw")
        f.write(json.dumps({{"label": j["label"], "line": j, "argv": args, "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "verify_ahead": os.environ.get("FC_VERIFY_AHEAD"),
                            "threads": os.environ.get("RAYON_NUM_THREADS")}}) + "\\n")
        f.flush()
        v = json.loads(f.readline())
        print("LIVE\\t" + json.dumps({{"label": j["label"], "accepted": v["accepted"], "run": n,
                                      "device": os.environ.get("CUDA_VISIBLE_DEVICES")}}), flush=True)
'''


@pytest.fixture
def binary(tmp_path, monkeypatch):
    for k in ("FAKE_SERVE_DIES", "FAKE_PROVE_DIES_AFTER", "FAKE_PROVE_EXIT"):
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


def test_provers_a_device_each_with_its_own_verifier_of_a_type(binary, tmp_path):
    """Two provers on one device, one type: each session goes to the prover with fewer unfinished sessions, and meets its
    own line at a verifier of that prover's (one verifier taking both provers' connections would pair them by arrival)."""
    pin = "55" * 64
    with ProofServer(binary, tmp_path / "ps", devices=[3], provers=2) as ps:
        sessions = [ps.submit(statement(tmp_path, pin), draw([i % 4]), f"s{i}") for i in range(8)]
        got = [s.result() for s in sessions]
    assert {s.prover for s in got} == {"prover-gpu3-0", "prover-gpu3-1"}
    assert set(ps.exits) == {"prover-gpu3-0", "prover-gpu3-1", f"verifier-{pin[:16]}-0", f"verifier-{pin[:16]}-1"}
    assert set(ps.pids) == set(ps.exits) and all(pid > 0 for pid in ps.pids.values())
    for s in got:
        assert s.accepted and s.device == 3 and s.live["device"] == "3"
        assert s.dir.parent.name == f"verifier-{pin[:16]}-{s.prover[-1]}" and record(s)["queue"]["label"] == s.label
    with pytest.raises(ValueError, match="at least 1"):
        ProofServer(binary, tmp_path / "ps0", provers=0)


@pytest.fixture
def node(tmp_path, monkeypatch):
    """A planted node for the provers' thread split: 12 CPUs of affinity, and a cgroup v2 hierarchy with the process in
    ``pod/job`` and no quota. Returns its root and ``cpu_max(level, text)``, which writes a level's ``cpu.max``."""
    root = tmp_path / "root"
    (root / "proc/self").mkdir(parents=True)
    (root / "proc/self/cgroup").write_text("0::/pod/job\n")
    (root / "sys/fs/cgroup/pod/job").mkdir(parents=True)
    (root / "sys/fs/cgroup/pod/job/cgroup.controllers").write_text("cpu\n")
    monkeypatch.setattr(PS, "_ROOT", root, raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(12)))
    return root, lambda level, text: (root / "sys/fs/cgroup" / level / "cpu.max").write_text(text + "\n")


def test_provers_that_share_the_node_split_its_threads(binary, tmp_path, monkeypatch, node):
    """More than one prover in all (``devices`` × ``provers``): each prover's ``RAYON_NUM_THREADS`` is its even share of the
    server's threads (the server's ``RAYON_NUM_THREADS``, else rayon's default, here the planted node's 12 CPUs), at least
    one, so each prover sizes its host witness's passes for its own cores and not for every core. A lone prover's
    environment is the server's."""
    pins = ["66" * 64, "77" * 64]

    def pools(name: str, rayon: str | None, **kw) -> set[str | None]:
        if rayon is None:
            monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
        else:
            monkeypatch.setenv("RAYON_NUM_THREADS", rayon)
        with ProofServer(binary, tmp_path / name, **kw) as ps:
            sessions = [ps.submit(statement(tmp_path, pins[i % 2]), draw([i % 4]), f"s{i}") for i in range(8)]
            got = [s.result() for s in sessions]
        assert all(s.accepted for s in got)
        return {record(s)["proved"]["threads"] for s in got}

    assert pools("two-devices", "16", devices=[0, 1], provers=2) == {"4"}
    assert pools("one-device", None, devices=[3], provers=2, env={"RAYON_NUM_THREADS": "7"}) == {"3"}
    assert pools("cpus", None, devices=[3], provers=2) == {"6"}
    assert pools("unset-is-cpus", "0", devices=[0, 1]) == {"6"}
    assert pools("at-least-one", "1", devices=[0, 1], provers=2) == {"1"}
    assert pools("lone", "16") == {"16"}
    assert pools("lone-unset", None, devices=[5]) == {None}


def test_verifiers_on_their_own_cpus_and_the_provers_on_the_rest(binary, tmp_path, monkeypatch, node):
    """``FC_VERIFIER_CPUS`` in the server's environment runs every verifier on those CPUs and every prover on the server's
    others (`taskset -c`, here a stand-in that logs each placement and runs the command), the provers splitting the threads
    of their own CPUs; unset, no process is placed and the provers split every CPU, as before. A list that isn't one, names
    a CPU outside the server's affinity, or leaves the provers none is refused before any process starts."""
    log = tmp_path / "taskset.log"
    fake = tmp_path / "taskset"
    fake.write_text('#!/bin/sh\n[ "$1" = -c ] || exit 9\nprintf "%s %s\\n" "$2" "$4" >> "$FAKE_TASKSET_LOG"\nshift 2\n'
                    'exec "$@"\n')
    fake.chmod(0o755)
    monkeypatch.setattr(PS, "TASKSET", str(fake))
    monkeypatch.setenv("FAKE_TASKSET_LOG", str(log))
    for k in ("RAYON_NUM_THREADS", "FC_VERIFIER_CPUS"):
        monkeypatch.delenv(k, raising=False)
    pins = ["88" * 64, "99" * 64]

    def run(name: str, **kw) -> tuple[str | None, set[str | None]]:
        with ProofServer(binary, tmp_path / name, devices=[0], provers=2, **kw) as ps:
            sessions = [ps.submit(statement(tmp_path, pins[i % 2]), draw([i % 4]), f"s{i}") for i in range(4)]
            got = [s.result() for s in sessions]
        assert all(s.accepted for s in got)
        return ps.verifier_cpus, {record(s)["proved"]["threads"] for s in got}

    assert run("placed", env={"FC_VERIFIER_CPUS": "10,11"}) == ("10-11", {"5"})
    assert sorted(set(log.read_text().splitlines())) == ["0-9 prove", "10-11 serve"]
    log.unlink()
    assert run("unplaced") == (None, {"6"})
    assert not log.exists()
    for bad, why in [("10-", "not a CPU list"), ("11-12", "CPUs 12 are outside"), ("0-11", "none is left for the provers")]:
        with pytest.raises(ValueError, match=why):
            ProofServer(binary, tmp_path / "refused", env={"FC_VERIFIER_CPUS": bad})
    assert not log.exists()


def test_a_cgroup_quota_bounds_the_threads_the_provers_split(binary, tmp_path, monkeypatch, node):
    """With ``RAYON_NUM_THREADS`` unset, the provers split rayon's default pool, Rust's `available_parallelism`: the
    affinity, at most the least whole-CPU quota of the process's cgroup or any above it, and at least one. The quota is v2's
    ``cpu.max``, or v1's ``cpu.cfs_quota_us`` over ``cpu.cfs_period_us``, a v1 cpu line taking precedence over the v2 one; a
    quota file that is there but doesn't parse raises. Counting the affinity alone, a quota pod's provers would split more
    threads than the pod gives them."""
    root, cpu_max = node
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    cpu_max("pod/job", "max 100000")
    cpu_max("pod", "400000 100000")
    with ProofServer(binary, tmp_path / "ps", devices=[3], provers=2) as ps:
        got = [s.result() for s in [ps.submit(statement(tmp_path, "88" * 64), draw([i % 4]), f"s{i}") for i in range(4)]]
    assert {record(s)["proved"]["threads"] for s in got} == {"2"}
    for level, text, want in [("pod", "2400000 100000", 12), ("pod", "50000 100000", 1), ("pod", "max 100000", 12),
                              ("pod/job", "300000 100000", 3)]:
        cpu_max(level, text)
        assert PS._threads({}) == want, (level, text)
    cpu_max("pod", "400000")
    with pytest.raises(ValueError):
        PS._threads({})
    v1 = root / "sys/fs/cgroup/cpu,cpuacct/job"
    v1.mkdir(parents=True)
    (v1 / "cpu.cfs_quota_us").write_text("200000\n")
    (v1 / "cpu.cfs_period_us").write_text("100000\n")
    (root / "proc/self/cgroup").write_text("4:cpu,cpuacct:/job\n0::/pod/job\n")
    assert PS._threads({}) == 2
    (v1 / "cpu.cfs_quota_us").write_text("-1\n")
    assert PS._threads({}) == 12
    (root / "proc/self/cgroup").unlink()
    assert PS._threads({}) == 12


def test_one_cpu_prover_without_devices(binary, tmp_path):
    with ProofServer(binary, tmp_path / "ps") as ps:
        s = ps.submit(statement(tmp_path, "ee" * 64), draw([3]), "cpu").result()
    assert s.device is None and "--gpu" not in record(s)["proved"]["argv"] and set(ps.exits) >= {"prover-cpu"}


def test_the_wide_leaf_is_carried_by_both_sides_or_by_neither(binary, tmp_path):
    """A session's prover and verifier name one statement version: `--wide-leaf` on both command lines of a wide server,
    and on neither of any other's."""
    for wide in (True, False):
        with ProofServer(binary, tmp_path / f"ps-{wide}", wide_leaf=wide) as ps:
            r = record(ps.submit(statement(tmp_path, "ab" * 64), draw([1]), "s").result())
        sides = (r["argv"], r["proved"]["argv"])
        assert [a.count("--wide-leaf") for a in sides] == [int(wide)] * 2 and all("--zk" in a for a in sides)
        assert [a[0] for a in sides] == ["serve", "prove"]


def test_a_templated_statements_unit_reaches_its_verifier_and_its_provers_line(binary, tmp_path):
    """A templated statement's verifier loads its circuit with the unit (`--tmpl-unit`) and the prover's line names it
    (``tmpl_unit``); a flat statement on the same server names none."""
    tmpl, flat = "66" * 64, "77" * 64
    st = statement(tmp_path, tmpl)
    st = Statement(st.pin, st.circuit, st.tables, st.instances, st.public, tmpl_unit=st.circuit.parent / "unit.placed")
    with ProofServer(binary, tmp_path / "ps") as ps:
        got = [ps.submit(x, draw([i]), f"s{i}").result() for i, x in enumerate((st, statement(tmp_path, flat), st))]
    for s in got:
        r = record(s)
        if s.pin == tmpl:
            assert r["argv"][r["argv"].index("--tmpl-unit") + 1] == str(st.tmpl_unit) == r["proved"]["line"]["tmpl_unit"]
        else:
            assert "--tmpl-unit" not in r["argv"] and "tmpl_unit" not in r["proved"]["line"]


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


def test_a_prover_that_exits_0_with_a_session_unrecorded_fails_that_session(binary, tmp_path, monkeypatch):
    """A session is done on its records and never on its prover's exit status: a prover that ends with status 0 and no
    ``LIVE`` record for a line it was handed fails that line's session, as one that ends with any other status does. The
    server is closed as soon as the session fails, and its close finds the verifier's retiring thread started."""
    monkeypatch.setenv("FAKE_PROVE_DIES_AFTER", "1")
    monkeypatch.setenv("FAKE_PROVE_EXIT", "0")
    st = statement(tmp_path, "ee" * 64)
    with ProofServer(binary, tmp_path / "ps") as ps:
        first = ps.submit(st, {"units": [0]}, "s0").result()
        left = ps.submit(st, {"units": [1]}, "s1")
        with pytest.raises(ProofServerError, match=r"prover-cpu exited \(rc 0\) before its records"):
            left.result()
    assert first.accepted and left.live is None and ps.exits["prover-cpu"] == 0


def test_a_verifier_that_never_serves_fails_its_session(binary, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SERVE_DIES", "1")
    ps = ProofServer(binary, tmp_path / "ps")
    s = ps.submit(statement(tmp_path, "11" * 64), draw([0]), "s0")
    with pytest.raises(ProofServerError, match="exited"):
        s.result()
    ps.close()
    with pytest.raises(ProofServerError, match="closed"):
        ps.submit(statement(tmp_path, "11" * 64), draw([0]), "s1")


def test_a_provers_exit_fails_its_sessions_before_the_server_retires_their_verifier(binary, tmp_path, monkeypatch):
    """A session its prover left unrecorded has failed on the prover's exit before the server does anything to the
    session's verifier: before the thread that retires the verifier starts, and so before the server ends the verifier's
    queue or terminates it. The verifier's own exit, which is the server's doing, is then never the reason the session
    carries. Stand-ins record at each of the three steps whether the session has failed. The prover ends with status 0 (a
    session is done on its records, never on its prover's exit status), and the server is closed as soon as the session
    fails: its close has joined the retiring thread."""
    monkeypatch.setenv("FAKE_PROVE_DIES_AFTER", "1")
    monkeypatch.setenv("FAKE_PROVE_EXIT", "0")
    st = statement(tmp_path, "ab" * 64)
    seen: list[tuple[str, bool]] = []

    def note(step: str, then=lambda: None):
        seen.append((step, ps._sessions["s1"]._error is not None))
        return then()

    class Retiring(threading.Thread):
        def start(self) -> None:
            if self.name.startswith("retire-"):
                note("retire")
            super().start()

    spy = types.ModuleType("threading")
    vars(spy).update(vars(threading), Thread=Retiring)
    monkeypatch.setattr(PS, "threading", spy)
    with ProofServer(binary, tmp_path / "ps") as ps:
        first = ps.submit(st, draw([0]), "s0").result()
        (v,) = ps._verifiers.values()
        monkeypatch.setattr(v, "end", lambda end=v.end: note("end", end))
        monkeypatch.setattr(v.p, "terminate", lambda terminate=v.p.terminate: note("terminate", terminate))
        left = ps.submit(st, draw([1]), "s1")
        with pytest.raises(ProofServerError) as e:
            left.result()
    assert seen == [("retire", True), ("end", True), ("terminate", True)], "False: the session had not failed at that step"
    assert "prover-cpu exited" in str(e.value) and left.live is None and ps.exits["prover-cpu"] == 0
    assert first.accepted and [t.is_alive() for t in ps._retiring] == [False]


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
