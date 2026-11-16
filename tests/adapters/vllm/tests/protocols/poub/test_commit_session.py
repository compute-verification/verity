"""The vLLM adapter's PoUB session on its Commit (`remote.CommitSession`, ``--remote-verifier HOST:PORT``): the engine
arguments and targets it serves and refuses, the store binder it installs before the build, its check of the built engine,
and its audit run in lockstep with the Commit's instrumented run (`lockstep`: each pass's engine step the run's next
step), over a CPU store and PoUB's verifier on a thread.  PoUB's serving loop under it, and the helpers these tests share
with its tests (`PassClock`, `verifying`, `store`, `engine`, `TickingBackend`), are PoUB's prover's
(`tests/verity/protocols/resource_accounting/memory/poub/test_remote_session.py`)."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pytest
from verity.protocols.resource_accounting.memory.poub.prover.node_store import serving
from verity.protocols.resource_accounting.memory.poub.prover.node_store.registry import (
    ArrayBacking,
    NodeMemory,
)
from tests.verity.protocols.resource_accounting.memory.poub.test_memory_map import BLOCK
from tests.verity.protocols.resource_accounting.memory.poub.test_registry import CODEC, RECEIPT
from tests.verity.protocols.resource_accounting.memory.poub.test_remote_session import (
    P512,
    PassClock,
    TickingBackend,
    answered,
    engine,
    store,
    verifying,
)
from verity.protocols.resource_accounting.memory.poub.verifier import audit, memory_map
from verity.protocols.resource_accounting.memory.poub.verifier.session import verifier, wire

from verity_vllm.protocols.poub import remote

VLLM = (Path(__file__).resolve().parents[6] / "adapters" / "vllm")


def test_the_commits_session_refuses_engines_it_does_not_serve_and_sets_the_rest():
    cs = remote.CommitSession("127.0.0.1:7781", kv_pages=8)
    assert cs.engine_args({"max_model_len": 128}) == {"max_model_len": 128, "async_scheduling": False,
                                                       "enable_prefix_caching": False, "num_gpu_blocks_override": 8}
    for bad, why in [({"tensor_parallel_size": 2}, "parallelism"), ({"async_scheduling": True}, "async"),
                     ({"enable_prefix_caching": True}, "prefix caching")]:
        with pytest.raises(SystemExit, match=why):
            cs.engine_args(bad)
    cs.target(type("T", (), {"async_scheduling": False})())
    for t in (None, type("T", (), {"async_scheduling": None})()):
        with pytest.raises(SystemExit, match="declare \"async_scheduling\": false"):
            cs.target(t)
    refused: list[tuple[dict[str, Any], str]] = [
        ({"plan": "serve,serve"}, "one mode"), ({"kv_pages": 0}, "a page"), ({"plan": "nope"}, "a session's mode"),
        ({"plan": "servedrop:2"}, "appends nothing"), ({"filler_gib": -1}, "nonnegative"), ({"node": "n" * 65}, "node id")]
    for kw, err in refused:
        with pytest.raises(Exception, match=err):
            remote.CommitSession("127.0.0.1:7781", **kw)
    assert 1 <= len(cs.node.encode()) <= audit.NODE_BYTES


class Binding:
    """A stand-in for the store's `Binder` that records how the Commit's session makes and uses it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.made = (args, kwargs)
        self.calls: list[Any] = []

    def install(self) -> Binding:
        self.calls.append("install")
        return self

    def register_filler(self, nbytes: int, device: Any) -> None:
        self.calls.append(("register_filler", nbytes, device))

    def stats(self) -> dict[str, Any]:
        self.calls.append("stats")
        return {"weights": 3}


def test_the_commits_session_installs_a_binder_of_its_store_before_the_build(monkeypatch):
    """`CommitSession.install` makes the store's binder with no passes of its own and the KV outside the store, over
    the session's store and its ``weight_blocks``, and keeps what its `install` returns; `before_build` sets the engine
    arguments, checks the target and installs, in that order, and installs nothing for a target it refuses."""
    monkeypatch.setattr(remote, "Binder", Binding)
    cs = remote.CommitSession("127.0.0.1:7781", kv_pages=8, weight_blocks=5)
    assert cs.binder is None
    cs.install()
    assert isinstance(cs.binder, Binding) and cs.binder.calls == ["install"]
    assert cs.binder.made == ((cs.store,), {"passes": False, "weight_blocks": 5, "kv": False})

    cs = remote.CommitSession("127.0.0.1:7781", kv_pages=None)
    ekw = {"engine_args": {"max_model_len": 128}, "target": type("T", (), {"async_scheduling": False})()}
    cs.before_build(ekw)
    assert ekw["engine_args"] == {"max_model_len": 128, "async_scheduling": False, "enable_prefix_caching": False}
    assert isinstance(cs.binder, Binding) and cs.binder.made == ((cs.store,), {"passes": False, "weight_blocks": None, "kv": False})

    cs = remote.CommitSession("127.0.0.1:7781")
    with pytest.raises(SystemExit, match="declare \"async_scheduling\": false"):
        cs.before_build({"engine_args": {}, "target": None})
    assert cs.binder is None


def built(core: Any = True, asynchronous: bool = False, prefix: bool = False, tp: int = 1, client: bool = True) -> Any:
    """A stand-in for a built engine as `CommitSession.check` reads it: the engine's client of its core (none without
    ``client``) with the core it holds in this process (``core``, None when it runs in another), and the scheduler's,
    the cache's and the parallel configuration's one field each."""
    def names(**kw: Any) -> Any:
        return type("N", (), kw)()

    config = names(scheduler_config=names(async_scheduling=asynchronous), cache_config=names(enable_prefix_caching=prefix),
                   parallel_config=names(tensor_parallel_size=tp))
    return names(llm_engine=names(vllm_config=config, **({"engine_core": names(engine_core=core)} if client else {})))


def test_the_commits_session_checks_the_built_engine_and_registers_its_filler():
    """`CommitSession.check` after the build: it refuses, with every reason in order, an engine whose core is in
    another process, that schedules asynchronously, caches prefixes or has a tensor parallel size other than 1, and a
    store that registered a KV pool; else it registers ``filler_gib`` GiB of filler on the runner's device, in whole
    bytes rounded down (none at 0), logs the binder's stats and returns them."""
    said: list[str] = []
    cs = remote.CommitSession("127.0.0.1:7781", filler_gib=0.3, log=said.append)
    cs.binder = Binding()
    runner = type("Runner", (), {"device": "cuda:0"})()
    assert cs.check(built(), runner) == {"weights": 3}
    assert cs.binder.calls == [("register_filler", 322_122_547, "cuda:0"), "stats"]
    assert said == ["[delta] remote session: the store binder on the engine, {'weights': 3}"]

    cs = remote.CommitSession("127.0.0.1:7781", log=said.append)
    cs.binder = Binding()
    assert cs.check(built(), runner) == {"weights": 3} and cs.binder.calls == ["stats"] and len(said) == 2

    head = "--remote-verifier: the engine is not one the session serves: "
    whys = [("the engine core in another process", built(core=None)), ("the engine core in another process", built(client=False)),
            ("async scheduling resolved on", built(asynchronous=True)), ("prefix caching on", built(prefix=True)),
            ("tensor parallel size 2", built(tp=2)), ("tensor parallel size 0", built(tp=0)),
            ("the engine core in another process; async scheduling resolved on; prefix caching on; tensor parallel size 4",
             built(core=None, asynchronous=True, prefix=True, tp=4))]
    for why, llm in whys:
        with pytest.raises(SystemExit) as e:
            cs.check(llm, runner)
        assert str(e.value) == head + why
    cs.store = store()
    with pytest.raises(SystemExit) as e:
        cs.check(built(), runner)
    assert str(e.value) == head + "a KV pool registered (served writes stay outside the registration)"
    assert cs.binder.calls == ["stats"] and len(said) == 2


def test_the_commits_session_names_the_step_that_comes_first():
    """`CommitSession.check` before `install`, and `end` before `begin`, each raise a `RuntimeError` naming the step
    that comes first; an engine the session doesn't serve is refused before `check` asks for its binder."""
    cs = remote.CommitSession("127.0.0.1:7781", log=lambda m: None)
    runner = type("Runner", (), {"device": "cuda:0"})()
    with pytest.raises(RuntimeError) as early:
        cs.check(built(), runner)
    assert str(early.value) == "the Commit's session checks its engine once its binder is installed (`install`)"
    with pytest.raises(SystemExit, match="^--remote-verifier: the engine is not one the session serves: prefix caching on$"):
        cs.check(built(prefix=True), runner)
    with pytest.raises(RuntimeError) as early:
        cs.end()
    assert str(early.value) == "the Commit's session ends once it has begun (`begin`)"
    assert cs.binder is None and cs.lockstep is None and cs.record is None


class Stats:
    def stats(self) -> dict:
        return {}


def served_store() -> NodeMemory:
    """The Commit's store: two weights chunks and two of filler, no KV pool (N = 4)."""
    s = NodeMemory()
    w = np.random.default_rng(1).integers(0, 256, 2 * BLOCK, np.uint8)
    s.register("w", "hbm", "weights", ArrayBacking(w), block_bytes=BLOCK, codec=CODEC, receipt=RECEIPT, source="w")
    s.register("filler", "hbm", "filler", ArrayBacking(np.zeros(2 * BLOCK, np.uint8)), block_bytes=BLOCK, codec=CODEC,
               receipt=RECEIPT, source="filler:zeros")
    return s


def served_engine(s: NodeMemory):
    """The binder's steps with the KV outside the store (`Binder(kv=False)`): each numbered, writing no chunk of it."""
    def step(k: int) -> None:
        s.kv_append([], k)
        s.end_step()
    return step


def lockstep(tmp_path, monkeypatch, run_steps: int, windows: int = 2, setup: Callable[[], None] | None = None,
             each: Callable[[int], None] | None = None, pace: dict | None = None, step_ms: float = 0.0,
             served: bool = False):
    """The Commit's audit run (`CommitSession.begin` … `end`) over a CPU store, its verifier on a thread at L = 2
    windows (paced at ``pace``, `verifier.pace_of`'s), the run ``run_steps`` steps of the scripted engine, each
    through the audit run's `Lockstep` and after ``each(k)``, after ``setup()``, the run's work between `begin` and its
    first step.  ``served``: the Commit's store and steps (`served_store`, `served_engine`), else a KV pool written as
    `engine` writes it."""
    clock = PassClock()
    monkeypatch.setattr(remote, "StoreBackend", lambda *a, **kw: TickingBackend(clock, *a, **kw))
    lsock = wire.listen("127.0.0.1", 0)
    cs = remote.CommitSession(f"127.0.0.1:{lsock.getsockname()[1]}", out=tmp_path, codec=P512, log=lambda m: None,
                              step_ms=step_ms)
    cs.store, cs.binder = served_store() if served else store(), Stats()
    seen: list = []
    over, ended = socket.socketpair()

    def verify() -> None:
        sock = answered(lsock, over)
        if sock is None:
            return
        try:
            with verifying(sock, clock, windows=windows, window_ns=5, samples=4096, lifetime=2) as v:
                v.register()
                seen.append(v.session(0, v.allowance(v.ping(4)), None, pace=pace))
        except Exception as e:  # noqa: BLE001  -- a side that stops mid-run drops the connection (`verifying`)
            seen.append(e)

    th = threading.Thread(target=verify, daemon=True)
    th.start()
    steps: list[int] = []
    try:
        lock = cs.begin()
        if setup is not None:
            setup()
        step = (served_engine if served else engine)(cs.store)

        def run(k: int):
            steps.append(k)
            if each is not None:
                each(k)
            return step(k)

        for k in range(run_steps):
            lock.step(lambda k=k: run(k))
        rec = cs.end()
    except BaseException:
        # a run that fails ends its audit run, so the verifier's thread ends with it and reaps its process
        if cs._thread is not None and cs.record is None:
            cs.end()
        raise
    finally:
        # the session is over: a verifier's thread it never connected to ends here (`answered`)
        ended.close()
        th.join()
    if not seen:
        pytest.fail(f"the Commit's session ended before it connected to its verifier: {rec['error']}")
    return rec, seen[0], cs, steps


def test_a_commits_session_that_cannot_connect_fails_its_test_where_the_verifiers_thread_would_wait(tmp_path, monkeypatch):
    """`lockstep`: a session whose prover cannot connect records why and its run goes on unaudited, so no connection
    comes to the verifier's thread.  The helper ends that thread once the session is over, and fails the test with
    what the session recorded; a run that fails as well raises as it did.  Each would wait for ever to join a thread
    at ``accept()``."""
    def unreachable(host: str, port: int) -> None:
        raise ConnectionRefusedError("no verifier")

    def stops(k: int) -> None:
        raise ValueError(f"the run's step {k}")

    monkeypatch.setattr(serving.session_module("prover").wire, "connect", unreachable)
    with pytest.raises(pytest.fail.Exception) as failed:
        lockstep(tmp_path, monkeypatch, 3)
    assert str(failed.value) == ("the Commit's session ended before it connected to its verifier: "
                                 "ConnectionRefusedError: no verifier")
    assert json.loads((tmp_path / "remote_session.json").read_text())["run"] == {"steps": 3, "audited": 0, "unaudited": 3}
    with pytest.raises(ValueError, match="^the run's step 0$"):
        lockstep(tmp_path, monkeypatch, 3, each=stops)


@pytest.mark.slow
def test_the_commits_session_runs_in_lockstep_with_its_run_and_its_map_stops_with_it(tmp_path, monkeypatch):
    """Each pass's engine step is the run's next step: the audit run's steps are the run's first steps, from its first,
    and the run's steps after the audit run are unaudited and in no map line."""
    rec, r, cs, steps = lockstep(tmp_path, monkeypatch, 40, windows=3)
    assert r["accepted"] and rec["accepted"] and rec["error"] is None and steps == list(range(40))
    mm, run = rec["memory_map"], rec["run"]
    assert mm["first_step"] == 0 and mm["steps"] == run["audited"] and mm["agrees"] and mm["rekeyed"] > 0
    assert 0 < run["audited"] < 40 and run["audited"] + run["unaudited"] == 40
    head, lines = memory_map.parse((tmp_path / "memory_map.jsonl").read_text().splitlines(keepends=True))
    # the last audited step may run past the close: its pass is dropped and its append never encoded, so its line,
    # with nothing freed or written, is not written
    last = max(x for w in r["record"]["profile"]["windows"] for x in w["steps"])
    assert last <= max(st.step for st in lines) <= run["audited"] - 1 and cs.store.memory_map is None
    assert mm["deferral"]["waiting"] == mm["unrecorded"]
    assert json.loads((tmp_path / "remote_session.json").read_text())["run"] == run


@pytest.mark.slow
def test_the_commits_session_arms_its_first_window_once_its_runs_first_step_has_run(tmp_path, monkeypatch):
    """The run's work between `begin` and its first step, and that step, pass 0's, are in no window: the audit run is
    registered by then, and arms its first window (the verifier's first coin follows `armed`) once the step has run,
    so the window's clock starts at pass 0's appends and read."""
    sent: list[str] = []
    at: dict[object, list[str]] = {}
    send = wire.Conn.send

    def logged(self, head, *a, **kw):
        sent.append(head["type"])
        return send(self, head, *a, **kw)

    def setup() -> None:
        at.setdefault("setup", list(sent))

    def each(k: int) -> None:
        at.setdefault(k, list(sent))

    monkeypatch.setattr(wire.Conn, "send", logged)
    rec, r, _, _ = lockstep(tmp_path, monkeypatch, 40, windows=3, setup=setup, each=each)
    assert r["accepted"] and rec["accepted"]
    assert "ready" in at["setup"] and not {"armed", "coin"} & {*at["setup"], *at[0]}
    assert "root" in at[1] and sent.index("armed") < sent.index("coin") < sent.index("append") < sent.index("root")
    steps = [x for w in r["record"]["profile"]["windows"] for x in w["steps"]]
    assert steps[0] == rec["memory_map"]["first_step"] == 0


@pytest.mark.slow
def test_a_paced_commits_session_runs_a_step_a_coin_and_tags_each_pass_with_its_step(tmp_path, monkeypatch):
    """At ``--paced 1:…`` each window is one pass under a coin of its own, its engine step inside it (no pass moves into
    the next window); the audit run's record tags each pass with the step the verifier's record has for it, a step of
    the map; the calibration's pass times carry ``step_ms``, so the allowance covers the step.  The Commit's store is
    the weights and filler (the KV outside it), so its steps write no registered chunk, every pass reads all N, and the
    declared line is the Lean verifier's: its γ terms decide both readings, so neither is undecided and no Python
    ledger's reading is recorded beside them."""
    pace = verifier.pace_of("1:1000:4096:2", 1000 / 3)
    assert pace["groups"] == 3
    rec, r, cs, _ = lockstep(tmp_path, monkeypatch, 40, pace=pace, step_ms=2.5, served=True)
    assert r["accepted"] and rec["accepted"] and r["record"]["public"]["pace"] == {"k": 1, "window_ns": 10**9}
    wins = r["record"]["profile"]["windows"]
    assert len(wins) == 9 and all(len(w["steps"]) == 1 for w in wins) and rec["prover"]["discarded"] == 0
    mm = rec["memory_map"]
    assert mm["pass_steps"] == [[w["window"], 0, w["steps"][0]] for w in wins] and mm["pass_steps_mapped"]
    assert mm["agrees"] and mm["recorded"] == 0 and mm["deferral"]["appends"] == 0
    assert list(memory_map.parse((tmp_path / "memory_map.jsonl").read_text().splitlines(keepends=True))[1]) == []
    assert {p.length for _, p in cs.backend.plans} == {4}
    assert rec["step_ms"] == 2.5 and cs.backend.calibrate()["step_ns"] == 2_500_000
    declared = r["ledger"]["declared"]
    assert set(declared) == {"accepted", "gamma", "memory", "windows"}
    assert declared["accepted"] is True and declared["windows"] == len(wins)
    assert not any(k in r["ledger"][n] for n in ("spec", "measured") for k in ("undecided", "python_ledger"))


def test_a_run_that_ends_before_its_session_fails_the_session(tmp_path, monkeypatch):
    rec, r, _, steps = lockstep(tmp_path, monkeypatch, 3)
    assert not rec["accepted"] and "the run ended after 3 steps" in rec["error"] and steps == [0, 1, 2]
    assert rec["run"] == {"steps": 3, "audited": 3, "unaudited": 0}
    assert isinstance(r, Exception) or not r["accepted"]


def test_a_commits_session_that_fails_is_recorded_rejected_and_its_run_goes_on_unaudited(monkeypatch, tmp_path):
    """The Commit writes its verdict whatever the audit run does: an audit run that raises is rejected, its error recorded,
    and the run's steps run unaudited."""
    def fails(backend, *a, **kw):
        raise ConnectionRefusedError("no verifier")

    monkeypatch.setattr(serving, "prove", fails)
    cs = remote.CommitSession("127.0.0.1:7781", out=tmp_path)
    cs.store, cs.binder = store(), Stats()
    lock, steps = cs.begin(), []
    for k in range(4):
        lock.step(lambda k=k: steps.append(k))
    rec = cs.end()
    assert rec["accepted"] is False and rec["error"] == "ConnectionRefusedError: no verifier" and rec["memory_map"] is None
    assert steps == [0, 1, 2, 3] and rec["run"] == {"steps": 4, "audited": 0, "unaudited": 4}
    assert (tmp_path / "remote_session.json").exists()
    with pytest.raises(RuntimeError, match="one session"):
        cs.begin()


@pytest.mark.parametrize(("result", "steps"), [(None, []), ({"accepted": True}, []), ({"accepted": True, "pass_steps": None}, []),
                                               ({"accepted": True, "pass_steps": []}, []),
                                               ({"accepted": True, "pass_steps": [[0, 0, 5]]}, [[0, 0, 5]])])
def test_a_session_whose_result_leaves_its_pass_steps_out_records_none(monkeypatch, tmp_path, result, steps):
    """The audit run's record of a session whose trace began: its map's ``pass_steps`` are the prover's result's, and the
    empty list where the result leaves the field out, holds None there, or was never returned (the prover raised:
    ``result`` None here); with no step in the map, none of them is mapped."""
    def proves(backend, *a, **kw):
        reg = audit.register(bytes(32), b"n", backend.blocks, backend.payload_root(b"n"), P512, nonce=bytes(32))
        backend.encode(reg)
        backend.begin_session()
        backend.serving(reg, 2)
        if result is None:
            raise ConnectionResetError("the verifier went away")
        return [result]

    monkeypatch.setattr(serving, "prove", proves)
    cs = remote.CommitSession("127.0.0.1:7781", out=tmp_path, codec=P512, log=lambda m: None)
    cs.store, cs.binder = store(), Stats()
    cs.begin()
    rec = cs.end()
    mm = rec["memory_map"]
    assert mm["pass_steps"] == steps and mm["pass_steps_mapped"] is False and mm["first_step"] is None and mm["steps"] == 0
    assert rec["prover"] == ({} if result is None else result) and rec["run"] == {"steps": 0, "audited": 0, "unaudited": 0}
    assert rec["error"] == ("ConnectionResetError: the verifier went away" if result is None else None)
    assert rec["accepted"] is (result is not None) and mm.get("agrees") is (None if result is None else True)
    assert json.loads((tmp_path / "remote_session.json").read_text())["memory_map"]["pass_steps"] == steps


FAILING = '''
import os
from pathlib import Path

from tests.protocols.poub import test_commit_session as C
from tests.verity.protocols.resource_accounting.memory.poub import test_remote_session as T


def deliberate(*a, **kw):
    raise AssertionError("deliberate")


def test_a_session_that_fails_on_the_tests_thread(monkeypatch):
    monkeypatch.setattr(T.verifier.Verifier, "session", deliberate)
    T.run("serve")


def test_a_registration_that_fails_on_the_lockstep_verifiers_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(T.verifier.Verifier, "register", deliberate)
    _, r, _, _ = C.lockstep(tmp_path, monkeypatch, 3)
    assert not isinstance(r, Exception), r


def test_a_run_that_fails_in_its_first_step(tmp_path, monkeypatch):
    C.lockstep(tmp_path, monkeypatch, 3, each=deliberate)


def test_no_verifier_process_is_left():
    comm = T.verifier.lv.EXE[:15]
    left = []
    for d in Path("/proc").iterdir():
        try:
            stat = (d / "stat").read_text() if d.name.isdigit() else ""
        except OSError:
            continue
        name, rest = stat.split("(", 1)[-1].rsplit(")", 1) if stat else ("", "")
        if name == comm and int(rest.split()[1]) == os.getpid():
            left.append(int(d.name))
    assert left == []
'''


@pytest.mark.slow
def test_a_failing_test_leaves_no_verifier_process(tmp_path):
    """A pytest whose tests fail with the Lean verifier's process live (the verifier's audit run raising on the test's
    thread, its registration raising on `lockstep`'s verifier thread, which hung the suite while the prover waited on
    the socket, and the run's first step raising) finishes, and leaves no `poub-verifier` child, running or
    unreaped, to the test after them."""
    (tmp_path / "test_failing.py").write_text(FAILING)
    r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:xdist",
                        f"--junitxml={tmp_path / 'report.xml'}", str(tmp_path / "test_failing.py")],
                       cwd=VLLM, capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    cases = {c.get("name"): [x.get("message", "") for x in c if x.tag == "failure"]
             for c in ElementTree.parse(tmp_path / "report.xml").iter("testcase")}
    failed = ["test_a_session_that_fails_on_the_tests_thread",
              "test_a_registration_that_fails_on_the_lockstep_verifiers_thread",
              "test_a_run_that_fails_in_its_first_step"]
    assert set(cases) == {*failed, "test_no_verifier_process_is_left"}, out
    assert all(len(cases[n]) == 1 and "deliberate" in cases[n][0] for n in failed), cases
    assert cases["test_no_verifier_process_is_left"] == [], out
