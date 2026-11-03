"""serve-openai's PoUB session (`serve_remote`) on CPU: its options and refusals before the model loads, the
engine's arguments under it, and, over the stand-in engine and a prover that drives the session's `Lockstep` as
`remote.prove` does, the server's engine steps paced by the session from its first, the steps after it unaudited, the
registered bytes a seal carries read before it, and a session that fails or outlasts the server rejected."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm import llm as L
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import serve_commit as sc
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import serve_remote as SR
from verity_vllm.protocols.remote_bandwidth import remote

from tests.pipeline.test_serve_openai import CERTIFIER_BODY, Serving

VERIFIER = ("--remote-verifier", "127.0.0.1:7781")


def parse(*more: str) -> so.Options:
    return cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--no-commit", *more], environ={})


def refusal(opts: so.Options) -> str | None:
    try:
        so.preflight(opts, which=lambda _: "/v/bin/ninja")
    except cli.UsageError as e:
        return str(e)
    return None


class Backend:
    """`remote.StoreBackend` as `CommitSession` builds it, its engine the session's `Lockstep`."""

    def __init__(self, store: Any, engine: Any, *, path: Any = None, device: Any = None) -> None:
        self.store, self.engine, self.path, self.device = store, engine, path, device
        self.trace, self.gates, self.step_ns = None, [], 0

    def layout(self) -> dict:
        return {}


def prover(passes: int, seen: list[str]):
    """`remote.prove` as the session's thread runs it: registered, then a pass an engine step, ``passes`` of them."""
    def prove(backend: Backend, *_a: Any, **_kw: Any) -> list[dict]:
        seen.append("registered")
        backend.engine.arrived()
        for s in range(passes):
            backend.engine(s)
        return [{"accepted": True, "pass_steps": []}]
    return prove


def session(tmp_path, monkeypatch, prove: Any) -> SR.ServeSession:
    monkeypatch.setattr(remote, "StoreBackend", Backend)
    monkeypatch.setattr(remote, "prove", prove)
    monkeypatch.setattr(remote.store_map, "agreement", lambda _b: {})
    rs = SR.session_of(parse(*VERIFIER), tmp_path)
    rs.cs.binder = SimpleNamespace(stats=lambda: {"blocks": 2})     # the binder's install patches vLLM's classes
    return rs


def serve(tmp_path, rs: SR.ServeSession, requests: int, order: list[str]) -> Serving:
    """``requests`` certifier requests, one after another, each 4 engine steps, on a server whose session began as `main`
    begins it: after the windows were built, before the first request."""
    s = Serving(tmp_path)
    s.windows.read_extras = lambda: order.append("extras")
    s.loop.engine = rs.paced(s.loop.engine)
    rs.begin(s.windows)
    with s:
        for i in range(requests):
            assert s.post("/v1/completions", dict(CERTIFIER_BODY, prompt=[1, 2 + i]))[0] == 200
    return s


def test_the_remote_options_parse_into_their_group_and_are_refused_before_the_model_loads():
    opts = parse(*VERIFIER, "--remote-filler-gib", "1.5")
    assert (opts.remote.remote_verifier, opts.remote.remote_plan, opts.remote.remote_filler_gib) == ("127.0.0.1:7781", "serve", 1.5)
    assert refusal(opts) is None and SR.session_of(parse(), "/tmp") is None and SR.accepted(None)
    assert "HOST:PORT" in refusal(parse("--remote-verifier", "nohost"))
    assert "is serve" in refusal(parse(*VERIFIER, "--remote-plan", "servedrop:1"))
    assert "kv_lib" in refusal(parse(*VERIFIER, "--remote-session-lib", "session.so"))
    assert "--prefix-caching" in refusal(parse(*VERIFIER, "--prefix-caching"))
    h1 = parse(*VERIFIER, "--remote-session-lib", "session.so", "--remote-kv-lib", "kv.so", "--protocols", "pouw:eps1-sm120-h1fx1")
    assert "host mirror" in SR.refusal(h1)
    assert SR.refusal(parse(*VERIFIER, "--remote-kv-lib", "kv.so", "--protocols", "pouw:eps1-sm120-h1fx1")) is None


def test_the_engine_is_built_synchronous_on_its_own_kv_pool_with_the_store_binder_installed(tmp_path, monkeypatch):
    installed: list[bool] = []
    monkeypatch.setattr(remote.CommitSession, "install", lambda self: installed.append(True))
    opts = parse(*VERIFIER)
    kw = SR.ServeSession(opts, tmp_path).engine_args(so.engine_options(opts))
    assert kw["async_scheduling"] is False and kw["enable_prefix_caching"] is False and "num_gpu_blocks_override" not in kw
    assert installed == [True] and {**so.engine_options(opts), "async_scheduling": False, "enable_prefix_caching": False} == kw
    assert L.resolve_options(kw)["async_scheduling"] is False and "async_scheduling" in L.ENGINE_ARGS


def test_the_server_steps_in_lockstep_with_its_session_from_its_first_step_and_unaudited_after(tmp_path, monkeypatch):
    """The session's passes are the server's first engine steps, one each; its record is written at the step after it
    ends, the steps after it run unaudited, and the registered bytes a seal carries were read before it registered."""
    order: list[str] = []
    rs = session(tmp_path, monkeypatch, prover(3, order))
    s = serve(tmp_path, rs, 2, order)
    assert order == ["extras", "registered"] and s.loop.steps == 8 and s.windows.summary()["passed"]
    rec = rs.end()
    assert rec["accepted"] and rec["error"] is None
    assert rec["run"] == {"steps": 8, "audited": 3, "unaudited": 5} and SR.accepted(rs)
    assert (tmp_path / "remote_session.json").is_file() and rs.cs.record["run"] == {"steps": 4, "audited": 3, "unaudited": 1}
    assert rs.describe()["store"] == {"blocks": 2} and rs.describe()["async_scheduling"] is False


def test_a_server_that_stops_while_its_session_still_asks_for_steps_fails_its_session(tmp_path, monkeypatch):
    rs = session(tmp_path, monkeypatch, prover(10, []))
    serve(tmp_path, rs, 1, [])
    rec = rs.end()
    assert not rec["accepted"] and "the run ended after 4 steps" in rec["error"] and not SR.accepted(rs)
    assert rec["run"] == {"steps": 4, "audited": 4, "unaudited": 0}


def test_a_session_that_fails_is_rejected_and_the_server_serves_on_unaudited(tmp_path, monkeypatch):
    def fails(*_a: Any, **_kw: Any) -> None:
        raise ConnectionRefusedError("no verifier")

    rs = session(tmp_path, monkeypatch, fails)
    s = serve(tmp_path, rs, 1, [])
    assert rs.record is not None and s.windows.summary()["passed"]
    rec = rs.end()
    assert rec["error"] == "ConnectionRefusedError: no verifier" and not SR.accepted(rs)
    assert rec["run"] == {"steps": 4, "audited": 0, "unaudited": 4}


def test_a_server_stopped_before_its_session_began_is_rejected(tmp_path, monkeypatch):
    rs = session(tmp_path, monkeypatch, prover(1, []))
    assert rs.end() == {"verifier": "127.0.0.1:7781", "plan": "serve", "accepted": False,
                        "error": "the server stopped before its session began", "run": None}


def test_a_sealing_server_reads_the_registered_bytes_once_under_its_weights_root(tmp_path, monkeypatch):
    calls: list[Any] = []
    monkeypatch.setattr(sc, "extras_of", lambda com, rc: calls.append(rc.weight_registration) or ("tree", ["bytes"]))
    com = SimpleNamespace(weights={"root": "ab"}, weights_tree={"root": "ab"})
    sc.CommitWindows(com, tmp_path, run_id="t", window_steps=4).read_extras()
    assert calls == [], "a server that seals nothing reads nothing"
    w = sc.CommitWindows(com, tmp_path, run_id="t", window_steps=4, seal={"label": "serve-openai"})
    w.read_extras()
    assert calls == [{"root": "ab"}]
    assert w._extras_of(com, SimpleNamespace(weight_registration={"root": "ab"})) == ("tree", ["bytes"]) and len(calls) == 1


@pytest.mark.parametrize("name", ["remote_session.json", "memory_map.jsonl"])
def test_the_sessions_records_are_mirrored(name):
    assert name in sc.MIRRORED
