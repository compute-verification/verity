"""The replay bundle of a `--replay-deferred` Commit (`pipeline.replay_bundle`): its pending verdict, sealing and refusal of any file
that is not the manifest's.  CPU only; the store dump itself is `tests/commit/test_store_dump.py`."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.check.test_commit_verdict_hopper_no_evaluator_gap import (
    BASE,
    SR_HOPPER,
    _bnd,
)
from verity_vllm.check.verdict import commit_verdict
from verity_vllm.pipeline import replay_bundle as RB

DEFERRED, _ = RB.deferred_records(RB.path("/o", 1))


def _commit(**kw):
    return {**BASE, "binding": _bnd(1), **kw}


def test_a_deferred_run_is_pending_and_the_same_run_replayed_passes():
    ok, why = RB.pending_verdict(_commit(sampled_replay=DEFERRED), True, min_openings=64)
    assert ok is None and why.startswith("PENDING"), why
    assert commit_verdict(_commit(sampled_replay=SR_HOPPER), True, min_openings=64)[0] is not None


@pytest.mark.parametrize("kw,tokens,why", [
    ({}, False, "tokens differ"),
    ({"weights_pin": {"result": False, "why": "pin differs"}}, True, "pin differs"),
    ({"value_correspondence": {"result": False, "method": "oracle", "why": "oracle FAIL"}}, True, "oracle FAIL"),
])
def test_a_deferred_run_that_fails_a_check_of_its_own_fails_now(kw, tokens, why):
    ok, reason = RB.pending_verdict(_commit(sampled_replay=DEFERRED, **kw), tokens, min_openings=64)
    assert ok is False and why in reason, reason


def test_the_commit_pass_is_none_only_while_every_failure_is_absent():
    _pass_of = RB.pass_of
    ok, pend = {"commit": {"commit_pass": True}}, {"commit": {"commit_pass": None, "sampled_replay": DEFERRED}}
    bad = {"commit": {"commit_pass": False}}
    rows = lambda *cs: [{"arm": "instrumented", **c} for c in cs] + [{"arm": "control", "commit": {}}]
    assert _pass_of(rows(ok, ok)) is True and _pass_of(rows(ok, pend)) is None
    assert _pass_of(rows(pend, bad)) is False and _pass_of(rows(bad)) is False
    assert _pass_of(rows({"commit": {"commit_pass": None}})) is False, "a pass that is None without a deferred replay is no pass"


def _sealed(tmp_path):
    p = RB.partial(tmp_path, 1)
    (p / "store" / "raw").mkdir(parents=True)
    (p / "store" / "raw" / "s0.bin").write_bytes(bytes(range(256)))
    (p / "store" / "store.json").write_text("{}")
    (tmp_path / "binding_map_p1.json").write_text(json.dumps({"entries": []}))
    return RB.seal(tmp_path, 1, {"pair": 1, "run_root": "ab" * 32}, tmp_path / "binding_map_p1.json")


def test_a_sealed_bundle_opens_and_its_partial_is_gone(tmp_path):
    m = _sealed(tmp_path)
    assert not RB.partial(tmp_path, 1).exists() and m["path"] == str(RB.path(tmp_path, 1))
    b = RB.open_bundle(m["path"])
    assert b["manifest_sha256"] == m["sha256"] and b["context"]["run_root"] == "ab" * 32 and b["context"]["schema"] == RB.SCHEMA
    assert set(b["manifest"]["files"]) == {"store/raw/s0.bin", "store/store.json", "binding_map.json", "context.json"}


def test_a_digest_taken_as_the_file_was_written_is_the_manifest_s_and_is_checked_on_opening(tmp_path):
    import hashlib
    p = RB.partial(tmp_path, 1)
    (p / "store" / "raw").mkdir(parents=True)
    (p / "store" / "raw" / "s0.bin").write_bytes(bytes(range(256)))
    (tmp_path / "binding_map_p1.json").write_text("{}")
    good = hashlib.sha256(bytes(range(256))).hexdigest()
    m = RB.seal(tmp_path, 1, {"pair": 1}, tmp_path / "binding_map_p1.json", hashed={"store/raw/s0.bin": good})
    assert m["files"]["store/raw/s0.bin"]["sha256"] == good and RB.open_bundle(m["path"])
    p.mkdir()
    (p / "s.bin").write_bytes(b"x")
    m = RB.seal(tmp_path, 1, {"pair": 1}, tmp_path / "binding_map_p1.json", hashed={"s.bin": "00" * 32})
    with pytest.raises(ValueError, match="sha256"):
        RB.open_bundle(m["path"])


@pytest.mark.parametrize("tamper,why", [
    (lambda b: (b / "store" / "raw" / "s0.bin").write_bytes(bytes(255) + b"\x01"), "sha256"),
    (lambda b: (b / "store" / "raw" / "s0.bin").unlink(), "missing"),
    (lambda b: (b / "store" / "raw" / "s1.bin").write_bytes(b"x"), "extra"),
])
def test_a_changed_missing_or_extra_file_is_refused(tmp_path, tamper, why):
    b = RB.path(tmp_path, 1)
    _sealed(tmp_path)
    tamper(b)
    with pytest.raises(ValueError, match=why):
        RB.open_bundle(b)


def _committed():
    pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from tests.commit.test_store_dump import STEPS

    return commit_steps(STEPS)


def _deferred(out, committed=None, **kw):
    """`defer` of a small committed store (`committed`, default a fresh commit), its write started beside the checks (`start`) unless
    `replay_bundle_overlap=0`."""
    torch = pytest.importorskip("torch")
    from verity_vllm.pipeline.timeline import Timeline

    com, rc = committed or _committed()
    com.weights_tree, com.model = {"root": (rc.weight_registration or {}).get("root"), "names": []}, torch.nn.Module()
    args = SimpleNamespace(**{"replay_deferred": "full", "replay_bundle_overlap": 1, "retain": "host", "required_manifest": None, "openings": 64, **kw})
    started, row, deferred = RB.start(out, 1, com, rc, args, None), {"commit": {}}, {}
    assert RB.defer(out, 1, com, rc, args, tl=Timeline(None), arm="instrumented", versions={}, reqs=[], sampling={}, tokens={}, stop_of_record=None,
                    ids_full=None, declaration=None, required_classes=[], row=row, deferred=deferred, started=started)
    return started, deferred[1], {p.relative_to(RB.partial(out, 1)).as_posix(): p.read_bytes() for p in RB.partial(out, 1).rglob("*") if p.is_file()}


def test_a_store_written_beside_the_checks_is_the_store_defer_writes(tmp_path):
    committed = _committed()                                         # one commit: its salts are in both stores
    started, ctx, files = _deferred(tmp_path / "beside", committed)
    inline, ctx0, files0 = _deferred(tmp_path / "inline", committed, replay_bundle_overlap=0)
    assert started is not None and started.future.done() and inline is None
    assert files == files0 and ctx[RB.HASHED] == ctx0[RB.HASHED] and set(ctx[RB.HASHED]) == {f for f in files if f.endswith(".bin") and f.startswith("store/")}
    assert {k: v for k, v in ctx.items() if k != "engine_of_record"} == {k: v for k, v in ctx0.items() if k != "engine_of_record"}


def test_the_bundle_reads_the_models_tensors_and_the_engines_facts_before_start_returns(tmp_path, monkeypatch):
    """A leased Commit gives its GPU back right after `start` (`gpu_lease.release`): nothing the bundle carries is read from the
    model or the engine after it returns."""
    torch = pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from tests.commit.test_store_dump import STEPS
    from verity_vllm.check.replay import driver
    from verity_vllm.pipeline.timeline import Timeline

    com, rc = commit_steps(STEPS)
    com.model = torch.nn.Module()
    com.model.register_buffer("cos", torch.arange(4, dtype=torch.int32))
    com.weights_tree = {"root": (rc.weight_registration or {}).get("root"), "names": [["w", "bfloat16", [2], 4], ["cos", "int32", [4], 16]]}
    args = SimpleNamespace(replay_deferred="full", replay_bundle_overlap=1, retain="host", required_manifest=None, openings=64)
    started, row, deferred = RB.start(tmp_path, 1, com, rc, args, None), {"commit": {}}, {}
    com.model = None
    monkeypatch.setattr(driver, "engine_facts_of_record", lambda: pytest.fail("the engine's facts were read after start"))
    assert RB.defer(tmp_path, 1, com, rc, args, tl=Timeline(None), arm="instrumented", versions={}, reqs=[], sampling={}, tokens={},
                    stop_of_record=None, ids_full=None, declaration=None, required_classes=[], row=row, deferred=deferred, started=started)
    w = RB.partial(tmp_path, 1) / "weights"
    assert deferred[1]["engine_of_record"] == started.engine and started.engine
    assert (w / "extra" / "1.bin").read_bytes() == torch.arange(4, dtype=torch.int32).numpy().tobytes()
    assert json.loads((w / "tree.json").read_text())["extra"] == {"cos": 1}


@pytest.mark.parametrize("kw", [{"replay_deferred": None}, {"replay_deferred": "slim"}, {"replay_bundle_overlap": 0}, {"retain": "device"},
                                {"replay_bundle_cap_gb": 0}])
def test_the_store_is_written_beside_the_checks_only_where_defer_writes_it_in_full(kw):
    pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from tests.commit.test_store_dump import STEPS

    com, rc = commit_steps(STEPS)
    args = SimpleNamespace(**{"replay_deferred": "full", "replay_bundle_overlap": 1, "retain": "host", **kw})
    assert RB.start("/nonexistent", 1, com, rc, args, None) is None


def _refusal_args(**kw):
    return SimpleNamespace(**{"replay_deferred": "full", "program_dir": "/p", "programs_root": None, "arms": None, "only_arm": None,
                              "padded_finalize": False, "compiled_taps": "off", "vu_export_dir": None, "serving_rows": None, **kw})


def test_a_deferred_commit_takes_serving_rows_and_still_refuses_what_its_bundle_cannot_hold():
    RB.refuse(_refusal_args(serving_rows="/run/serving_rows/request.json"))
    with pytest.raises(SystemExit, match="--vu-export-dir"):
        RB.refuse(_refusal_args(serving_rows="/run/serving_rows/request.json", vu_export_dir="/vu"))


def test_the_serving_rows_commit_after_the_store_s_write_on_the_tensors_start_read(tmp_path):
    """After `start` the GPU is given back: the serving rows commit at the deferral site once the store is written, reading RoPE's
    ``cos_sin_cache`` from the registered tensors `start` read, nothing from the model."""
    import numpy as np
    torch = pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from tests.commit.test_store_dump import STEPS
    from verity_vllm.pipeline.timeline import Timeline

    com, rc = commit_steps(STEPS)
    cache = torch.arange(1, 5, dtype=torch.bfloat16)
    com.model = torch.nn.Module()
    com.model.rotary = torch.nn.Module()
    com.model.rotary.register_buffer("cos_sin_cache", cache.clone())
    com.weights_tree = {"root": (rc.weight_registration or {}).get("root"), "names": [["rotary.cos_sin_cache", "bfloat16", [4], 8]]}
    args = SimpleNamespace(replay_deferred="full", replay_bundle_overlap=1, retain="host", required_manifest=None, openings=64)
    started, row, deferred = RB.start(tmp_path, 1, com, rc, args, None), {"commit": {}}, {}
    com.model = None
    seen = []
    assert RB.defer(tmp_path, 1, com, rc, args, tl=Timeline(None), arm="instrumented", versions={}, reqs=[], sampling={}, tokens={},
                    stop_of_record=None, ids_full=None, declaration=None, required_classes=[], row=row, deferred=deferred, started=started,
                    serving=lambda w: seen.append((started.future.done(), w("model.")("model.rotary.cos_sin_cache", 1, 3))))
    (done, words), = seen
    assert done and words.dtype == np.uint16 and words.tolist() == cache.view(torch.int16).numpy().view(np.uint16)[1:3].tolist()


def test_the_serving_rows_commit_before_the_plan_s_reads_on_the_live_weights(tmp_path):
    """No write began at `start` (a slim bundle): the GPU is kept, the serving rows read the live weights (None here), and they
    commit before the plan, whose reads come from the replay's seed and make the slim store."""
    torch = pytest.importorskip("torch")
    from verity_vllm.pipeline.timeline import Timeline

    com, rc = _committed()
    com.weights_tree, com.model = {"root": (rc.weight_registration or {}).get("root"), "names": []}, torch.nn.Module()
    args = SimpleNamespace(replay_deferred="slim", replay_bundle_overlap=1, retain="host", required_manifest=None, openings=64)
    assert RB.start(tmp_path, 1, com, rc, args, None) is None
    order = []
    assert RB.defer(tmp_path, 1, com, rc, args, tl=Timeline(None), arm="instrumented", versions={}, reqs=[], sampling={}, tokens={},
                    stop_of_record=None, ids_full=None, declaration=None, required_classes=[], row={"commit": {}}, deferred={},
                    plan=lambda: order.append("plan") or {"replay": {}, "linkage": None}, serving=lambda w: order.append(("serving", w)))
    assert order == [("serving", None), "plan"]


def _site(tmp_path, request):
    from verity_vllm.pipeline import c2_replay as C2

    (tmp_path / "binding_map_p0.json").write_text(json.dumps({"entries": []}))
    return C2.Site(args=SimpleNamespace(serving_rows=request, label="row"), out=tmp_path, pair=0, com="com", rc="rc", row={"commit": {}},
                   reqs=[{"request_id": "a", "arrive_step": 2}], sampling={}, tokens={}, manifest=None, stop_of_record={}, ids_full=None,
                   versions={}, components={}, vc_result=None, weights_of=lambda p: ("live", p), weights_check=dict, tl=None)


def test_a_deferred_run_s_serving_rows_commit_once_and_the_replay_on_its_site_commits_none(tmp_path, monkeypatch):
    from verity_vllm.check.replay import driver as SR
    from verity_vllm.pipeline import c2_replay as C2
    from verity_vllm.pipeline import serving_rows as SRW

    monkeypatch.setattr(C2, "programs_of_record", lambda s, rl: ({"r0": {"instances": {"rows": []}, "dir": "/p"}}, {"r0": "d"}))
    monkeypatch.setattr(SR, "ProgramIndex", lambda inst, program_dir=None: SimpleNamespace(prefix="model."))
    monkeypatch.setattr(C2, "linkage_facts", lambda *a: {})
    calls, readers = [], []
    monkeypatch.setattr(SRW, "commit_request", lambda *a: calls.append(a))

    def stop(request, *a):
        readers.append(request)
        raise RuntimeError("the replay goes on from here")

    monkeypatch.setattr(SRW, "readers", stop)
    s = _site(tmp_path, "/run/serving_rows/request.json")
    C2.serving_rows(s, lambda p: ("start's tensors", p))
    C2.serving_rows(s, lambda p: ("start's tensors", p))
    (req, _com, _rc, w, progs, bm, arr, _loaded, _man, row), = calls
    assert s.serving_committed and (req, w, progs, bm, arr, row) == (
        "/run/serving_rows/request.json", ("start's tensors", "model."), {"r0": {"rows": []}}, {"entries": []}, {"r0": 2}, "row")
    for site in (s, _site(tmp_path, "/run/serving_rows/request.json")):
        with pytest.raises(RuntimeError, match="goes on"):
            C2.evaluate(site, {})
    assert readers == [None, "/run/serving_rows/request.json"], "the in-process replay of a run whose rows are not committed commits them"


def test_a_serving_rows_failure_at_the_deferral_site_is_written_beside_them_and_raises_nothing(tmp_path, monkeypatch):
    from verity_vllm.pipeline import c2_replay as C2

    def no_programs(s, rl):
        raise RuntimeError("no Program digest of record")

    monkeypatch.setattr(C2, "programs_of_record", no_programs)
    req = tmp_path / "request.json"
    req.write_text(json.dumps({"dir": str(tmp_path / "rows")}))
    s = _site(tmp_path, str(req))
    C2.serving_rows(s)
    assert s.serving_committed and "no Program digest of record" in (tmp_path / "rows" / "error.txt").read_text()


def test_a_discarded_bundle_leaves_nothing(tmp_path):
    (RB.partial(tmp_path, 1) / "store").mkdir(parents=True)
    RB.discard(tmp_path, 1)
    assert not RB.partial(tmp_path, 1).exists() and not RB.path(tmp_path, 1).exists()
