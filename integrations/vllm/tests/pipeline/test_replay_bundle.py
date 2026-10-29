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
from verity_vllm.check.commit_pass import commit_verdict
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


def _deferred(out, **kw):
    """`defer` of a small committed store, its write started beside the checks (`start`) unless `replay_bundle_overlap=0`."""
    torch = pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from tests.commit.test_store_dump import STEPS
    from verity_vllm.pipeline.timeline import Timeline

    com, rc = commit_steps(STEPS)
    com.weights_tree, com.model = {"root": (rc.weight_registration or {}).get("root"), "names": []}, torch.nn.Module()
    args = SimpleNamespace(**{"replay_deferred": "full", "replay_bundle_overlap": 1, "retain": "host", "required_manifest": None, "openings": 64, **kw})
    started, row, deferred = RB.start(out, 1, com, rc, args, None), {"commit": {}}, {}
    assert RB.defer(out, 1, com, rc, args, tl=Timeline(None), arm="instrumented", versions={}, reqs=[], sampling={}, tokens={}, stop_of_record=None,
                    ids_full=None, declaration=None, required_classes=[], row=row, deferred=deferred, started=started)
    return started, deferred[1], {p.relative_to(RB.partial(out, 1)).as_posix(): p.read_bytes() for p in RB.partial(out, 1).rglob("*") if p.is_file()}


def test_a_store_written_beside_the_checks_is_the_store_defer_writes(tmp_path):
    started, ctx, files = _deferred(tmp_path / "beside")
    inline, ctx0, files0 = _deferred(tmp_path / "inline", replay_bundle_overlap=0)
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


def test_serving_rows_defer_in_full_and_a_slim_bundle_with_them_is_refused_by_name():
    """A full deferral takes ``--serving-rows`` (its C2 site commits the window, `c2_replay.serving_rows`), and the bundle's CPU
    replay commits none over it; a slim one, whose plan would commit the window and seal every unit it reads, is refused."""
    from verity_vllm.pipeline import cpu_replay as CPU

    def args(**kw):
        return SimpleNamespace(**{"replay_deferred": "full", "program_dir": "/p", "programs_root": None, "arms": None, "only_arm": None,
                                  "compiled_taps": "off", "vu_export_dir": None,
                                  "serving_rows": "/r/request.json", **kw})

    RB.refuse(args())
    RB.refuse(args(replay_deferred=None))
    RB.refuse(args(replay_deferred="slim", serving_rows=None))
    with pytest.raises(SystemExit, match=r"--replay-deferred: slim with --serving-rows: .*c2_replay\.serving_rows"):
        RB.refuse(args(replay_deferred="slim"))
    ctx = {"args": {k: getattr(args(), k, None) for k in RB.ARGS}}
    assert CPU.replay_args(SimpleNamespace(row_dir="/row", replay_workers=None), ctx).serving_rows is None


def test_a_discarded_bundle_leaves_nothing(tmp_path):
    (RB.partial(tmp_path, 1) / "store").mkdir(parents=True)
    RB.discard(tmp_path, 1)
    assert not RB.partial(tmp_path, 1).exists() and not RB.path(tmp_path, 1).exists()
