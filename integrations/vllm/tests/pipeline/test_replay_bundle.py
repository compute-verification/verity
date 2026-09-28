"""The replay bundle of a `--replay-deferred` Commit (`pipeline.replay_bundle`): its pending verdict, sealing and refusal of any file
that is not the manifest's.  CPU only; the store dump itself is `tests/commit/test_store_dump.py`."""
from __future__ import annotations

import json

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


def test_a_discarded_bundle_leaves_nothing(tmp_path):
    (RB.partial(tmp_path, 1) / "store").mkdir(parents=True)
    RB.discard(tmp_path, 1)
    assert not RB.partial(tmp_path, 1).exists() and not RB.path(tmp_path, 1).exists()
