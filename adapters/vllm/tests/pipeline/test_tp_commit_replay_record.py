"""`verity_vllm.pipeline.tp.commit_replay` on a record that holds less than a sealed TP Commit's.  `_of_record`: a deferred record
without its bundles, a run that never reached its commit, and a commit without its commitment vector, each refused before the
bundle's store is opened; and a rank committed without a binding, on a stand-in store.  `replay_rank`: a rank committed without a
weight registration.  `fold`: a pair with no cross-rank verdict, a rank whose finalize recorded nothing of its replay, and a run
without a commit."""
from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import pytest

from verity_vllm.errors import Refused
from verity_vllm.pipeline.tp import c2 as C2
from verity_vllm.pipeline.tp import commit_replay as TPR
from verity_vllm.pipeline.tp import rank_replay as RR
from verity_vllm.pipeline.tp import run_root as RT

BUNDLE = {"context": {"pair": 0, "tp": {"rank": 1, "world": 2}, "tp_run_root": "aa"}, "path": "/nowhere/bundle", "manifest_sha256": "5e" * 32}
DEFERRED = {"world": 2, "bundles": [{"pair": 0, "rank": 1, "manifest_sha256": BUNDLE["manifest_sha256"]}]}
PENDING = {"run_root": "bb", "commit_pass": None, "replay_deferred": {"pre": [True, None]}}


def _record(out: Path, *rows: dict) -> Path:
    (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "versions.json").write_text(json.dumps({"root_tag": RT.TP_ROOT_TAG.decode()}))
    return out


@pytest.mark.parametrize("listed", [{}, {"bundles": None}, {"bundles": []}, {"bundles": [{"pair": 0, "rank": 0, "manifest_sha256": "5e" * 32}]}],
                         ids=["no list", "a null list", "an empty list", "another rank's bundle"])
def test_a_bundle_the_deferred_record_does_not_name_is_refused(tmp_path, listed):
    with pytest.raises(Refused, match="names no bundle of pair 0 rank 1 with manifest sha256 5e5e5e5e5e5e5e5e"):
        TPR._of_record(tmp_path, {"world": 2, **listed}, BUNDLE, 0, 1)


@pytest.mark.parametrize("row", [{"pair": 0, "arm": "instrumented"}, {"pair": 0, "arm": "instrumented", "commit": None},
                                 {"pair": 0, "arm": "instrumented", "commit": {"per_rank": {"0": PENDING}}}],
                         ids=["no commit block", "a null commit block", "a commit without the rank"])
def test_a_run_whose_record_holds_no_commit_of_the_rank_awaits_no_replay(tmp_path, row):
    out = _record(tmp_path, {"pair": 0, "arm": "control"}, row)
    with pytest.raises(Refused) as e:
        TPR._of_record(out, DEFERRED, BUNDLE, 0, 1)
    assert str(e.value) == f"pair 0 rank 1: no rank awaiting its deferred replay in {out / 'runs.jsonl'}"


@pytest.mark.parametrize("commit", [{}, {"commitment_vector": None}, {"commitment_vector": []}], ids=["absent", "null", "empty"])
def test_a_commit_without_its_commitment_vector_combines_into_no_tp_run_root(tmp_path, commit):
    out = _record(tmp_path, {"pair": 0, "arm": "instrumented", "commit": {"per_rank": {"1": PENDING}, "tp_run_root": "aa", **commit}})
    with pytest.raises(Refused, match="pair 0: the commitment vector does not combine into"):
        TPR._of_record(out, DEFERRED, BUNDLE, 0, 1)


ROOTS = {0: bytes(64), 1: bytes(range(64))}
MAP_DIGEST = hashlib.sha256(b"[]").hexdigest()            # of a binding map with no entry


def _bundle_of_record(tmp_path: Path, monkeypatch, block: dict) -> tuple[Path, dict, dict]:
    """A two-rank record whose rank 1 awaits its replay with ``block`` beside its run root, and rank 1's bundle of that record: its
    binding map on disk, its store a stand-in that folds to the rank's run root."""
    out, bdir, tag = tmp_path / "commit", tmp_path / "bundle", RT.TP_ROOT_TAG.decode()
    for d in (out, bdir):
        d.mkdir()
    tp_root = RT.combine(tag, 2, ROOTS).hex()
    _record(out, {"pair": 0, "arm": "instrumented", "commit": {
        "per_rank": {"1": {**PENDING, "run_root": ROOTS[1].hex(), **block}}, "tp_run_root": tp_root,
        "commitment_vector": [{"rank": r, "run_root": root.hex()} for r, root in ROOTS.items()]}})
    (bdir / "binding_map.json").write_text(json.dumps({"entries": [], "digest": MAP_DIGEST, "run_root": ROOTS[1].hex(), "rank": 1}))
    monkeypatch.setattr(TPR.SD, "load", lambda path: (types.SimpleNamespace(_run_root=lambda run: ROOTS[1]), types.SimpleNamespace(run_root=ROOTS[1])))
    bundle = {"context": {"pair": 0, "tp": {"rank": 1, "world": 2}, "tp_run_root": tp_root, "run_root": ROOTS[1].hex()},
              "path": str(bdir), "manifest_sha256": BUNDLE["manifest_sha256"]}
    return out, bundle, {"world": 2, "bundles": [{"pair": 0, "rank": 1, "manifest_sha256": BUNDLE["manifest_sha256"]}]}


def test_a_bundle_whose_binding_map_is_the_ranks_map_of_record_is_the_records(tmp_path, monkeypatch):
    registered = {"root": "ee" * 32}
    out, bundle, dd = _bundle_of_record(tmp_path, monkeypatch, {"binding": {"map_digest": MAP_DIGEST}, "weight_registration": registered})
    _row, c, _store, run, bmap = TPR._of_record(out, dd, bundle, 0, 1)
    assert c["run_root"] == ROOTS[1].hex() and bmap["digest"] == MAP_DIGEST and run.weight_registration == registered


@pytest.mark.parametrize("block", [{}, {"binding": None}, {"binding": {}}, {"binding": {"map_digest": None}}],
                         ids=["no binding", "a null binding", "an empty binding", "a binding without its digest"])
def test_a_rank_committed_without_a_binding_has_no_map_of_record_and_its_bundles_map_is_refused(tmp_path, monkeypatch, block):
    out, bundle, dd = _bundle_of_record(tmp_path, monkeypatch, block)
    with pytest.raises(Refused) as e:
        TPR._of_record(out, dd, bundle, 0, 1)
    assert str(e.value) == f"rank 1: the bundle's binding map {MAP_DIGEST[:16]} is not the rank's map of record None"


@pytest.mark.parametrize("block, root", [({"weight_registration": {"root": "ee" * 32}}, "ee" * 32), ({}, None), ({"weight_registration": None}, None),
                                         ({"weight_registration": {}}, None)],
                         ids=["a registered root", "no registration", "a null registration", "an empty registration"])
def test_a_ranks_shards_are_opened_against_the_root_its_commit_registered_and_against_none_without_one(tmp_path, monkeypatch, block, root):
    """The stand-in checkpoint stops the replay where it opens the weights, so the rank's record is the one of a replay that did
    not run; what it was asked to open against is what the test reads."""
    out, bundle, dd = _bundle_of_record(tmp_path, monkeypatch, {"binding": {"map_digest": MAP_DIGEST}} | block)
    bdir, opened = Path(bundle["path"]), []
    (bdir / "weights").mkdir()
    (bdir / "weights" / "tree.json").write_text(json.dumps({"names": [], "root": "ab" * 32}))
    bundle["context"]["replay"] = {}

    def open_weights(bundle_dir, tree, root_of_record, index, **kw):
        opened.append(root_of_record)
        raise RuntimeError("no checkpoint here")

    monkeypatch.setattr(TPR.RB, "open_bundle", lambda path: bundle)
    monkeypatch.setattr(TPR.RR, "load", lambda cfg, rank, world: types.SimpleNamespace(loaded={}, index=types.SimpleNamespace(prefix="")))
    monkeypatch.setattr(TPR.CPU, "open_weights", open_weights)
    doc = TPR.replay_rank(TPR.Options(commit_out=str(out), checkpoint_dir="/ck"), out, dd, 0, 1, ck=lambda: None)
    assert opened == [root]
    assert doc["sampled_replay"]["grade"] == "not-evaluated" and doc["sampled_replay"]["error"] == "RuntimeError('no checkpoint here')"
    assert doc["weights_check"] is None and json.loads(TPR.result_path(out, 0, 1).read_text())["sampled_replay"]["result"] is None


def test_the_fold_decides_a_record_with_no_cross_rank_verdict_no_finalize_block_and_a_run_without_a_commit(tmp_path):
    """Two ranks whose replay could not run, on a record whose pair has no cross-rank verdict: each rank fails by its replay's own
    reason and no collective family is waived; the rank's finalize gets the result alone; the run without a commit stays None."""
    out = tmp_path
    dd = {"schema": C2.SCHEMA, "world": 2, "pending_pairs": [0], "bundles": [{"pair": 0, "rank": r, "manifest_sha256": f"m{r}"} for r in (0, 1)],
          "all_pass": True, "vc_base": None, "components": {}, "xrank": {}, "xrank_families": [], "required_manifest": None, "replayed": []}
    (out / C2.DEFERRED).write_text(json.dumps(dd))
    for r in (0, 1):
        sr, lk = RR.not_run(r, RuntimeError("no checkpoint here"), "traceback")
        TPR.result_path(out, 0, r).write_text(json.dumps({"pair": 0, "rank": r, "manifest_sha256": f"m{r}", "sampled_replay": sr,
                                                          "boundary_linkage": lk, "wall_s": 1.5, "where": "cpu"}))
    _record(out, {"pair": 0, "arm": "control", "tokens_equal_ref": True},
            {"pair": 0, "arm": "instrumented", "tokens_equal_ref": True, "commit": {"per_rank": {"0": dict(PENDING), "1": dict(PENDING)}, "commit_pass": None}},
            {"pair": 1, "arm": "instrumented", "tokens_equal_ref": True})
    (out / "finalize_pair0.json").write_text(json.dumps([{"rank": 0}, {"rank": 1, "sampled_replay": None}]))
    (out / "summary.json").write_text("{}")
    assert TPR.fold(out) is False
    summ = json.loads((out / "summary.json").read_text())
    assert summ["commit_pass"] == [False, None] and summ["pass"] is False
    decided = [json.loads(ln) for ln in (out / "runs.jsonl").read_text().splitlines()][1]["commit"]
    said = "sampled replay (C2 of record, rank {r}): not-evaluated -- rank {r}: sampled replay not run: RuntimeError('no checkpoint here')"
    assert decided["commit_fail_reason"] == "; ".join(f"rank {r}: " + said.format(r=r) for r in (0, 1))
    assert [decided["per_rank"][r]["no_evaluator_gap"]["cross_rank_pass"] for r in ("0", "1")] == [False, False]
    for f in json.loads((out / "finalize_pair0.json").read_text()):
        assert sorted(f["sampled_replay"]) == ["boundary_linkage", "sampled_replay", "wall_s", "where"] and f["sampled_replay"]["where"] == "cpu"
