"""`verity_vllm.pipeline.tp.commit_replay` on a record that holds less than a sealed TP Commit's.  `_of_record`: a deferred record
without its bundles, a run that never reached its commit, and a commit without its commitment vector; each is refused before the
bundle's store is opened, so no bundle is on disk here.  `fold`: a pair with no cross-rank verdict, a rank whose finalize recorded
nothing of its replay, and a run without a commit."""
from __future__ import annotations

import json
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
