"""CPU tests for TP-08: the per-rank sampled-replay record (manbind MAN-06 form (A) run inside each rank's worker) -- the driver's fold over
ranks/pairs (`tp2_commit._sampled_replay_components`) and the worker's fail-closed paths (`_tp2_sampled_replay`: no map, no Build dir,
a Program that is not of record is REFUSED by name before any evaluation)."""
from __future__ import annotations

import gzip
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from verity_vllm.pipeline.tp.commit import _sampled_replay_components  # noqa: E402


def _sr(result, grade, vus=10, evaluated=4, why=None, unrep=None):
    return {"result": result, "grade": grade, "why": why, "rank": 0,
            "population": {"vus": vus, "strata_n": 3, "source": "rows", "by_family": {"Gemm_v1": vus}},
            "sample": {"picked": evaluated, "evaluated": evaluated, "equal": evaluated if result else evaluated - 1, "mismatch_n": 0 if result else 1,
                       "not_evaluated_n": 0, "per_stratum": 2, "seed": 7, "seed_source": "run root"},
            "strata_keys": {"mechanisms": ["Gemm_v1"], "requests": ["r0"], "phases": ["prefill"], "steps": {"early": [0], "late": [5]},
                            "lifetime": {"values": {"fresh-window": vus}, "ring_slots": 8}, "key": "spec | request | rank | phase | module op_path | lifetime (window)"},
            "unreplayable": unrep or [{"family": "AllReduce2_v1", "n": 3, "why": "no registered evaluator"}], "weights": {"equal": True}, "mismatch": [], "strata": {"a": {}}}


def _rows(per_rank_pairs):
    return [{"arm": "control"}] + [{"arm": "treatment", "commit": {"per_rank": p}} for p in per_rank_pairs]


VERS = {"sampled_replay": {"of_record": {"0": ["a" * 64], "1": ["b" * 64]}}}


def test_both_ranks_true_folds_true_and_lists_true_ranks_and_families():
    p = {0: {"sampled_replay": _sr(True, "complete"), "boundary_linkage": {"result": True, "pairs": 9, "linked": 9}},
         1: {"sampled_replay": _sr(True, "complete", vus=12), "boundary_linkage": {"result": True, "pairs": 9, "linked": 9}}}
    sr, lk = _sampled_replay_components(_rows([p]), VERS)
    assert sr["result"] is True and lk["result"] is True
    assert sr["population"]["n"] == 22 and sr["n"] == 8 and sr["strata"]["ranks"] == ["rank 0", "rank 1"]
    assert sr["unreplayable"] == [{"family": "AllReduce2_v1", "n": 6, "why": "no registered evaluator", "ranks": [0, 1]}]
    assert sr["strata"]["lifetime"]["values"] == {"fresh-window": 22} and lk["checked"] == 18 and sr["why"] is None


def test_one_rank_not_evaluated_is_not_true_and_named():
    p = {0: {"sampled_replay": _sr(True, "complete"), "boundary_linkage": {"result": True, "pairs": 9, "linked": 9}},
         1: {"sampled_replay": _sr(None, "not-evaluated", why="rank 1: REFUSED (not the Program of record): r0: dddd"), "boundary_linkage": {"result": None, "why": "rank 1: not run"}}}
    sr, lk = _sampled_replay_components(_rows([p]), VERS)
    assert sr["result"] is False and "rank 1" in sr["why"] and "REFUSED" in sr["why"]
    assert lk["result"] is False and "rank 1" in lk["why"]


def test_earlier_pair_failure_is_not_forgotten():
    good = {r: {"sampled_replay": _sr(True, "complete"), "boundary_linkage": {"result": True, "pairs": 9, "linked": 9}} for r in (0, 1)}
    bad = {0: {"sampled_replay": _sr(False, "mismatch", why="rank 0: mismatch"), "boundary_linkage": {"result": True, "pairs": 9, "linked": 9}},
           1: good[1]}
    sr, _lk = _sampled_replay_components(_rows([bad, good]), VERS)
    assert sr["result"] is False and sr["per_run"][0]["0"]["sampled_replay"] is False if "0" in sr["per_run"][0] else sr["per_run"][0][0]["sampled_replay"] is False
    assert sr["why"] == "a rank of an earlier pair failed"


def test_not_run_shape_when_replay_disabled():
    sr, lk = _sampled_replay_components(_rows([{0: {"sampled_replay": None}}]), {})
    assert sr["result"] is None and "not run" in sr["why"] and lk["result"] is None


# ------------------------------------------------------------------------------------------------------------ worker fail-closed paths

def _worker(cfg, bmap, rc=None, model=None):
    from verity_vllm.engine import rank_worker as W

    class Stub:
        _tp2_sampled_replay = W.TP2CaptureWorkerExtension._tp2_sampled_replay

    s = Stub()
    s._tp2_commit = {"rank": 1, "replay": cfg, "bmap": bmap, "com": types.SimpleNamespace(_layouts={}, weights=None, register_weights=lambda: {"root": "x"}),
                     "eng": types.SimpleNamespace(model=model)}
    return s


def _rc():
    return types.SimpleNamespace(run_root=b"\x01" * 32, weight_registration={"root": "x"})


def test_no_binding_map_is_not_evaluated_never_pass():
    r = _worker({"programs": {"1": "/nonexistent"}, "of_record": {"1": ["c" * 64]}, "requests": []}, None)._tp2_sampled_replay(_rc())
    assert r["rank"] == 1 and r["sampled_replay"]["result"] is None and "no binding map" in r["sampled_replay"]["why"]


def test_program_not_of_record_is_refused_by_name(tmp_path):
    d = tmp_path / "build_request_r1"
    d.mkdir()
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump({"rows": [], "params": [], "digest": "d" * 64}, f)
    (d / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": "d" * 64}, "LP": 4, "T": 1}))
    cfg = {"programs": {"1": str(d)}, "of_record": {"1": ["c" * 64]},
           "requests": [{"request_id": "r0", "arrive_step": 0, "prompt_len": 4, "max_tokens": 2, "prompt_token_ids": [1, 2, 3, 4]}]}
    r = _worker(cfg, {"entries": [], "digest": "m"})._tp2_sampled_replay(_rc())
    sr = r["sampled_replay"]
    assert sr["result"] is None and sr["grade"] == "not-evaluated"
    assert "REFUSED (not the Program of record)" in sr["why"] and ("d" * 16) in sr["why"] and ("c" * 16) in sr["why"]
    assert r["boundary_linkage"]["result"] is None


def test_missing_build_dir_for_the_rank_is_named(tmp_path):
    cfg = {"programs": {"0": str(tmp_path)}, "of_record": {"1": ["c" * 64]}, "requests": []}
    r = _worker(cfg, {"entries": [], "digest": "m"})._tp2_sampled_replay(_rc())
    assert r["sampled_replay"]["result"] is None and "no Build dir for rank 1" in r["sampled_replay"]["why"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---- [MAN-09 per rank] the weights-pin fold ------------------------------------------------------------------------------------------

def _rows_with_pins(p0, p1):
    from verity_vllm.pipeline.tp.commit import _weights_pin_component
    rows = [{"arm": "control"}, {"arm": "instrumented", "commit": {"per_rank": {"0": {"weights_pin": p0}, "1": {"weights_pin": p1}}}}]
    return _weights_pin_component(rows, {"weights_pin": {"checkpoint_dir": "/ck"}})


def test_weights_pin_true_only_when_every_rank_compared_equal():
    ok = {"result": True, "root_of_record": "a" * 64, "compared": 98, "mismatched": [], "missing": [], "unpinned": ["x"], "all_shards_pinned": True}
    c = _rows_with_pins(ok, dict(ok, root_of_record="b" * 64))
    assert c["result"] is True and c["per_rank"]["1"]["root_of_record"] == "b" * 64 and c["why"] is None


def test_weights_pin_one_rank_flipped_is_false_and_one_rank_unpinned_is_none():
    ok = {"result": True, "root_of_record": "a" * 64, "compared": 98, "mismatched": [], "missing": []}
    bad = {"result": False, "root_of_record": "b" * 64, "compared": 98, "mismatched": [{"served_name": "model.layers.5.mlp.down_proj.weight"}], "missing": [],
           "why": "1 served weight(s) differ from the checkpoint of record: model.layers.5.mlp.down_proj.weight"}
    c = _rows_with_pins(ok, bad)
    assert c["result"] is False and "differ" in c["why"]
    c2 = _rows_with_pins(ok, {"result": None, "why": "weights unpinned (MAN-09): no checkpoint of record given to this rank"})
    assert c2["result"] is None and "unpinned" in c2["why"]
