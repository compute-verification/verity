"""CPU tests for the per-rank sampled-replay record (run inside each rank's worker) -- the driver's fold over
ranks/pairs (`tp.c2.sampled_replay_components`) and the worker's fail-closed paths (`_tp2_sampled_replay`: no map, no Build dir,
a Program that is not of record is REFUSED by name before any evaluation)."""
from __future__ import annotations

import gzip
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from verity_vllm.pipeline.tp.c2 import sampled_replay_components


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
    sr, lk = sampled_replay_components(_rows([p]), VERS)
    assert sr["result"] is True and lk["result"] is True
    assert sr["population"]["n"] == 22 and sr["n"] == 8 and sr["strata"]["ranks"] == ["rank 0", "rank 1"]
    assert sr["unreplayable"] == [{"family": "AllReduce2_v1", "n": 6, "why": "no registered evaluator", "ranks": [0, 1]}]
    assert sr["strata"]["lifetime"]["values"] == {"fresh-window": 22} and lk["checked"] == 18 and sr["why"] is None


def test_one_rank_not_evaluated_is_not_true_and_named():
    p = {0: {"sampled_replay": _sr(True, "complete"), "boundary_linkage": {"result": True, "pairs": 9, "linked": 9}},
         1: {"sampled_replay": _sr(None, "not-evaluated", why="rank 1: REFUSED (not the Program of record): r0: dddd"), "boundary_linkage": {"result": None, "why": "rank 1: not run"}}}
    sr, lk = sampled_replay_components(_rows([p]), VERS)
    assert sr["result"] is False and "rank 1" in sr["why"] and "REFUSED" in sr["why"]
    assert lk["result"] is False and "rank 1" in lk["why"]


def test_earlier_pair_failure_is_not_forgotten():
    good = {r: {"sampled_replay": _sr(True, "complete"), "boundary_linkage": {"result": True, "pairs": 9, "linked": 9}} for r in (0, 1)}
    bad = {0: {"sampled_replay": _sr(False, "mismatch", why="rank 0: mismatch"), "boundary_linkage": {"result": True, "pairs": 9, "linked": 9}},
           1: good[1]}
    sr, _lk = sampled_replay_components(_rows([bad, good]), VERS)
    assert sr["result"] is False and sr["per_run"][0]["0"]["sampled_replay"] is False if "0" in sr["per_run"][0] else sr["per_run"][0][0]["sampled_replay"] is False
    assert sr["why"] == "a rank of an earlier pair failed"


def test_every_collective_the_cross_rank_pass_evaluates_is_annotated_with_its_site_coverage_and_the_rest_are_not():
    """The world-8 routed all-reduces (GLM-5.3 TP8: Mnnvl, Multimem, the fused RMSNorm) are evaluated by the cross-rank pass like AllReduce_v2 and
    carry its record into the fold; a family the pass does not evaluate (AllGather2_v1 at world 8) stays a named gap."""
    from verity_vllm.pipeline.tp import c2 as C2
    from verity_vllm.program.frontend import collective_kinds as CK

    fams = ["AllGather_v1", "AllReduceMnnvl_v1", "AllReduceMultimem_v1", "AllReduceRmsNormMnnvl_v1", "AllGather2_v1"]
    xr = {"result": True, "by_family": {"AllReduceRmsNormMnnvl_v1": {"picks": 156, "equal": 156}},
          "coverage_of_unreplayable": {"families": {"AllReduceRmsNormMnnvl_v1": {"sites": 78, "covered": 78, "missing": []}}}}
    per_rank = {}
    for r_ in (0, 1):
        s = _sr(True, "complete", unrep=[{"family": f, "n": 2, "why": "no registered evaluator"} for f in fams])
        C2.annotate_unreplayable([s], xr, 0, CK.cross_rank_families(8))
        per_rank[r_] = {"sampled_replay": s, "boundary_linkage": {"result": True, "pairs": 9, "linked": 9}}
    sr, _lk = sampled_replay_components(_rows([per_rank]), VERS)
    by = {u["family"]: u for u in sr["unreplayable"]}
    assert sorted(f for f, u in by.items() if "evaluated_by" in u) == fams[:4]
    fused = by["AllReduceRmsNormMnnvl_v1"]["evaluated_by"]
    assert (fused["picks"], fused["equal"], fused["sites"], fused["sites_covered"], fused["sites_missing"]) == (156, 156, 78, 78, [])
    assert by["AllGather2_v1"] == {"family": "AllGather2_v1", "n": 4, "why": "no registered evaluator", "ranks": [0, 1]}


def test_evaluated_cross_rank_is_the_worlds_families_and_every_all_reduce_route():
    from verity_vllm.program.frontend import collective_kinds as CK

    w8 = CK.cross_rank_families(8)
    assert all(CK.evaluated_cross_rank(f, w8) for f in w8 | {"AllReduceMnnvl_v1", "AllReduceMultimem_v1", "AllReduceCustom_v1",
                                                             "AllReduceRmsNormMnnvl_v1{WORLD=8,N=6144}"})
    assert not any(CK.evaluated_cross_rank(f, w8) for f in ("BatchedGemm_v1", "AllGather2_v1", "Const32[0x00000001]_v1", "Gemm_v2"))


def test_not_run_shape_when_replay_disabled():
    sr, lk = sampled_replay_components(_rows([{0: {"sampled_replay": None}}]), {})
    assert sr["result"] is None and "not run" in sr["why"] and lk["result"] is None


# ------------------------------------------------------------------------------------------------------------ worker fail-closed paths

def _worker(cfg, bmap, rc=None, model=None):
    from verity_vllm.pipeline.tp import rank_worker as W

    class Stub:
        _tp2_sampled_replay = W.TP2CaptureWorkerExtension._tp2_sampled_replay

    s = Stub()
    s._tp2_commit = {"rank": 1, "replay": cfg, "bmap": bmap, "com": types.SimpleNamespace(_layouts={}, weights=None, register_weights=lambda: {"root": "x"}),
                     "eng": types.SimpleNamespace(model=model), "weights": model}
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


def test_the_ranks_linkage_reads_each_requests_declared_seed_and_its_programs_s_of_record(tmp_path, monkeypatch):
    """TP1's linkage links the committed seed to the request's declared seed and the committed splits to SplitsFor_v1 over the
    Programs' num_SMs (`c2_replay.linkage_requests`, `linkage.splits_of_record`); a rank's linkage takes both from its replay config
    and its Programs.  Without them phi4-14b TP2 B8 stoch (cov-lw04-gm357) linked 112/112 boundary pairs and still failed: 112 committed
    seed words with no declared workload seed."""
    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.check.replay import driver as SR

    d = tmp_path / "build_request_r1"
    d.mkdir()
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump({"rows": [], "params": [], "digest": "d" * 64}, f)
    (d / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": "d" * 64}, "LP": 4, "T": 1, "tp": {"world": 2, "rank": 1},
                                               "sampling": {"splits": "derived", "geometry": {"num_sms": 188}}}))
    seen: dict = {}
    monkeypatch.setattr(SR, "ProgramIndex", lambda *a, **k: types.SimpleNamespace(rows=[], prefix="", embed_op_path=lambda: "model.embed_tokens"))
    monkeypatch.setattr(SR, "model_weights_provider", lambda *a: None)
    monkeypatch.setattr(SR, "sampled_replay", lambda *a, **k: {"result": True})
    monkeypatch.setattr(SR, "boundary_linkage", lambda bmap, rd, wreq, **k: seen.update(wreq=wreq, **k) or {"result": True})
    monkeypatch.setattr(OC, "OpenedReader", lambda com, rc: types.SimpleNamespace(record=dict))
    cfg = {"programs": {"1": str(d)}, "of_record": {"1": ["d" * 64]},
           "requests": [{"request_id": "r0", "arrive_step": 0, "prompt_len": 4, "max_tokens": 2, "prompt_token_ids": [1, 2, 3, 4], "seed": 1234}]}
    w = _worker(cfg, {"entries": [], "digest": "m"})
    w._tp2_commit["world"] = 2
    r = w._tp2_sampled_replay(_rc())
    assert r["boundary_linkage"] == {"result": True, "rank": 1}, r
    assert [(q["request_id"], q["seed"], q["steps"]) for q in seen["wreq"]] == [("r0", 1234, [0, 1])]
    assert seen["splits_of_record"] == {"r0": {"derived": 188}} and seen["embed_op_path"] == "model.embed_tokens"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---- the per-rank weights-pin fold ------------------------------------------------------------------------------------------

def _rows_with_pins(p0, p1):
    from verity_vllm.pipeline.tp.c2 import weights_pin_component
    rows = [{"arm": "control"}, {"arm": "instrumented", "commit": {"per_rank": {"0": {"weights_pin": p0}, "1": {"weights_pin": p1}}}}]
    return weights_pin_component(rows, {"weights_pin": {"checkpoint_dir": "/ck"}})


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
