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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from verity_capture.tp.commit import _sampled_replay_components  # noqa: E402


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
    from verity_capture.tp import worker as W

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


# ---- [TP-08d] the replay's ProgramIndex describes a RANK Program's rows with the manifest's TP members -------------------------------

_RANK_ROWS = [  # one step of a rank Program (shapes of Llama-3.2-1B TP2 rank 1's build_request: i, spec, name)
    {"i": 0, "spec": "EmbeddingShard_v1{VS=64128,H=2048,START=64128}", "name": "inner.model.model.embed_tokens/embedding_shard[0]/shard", "args": "[]"},
    {"i": 1, "spec": "AllReduce2_v1{N=2048}", "name": "inner.model.model.embed_tokens/all_reduce2[0]/all_reduce", "args": "[]"},
    {"i": 2, "spec": "Gemm_v1{K=2048,N=1536}", "name": "inner.model.model.layers.0.self_attn.qkv_proj/triton_launch_1[0]", "args": "[]"},
    {"i": 3, "spec": "Attention_v3{NH=16,KVH=4,D=64}", "name": "inner.model.model.layers.0.self_attn.attn/attn[t=0]", "args": "[]"},
    {"i": 4, "spec": "Gemm_v1{K=1024,N=2048}", "name": "inner.model.model.layers.0.self_attn.o_proj/triton_launch_2[0]", "args": "[]"},
    {"i": 5, "spec": "AllReduce2_v1{N=2048}", "name": "inner.model.model.layers.0.self_attn.o_proj/all_reduce2_1[0]/all_reduce", "args": "[]"},
    {"i": 6, "spec": "Gemm_v1{K=2048,N=64128}", "name": "inner.model.logits_processor/triton_launch_9[0]", "args": "[]"},
    {"i": 7, "spec": "AllGather2_v1{N=64128}", "name": "inner.model.logits_processor/all_gather2[0]/all_gather", "args": "[]"},
    {"i": 8, "spec": "TokenSelect_v1{V=128256}", "name": "sampler/token_select[0]", "args": "[]"},
]


def _members(tp):
    from verity_capture.commit import sampled_replay as SR
    pi = SR.ProgramIndex({"rows": _RANK_ROWS, "params": [], "digest": "e" * 64}, attention_paths=["model.layers.0.self_attn.attn"], tp=tp)
    out = {}
    for r in _RANK_ROWS:
        d = pi.describe(r)
        if d:
            out[(d["family"], d["op_path"])] = [m[0] for m in d["members"]]
    return pi, out


def test_rank_program_rows_describe_the_tp_members_of_that_rank():
    pi, m = _members({"world": 2, "rank": 1})
    assert pi.prefix == "inner.model."                                           # determined from the observed attention path
    assert m[("Gemm_v1", "model.layers.0.self_attn.o_proj")] == ["part_rank1"]   # row-parallel Gemm feeding the module's AllReduce2 = the PARTIAL
    assert m[("AllReduce2_v1", "model.layers.0.self_attn.o_proj")] == ["0"]      # the collective row = the module's bound (all-reduced) output
    assert m[("EmbeddingShard_v1", "model.embed_tokens")] == ["part_rank1"]
    assert m[("Gemm_v1", "logits_processor")] == ["shard_rank1"]           # lm_head shard feeding AllGather2
    assert m[("AllGather2_v1", "logits_processor")] == ["out"]
    assert m[("Gemm_v1", "model.layers.0.self_attn.qkv_proj")] == ["0"]          # column-parallel: no collective under it, the module's output
    _pi0, m0 = _members({"world": 2, "rank": 0})
    assert m0[("Gemm_v1", "model.layers.0.self_attn.o_proj")] == ["part_rank0"]  # the rank label is the Program's rank, never a default


def test_without_tp_context_the_row_parallel_gemm_is_the_module_output_the_measured_false_mismatch():
    _pi, m = _members(None)
    assert m[("Gemm_v1", "model.layers.0.self_attn.o_proj")] == ["0"]            # what tp10 replayed: F_V(partial) vs the all-reduced member
    _pi1, m1 = _members({"world": 1, "rank": 0})                                 # a TP1 Program: tp context is dropped, TP1 semantics unchanged
    assert m1[("Gemm_v1", "model.layers.0.self_attn.o_proj")] == ["0"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---- [MAN-09 per rank] the weights-pin fold ------------------------------------------------------------------------------------------

def _rows_with_pins(p0, p1):
    from verity_capture.tp.commit import _weights_pin_component
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
