"""The cached prefix of each request of a prefix-caching row (`program.prefix_plan`), and the launch rows it leaves its prefill
(`pipeline.launch_context`): the uncached suffix, at positions C.."""
from __future__ import annotations

import json

import pytest

from verity_vllm.pipeline import launch_context as LC
from verity_vllm.program import prefix_plan as PP


def _wl(reqs, *, on=True, bs=None):
    args = {"enable_prefix_caching": on} | ({"block_size": bs} if bs else {})
    return {"engine": {"args": args, "target": {"batch_invariant": True}},
            "requests": [{"request_id": f"r{i}", "prompt_len": len(ids), "max_tokens": c, "arrive_step": a, "prompt_token_ids": ids}
                         for i, (ids, c, a) in enumerate(reqs)]}


P, X, Y = list(range(1000, 1128)), list(range(1, 41)), list(range(2001, 2041))


def test_a_later_request_hits_the_full_blocks_an_earlier_one_computed():
    w = _wl([(P + X, 10, 0), (P + Y, 10, 4)])
    assert PP.plan(w) == [{"cached": 0, "producers": []}, {"cached": 128, "producers": [0] * 8}]
    assert PP.cached_of(_wl([(P + X, 10, 0), (P + Y, 10, 4)], on=False)) == [0, 0]


def test_the_hit_is_whole_blocks_and_never_the_whole_prompt():
    assert PP.cached_of(_wl([(P + X, 4, 0), (P[:120] + Y, 4, 1)])) == [0, 112]       # 120 shared rows: 7 full blocks
    assert PP.cached_of(_wl([(P, 4, 0), (P, 4, 1)])) == [0, 112]                     # the same 128 rows: the last row is computed
    assert PP.cached_of(_wl([(P + X, 4, 0), (P + Y, 4, 1)], bs=64)) == [0, 128]
    assert PP.cached_of(_wl([(P[:100], 4, 0), (P + Y, 4, 1)])) == [0, 96]            # the producer's prompt holds 6 full blocks


def test_schedule_order_is_arrival_then_index_and_a_same_step_request_hits():
    w = _wl([(P + Y, 4, 3), (P + X, 4, 0), (P + X[:5], 4, 3)])
    assert PP.plan(w) == [{"cached": 128, "producers": [1] * 8}, {"cached": 0, "producers": []},
                          {"cached": 128, "producers": [1] * 8}]
    assert PP.cached_of(_wl([(P + X, 4, 0), (P + Y, 4, 0)])) == [0, 128]


def test_each_block_names_the_request_that_computed_it():
    A = P[:64] + list(range(5000, 5064))
    w = _wl([(P[:64] + X, 4, 0), (P + X, 4, 1), (P + Y, 4, 2), (A, 4, 3)])
    assert [p["producers"] for p in PP.plan(w)] == [[], [0] * 4, [0] * 4 + [1] * 4, [0] * 4]


def test_a_prefix_row_without_prompt_ids_is_refused():
    w = _wl([(P + X, 4, 0), (P + Y, 4, 1)])
    del w["requests"][1]["prompt_token_ids"]
    with pytest.raises(ValueError, match=r"requests \[1\]"):
        PP.plan(w)


def test_shape_keys_and_names_keep_every_uncached_shape():
    assert PP.shape(168, 9) == (168, 9) and PP.shape(168, 9, 0) == (168, 9) and PP.shape(168, 9, 128) == (168, 9, 128)
    assert PP.shape_name((168, 9)) == "LP168_T9" and PP.shape_name((168, 9, 128)) == "LP168_T9_C128"
    assert PP.shape_key((168, 9, 128)) == "168,9,128" and PP.parse_shape_key("168,9") == (168, 9)
    assert PP.cached_rows(PP.parse_shape_key("168,9,128")) == 128 and PP.cached_rows((168, 9)) == 0


def test_a_warm_prefill_launches_its_uncached_suffix_from_row_c():
    w = _wl([(P + X, 10, 0), (P + Y, 10, 4)])
    ctx = LC.shape_contexts(w)
    assert ctx == {(168, 9): [168, 1, 1, 1, 40, 1, 1, 1, 1, 1], (168, 9, 128): [40] + [1] * 9}
    assert LC.spec_for(w, 168, 9, 128) == json.dumps([40] + [1] * 9, separators=(",", ":"))
    assert LC.spec_for(w, 168, 9) == json.dumps(ctx[(168, 9)], separators=(",", ":"))
    w["engine"]["target"]["batch_invariant"] = False
    b = LC.shape_contexts(w)
    assert b[(168, 9, 128)][0] == [[1, 171], [40, 128]]          # r0 decodes row 171 beside r1's suffix rows 128..167
    assert b[(168, 9)][0] == [[168, 0]]


def test_a_warm_request_finds_its_own_program_and_a_cold_one_never_takes_it(tmp_path):
    from verity_vllm.pipeline import manifest as PM
    from verity_vllm.query.manifest import format as MF
    for name, res in (("build_request", {"LP": 168, "T": 9}), ("build_request_LP168_T9_C128", {"LP": 168, "T": 9, "C": 128})):
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text(json.dumps(dict(res, ok=True)))
    fx = {"requests": PP.with_cached(_wl([(P + X, 10, 0), (P + Y, 10, 4)]))}
    want = {"r0": str(tmp_path / "build_request"), "r1": str(tmp_path / "build_request_LP168_T9_C128")}
    assert MF.program_dirs_for(fx, str(tmp_path)) == want and PM._program_dirs_for(fx, str(tmp_path)) == want
    (tmp_path / "build_request_LP168_T9_C128").rename(tmp_path / "build_request_warm")      # found by its result.json's C
    assert MF.program_dirs_for(fx, str(tmp_path))["r1"] == str(tmp_path / "build_request_warm")
    (tmp_path / "build_request_warm" / "result.json").write_text(json.dumps({"ok": True, "LP": 168, "T": 9}))
    with pytest.raises(SystemExit, match="r1 .*168,9,128"):
        MF.program_dirs_for(fx, str(tmp_path))
    with pytest.raises(SystemExit, match="r1 .*168,9,128"):
        PM._program_dirs_for(fx, str(tmp_path))


def test_a_warm_programs_cached_rows_are_not_token_ids():
    from types import SimpleNamespace as NS
    from verity_vllm.query.required import ValueRef, cached_prefix_inputs
    P_ = NS(params=[NS(id="prompt", port_table=[NS(name="out")]), NS(id="prefix", port_table=[NS(name="out")]),
                    NS(id="weights", port_table=[NS(name="w0"), NS(name="w1")])])
    assert cached_prefix_inputs(P_, {"prefix": {"input": 1, "C": 128}}) == {ValueRef("prefix", "out")}
    assert cached_prefix_inputs(P_, {"LP": 168}) == frozenset()
