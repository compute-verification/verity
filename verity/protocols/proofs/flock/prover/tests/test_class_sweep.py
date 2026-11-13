"""The backend sweep's work lists and whole-row figures (`verity_flock.class_sweep`), on synthetic graphs and records."""
from __future__ import annotations

import math

from verity_flock import class_sweep as CW

ATT = "Attention_v3{{T={t},NH={nh},KVH=8,D=64,BN=64}}"


def _graph(groups):
    return {"groups": groups}


def test_graph_calls_expands_a_varying_group():
    g = _graph([{"module": "m.layers.0.attn", "spec": None, "calls": 6, "varying": {"T": [1, 3]},
                 "varying_calls": {"1": 1, "2": 1, "3": 1}, "definition_ids": [ATT.format(t=1, nh=32)]},
                {"module": "m.layers.0.mlp", "spec": "SiluMul_v1{I=512}", "calls": 4},
                {"module": "m.lm_head", "spec": "SiluMul_v1{I=512}", "calls": 1}])
    assert CW.graph_calls(g) == {ATT.format(t=1, nh=32): 2, ATT.format(t=2, nh=32): 2, ATT.format(t=3, nh=32): 2, "SiluMul_v1{I=512}": 5}
    assert CW.graph_calls(g, "m.layers.0.mlp") == {"SiluMul_v1{I=512}": 4}


def test_grid_keeps_every_tail_the_largest_and_a_geometric_step():
    ts = list(range(1, 1001))
    got = CW.grid(ts, 64, 24)
    assert got[:128] == list(range(1, 129)) and got[-1] == 1000 and set(got) <= set(ts) and got == sorted(set(got))
    assert len(got) - 128 == math.ceil(24 * math.log2(1000 / 128))
    assert CW.grid([5, 9], 64, 24) == [5, 9]


def test_plan_dedupes_heads_across_rows_leaves_the_full_row_out_and_splits_by_hardware():
    rows = {"a": {"row": 1, "hw": "l40s", "calls": {ATT.format(t=t, nh=32): 2 for t in range(1, 201)} | {"Gemm_v1{K=64,N=8}": 3,
                                                                                                        "GumbelTopPTokenSelect_v1": 1}},
            "b": {"row": 2, "hw": "l40s", "calls": {ATT.format(t=t, nh=9): 1 for t in range(1, 401)} | {"Gemm_v1{K=64,N=8}": 1}},
            "c": {"row": 3, "hw": "h100", "calls": {ATT.format(t=t, nh=9): 1 for t in range(1, 11)}}}
    got = CW.plan(rows, {"1"}, 4, {"l40s": 2, "h100": 1})
    l40s = {**got["lists"]["l40s-1"], **got["lists"]["l40s-2"]}
    assert not set(got["lists"]["l40s-1"]) & set(got["lists"]["l40s-2"])
    assert "Gemm_v1{K=64,N=8}" not in l40s and not any(f.startswith("Gumbel") for f in l40s)   # row 1's own run proves it
    ts = sorted(CW.t_of(f) for f in l40s)
    assert ts[0] == 201 and ts[-1] == 400                   # the full row covers the head up to T = 200
    assert all(f.endswith("NH=9,KVH=8,D=64,BN=64}") for f in l40s)
    assert set(got["lists"]["h100-1"]) == {ATT.format(t=t, nh=9) for t in range(1, 11)}   # same head, other hardware: proved there too


def _rec(shape, n, b, prove_s, ands, **kw):
    return {"shape": shape, "definition": f"T{shape}", "units": n, "row_units": n, "ands": ands, "stage": {"n": b},
            "live": {"accepted": True, "prove_total_s": prove_s, "proof_bytes": [100, 100], "verify_s": 0.5}, **kw}


def test_rollup_sums_row_units_over_throughput_and_lists_what_was_not_proved():
    recs = [_rec("a", 1000, 256, 2.0, 10), _rec("b", 10, 16, 1.0, 1000),
            {"shape": "c", "definition": "Tc", "units": 5, "row_units": 5, "skipped": "reads", "ands": 7},
            {"shape": "d", "definition": "Td", "units": 1, "broke": {"at": "size", "why": "too big"}}]
    got = CW.rollup(recs, total_ands=40_000, total_units=1016, natives={"tensor_peak": 2.0})
    assert got["proving_s"] == 1000 * 2.0 / 256 + 10 * 1.0 / 16
    assert got["proof_bytes"] == 4 * 200 + 1 * 200 and got["verify_s"] == 5 * 0.5
    assert got["ands_proved"] == 20_000 and got["and_coverage"] == 0.5
    assert got["overhead"]["tensor_peak"]["overhead"] == got["proving_s"] / 1.0
    assert [(x["shape"], x["why"]) for x in got["not_proved"]] == [("c", "skipped: reads"), ("d", "broke at size: too big")]


def test_regroup_carries_known_definitions_over_and_names_the_ones_to_cut():
    known = CW.merge_counts([{"calls": {"A": 2}, "per_definition": {"A": {"s1": 3, "s2": 1}}, "shapes": {"s1": {"row_units": 6, "definition": "a"},
                                                                                                         "s2": {"row_units": 2, "definition": "a"}}},
                             {"calls": {"B": 1}, "per_definition": {"B": {"s2": 5}}, "shapes": {"s2": {"row_units": 5, "definition": "b"}}}])
    assert known["shapes"]["s2"]["row_units"] == 7
    got, missing = CW.regroup({"A": 10, "B": 1, "C": 4}, known)
    assert missing == ["C"] and got["shapes"]["s1"]["row_units"] == 30 and got["shapes"]["s2"]["row_units"] == 15


def test_rollup_with_counts_lists_every_shape_of_the_row():
    got = CW.rollup([_rec("a", 1, 256, 2.0, 10)], total_ands=1e6, counts={"a": {"row_units": 512}, "z": {"row_units": 7, "definition": "Tz"}})
    assert got["row_units_proved"] == 512 and got["proving_s"] == 512 * 2.0 / 256
    assert [(x["shape"], x["why"]) for x in got["not_proved"]] == [("z", "not run")]


def test_rollup_prefers_an_accepted_record_across_runs():
    recs = [_rec("a", 10, 16, 1.0, 5), {"shape": "a", "definition": "Ta", "units": 10, "row_units": 10, "skipped": "table-free", "ands": 5}]
    assert CW.rollup(recs, total_ands=100)["shapes_proved"] == 1
