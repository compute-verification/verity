"""STRUCT-B (2026-09-10): the ancestor cones of two attention output coordinates OVERLAP on the existing program.

TINY only (``e2e.population.TINY``: LP=4, STEPS=8, NL=2, NH=2, KVH=1, D=16).  The numbers asserted here are the ones
reported in ``docs/vllm-poc/reports/STRUCT-B-2026-09-10.md`` §1 for the head ``t=11/layers.0/attn`` member 0
(``AttentionHead_v1{T=12, D=16}``); every count follows from the ``AttnBlock_v1`` / ``AttentionHead_v1`` bodies:

* shared by every coordinate: the T q.k dots (``DotBf16{K=16}`` = 1 tc_dot16 + 1 zero each), T-1 row-max gates, the guard,
  the scale constant and ``m_use*scale``, T ``F32FmaSubFtz``, T ``MufuEx2Ftz``, T ``F2fpBf16`` (P -> bf16), the lane sums
  (T-4 ``F32AddFtz`` + 3 epilogue adds), the two ``Const16`` pads, ``Fa2InvSum``;
* exclusive to coordinate d: its accumulator zero, ceil(T/16) P.V ``tc_dot16`` steps, ``o[d]*inv`` and the output cast.

A bare ``AttentionHead_v1{T=47, D=64}`` (the B0 head shape at the longest B0 context) is checked the same way, and the
whole-B0 sweep must reproduce the Table-3(a) census attention rows kind for kind.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

POC = Path(__file__).resolve().parents[1]
for p in (str(POC), str(POC.parent), str(POC.parent / "src"), str(POC / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import struct_b_cones as sbc  # noqa: E402

CENSUS = POC.parent / "docs" / "data" / "table3a" / "B0-serve-lp32-s15.json"


@pytest.fixture(scope="module")
def tiny():
    program, adapted, circuit = sbc.build_tiny()
    calls = sbc.head_calls(circuit)
    return program, circuit, calls


def _head(calls, circuit, t: int, layer: int, member: int):
    for c in calls:
        parent = c.parent
        node = parent.parent.fn.nodes[parent.node_idx]
        if node.name == f"t={t}/layers.{layer}/attn" and c.member == member:
            return sbc.HeadCones.from_call(circuit, c)
    raise AssertionError("head not found")


def test_tiny_two_coordinates_share_the_score_softmax_cone(tiny):
    program, circuit, calls = tiny
    assert len(calls) == 48  # 12 positions x 2 layers x 2 heads
    hc = _head(calls, circuit, t=11, layer=0, member=0)
    assert (hc.T, hc.D) == (12, 16)
    r = hc.analyse([0, 1])
    T = 12
    # cones: 92 gates each, 88 shared, 4 exclusive each
    assert r["per_coord"]["0"]["cone_gates"] == r["per_coord"]["1"]["cone_gates"] == 92
    assert r["shared_gates"] == 88 and r["shared_p1_units"] == 76 and r["shared_nonconst_gates"] == 73
    assert r["union_gates"] == 88 + 2 * 4
    for d in ("0", "1"):
        pc = r["per_coord"][d]
        assert pc["exclusive_gates"] == 4
        assert pc["exclusive_by_kind"] == {"AmpereBF16TcDot16_v1[pv]": 1, "Const32[0x00000000]_v1": 1, "F2fpBf16_v1": 1,
                                           "F32MulFtz_v1[epilogue o*inv]": 1}
        assert pc["output_prim"] == "F2fpBf16_v1" and pc["output_bits"] == 16
        inc = pc["incoming"]
        assert inc["q_coord_count"] == 16 and inc["k_row_count"] == T and inc["k_coord_count"] == 16
        assert inc["v_row_count"] == T and inc["v_coords"] == [int(d)]  # V column d only
        assert inc["unclassified_gates"] == 0
    shared = r["shared_by_kind"]
    assert shared["AmpereBF16TcDot16_v1[qk]"] == T and shared["Const32[0x00000000]_v1[qk-acc0]"] == T
    assert shared["F32Max_v1"] == T - 1 and shared["F32FmaSubFtz_v1"] == T and shared["MufuEx2Ftz_v1"] == T
    assert shared["F2fpBf16_v1"] == T and shared["F32AddFtz_v1"] == (T - 4) + 3
    assert shared["GuardNegInfZero_v1"] == 1 and shared["Fa2InvSum_v1"] == 1 and shared["F32MulFtz_v1[*scale_log2]"] == 1
    assert shared["Const16[0x0000]_v1"] == 2
    # (b2): the owner's outgoing interface = T probabilities (bf16) + 1/l (+ the two pad constants)
    b2 = r["b2_owner"]
    assert b2["interface_nonconst_bits"] == 16 * T + 32 and b2["interface_bits"] == 16 * T + 32 + 2 * 16
    assert b2["interface_nonconst_bits"] > 32


def test_tiny_third_coordinate_and_whole_head(tiny):
    program, circuit, calls = tiny
    hc = _head(calls, circuit, t=11, layer=0, member=0)
    r3 = hc.analyse([0, 1, 15])
    assert r3["shared_gates"] == 88  # the same 88 gates are shared by d = 0, 1 and 15
    assert r3["pairwise_overlap_gates"] == {"0,1": 88, "0,15": 88, "1,15": 88}
    r_all = hc.analyse(list(range(16)))
    assert r_all["shared_gates"] == 88 and r_all["union_gates"] == 88 + 16 * 4
    assert r_all["b3_duplication"]["sum_cone_gates"] == 16 * 92
    assert r_all["dead_gates_in_head"] == 1  # the ignored FIRST-block state zero
    assert r_all["dead_by_kind"] == {"Const32[0x00000000]_v1": 1}


def test_tiny_different_heads_are_disjoint_but_read_the_same_kv_rows(tiny):
    program, circuit, calls = tiny
    h0 = _head(calls, circuit, t=11, layer=0, member=0)
    h1 = _head(calls, circuit, t=11, layer=0, member=1)
    c0, in0 = h0.cone(h0.outputs[0])
    c1, in1 = h1.cone(h1.outputs[0])
    assert not (c0 & c1)
    assert h0.hi <= h1.lo or h1.hi <= h0.lo
    # GQA with KVH = 1: both heads read the SAME committed K/V rows (shared inputs, disjoint gates)
    assert in0 & in1
    s0, s1 = h0.incoming_summary(in0), h1.incoming_summary(in1)
    assert s0["k_row_count"] == s1["k_row_count"] == 12


def test_bare_b0_head_shape():
    program, adapted, circuit = sbc.build_head(47, 64)
    hc = sbc.HeadCones.from_root(circuit, program)
    r = hc.analyse([0, 1, 63])
    assert r["shared_gates"] == 474 and r["shared_p1_units"] == 286
    for d in ("0", "1", "63"):
        assert r["per_coord"][d]["cone_gates"] == 480 and r["per_coord"][d]["exclusive_gates"] == 6
        assert r["per_coord"][d]["incoming"]["v_coords"] == [int(d)]
    assert r["b2_owner"]["interface_nonconst_bits"] == 16 * 47 + 32


@pytest.mark.skipif(not CENSUS.exists(), reason="census file absent")
def test_b0_sweep_reproduces_the_census_attention_rows():
    doc = sbc.sweep(1, 47, 64, 270, CENSUS)
    assert doc["census_check"]["all_match"], json.dumps(doc["census_check"]["rows"], indent=1)
    s = doc["totals_x_mult"]
    assert s["heads"] == 12690 and s["coordinates"] == 812160
    assert s["union_gates"] == 7145280 and s["union_p1_units"] == 5927040
    assert s["shared_p1_units"] == 1883520 and s["shared_nonconst_p1_units"] == 1843290
    assert s["b3_sum_cone_gates"] == 202556160
