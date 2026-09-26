"""`Q_word_v1{X,W,R=no-recompute}` (`verity_vllm.query.word`): every computation gate certified by exactly one unit, cross-unit reads
only through committed values (`verity.ir.partition.validate_unit_cut`), wiring and constants as structure, the widths as the IR's own
|out(S)| where no wiring sits between units, the committed interior values of a norm, a Gemm and FA2 attention, and failures by name."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ir.layout import GateSelection
from verity_vllm.query import word as W


def _fn(fam: str, **statics):
    return W.specialization(fam, statics)


@pytest.fixture(scope="module")
def norm():
    return _fn("RMSNormTriton_v1", N=2048, EPS=1e-05)


@pytest.fixture(scope="module")
def attn():
    return _fn("Attention_v3", T=21, NH=4, KVH=2, D=64, BN=128)


def test_query_id_and_parse():
    assert W.query_id() == "Q_word_v1{X=16,W=32,R=no-recompute}"
    assert W.parse("16") == (16, 32, 0) and W.parse("16/32+2") == (16, 32, 2) and W.parse("8/8") == (8, 8, 0)


@pytest.mark.parametrize("fam,statics", [("RMSNormTriton_v1", {"N": 2048, "EPS": 1e-05}), ("Gemm_v1", {"K": 64, "N": 8}),
                                         ("Attention_v3", {"T": 21, "NH": 4, "KVH": 2, "D": 64, "BN": 128}), ("TokenSelect_v1", {"V": 8})])
def test_every_computation_gate_is_certified_by_one_unit(fam, statics):
    fn = _fn(fam, **statics)
    G = W.Graph(fn)
    R = W.units(G, 16, 32)
    assert R["cut"].ok, R["cut"].detail
    assert G.n + G.free == fn.gates and R["owner"].min() >= 0 and int(R["gates"].sum()) == G.n
    assert R["cut"].detail["input_gates"] == len(G.input_ids) > 0 and R["cut"].detail["certified"] == G.n
    from verity.ir.defs import REGISTRY
    prims = {G.prim_names[p] for p in set(G.prim.tolist())}
    assert not prims & W.WIRING and all(REGISTRY.defs[p].params for p in prims)     # structure is in no unit


@pytest.mark.parametrize("fam,statics", [("Gemm_v1", {"K": 64, "N": 8}), ("RoPE_v1", {"NHEADS": 2, "D": 8}), ("SiluMul_v1", {"I": 8})])
def test_widths_are_the_irs_out_where_no_wiring_sits_between_units(fam, statics):
    fn = _fn(fam, **statics)
    G = W.Graph(fn)
    R = W.units(G, 16, 32)
    c, call = W._call_scope(fn)
    for u in range(len(R["kinds"])):
        gates = (G.keep[np.flatnonzero(R["owner"] == u)] + call.offset).tolist()
        out = GateSelection(c, gates).out
        assert sum(c.gate(g).prim.ret.w for g in out) == R["out_bits"][u] and len(out) == R["out_gates"][u]


def test_the_checker_names_gates_that_break_the_invariant():
    from verity.ir.partition import validate_unit_cut
    v = validate_unit_cut(10, 2, [0, 0, 1, -1, 1, 1, 2, 2], [(0, 2), (1, 2), (2, 6), (4, 6)], [7], {1, 7})
    assert v.codes == ("gate-not-certified-once", "gate-count", "read-uncommitted")
    assert v.detail["gate-not-certified-once"]["gates"] == [3] and v.detail["read-uncommitted"]["gates"] == [[0, 2], [2, 6], [4, 6]]
    assert validate_unit_cut(9, 2, [0, 0, 1, 1, 1, 1, 2], [(1, 2), (5, 6)], [6], {1, 5, 6}).ok
    assert validate_unit_cut(9, 2, [0, 0, 1, 1, 1, 1, 2], [(1, 2), (5, 6)], [6], {1, 5, 6, 3}).codes == ("committed-unread",)
    assert validate_unit_cut(9, 2, [0, 0, 1, 1, 1, 1, 2], [(1, 2), (5, 6)], [6], {1, 5}).codes == ("output-uncommitted",)
    # the input unit: every input gate in it, read by any unit through its committed value
    from verity.ir.partition import INPUT_UNIT
    own = [0, 1, INPUT_UNIT, INPUT_UNIT]
    assert validate_unit_cut(5, 1, own, [(2, 0), (3, 1), (0, 1)], [1], {0, 1, 2, 3}, input_gates=[2, 3]).ok
    v = validate_unit_cut(5, 1, [0, 1, 0, INPUT_UNIT], [(2, 0), (3, 1), (0, 1)], [1], {0, 1, 3}, input_gates=[2, 3])
    assert v.codes == ("input-outside-input-unit",) and v.detail["input-outside-input-unit"]["gates"] == [2]
    assert validate_unit_cut(5, 1, own, [(2, 0), (3, 1), (0, 1)], [1], {0, 1, 2}, input_gates=[2, 3]).codes == ("read-uncommitted",)


def _copies_bits(prim, rng) -> bool:
    ins = [t.w for _n, t in prim.params]
    specials = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x3F800000, 1, 0xFFFFFFFF]
    tries = [[int(rng.integers(0, 1 << w)) for w in ins] for _ in range(64)] + [[sv & ((1 << w) - 1) for w in ins] for sv in specials]
    src: dict[int, set] = {}
    for a in tries:
        y = prim.evaluate(*a)
        for j in range(prim.ret.w):
            yj = (y >> j) & 1
            ok = {("c", yj)} | {(i, b) for i, w in enumerate(ins) for b in range(w) if ((a[i] >> b) & 1) == yj}
            src[j] = ok if j not in src else src[j] & ok
            if not src[j]:
                return False
    return True


def test_wiring_only_rearranges_bits():
    from verity.ir.defs import REGISTRY
    W.specialization("Gemm_v1", {"K": 16, "N": 1})
    rng = np.random.default_rng(0)
    assert all(_copies_bits(REGISTRY.defs[p], rng) for p in W.WIRING)
    assert not _copies_bits(REGISTRY.defs["F32Add_v1"], rng) and not _copies_bits(REGISTRY.defs["GuardNegInfZero_v1"], rng)


def test_a_norm_commits_only_its_scale(norm):
    r = W.unit_rule(norm)
    assert r["committed_interior_words"] == 1 and not r["violations"]
    assert r["by_kind"] == {"output": 2048, "committed": 1} and r["free_gates"] == 4100      # 2048 upcasts + 2048 cvt, 4 constants
    out = next(c for c in r["classes"] if c["kind"] == "output")
    assert out["out_bits"] == 16 and out["out_gates"] == 1


def test_a_gemm_is_one_unit_per_output():
    r = W.unit_rule(_fn("Gemm_v1", K=64, N=8))
    assert r["by_kind"] == {"output": 8} and r["committed_interior_words"] == 0 and r["classes"][0]["out_bits"] == 16


def test_attention_commits_scores_p_row_values_and_inverse_sum(attn):
    r = W.unit_rule(attn)
    assert not r["violations"]
    heads = {(i["scope"].split("{")[0], i["head"]): i["words"] for i in r["committed_interior"]}
    assert heads[("DotBf16_v1", "AmpereBF16TcDot16_v1")] == 4 * 21                  # S: one score per (head, key)
    assert heads[("AttnBlock_v3", "F2fpBf16_v1")] == 4 * 21                          # P (bf16): what the PV products read
    assert heads[("AttentionHead_v3", "Fa2InvSum_v1")] == 4                          # FIN inv_sum, one per head
    assert heads[("AttnBlock_v3", "MufuEx2Ftz_v1")] == 4 * 21                        # exp2(s * scale - max), f32: the stream's P plane
    assert "recompute" not in r["by_kind"]


def test_a_width_over_x_and_w_is_named():
    with pytest.raises(W.QueryRuleViolation, match=r"Gemm_v1\{K=64,N=8\}: output:F2fpBf16_v1"):
        W.check_calls([{"family": "Gemm_v1", "statics": {"K": 64, "N": 8}, "definition": "Gemm_v1{K=64,N=8}", "calls": 3, "where": "m.proj"}], 8, 8)


def test_uncommitted_boundaries_are_named():
    g = {"family": "Gemm_v1", "statics": {"K": 64, "N": 8}, "definition": "Gemm_v1{K=64,N=8}", "calls": 2, "where": "m.proj",
         "outputs_committed": False}
    r = W.check_calls([g], strict=False)
    assert [v["class"] for v in r["violations"]] == ["output-not-committed"] and r["violations"][0]["where"] == ["m.proj"]
    with pytest.raises(W.QueryRuleViolation, match="output-not-committed, 2 Call"):
        W.check_calls([g])
    ok = W.check_calls([dict(g, outputs_committed=True)])
    assert ok["ok"] and ok["units"] == 16 and ok["gates"] == 2 * _fn("Gemm_v1", K=64, N=8).gates


def test_the_manifest_build_runs_the_check(tmp_path):
    from tests.check.test_sampled_replay import _program
    from verity_vllm.pipeline import manifest as M
    from verity_vllm.query.program_view import from_instances
    from verity_vllm.correspondence.reader_for_query import Correspondence
    from verity_vllm.query.required import population
    inst, _w = _program(attn=True)
    P = from_instances(inst)
    corr = Correspondence(P)
    pop = population(P, corr)
    r = M.word_check(P, corr, pop.required, "16/32", strict=False)
    assert r["query"] == "Q_word_v1{X=16,W=32,R=no-recompute}" and r["calls"] == len(inst["rows"]) - len(pop.literals)
    assert r["units"] > r["calls"]
    # the test Program's o_proj output feeds no other module: interior to its body, so no unit can end at it
    assert {(v["definition"].split("{")[0], v["class"]) for v in r["violations"]} == {("Gemm_v1", "output-not-committed")}
    with pytest.raises(W.QueryRuleViolation, match="o_proj"):
        M.word_check(P, corr, pop.required, "16/32")


def test_a_dead_gate_joins_the_unit_it_reads():
    r = W.unit_rule(_fn("TokenSelect_v1", V=8))
    assert r["by_kind"] == {"output": 1} and r["classes"][0]["out_bits"] == 32 and not r["violations"]
