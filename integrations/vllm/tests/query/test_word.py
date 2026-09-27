"""`Q_word_v1{X,W,R=no-recompute}` (`verity_vllm.query.word`): every computation gate certified by exactly one unit, cross-unit reads
only through committed values (`verity.ir.partition.validate_unit_cut`), wiring and constants as structure, the widths as the IR's own
|out(S)| where no wiring sits between units, the committed interior values of a norm, a Gemm and FA2 attention, and failures by name."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ir.layout import GateSelection
from verity.ir.cut import call_scope
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
    c, call = call_scope(fn)
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
    assert r["by_kind"] == {"output": 2048, "committed": 1} and r["free_gates"] == 4101      # 4096 upcasts, 4 constants, rcp(N)
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


@pytest.mark.parametrize("name,E,committed", [("MoeRouterTopKOrdered", 64, 2 * 64 + 2), ("MoeRouterTopKOrderedNorm", 128, 2 * 128 + 2 + 8 + 1)])
def test_the_ordered_router_commits_its_softmax_once_and_recomputes_nothing(name, E, committed):
    from verity.ir.defs import bind
    from verity_vllm.program.registry import moe
    f = bind(getattr(moe, name), E=E, TOPK=8, VPT=8)
    G = W.Graph(f)
    R = W.units(G, 16, 32)
    assert R["cut"].ok and G.recomputed == [], R["cut"].codes
    r = W.unit_rule(f)
    assert r["committed_interior_words"] == committed and not r["violations"] and r["max_out_bits"] == 32
    kernel = W.unit_rule(bind(moe.MoeRouterTopKNorm if name.endswith("Norm") else moe.MoeRouterTopK, E=E, TOPK=8, VPT=8))
    assert kernel["committed_interior_words"] > 10 * committed          # kernel order commits every round's masked vector


def test_a_recompute_fails_the_cut():
    from verity.ir.defs import CompositeDefinition, bind
    from verity.ir.refs import tuple_of
    from verity.ir.types import Array, Tuple
    from verity_vllm.program.registry import prims as P
    F32 = P.F32

    def body(B, S, x):
        a = B.call(P.F32Add, x[0], x[1])
        b = B.call(P.F32Add, x[0], x[1])                              # the same value computed a second time
        return tuple_of(B.call(P.F32Mul, a, x[2]), B.call(P.F32Mul, b, x[3]))

    f = bind(CompositeDefinition("Twice", 1, (), lambda S: ((("x", Array(4, F32)),), Tuple(F32, F32)), body, register=False))
    G = W.Graph(f)
    R = W.units(G, 16, 32)
    assert R["cut"].codes == ("gate-recomputed",) and R["cut"].detail["gate-recomputed"]["gates"] == [[1, 0]]


def test_a_value_computed_again_in_its_own_unit_is_redundant_not_a_recompute():
    from verity.ir.partition import validate_unit_cut
    assert validate_unit_cut(3, 0, [0, 0, 1], [(0, 2), (1, 2)], [2], {0, 2}, recomputed=[(1, 0)]).codes == ("read-uncommitted",)
    v = validate_unit_cut(3, 0, [0, 0, 0], [(0, 2), (1, 2)], [2], {2}, recomputed=[(1, 0)])
    assert v.ok and v.detail["redundant_gates"] == 1
    v = validate_unit_cut(3, 0, [0, 1, 2], [(0, 2), (1, 2)], [2], {0, 1, 2}, recomputed=[(1, 0)])
    assert v.codes == ("gate-recomputed",) and v.detail["gate-recomputed"]["gates"] == [[1, 0]] and v.detail["redundant_gates"] == 0


def test_the_sampler_computes_its_greedy_flag_twice_inside_its_one_unit():
    from verity.ir.defs import bind
    from verity_vllm.program.registry import sampling as SM
    G = W.Graph(bind(SM.GumbelTopPTokenSelect, V=16))
    R = W.units(G, 16, 32)
    (a, b), = G.recomputed                                            # temp == 0, in TemperatureScale and in GumbelSelectF32
    assert {G.prim_names[int(G.prim[a])], G.prim_names[int(G.prim[b])]} == {"F32Eq_v1"} and R["owner"][a] == R["owner"][b]
    assert R["kinds"] == ["output"] and R["cut"].ok and R["cut"].detail["redundant_gates"] == 1


def test_a_call_that_only_rearranges_bits_returns_its_inputs():
    # every gate is wiring: the returned values are the committed inputs themselves (not committed and unread)
    from verity.ir.defs import CompositeDefinition, bind
    from verity.ir.types import Array
    from verity_vllm.program.registry import prims as P
    f = bind(CompositeDefinition("Widen", 1, ("N",), lambda S: ((("a", Array(S.N, P.BF16)),), Array(S.N, P.F32)),
                                 lambda B, S, a: B.batch(P.Bf16ToF32, a, axes=(0,)), register=False), N=8)
    G = W.Graph(f)
    R = W.units(G, 16, 32)
    assert G.n == 0 and len(G.returned_inputs) == 8 and R["cut"].ok, R["cut"].codes
    assert W.unit_rule(f)["violations"] == []


def test_the_graph_is_the_core_reference_and_a_programs_partition_object_verifies_in_core():
    # word.Graph is verity.ir.cut.CallGraph with numpy views; a Program's cut, as verity/partition/v1, checks in core
    from verity.ir import cut, partition_object as PO
    from verity.ir.defs import CompositeDefinition, bind
    from verity.ir.program import Program
    from verity.ir.refs import tuple_of
    from verity.ir.types import Tuple
    fns = [_fn("RMSNormTriton_v1", N=64, EPS=1e-05), _fn("TokenSelect_v1", V=8), _fn("Attention_v3", T=3, NH=2, KVH=1, D=16, BN=128)]
    for f in fns:
        c, g = cut.CallGraph(f), W.Graph(f)
        assert (c.n, c.free, c.outputs, c.recomputed, c.input_reads) == (g.n, g.free, g.outputs, g.recomputed, g.input_reads)
        assert list(c.src) == g.src.tolist() and list(c.width) == g.width.tolist()
        R = W.units(g, 16, 32)
        ob, og = cut.boundary_widths(c, R["owner"].tolist(), len(R["kinds"]))
        assert ob == R["out_bits"].tolist() and og == R["out_gates"].tolist()
    sig = [f.signature() for f in fns]
    params = tuple((f"p{i}_{n}", t) for i, (ps, _r) in enumerate(sig) for n, t in ps)

    def body(B, S, *a):
        outs, i = [], 0
        for f, (ps, _r) in zip(fns, sig):
            outs.append(B.call(f, *a[i:i + len(ps)]))
            i += len(ps)
        return tuple_of(*outs)

    P = Program(bind(CompositeDefinition("PartitionTop", 1, (), lambda S: (params, Tuple(*[r for _p, r in sig])), body, register=False)))
    obj = W.partition_object(P)
    assert PO.validate(obj).ok and PO.verify(obj, P).ok and len(obj["calls"]) == 3
    assert len(obj["units"]) == sum(W.unit_rule(f)["units_per_call"] for f in fns[:2]) + len(set(obj["calls"][2]["owner"]))
    assert PO.digest(obj) == PO.digest(W.partition_object(P))
