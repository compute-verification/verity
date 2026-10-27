"""`Q_word_v1{X,W,R=no-recompute}` (`verity.ml.definitions.word`): every computation gate certified by exactly one unit, cross-unit reads
only through committed values (`verity.ml.circuits.partition.validate_unit_cut`), wiring and constants as structure, the widths as the circuit's own
|out(S)| where no wiring sits between units, the committed interior values of a norm, a Gemm and FA2 attention, failures by name, and
the rule cache (`word.use_rule_cache`, the manifest CLI's `--unit-rule-cache`): a read rule is the computed one, keyed by the Definition's content, the query, the limits and
the code, and the manifest is the same with it."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.layout import GateSelection
from verity.ml.circuits.cut import call_scope
from verity.ml.definitions import word as W


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


def test_max_scaled_is_not_in_the_fa_stream():
    # ROW word 2 is the kernel's visit index (vt_step): max * scale needs the MS plane
    for scope in ("AttnBlock_v3{D=64,NVIS=64,FIRST=False,BN=128}", "AttnBlockSoftcap_v1{D=256,NVIS=64,FIRST=False,BN=64,CAP=50.0}"):
        assert W.committed_today(scope, "F32MulFtz_v1") is None
        assert W.tap_kernel(scope, "F32MulFtz_v1") == "FA2 / FA3 tap build (MS plane, `max_scaled`)"
        assert W.committed_today(scope, "F32Max_v1") == "ROW row_max"
        assert W.committed_today(scope, "GuardNegInfZero_v1") is None                           # opt-in taps stay new taps (norm scales, router)
        assert W.tap_kernel(scope, "GuardNegInfZero_v1") == "FA2 / FA3 tap build (ROW word 3, opt-in `GUARDED_MAX_TAP=1`, #95)"
    assert W.tap_kernel("RMSNormTriton_v1{N=64,EPS=1e-05}", "F32MulFtz_v1") == "vLLM batch-invariant Triton _rms_norm_kernel"


@pytest.mark.parametrize("fam,statics", [("RMSNormTriton_v1", {"N": 2048, "EPS": 1e-05}), ("Gemm_v1", {"K": 64, "N": 8}),
                                         ("Attention_v3", {"T": 21, "NH": 4, "KVH": 2, "D": 64, "BN": 128}), ("TokenSelect_v1", {"V": 8})])
def test_every_computation_gate_is_certified_by_one_unit(fam, statics):
    fn = _fn(fam, **statics)
    G = W.Graph(fn)
    R = W.units(G, 16, 32)
    assert R["cut"].ok, R["cut"].detail
    assert G.n + G.free == fn.gates and R["owner"].min() >= 0 and int(R["gates"].sum()) == G.n
    assert R["cut"].detail["input_gates"] == len(G.input_ids) > 0 and R["cut"].detail["certified"] == G.n
    from verity.ml.circuits.defs import REGISTRY
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
    from verity.ml.circuits.partition import validate_unit_cut
    v = validate_unit_cut(10, 2, [0, 0, 1, -1, 1, 1, 2, 2], [(0, 2), (1, 2), (2, 6), (4, 6)], [7], {1, 7})
    assert v.codes == ("gate-not-certified-once", "gate-count", "read-uncommitted")
    assert v.detail["gate-not-certified-once"]["gates"] == [3] and v.detail["read-uncommitted"]["gates"] == [[0, 2], [2, 6], [4, 6]]
    assert validate_unit_cut(9, 2, [0, 0, 1, 1, 1, 1, 2], [(1, 2), (5, 6)], [6], {1, 5, 6}).ok
    assert validate_unit_cut(9, 2, [0, 0, 1, 1, 1, 1, 2], [(1, 2), (5, 6)], [6], {1, 5, 6, 3}).codes == ("committed-unread",)
    assert validate_unit_cut(9, 2, [0, 0, 1, 1, 1, 1, 2], [(1, 2), (5, 6)], [6], {1, 5}).codes == ("output-uncommitted",)
    # the input unit: every input gate in it, read by any unit through its committed value
    from verity.ml.circuits.partition import INPUT_UNIT
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
    from verity.ml.circuits.defs import REGISTRY
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
    assert heads[("DotBf16_v1", "AmpereBF16TcDot16_v2")] == 4 * 21                  # S: one score per (head, key)
    assert heads[("AttnBlock_v3", "F2fpBf16_v1")] == 4 * 21                          # P (bf16): what the PV products read
    assert heads[("AttentionHead_v3", "Fa2InvSum_v1")] == 4                          # FIN inv_sum, one per head
    assert heads[("AttnBlock_v3", "MufuEx2Ftz_v1")] == 4 * 21                        # exp2(s * scale - max), f32: the stream's P plane
    assert "recompute" not in r["by_kind"]


def test_a_width_over_x_and_w_is_named():
    with pytest.raises(W.QueryRuleViolation, match=r"Gemm_v1\{K=64,N=8\}: output:F2fpBf16_v1"):
        W.check_calls([{"family": "Gemm_v1", "statics": {"K": 64, "N": 8}, "definition": "Gemm_v1{K=64,N=8}", "calls": 3, "where": "m.proj"}], 8, 8)


def test_the_gate_limit_is_the_runs_not_the_querys(norm):
    g = {"family": "RMSNormTriton_v1", "statics": {"N": 2048, "EPS": 1e-05}, "definition": norm.id, "calls": 1}
    low = norm.gates - 1
    capped = W.check_calls([g], strict=False, max_gates=f"{low:_}")
    assert [(v["class"], v["limit"]) for v in capped["violations"]] == [("too-large", low)]
    assert W.check_calls([g], strict=False)["ok"] and W.gate_limits() == {"*": W.MAX_GATES}
    for raised in (f"{low},RMSNormTriton_v1={norm.gates}", {"RMSNormTriton_v1": norm.gates}, norm.gates):
        r = W.check_calls([g], strict=False, max_gates=raised)
        assert r["ok"] and r["units"] == 2049 and r["query"] == capped["query"] == W.query_id()
    assert W.gate_limit(norm, f"{low},Gemm_v1=5") == low and W.gate_limit(norm, {norm.id: 7, "RMSNormTriton_v1": 9}) == 7
    from verity_vllm.pipeline import cli
    for cmd, argv in (("manifest", ["build", "--program", "p", "--out", "o"]), ("program-graph", ["o", "i"])):
        assert cli.parse(cmd, argv, {W.MAX_GATES_ENV: "GumbelTopPTokenSelect_v2=120000000"}).word_max_gates == "GumbelTopPTokenSelect_v2=120000000"
        assert cli.parse(cmd, [*argv, "--word-max-gates", "1"], {}).word_max_gates == "1"


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


def _program_dir(tmp_path, **kw) -> str:
    import gzip
    import json
    from tests.check.test_sampled_replay import _program
    inst, _w = _program(**kw)
    with gzip.open(tmp_path / "instances.json.gz", "wt") as f:
        json.dump(inst, f)
    (tmp_path / "result.json").write_text(json.dumps({"program": {"digest": "0" * 128, "function": "f"},
                                                      "target": {"declared": {"compute_capability": [8, 9]}}}))
    return str(tmp_path)


def test_the_manifest_records_the_check_it_passed_in_the_header_and_its_rows_are_unchanged(tmp_path):
    """The header's `query.word_check` is what the Commit requires (`commit.partition.word_check_refusal`); it is no identity, so the
    rows digest of a checked build is the unchecked one's.  A check that fails raises before the manifest is written."""
    from verity_vllm.commit.partition import word_check_refusal
    from verity_vllm.pipeline import manifest as M
    d = _program_dir(tmp_path)
    checked, plain = (M.build(d, request_id="r0", word=w) for w in ("16/32", None))
    wc = checked["query"]["word_check"]
    assert wc["query"] == checked["query"]["query_id"] == "Q_word_v1{X=16,W=32,R=no-recompute}"
    assert wc["strict"] is True and wc["ok"] is True and wc["violations"] == 0 and wc["units"] > wc["calls"] > 0
    assert wc["max_gates"] == W.gate_limits(None) and "groups" not in wc and word_check_refusal(checked["query"]) is None
    assert "word_check" not in plain["query"] and "no word check recorded" in word_check_refusal(plain["query"])
    assert checked["manifest_digest"] == plain["manifest_digest"] and checked["identities"] == plain["identities"]
    assert {k: v for k, v in checked["query"].items() if k != "word_check"} == plain["query"]


def test_a_dead_output_passes_the_check_only_as_unreturned_and_listed(tmp_path):
    """The attention Program's o_proj output feeds nothing and no policy requires it: UNRETURNED, so the check skips it and the
    manifest lists it by identity, outside the identities.  A Call with a required output gets no such pass: without the return-slot
    policy the final fused norm's residual is read by nothing while its normed row feeds the logits, and the check fails it."""
    from verity_vllm.pipeline import manifest as M
    (tmp_path / "attn").mkdir()
    attn = M.build(_program_dir(tmp_path / "attn", attn=True), request_id="r0", word="16/32")
    dead = [r for r in attn["unreturned"] if "o_proj" in r["op_path"]]
    assert attn["query"]["word_check"]["ok"] and dead and {r["spec"] for r in dead} == {"Gemm_v1"}
    assert {r["step"] for r in dead} == set(attn["steps"])
    assert not [r for r in attn["identities"] if "o_proj" in r["op_path"] and r["family"] != "weights"]
    assert attn["populations"]["unreturned_calls"] == sum(r["rows"] for r in dead) and len(attn["unreturned_digest"]) == 64
    with pytest.raises(W.QueryRuleViolation, match="RMSNormFusedCuda_v2.*output-not-committed"):
        M.build(_program_dir(tmp_path, attn=True), request_id="r0", word="16/32", return_slots=False)


def test_a_dead_gate_joins_the_unit_it_reads():
    r = W.unit_rule(_fn("TokenSelect_v1", V=8))
    assert r["by_kind"] == {"output": 1} and r["classes"][0]["out_bits"] == 32 and not r["violations"]


@pytest.mark.parametrize("name,E,committed", [("MoeRouterTopKOrdered", 64, 2 * 64 + 2), ("MoeRouterTopKOrderedNorm", 128, 2 * 128 + 2 + 8 + 1)])
def test_the_ordered_router_commits_its_softmax_once_and_recomputes_nothing(name, E, committed):
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import moe
    f = bind(getattr(moe, name), E=E, TOPK=8, VPT=8)
    G = W.Graph(f)
    R = W.units(G, 16, 32)
    assert R["cut"].ok and G.recomputed == [], R["cut"].codes
    r = W.unit_rule(f)
    assert r["committed_interior_words"] == committed and not r["violations"] and r["max_out_bits"] == 32
    kernel = W.unit_rule(bind(moe.MoeRouterTopKNorm if name.endswith("Norm") else moe.MoeRouterTopK, E=E, TOPK=8, VPT=8))
    assert kernel["committed_interior_words"] > 10 * committed          # kernel order commits every round's masked vector


def test_a_recompute_fails_the_cut():
    G = W.Graph(_twice())
    R = W.units(G, 16, 32)
    assert R["cut"].codes == ("gate-recomputed",) and R["cut"].detail["gate-recomputed"]["gates"] == [[1, 0]]


def test_a_value_computed_again_in_its_own_unit_is_redundant_not_a_recompute():
    from verity.ml.circuits.partition import validate_unit_cut
    assert validate_unit_cut(3, 0, [0, 0, 1], [(0, 2), (1, 2)], [2], {0, 2}, recomputed=[(1, 0)]).codes == ("read-uncommitted",)
    v = validate_unit_cut(3, 0, [0, 0, 0], [(0, 2), (1, 2)], [2], {2}, recomputed=[(1, 0)])
    assert v.ok and v.detail["redundant_gates"] == 1
    v = validate_unit_cut(3, 0, [0, 1, 2], [(0, 2), (1, 2)], [2], {0, 1, 2}, recomputed=[(1, 0)])
    assert v.codes == ("gate-recomputed",) and v.detail["gate-recomputed"]["gates"] == [[1, 0]] and v.detail["redundant_gates"] == 0


def test_the_sampler_computes_its_greedy_flag_twice_inside_its_one_unit():
    G = W.Graph(_sampler())
    R = W.units(G, 16, 32)
    (a, b), = G.recomputed                                            # temp == 0, in TemperatureScale and in GumbelSelectF32
    assert {G.prim_names[int(G.prim[a])], G.prim_names[int(G.prim[b])]} == {"F32Eq_v1"} and R["owner"][a] == R["owner"][b]
    assert R["kinds"] == ["output"] and R["cut"].ok and R["cut"].detail["redundant_gates"] == 1


def test_a_call_that_only_rearranges_bits_returns_its_inputs():
    # every gate is wiring: the returned values are the committed inputs themselves (not committed and unread)
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.types import Array
    from verity.ml.definitions.registry import prims as P
    f = bind(CompositeDefinition("Widen", 1, ("N",), lambda S: ((("a", Array(S.N, P.BF16)),), Array(S.N, P.F32)),
                                 lambda B, S, a: B.batch(P.Bf16ToF32, a, axes=(0,)), register=False), N=8)
    G = W.Graph(f)
    R = W.units(G, 16, 32)
    assert G.n == 0 and len(G.returned_inputs) == 8 and R["cut"].ok, R["cut"].codes
    assert W.unit_rule(f)["violations"] == []


def test_the_graph_is_the_core_reference_and_a_programs_partition_object_verifies_in_core():
    # word.Graph is verity.ml.circuits.cut.CallGraph's graph in flat arrays; a Program's cut, as verity/partition/v1, checks in core
    from verity.ml.circuits import cut, partition_object as PO
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.refs import tuple_of
    from verity.ml.circuits.types import Tuple
    fns = [_fn("RMSNormTriton_v1", N=64, EPS=1e-05), _fn("TokenSelect_v1", V=8), _fn("Attention_v3", T=3, NH=2, KVH=1, D=16, BN=128)]
    for f in fns:
        c, g = cut.CallGraph(f), W.Graph(f)
        assert (c.n, c.free, c.outputs, c.recomputed, c.input_reads) == (g.n, g.free, g.outputs, g.recomputed, g.input_reads)
        assert list(c.src) == g.src.tolist() and list(c.width) == g.width.tolist()
        R = W.units(g, 16, 32)
        own, _kinds, head = W.partition(g, 16)
        assert cut.derived_committed(c, own.tolist()) == np.flatnonzero(head >= 0).tolist()   # the heads are the derived committed set
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
    obj = PO.build(P)
    assert PO.validate(obj).ok and PO.verify(obj, P).ok
    part = PO.evaluate(P)
    assert part.population == sum(W.unit_rule(f)["units_per_call"] for f in fns) and len(part.calls) == 3
    assert PO.owners_digest(part) == PO.owners_digest(PO.evaluate(P))


def test_a_wide_gate_skips_the_calls_parameters_unresolved_and_its_graph_is_unchanged(monkeypatch):
    """`EmbeddingShard_v1`'s gathers are wide gates over a strided (transposed) view of the Call's own table: operands a wide gate never
    lists.  Skipping them unresolved gives the graph that resolving every operand gives (the OLMoE shard's cut: 29 s -> 0.06 s)."""
    from verity.ml.circuits import cut
    from verity.ml.circuits.layout import resolve
    fn = _fn("EmbeddingShard_v1", VS=cut.WIDE_FAN_IN + 3, H=4, START=0)
    enumerated = []

    def every_operand(gate, call):
        s, body = gate.scope, gate.scope.spec.body
        k = body.node_at(gate.index - s.offset)
        node = body.nodes[k]
        m = (gate.index - s.offset - body.offsets[k]) // node.fn.gates if node.form == "batch" else 0
        for a, ax in zip(node.args, node.axes if node.form == "batch" else (None,) * len(node.args)):
            refs = cut.member_refs(a, ax, m)
            for j in range(len(refs)):
                enumerated.append(1)
                yield resolve(s, refs[j])

    fast = cut.CallGraph(fn)
    monkeypatch.setattr(cut, "wide_operands", every_operand)
    full = cut.CallGraph(fn)
    assert len(enumerated) > 4 * cut.WIDE_FAN_IN
    for a in ("n", "keep", "width", "prim", "scope", "external", "src", "dst", "input_ids", "input_reads", "outputs", "returned_inputs",
              "recomputed", "total", "free", "free_by_prim"):
        assert getattr(fast, a) == getattr(full, a), a


# ------------------------------------------------------------- the graph in flat arrays and its verdict, held to core's reference

def _moe(routed: bool, E: int, K: int, N: int, swap: bool):
    """GLM-5.3's MoE expert FP8 GEMM (w13; w2 `routed`) on Hopper's tiles (G = KB = 128), scaled down in E, K and N."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.prims import HopperE4m3MmaV2Dot32, HopperE4m3QgmmaDot32
    from verity.ml.definitions.registry import fp8_moe_experts as XM
    return bind(XM.MoeExpertFp8GemmW if routed else XM.MoeExpertFp8Gemm, E=E, K=K, N=N, G=128, KB=128, SWAP=swap,
                DOT=HopperE4m3QgmmaDot32 if swap else HopperE4m3MmaV2Dot32)


def _twice():
    """The same F32Add computed twice, each copy read by its own output: a value computed in two units."""
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.refs import tuple_of
    from verity.ml.circuits.types import Array, Tuple
    from verity.ml.definitions.registry import prims as P

    def body(B, S, x):
        a = B.call(P.F32Add, x[0], x[1])
        b = B.call(P.F32Add, x[0], x[1])
        return tuple_of(B.call(P.F32Mul, a, x[2]), B.call(P.F32Mul, b, x[3]))
    return bind(CompositeDefinition("Twice", 1, (), lambda S: ((("x", Array(4, P.F32)),), Tuple(P.F32, P.F32)), body, register=False))


def _sampler():
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import sampling as SM
    return bind(SM.GumbelTopPTokenSelect, V=16)


def _dead_input():
    """A parameter read only by wiring nothing reads (committed, unread), beside one returned through wiring."""
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.refs import tuple_of
    from verity.ml.circuits.types import Array, Tuple
    from verity.ml.definitions.registry import prims as P

    def body(B, S, x, y):
        B.call(P.Bf16ToF32, y[0])
        return tuple_of(B.call(P.F32Add, B.call(P.Bf16ToF32, x[0]), B.call(P.Bf16ToF32, x[1])), B.call(P.Bf16ToF32, y[1]))
    return bind(CompositeDefinition("DeadInput", 1, (), lambda S: ((("x", Array(2, P.BF16)), ("y", Array(2, P.BF16))), Tuple(P.F32, P.F32)),
                                    body, register=False))


#: Definitions over every branch of the graph and its check: wiring and constants, a wide gate, a value computed twice in one unit
#: (the sampler's greedy flag) and in two, a parameter returned through wiring and one nobody reads, and the MoE expert FP8 GEMMs
#: over two K tiles and a partial weight block
CORPUS = {"norm": lambda: _fn("RMSNormTriton_v1", N=64, EPS=1e-05), "select": lambda: _fn("TokenSelect_v1", V=8),
          "attention": lambda: _fn("Attention_v3", T=3, NH=2, KVH=1, D=16, BN=128), "gemm": lambda: _fn("Gemm_v1", K=64, N=8),
          "wide": lambda: _fn("EmbeddingShard_v1", VS=4099, H=4, START=0), "sampler": lambda: _sampler(), "twice": _twice,
          "dead-input": _dead_input,
          "moe-w13": lambda: _moe(False, E=4, K=256, N=130, swap=True), "moe-w2": lambda: _moe(True, E=4, K=256, N=48, swap=False)}


class _Reference:
    """main's `word.Graph` before the flat arrays: core's `CallGraph` itself, with numpy views of the arrays the cut computes on."""

    def __new__(cls, fn):
        from verity.ml.circuits import cut
        g = cut.CallGraph(fn)
        for a in ("keep", "width", "prim", "scope", "src", "dst"):
            setattr(g, a, np.asarray(getattr(g, a), np.int64))
        g.external = np.frombuffer(g.external, np.uint8).astype(bool) if len(g.external) else np.zeros(0, bool)
        return g


def _reference_verdict(G, owner, head):
    """main's check of a cut: `validate_unit_cut` over a Python pair per read, the Call's parameters standing for the input unit."""
    from verity.ml.circuits.partition import INPUT_UNIT, validate_unit_cut
    k = len(G.input_ids)
    reads = list(zip(G.src.tolist(), G.dst.tolist())) + [(G.n + i, q) for i, q in G.input_reads]
    return validate_unit_cut(G.total + k, G.free, np.asarray(owner).tolist() + [INPUT_UNIT] * k, reads,
                             list(G.outputs) + [G.n + i for i in G.returned_inputs],
                             np.flatnonzero(np.asarray(head) >= 0).tolist() + list(range(G.n, G.n + k)), range(G.n, G.n + k), G.recomputed)


@pytest.mark.parametrize("name", CORPUS)
def test_the_graph_is_core_references_field_by_field(name):
    from verity.ml.circuits import cut
    f = CORPUS[name]()
    ref, g = cut.CallGraph(f), W.Graph(f)
    for a in ("n", "total", "free", "outputs", "returned_inputs", "recomputed", "input_reads", "free_by_prim", "prims", "scopes",
              "prim_names", "scope_names"):
        assert getattr(g, a) == getattr(ref, a), a
    for a in ("keep", "width", "prim", "scope", "src", "dst", "input_ids"):
        assert getattr(g, a).tolist() == list(getattr(ref, a)), a
    assert g.external.tolist() == [bool(x) for x in ref.external]
    assert g.input_read.tolist() == [i for i, _q in ref.input_reads] and g.input_reader.tolist() == [q for _i, q in ref.input_reads]


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.slow) if n == "moe-w13" else n for n in CORPUS])
def test_the_cut_verdict_is_validate_unit_cuts_on_the_cut_and_on_broken_ones(name):
    """`word.cut_verdict` gives `validate_unit_cut`'s verdict, its codes, counts, listed gates and detail order, on the cut and on cuts
    broken every way it names: gates in no unit, a committed value dropped (an uncommitted read or output), a value committed that no
    other unit reads, units merged and split (a recompute across units), and random owners."""
    import json
    G = W.Graph(CORPUS[name]())
    owner, kinds, head = W.partition(G, 16)
    rng = np.random.default_rng(0)
    cuts = [(owner, head)]
    heads = np.flatnonzero(head >= 0)
    for _ in range(2):
        o, h = owner.copy(), head.copy()
        if G.n:
            o[rng.integers(0, G.n, max(1, G.n // 40))] = -1
            h[rng.integers(0, G.n, max(1, G.n // 40))] = len(kinds)
        if len(heads):
            h[rng.choice(heads, max(1, len(heads) // 3), replace=False)] = -1
        cuts += [(o, head), (owner, h), (o, h), (rng.integers(0, max(1, len(kinds)), G.n), head), (np.zeros(G.n, np.int64), head),
                 (np.arange(G.n, dtype=np.int64), np.arange(G.n, dtype=np.int64))]
    codes = set()
    for o, h in cuts:
        got, want = W.cut_verdict(G, o, h), _reference_verdict(G, o, h)
        assert json.dumps(got.to_json()) == json.dumps(want.to_json())
        codes |= set(got.codes)
    if G.n > 8:
        assert {"gate-not-certified-once", "gate-count", "read-uncommitted", "committed-unread"} <= codes, codes


@pytest.mark.parametrize("name", ["moe-w13", "moe-w2", "twice", "dead-input", "sampler", "wide"])
def test_a_rule_is_the_reference_cuts_byte_for_byte(name, monkeypatch):
    """The unit rule, as the rule cache writes it, is the one main cut with core's graph and `validate_unit_cut` (`_Reference`,
    `_reference_verdict`): GLM-5.3's MoE expert FP8 GEMMs scaled down, and a failing cut's violation detail, gate ids and all."""
    import hashlib
    import json
    from verity.ml.circuits.codec import canonical_json
    f = CORPUS[name]()
    new = W.unit_rule(f, memo={})
    monkeypatch.setattr(W, "Graph", _Reference)
    monkeypatch.setattr(W, "cut_verdict", _reference_verdict)
    old = W.unit_rule(f, memo={})
    assert json.dumps(new) == json.dumps(old)
    assert hashlib.sha256(canonical_json(new)).hexdigest() == hashlib.sha256(canonical_json(old)).hexdigest()
    assert (name in ("twice", "dead-input")) == bool(new["violations"])


def test_the_graph_holds_a_few_bytes_per_operand_read():
    """A MoE expert FP8 GEMM's gathers read every expert's weight word (E + 1 operands a gate): the graph's peak is a few bytes per
    read (19 here), where core's reference, a Python object per read, peaks at 435 (GLM-5.3's w13 at E=256: 468 GB)."""
    import tracemalloc
    f = _moe(False, E=64, K=256, N=4, swap=True)
    W.Graph(f)
    tracemalloc.start()
    try:
        g = W.Graph(f)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    reads = len(g.src) + len(g.input_read)
    assert reads > 60 * f.gates and peak < 32 * reads, peak / reads


#: forking happens in a fresh interpreter: a pytest-xdist worker has threads, and a fork beside them may deadlock
PREFETCH_PROBE = """
import json, multiprocessing
from verity.ml.definitions import word as W
from verity_vllm.query.program_view import spec_base, spec_statics
specs = ["Attention_v3{T=5,NH=2,KVH=1,D=64,BN=128}", "batch:RMSNormTriton_v1{N=64,EPS=1e-05}", "EmbeddingShard_v1{VS=4099,H=4,START=0}",
         "NoSuch_v1{N=1}"]
n = W.prefetch(specs, jobs=2)
memo = W._RULES.get(tuple(sorted(W.gate_limits(None).items())), {})
same = all(rule == W.unit_rule(W.specialization(spec_base(fid), spec_statics(fid)), X, W_, e, memo={}) for (fid, X, W_, e), rule in memo.items())
print(json.dumps({"fork": "fork" in multiprocessing.get_all_start_methods(), "n": n, "families": sorted({k[0].split("{")[0] for k in memo}),
                  "same": same}))
"""


def test_prefetch_memos_the_rules_the_check_computes_in_place():
    """`word.prefetch` cuts in forked processes what `check_calls` would cut one after another -- the bodies cut as a whole, then each
    separable body over its memoed members -- and memos exactly those rules; a spec that does not resolve is left to the check."""
    import json
    import subprocess
    import sys
    r = json.loads(subprocess.run([sys.executable, "-c", PREFETCH_PROBE], capture_output=True, text=True, check=True).stdout)
    if not r["fork"]:
        assert r["n"] == 0 and r["families"] == []
        return
    assert r["families"] == ["AttentionHead_v3", "Attention_v3", "EmbeddingShard_v1", "RMSNormTriton_v1"] and r["n"] == 4 and r["same"], r


#: the probe's specs prefetched under a budget no cut fits, each cut counted in its worker while it runs, then without one
PREFETCH_BUDGET_PROBE = """
import json, multiprocessing
from verity.ml.definitions import word as W
specs = ["Attention_v3{T=5,NH=2,KVH=1,D=64,BN=128}", "batch:RMSNormTriton_v1{N=64,EPS=1e-05}", "EmbeddingShard_v1{VS=4099,H=4,START=0}"]
live, most = multiprocessing.Value("i", 0), multiprocessing.Value("i", 0)
cut = W._rule_of_pending


def counted(i):
    with live.get_lock():
        live.value += 1
        most.value = max(most.value, live.value)
    try:
        return cut(i)
    finally:
        with live.get_lock():
            live.value -= 1


W._rule_of_pending = counted
key = tuple(sorted(W.gate_limits(None).items()))
n = [W.prefetch(specs, jobs=4, budget=1)]
budgeted, at_once = W._RULES.pop(key, {}), most.value
n.append(W.prefetch(specs, jobs=4))
free = W._RULES.get(key, {})
print(json.dumps({"fork": "fork" in multiprocessing.get_all_start_methods(), "n": n, "at_once": at_once,
                  "same": list(budgeted.items()) == list(free.items())}))
"""


def test_prefetch_under_a_memory_budget_runs_only_the_cuts_that_fit_and_memos_the_same_rules():
    """`word.prefetch(budget=...)`: the cuts in flight are those whose estimated peaks fit the budget, and one over it runs alone, so
    under a budget no cut fits the three bodies cut as a whole run one at a time; the rules memoed, and their order, are those of a
    prefetch without a budget."""
    import json
    import subprocess
    import sys
    r = json.loads(subprocess.run([sys.executable, "-c", PREFETCH_BUDGET_PROBE], capture_output=True, text=True, check=True).stdout)
    if not r["fork"]:
        assert r["n"] == [0, 0]
        return
    assert r["n"] == [4, 4] and r["at_once"] == 1 and r["same"], r


@pytest.mark.slow
def test_build_global_builds_the_same_manifest_in_one_process_and_in_several(tmp_path):
    """`--jobs`: the word check's rules prefetched and the components built in forked processes (`manifest.build_components`) give the
    manifest one process gives, byte for byte; the first component to fail, in declaration order, is the one reported.  A workload
    digest read from stdin once the components are built (`--workload-digest -`) gives the manifest the same digest named on the
    command line gives, and none read fails it."""
    import gzip
    import json
    import shutil
    import subprocess
    import sys
    from collections import Counter
    from pathlib import Path
    from tests.check.test_sampled_replay import LP, T
    d = tmp_path / "row" / f"build_request_LP{LP}_T{T}"
    d.mkdir(parents=True)
    _program_dir(d)
    with gzip.open(d / "instances.json.gz", "rt") as f:
        hist = Counter(r["spec"] for r in json.load(f)["rows"])
    (d / "result.json").write_text(json.dumps(dict(json.loads((d / "result.json").read_text()), spec_histogram=dict(hist))))
    fx = {"fixture": "t", "requests": [{"request_id": q, "prompt_len": LP, "max_tokens": T + 1, "arrive_step": 0} for q in "abc"]}
    (tmp_path / "wl.json").write_text(json.dumps(fx))

    def build_global(jobs: int, out: str, *extra: str, stdin: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "manifest", "build-global", "--fixture", str(tmp_path / "wl.json"),
                               "--programs-root", str(tmp_path / "row"), "--jobs", str(jobs), *extra, "--out", str(tmp_path / out)],
                              cwd=Path(__file__).resolve().parents[2], input=stdin, capture_output=True, text=True)

    for jobs in (1, 3):
        assert build_global(jobs, f"m{jobs}.json").returncode == 0
    assert (tmp_path / "m1.json").read_bytes() == (tmp_path / "m3.json").read_bytes()
    assert build_global(3, "named.json", "--workload-digest", "w" * 64).returncode == 0
    assert build_global(3, "stdin.json", "--workload-digest", "-", stdin="w" * 64 + "\n").returncode == 0
    assert (tmp_path / "named.json").read_bytes() == (tmp_path / "stdin.json").read_bytes() != (tmp_path / "m1.json").read_bytes()
    p = build_global(3, "none.json", "--workload-digest", "-", stdin="")
    assert p.returncode != 0 and "no workload digest on stdin" in p.stderr and not (tmp_path / "none.json").exists(), p.stderr[-2000:]
    shutil.copytree(d, tmp_path / "row" / f"build_request_LP{LP}_T{T - 1}")    # a wrong component for r1
    fx["requests"][1]["max_tokens"] = T
    (tmp_path / "wl.json").write_text(json.dumps(fx))
    p = build_global(3, "bad.json")
    assert p.returncode != 0 and f"request r1: Program has {T + 1} steps" in p.stderr, p.stderr[-2000:]


# ---------------------------------------------------------------------------------------------------------------- the rule cache

def _no_cut(fn):
    raise AssertionError(f"{fn.id} cut again: its cached rule was not read")


def test_a_cached_rule_is_the_rule_computed_and_a_later_process_reads_it_back(tmp_path, monkeypatch, attn):
    computed = W.unit_rule(attn, memo={})
    monkeypatch.setattr(W, "_RULE_CACHE", [str(tmp_path)])
    assert W.unit_rule(attn, memo={}) == computed and list(tmp_path.rglob("*.json"))
    monkeypatch.setattr(W, "Graph", _no_cut)
    assert W.unit_rule(attn, memo={}) == computed


def test_an_entry_is_keyed_by_the_definitions_content_the_query_the_limits_and_the_code(tmp_path, monkeypatch, norm):
    monkeypatch.setattr(W, "_RULE_CACHE", [str(tmp_path)])
    limits = W.gate_limits(None)

    def at(fn, X=16, lim=limits):
        return W._rule_entry(fn, X, 32, 0, lim)[0]
    base = at(norm)
    assert len({base, at(_fn("RMSNormTriton_v1", N=1024, EPS=1e-05)), at(norm, X=8), at(norm, lim=W.gate_limits(1000))}) == 4
    assert W.definition_digest(norm) != W.definition_digest(_fn("RMSNormTriton_v1", N=2048, EPS=1e-06))
    monkeypatch.setattr(W, "_CODE", ["another cut"])
    assert at(norm) != base


def test_an_entry_under_another_key_or_unreadable_is_not_read(tmp_path, monkeypatch, norm):
    import json
    monkeypatch.setattr(W, "_RULE_CACHE", [str(tmp_path)])
    rule = W.unit_rule(norm, memo={})
    path, key = W._rule_entry(norm, 16, 32, 0, W.gate_limits(None))
    path.write_text(json.dumps({"key": {**key, "digest": "0" * 64}, "rule": {**rule, "units_per_call": 0}}))
    assert W.unit_rule(norm, memo={}) == rule
    path.write_text("{")
    assert W.unit_rule(norm, memo={}) == rule


def test_a_rule_json_would_not_give_back_equal_is_not_written(tmp_path, monkeypatch, norm):
    monkeypatch.setattr(W, "_RULE_CACHE", [str(tmp_path)])
    W._cache_rule(norm, 16, 32, 0, W.gate_limits(None), {"violations": [("tuple", 1)]})
    W._cache_rule(norm, 16, 32, 0, W.gate_limits(None), {"gates": np.int64(3)})
    assert not list(tmp_path.rglob("*.json"))


def test_the_manifest_is_the_same_without_the_cache_as_with_it_cold_and_read_back(tmp_path, monkeypatch):
    from verity_vllm.pipeline import manifest as M
    d = _program_dir(tmp_path)

    def build():
        monkeypatch.setattr(W, "_RULES", {})
        return M.build(d, request_id="r0", word="16/32")
    plain = build()
    monkeypatch.setattr(W, "_RULE_CACHE", [str(tmp_path / "rules")])
    cold = build()
    monkeypatch.setattr(W, "Graph", _no_cut)
    assert plain == cold == build() and plain["query"]["word_check"]["units"] > 0


def test_the_manifest_cli_takes_the_cache_directory_from_its_flag_or_the_environment():
    from verity_vllm.pipeline import cli
    argv = ["build", "--program", "p", "--out", "o"]
    assert cli.parse("manifest", [*argv, "--unit-rule-cache", "/r"], {}).unit_rule_cache == "/r"
    assert cli.parse("manifest", argv, {W.RULE_CACHE_ENV: "/e"}).unit_rule_cache == "/e"
    assert cli.parse("manifest", ["build-global", "--fixture", "f", "--programs-root", "p", "--out", "o"], {}).unit_rule_cache is None
