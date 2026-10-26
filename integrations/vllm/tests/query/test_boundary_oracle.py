"""Differential tests of `verity.ml.circuits.boundary` on the registry's Programs against the explicit oracle `GateSelection.out` / `GateSelection.in_`.

Every exact answer must equal the oracle on the expanded circuit; every bound must be >= the exact `w_out`; every
`Unsupported` must be on a case named here (in the natural fixtures there are none; the named ones are produced by
tightening the module caps).  Fixture families: GEMM (members, coordinate interior, DotBf16 chain), RMSNorm (nodes,
members, merged normalize path), SiluMul batch members, AttentionHead_v3 nodes (dead parameter: `Const32[0]` passed as
`m_prev/l_prev/o_prev` into `AttnBlock_v3{FIRST=True}`), AttnBlock_v3 FIRST=True/False members and the PV chain, nested
DotBf16 interiors, MoE router scan iterations and step interiors, MoE expert GEMM, Gumbel (64-bit gates), fan-out, dead
return, re-returned nested returns, pass-through returns, scan carries (including a pass-through carry), Input gates,
axis-1 and strided batch arguments.  Small statics keep the whole file well under a minute."""
from __future__ import annotations

import pytest

from verity.ml.circuits import boundary as B
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.liveness import Liveness
from verity.ml.circuits.layout import FunctionCall, Scope, GateSelection
from verity.ml.circuits.parts import Limits, Part, Unsupported, n_members
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, array_of, strided_view, tuple_of
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.prims import BF16, F32
from verity.ml.circuits.types import Array, Tuple

# --------------------------------------------------------------------------------------------------------------
# oracle
# --------------------------------------------------------------------------------------------------------------


class Oracle:
    """`GateSelection.in_` / `.out` computed from the expanded circuit's operands, cached once per program.  The formula is
    the oracle's (`layout.py`): out = gates of S consumed by a gate outside S or protected; in = operands of S outside S.
    `explicit()` re-checks a part against `GateSelection` itself."""

    def __init__(self, program: Program) -> None:
        C = program.circuit
        self.C = C
        self.ops: list[tuple[int, ...]] = []
        self.width: list[int] = []
        cons: list[list[int]] = [[] for _ in range(C.size)]
        for g in range(C.size):
            gate = C.gate(g)
            ops = tuple(gate.operands)
            self.ops.append(ops)
            self.width.append(gate.prim.ret.w)
            for o in ops:
                cons[o].append(g)
        self.cons = cons
        self.roots = frozenset(C.root_outputs())

    def out(self, gates: frozenset[int]) -> frozenset[int]:
        return frozenset(g for g in gates if g in self.roots or any(c not in gates for c in self.cons[g]))

    def in_(self, gates: frozenset[int]) -> frozenset[int]:
        return frozenset(o for g in gates for o in self.ops[g] if o not in gates)

    def explicit(self, gates: frozenset[int]) -> None:
        sub = GateSelection(self.C, sorted(gates))
        assert frozenset(sub.out) == self.out(gates)
        assert frozenset(sub.in_) == self.in_(gates)


def part_gates(part: Part) -> frozenset[int]:
    return frozenset(g for lo, hi in part.intervals() for g in range(lo, hi))


def body_parts(sc: Scope) -> list[Part]:
    """Every node / member of body `sc` as its own part (the finest hierarchy parts, Input members included)."""
    return [Part(sc, frozenset({(j, m)})) for j, node in enumerate(sc.spec.body.nodes) for m in range(n_members(node))]


def merged(sc: Scope, items) -> Part:
    return Part(sc, frozenset(items))


class Tally:
    """exact / bound / unsupported counts per family (reported in the status file)."""

    families: dict[str, dict[str, int]] = {}

    @classmethod
    def add(cls, family: str, **kw: int) -> None:
        d = cls.families.setdefault(family, {"parts": 0, "exact": 0, "bound_tight": 0, "bound_loose": 0, "unsupported": 0})
        for k, v in kw.items():
            d[k] += v


def run_family(family: str, oracle: Oracle, parts: list[Part], *, allow_unsupported: tuple[str, ...] = (),
               explicit_samples: int = 3) -> dict[str, int]:
    """Check out_gates / in_gates / w_out / w_out_bound / explain of every part against the oracle."""
    n_exact = n_unsup = n_tight = n_loose = 0
    for i, part in enumerate(parts):
        gates = part_gates(part)
        o_out, o_in = oracle.out(gates), oracle.in_(gates)
        if i < explicit_samples:
            oracle.explicit(gates)
        o_w = sum(oracle.width[g] for g in o_out)
        out = B.out_gates(part)
        bound = B.w_out_bound(part)
        assert isinstance(bound, int) and bound >= o_w, (family, sorted(part.items)[:4], bound, o_w)
        if isinstance(out, Unsupported):
            assert out.reason in allow_unsupported, (family, sorted(part.items)[:4], out)
            assert isinstance(B.w_out(part), Unsupported)
            n_unsup += 1
            continue
        assert out == o_out, (family, sorted(part.items)[:4], sorted(o_out)[:8], sorted(out)[:8])
        assert out <= gates
        inn = B.in_gates(part)
        if isinstance(inn, Unsupported):  # in(S) may be undetermined while out(S) is exact (a callee's inexact reads)
            assert inn.reason in allow_unsupported, (family, sorted(part.items)[:4], inn)
        else:
            assert inn == o_in, (family, sorted(part.items)[:4], sorted(o_in)[:8], sorted(inn)[:8])
            assert not (inn & gates)
        w = B.w_out(part)
        assert w == o_w, (family, sorted(part.items)[:4], w, o_w)
        n_exact += 1
        if bound == w:
            n_tight += 1
        else:
            n_loose += 1
        if i % 97 == 0:
            ex = B.explain(part)
            assert [e.gate for e in ex] == sorted(out)
            assert all(e.width == oracle.width[e.gate] for e in ex)
    counts = {"parts": len(parts), "exact": n_exact, "bound_tight": n_tight, "bound_loose": n_loose, "unsupported": n_unsup}
    Tally.add(family, **counts)
    return counts


def check_widths(program: Program, oracle: Oracle) -> None:
    for g in range(program.circuit.size):
        assert B.gate_width(program, g) == oracle.width[g]


def check_used_against_liveness(spec) -> None:
    """Second oracle for the used-parameter summary on specializations without pass-through returns: the liveness
    analysis's per-parameter reads (every consumer counts) are the same leaves."""
    n = spec.body.ret_type.leaves
    reads = Liveness().analyse(spec, ((0, n),) if n else ()).reads
    assert B.used_param_intervals(spec) == reads
    leaves = B.used_param_leaves(spec)
    assert leaves == frozenset((pi, l) for pi, iv in enumerate(reads) for lo, hi in iv for l in range(lo, hi))


# --------------------------------------------------------------------------------------------------------------
# registry families
# --------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gemm():
    P_ = Program(bind(b1.Gemm, K=32, N=6))
    return P_, Oracle(P_)


def test_gemm_members_and_interiors(gemm):
    P_, oracle = gemm
    sc = P_.circuit.root_scope
    c = run_family("Gemm_v1{K=32,N=6} root nodes/members (Input members included)", oracle, body_parts(sc))
    assert c["unsupported"] == 0 and c["parts"] == 224 + 6
    jb = [j for j, n in enumerate(sc.spec.body.nodes) if n.form == "batch" and not isinstance(n.fn, PrimitiveDefinition)][0]
    # two adjacent coordinate members merged, and the whole node
    run_family("  Gemm merged members", oracle, [merged(sc, {(jb, 0), (jb, 1)}), merged(sc, {(jb, m) for m in range(6)})])
    sub = sc.child(jb, 2)
    run_family("  GemmCoordinate#2 body nodes", oracle, body_parts(sub))
    sub2 = sub.child(0, 0)
    run_family("  DotBf16 body nodes (accumulator chain)", oracle, body_parts(sub2) + [merged(sub2, {(0, 0), (1, 0)})])
    check_widths(P_, oracle)
    for spec in (bind(b1.DotBf16, K=32), bind(b1.GemmCoordinate, K=32), bind(b1.Gemm, K=32, N=6)):
        check_used_against_liveness(spec)
    # the coordinate member is a FunctionCall: its declared interface is a superset of the graph boundary
    fc = FunctionCall(P_.circuit, sc, jb, 2)
    assert B.out_gates(Part(sc, frozenset({(jb, 2)}))) == frozenset(fc.out) <= frozenset(fc.distinct_returns)


def test_rmsnorm_nodes_members_and_merged_path():
    P_ = Program(bind(b1.RMSNormTriton, N=32, EPS=1e-5))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    c = run_family("RMSNormTriton_v1{N=32} root nodes/members", oracle, body_parts(sc))
    assert c["unsupported"] == 0
    nodes = sc.spec.body.nodes
    tail = [j for j, n in enumerate(nodes) if n.form == "batch"][-4:]
    c2 = run_family("  RMSNorm normalize-path merge per output", oracle, [merged(sc, {(j, m) for j in tail}) for m in range(nodes[tail[0]].n)])
    assert all(B.w_out(merged(sc, {(j, m) for j in tail})) == 16 for m in range(nodes[tail[0]].n))
    assert c2["bound_tight"] == 32  # the merged path's only escaping value is the bf16 output
    check_used_against_liveness(bind(b1.RMSNormTriton, N=32, EPS=1e-5))


def test_silumul_members():
    P_ = Program(bind(b1.SiluMul, I=8))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    c = run_family("SiluMul_v1{I=8} root nodes/members", oracle, body_parts(sc))
    assert c["unsupported"] == 0 and c["parts"] == 16 + 8
    jb = [j for j, n in enumerate(sc.spec.body.nodes) if n.fn is P.SiluMulBf16][0]
    for m in range(8):  # member m reads gu[m] and gu[8 + m]: two affine edges with different bases
        assert B.in_gates(Part(sc, frozenset({(jb, m)}))) == frozenset({m, 8 + m})
        assert B.w_out(Part(sc, frozenset({(jb, m)}))) == 16
    check_widths(P_, oracle)


@pytest.fixture(scope="module")
def attn_head():
    P_ = Program(bind(b1.AttentionHeadV3, T=40, D=16, BN=16))
    return P_, Oracle(P_)


def test_attention_head_nodes_dead_parameter(attn_head):
    P_, oracle = attn_head
    sc = P_.circuit.root_scope
    c = run_family("AttentionHead_v3{T=40,D=16,BN=16} root nodes/members", oracle, body_parts(sc))
    assert c["unsupported"] == 0
    nodes = sc.spec.body.nodes
    # the dead parameter: Const32[0] feeds m_prev / l_prev / o_prev of the FIRST block, whose body never reads them
    first = [j for j, n in enumerate(nodes) if n.form == "call" and "FIRST=True" in n.fn.id][0]
    z = first - 1
    assert nodes[z].fn.id.startswith("Const32") and nodes[z].form == "primitive"
    assert B.out_gates(Part(sc, frozenset({(z, 0)}))) == frozenset()
    assert B.w_out(Part(sc, frozenset({(z, 0)}))) == 0 and B.w_out_bound(Part(sc, frozenset({(z, 0)}))) >= 0
    used = B.used_param_leaves(nodes[first].fn)
    assert {pi for pi, _ in used} == {0, 1, 2}
    later = [j for j, n in enumerate(nodes) if n.form == "call" and "FIRST=False" in n.fn.id][0]
    assert {pi for pi, _ in B.used_param_leaves(nodes[later].fn)} == {0, 1, 2, 3, 4, 5}
    # the block call as a part: in_ excludes the dead constant, out is the state read by the next block
    blk = Part(sc, frozenset({(first, 0)}))
    z_gate = sc.spec.body.offsets[z]
    fc = FunctionCall(P_.circuit, sc, first, 0)
    assert z_gate not in B.in_gates(blk) and z_gate in fc.args
    assert B.in_gates(blk) < frozenset(fc.in_)  # the interface adapter is a strict superset here (dead parameter)
    assert B.out_gates(blk) == frozenset(fc.out)
    assert B.w_out(blk) == 32 + 4 * 32 + 16 * 32  # m, l[4], o[16]
    check_widths(P_, oracle)
    for j in (first, later):
        check_used_against_liveness(nodes[j].fn)


def test_attn_block_members_pv_chain_and_nested_dot(attn_head):
    P_, oracle = attn_head
    sc = P_.circuit.root_scope
    blocks = [j for j, n in enumerate(sc.spec.body.nodes) if n.form == "call" and "AttnBlock" in n.fn.id]
    assert len(blocks) == 3
    for jb in blocks[:2]:
        sb = sc.child(jb, 0)
        c = run_family(f"  {sb.spec.id} body nodes/members", oracle, body_parts(sb))
        assert c["unsupported"] == 0
        pv = [j for j, n in enumerate(sb.spec.body.nodes) if n.form == "batch" and "Ampere" in n.fn.id]
        chain = [merged(sb, {(j, d) for j in pv}) for d in range(sb.spec.body.nodes[pv[0]].n)]
        c2 = run_family(f"    PV chain per d ({len(pv)} k-steps)", oracle, chain)
        assert all(B.w_out(p) == 32 for p in chain) and c2["bound_tight"] == len(chain)
        # DotBf16 calls nested two levels below the root: their accumulator chain, exact through the parent walk
        dots = [j for j, n in enumerate(sb.spec.body.nodes) if n.form == "call" and "DotBf16" in n.fn.id]
        for jd in dots[:3]:
            sd = sb.child(jd, 0)
            run_family("    DotBf16 inside AttnBlock (2 levels nested)", oracle, body_parts(sd) + [merged(sd, {(0, 0), (1, 0)})])
        # a merged interior: every s[c] dot with the max chain (nothing escapes but the max and the s values used by ex2)
        run_family("    AttnBlock dots+max merged", oracle, [merged(sb, {(j, 0) for j in dots} | {(j, 0) for j, n in enumerate(sb.spec.body.nodes) if n.form == "primitive" and n.fn is P.F32Max})])


def test_moe_router_scans_and_step_interiors():
    from verity_vllm.program.quarantine.ov_moe import moe as Q
    P_ = Program(bind(Q.MoeRouterTopK, E=8, TOPK=2, VPT=8))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    c = run_family("MoeRouterTopK_v1{E=8,TOPK=2,VPT=8} root nodes/iterations", oracle, body_parts(sc))
    assert c["unsupported"] == 0
    nodes = sc.spec.body.nodes
    scans = [j for j, n in enumerate(nodes) if n.form == "scan"]
    assert len(scans) == 2
    for js in scans:
        n = nodes[js].n
        # an iteration's boundary is its carry (best, best_i, i) = 96 bits, or less for the last one
        for m in range(n - 1):
            assert B.w_out(Part(sc, frozenset({(js, m)}))) == 96
        # merged iterations: only the last carry escapes
        run_family("  scan iterations merged [0, k)", oracle, [merged(sc, {(js, m) for m in range(k)}) for k in range(1, n + 1)])
        run_family("  scan iterations merged [k, n)", oracle, [merged(sc, {(js, m) for m in range(k, n)}) for k in range(n)])
        # inside the step body of the first, a middle and the last iteration (parent walk through the carry edges)
        for m in (0, n // 2, n - 1):
            si = sc.child(js, m)
            run_family("  RouterArgmaxStep body nodes (iteration interiors)", oracle, body_parts(si) + [merged(si, {(0, 0), (1, 0)})])
    check_widths(P_, oracle)
    check_used_against_liveness(nodes[scans[0]].fn)


def test_moe_expert_gemm():
    from verity_vllm.program.quarantine.ov_moe import moe as Q
    P_ = Program(bind(Q.MoeExpertGemm, E=4, K=16, N=3))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    c = run_family("MoeExpertGemm_v1{E=4,K=16,N=3} root nodes/members", oracle, body_parts(sc))
    assert c["unsupported"] == 0
    jb = [j for j, n in enumerate(sc.spec.body.nodes) if n.form == "batch" and not isinstance(n.fn, PrimitiveDefinition)][0]
    for m in range(3):
        assert B.w_out(Part(sc, frozenset({(jb, m)}))) == 16
        sub = sc.child(jb, m)  # MoeExpertCoordinate: the row gather call + the GemmCoordinate call
        run_family("  MoeExpertCoordinate body nodes", oracle, body_parts(sub))
        row = sub.child(0, 0)  # MoeExpertRow: gather batch members (wide fan-in over the E candidates)
        run_family("  MoeExpertRow gather members", oracle, body_parts(row) + [merged(row, {(0, k) for k in range(16)})])
    check_widths(P_, oracle)


def test_gumbel_64bit_gates_reported():
    from verity.ml.definitions.registry import sampling as G
    P_ = Program(bind(G.GumbelArgmaxBf16, V=8, SEED=12345, POS=3))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    c = run_family("GumbelArgmaxBf16_v1{V=8} root nodes/iterations", oracle, body_parts(sc))
    assert c["unsupported"] == 0
    nodes = sc.spec.body.nodes
    wide = [j for j, n in enumerate(nodes) if getattr(n.fn, "ret", None) is not None and n.fn.ret.w > 32]
    assert len(wide) == 2  # Const64[SEED], Const64[POS]
    for j in wide:
        part = Part(sc, frozenset({(j, 0)}))
        g = next(iter(B.out_gates(part)))
        assert B.gate_width(P_, g) == 64 and B.w_out(part) == 64 and B.w_out_bound(part) == 64
    js = [j for j, n in enumerate(nodes) if n.form == "scan"][0]
    for m in (0, nodes[js].n - 1):
        run_family("  GumbelArgmaxStep body nodes", oracle, body_parts(sc.child(js, m)))
    check_widths(P_, oracle)


def test_attention_v3_batch_of_heads():
    P_ = Program(bind(b1.AttentionV3, T=8, NH=2, KVH=1, D=16, BN=16))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    c = run_family("Attention_v3{T=8,NH=2,KVH=1,D=16,BN=16} root nodes/members", oracle, body_parts(sc))
    assert c["unsupported"] == 0
    jb = [j for j, n in enumerate(sc.spec.body.nodes) if n.form == "batch" and not isinstance(n.fn, PrimitiveDefinition)][0]
    for m in range(2):
        assert B.w_out(Part(sc, frozenset({(jb, m)}))) == 16 * 16
        run_family("  AttentionHead_v3 member body nodes", oracle, body_parts(sc.child(jb, m)))


# --------------------------------------------------------------------------------------------------------------
# hand-built counterexample programs (unregistered definitions)
# --------------------------------------------------------------------------------------------------------------


def comp(name: str, params, ret, body) -> CompositeDefinition:
    return CompositeDefinition(name, 1, (), lambda S: (tuple(params), ret), body, register=False)


def test_fan_out_counts_producer_once():
    def body(Bd, S, x):
        a = Bd.call(P.F32Add, x[0], x[1])
        b = Bd.call(P.F32Mul, a, a)  # a used twice by one gate
        c = Bd.call(P.F32Add, a, x[0])  # and once more by another
        return tuple_of(b, c)
    P_ = Program(comp("QbFanOut", [("x", Array(2, F32))], Tuple(F32, F32), body))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    run_family("fan-out", oracle, body_parts(sc) + [merged(sc, {(1, 0), (2, 0)}), merged(sc, {(0, 0), (1, 0)})])
    a = Part(sc, frozenset({(1, 0)}))
    assert B.out_gates(a) == frozenset({2}) and B.w_out(a) == 32 and B.w_out_bound(a) == 32
    assert B.in_gates(Part(sc, frozenset({(2, 0)}))) == frozenset({2})  # b's two operands are one gate
    assert B.in_gates(Part(sc, frozenset({(0, 0)}))) == frozenset()  # Input members read nothing


def test_dead_return_and_re_returned_nested_returns():
    inner = comp("QbInner", [("x", F32)], Tuple(F32, F32),
                 lambda Bd, S, x: tuple_of(Bd.call(P.F32Add, x, x), Bd.call(P.F32Mul, x, x)))
    middle = comp("QbMiddle", [("x", F32)], Tuple(F32, F32),
                  lambda Bd, S, x: (lambda r: tuple_of(r[1], r[0]))(Bd.call(inner, x)))

    def root_dead(Bd, S, x):  # reads only Inner's u; v is a dead return
        r = Bd.call(inner, x[0])
        return Bd.call(P.F32Add, r[0], r[0])

    def root_rereturn(Bd, S, x):  # Middle re-returns Inner's (v, u); the root keeps only Middle[0] = v
        s = Bd.call(middle, x[0])
        return s[0]

    for name, body, live in (("dead return", root_dead, "u"), ("re-returned nested return", root_rereturn, "v")):
        P_ = Program(comp("QbRoot" + name.replace(" ", ""), [("x", Array(1, F32))], F32, body))
        oracle = Oracle(P_)
        sc = P_.circuit.root_scope
        run_family(name, oracle, body_parts(sc))
        call = sc.child(1, 0)
        if name != "dead return":
            run_family(name + " (Middle body)", oracle, body_parts(call))
            call = call.child(0, 0)
        run_family(name + " (Inner body)", oracle, body_parts(call) + [merged(call, {(0, 0), (1, 0)})])
        u, v = Part(call, frozenset({(0, 0)})), Part(call, frozenset({(1, 0)}))
        if live == "u":
            assert B.w_out(u) == 32 and B.w_out(v) == 0 and B.out_gates(v) == frozenset()
        else:
            assert B.w_out(v) == 32 and B.w_out(u) == 0 and B.out_gates(u) == frozenset()
        # the bound charges every returned leaf: 32 for the dead one too, never below the exact value
        assert B.w_out_bound(u) == 32 and B.w_out_bound(v) == 32
        # declared interface vs boundary: distinct_returns has both, out has one
        fc = FunctionCall(P_.circuit, sc, 1, 0)
        assert len(fc.distinct_returns) == 2 and len(fc.out) == 1
        assert B.out_gates(Part(sc, frozenset({(1, 0)}))) == frozenset(fc.out)


def test_pass_through_return_is_transparent():
    # Wrap2 passes x through unread; only y is read by a gate
    wrap = comp("QbWrap2", [("x", F32), ("y", F32)], Tuple(F32, F32), lambda Bd, S, x, y: tuple_of(Bd.call(P.F32Mul, y, y), x))

    def body(Bd, S, x):
        a = Bd.call(P.F32Add, x[0], x[1])
        w = Bd.call(wrap, a, x[0])
        g = Bd.call(P.F32Add, w[1], w[1])  # reads a through the pass-through
        return tuple_of(w[0], g)
    P_ = Program(comp("QbPassThrough", [("x", Array(2, F32))], Tuple(F32, F32), body))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    run_family("pass-through return", oracle, body_parts(sc) + [merged(sc, {(1, 0), (2, 0)}), merged(sc, {(2, 0), (3, 0)})])
    assert B._ret_canon(sc.spec.body.nodes[2].fn).passthrough
    a, w = Part(sc, frozenset({(1, 0)})), Part(sc, frozenset({(2, 0)}))
    assert B.out_gates(a) == frozenset({2})  # a is out: g reads it through Wrap2's second return
    assert B.in_gates(w) == frozenset({0})  # Wrap2's only gate reads x[0]; a is not a wire into the call
    assert B.used_param_leaves(sc.spec.body.nodes[2].fn) == frozenset({(1, 0)})
    assert B.w_out(w) == 32  # f; the passed-through a is produced outside the call
    run_family("pass-through return (Wrap2 body)", oracle, body_parts(sc.child(2, 0)))


def test_scan_with_pass_through_carry_and_designated_input():
    # carry = (c0, c1): c0 is passed through every iteration unread; the root returns it (an Input gate is designated)
    step = comp("QbStepPT", [("carry", Tuple(F32, F32)), ("x", F32)], Tuple(Tuple(F32, F32), F32),
                lambda Bd, S, carry, x: (lambda s: tuple_of(tuple_of(carry[0], s), s))(Bd.call(P.F32Add, carry[1], x)))

    def body(Bd, S, init, xs):
        res = Bd.scan(step, init, xs=(xs,))
        return tuple_of(res[0][0], res[0][1], res[1][0])
    P_ = Program(comp("QbScanPT", [("init", Tuple(F32, F32)), ("xs", Array(3, F32))], Tuple(F32, F32, F32), body))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    nodes = sc.spec.body.nodes
    js = [j for j, n in enumerate(nodes) if n.form == "scan"][0]
    run_family("scan with pass-through carry", oracle, body_parts(sc) + [merged(sc, {(js, 0), (js, 1)}), merged(sc, {(js, 1), (js, 2)})])
    assert B._ret_canon(nodes[js].fn).passthrough
    assert B.used_param_leaves(nodes[js].fn) == frozenset({(0, 1), (1, 0)})
    x0 = Part(sc, frozenset({(0, 0)}))  # init[0]: never read by a gate, but designated through the pass-through carries
    assert B.out_gates(x0) == frozenset({0}) and B.in_gates(x0) == frozenset()
    x1 = Part(sc, frozenset({(0, 1)}))
    assert B.out_gates(x1) == frozenset({1})  # read by iteration 0's add
    for m in range(3):
        it = Part(sc, frozenset({(js, m)}))
        assert B.w_out(it) == 32  # each iteration's add feeds the next iteration (or the root's return + y[0] for m = 0)
        run_family("  step body inside the pass-through scan", oracle, body_parts(sc.child(js, m)))


def test_pass_through_batch_members_scan_xs_and_straddling_runs():
    """Pass-through returns of batch members (axis 0 + shared argument, axis 1), of a scan step's xs element, and a
    step whose returned run of one node straddles the carry / y boundary; all re-returned wholesale by a middle
    composite, so the root's reads resolve two levels down to the true producers (Input gates / inner gates)."""
    wrap = comp("QbWrapB", [("x", F32), ("s", Array(2, F32))], Tuple(F32, F32, F32),
                lambda Bd, S, x, s: tuple_of(Bd.call(P.F32Mul, x, s[0]), x, s[1]))  # x and s[1] pass through
    wrap2 = comp("QbWrapB2", [("x", Array(2, F32))], Tuple(F32, Array(2, F32)),
                 lambda Bd, S, x: tuple_of(Bd.call(P.F32Add, x[0], x[1]), x))  # the whole column passes through
    inner = comp("QbInnerS", [("a", F32), ("b", F32)], Tuple(F32, F32),
                 lambda Bd, S, a, b: tuple_of(Bd.call(P.F32Add, a, b), Bd.call(P.F32Mul, a, b)))
    step = comp("QbStepS", [("carry", Tuple(F32, F32)), ("x", F32)], Tuple(Tuple(F32, F32), Tuple(F32, F32)),
                lambda Bd, S, carry, x: (lambda r: tuple_of(tuple_of(carry[0], r[0]), tuple_of(r[1], x)))(Bd.call(inner, carry[1], x)))
    # step pieces: carry[0] (p) | inner[0:2] (a run straddling c = 2) | x (p, the xs element passes through as y[1])

    def mid(Bd, S, X, shared, M, init, xs):
        w = Bd.batch(wrap, X, shared, axes=(0, None))
        v = Bd.batch(wrap2, M, axes=(1,))
        res = Bd.scan(step, init, xs=(xs,))
        return tuple_of(w, v, res[0], res[1])

    mid_t = Tuple(Array(2, Tuple(F32, F32, F32)), Array(3, Tuple(F32, Array(2, F32))), Tuple(F32, F32), Array(3, Tuple(F32, F32)))
    mid_def = comp("QbMidPT", [("X", Array(2, F32)), ("shared", Array(2, F32)), ("M", Array(2, Array(3, F32))),
                               ("init", Tuple(F32, F32)), ("xs", Array(3, F32))], mid_t, mid)

    def body(Bd, S, X, shared, M, init, xs):
        r = Bd.call(mid_def, X, shared, M, init, xs)
        g1 = Bd.call(P.F32Add, r[0][1][1], r[0][0][2])  # X[1] and shared[1] through wrap members
        g2 = Bd.call(P.F32Add, r[1][2][1][0], r[1][0][0])  # M[0][2] through wrap2 column 2; wrap2 member 0's add
        g3 = Bd.call(P.F32Add, r[3][1][1], r[3][2][0])  # xs[1] through iteration 1's y; iteration 2's inner mul
        return tuple_of(g1, g2, g3, r[2][0], r[0][1][0], r[1][1][1][1])  # init[0] (carry pass-through) and M[1][1] designated
    P_ = Program(comp("QbRootPT", [("X", Array(2, F32)), ("shared", Array(2, F32)), ("M", Array(2, Array(3, F32))),
                                   ("init", Tuple(F32, F32)), ("xs", Array(3, F32))], Tuple(F32, F32, F32, F32, F32, F32), body))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    assert B._ret_canon(sc.spec.body.nodes[5].fn).passthrough
    run_family("pass-through: batch members / scan xs / straddling run (root)", oracle,
               body_parts(sc) + [merged(sc, {(5, 0), (6, 0)}), merged(sc, {(0, 1), (1, 1), (5, 0)})])
    off = sc.spec.body.offsets
    X1, sh1, M02, M11, init0, xs1 = off[0] + 1, off[1] + 1, off[2] + 2, off[2] + 4, off[3], off[4] + 1
    # exact escapes through two pass-through levels (Input gate -> mid -> wrap member / step iteration -> root gate)
    assert B.out_gates(Part(sc, frozenset({(0, 1)}))) == frozenset({X1})  # X[1]: read by g1 through mid / wrap member 1
    assert B.out_gates(Part(sc, frozenset({(1, 1)}))) == frozenset({sh1})  # shared[1]: g1 via wrap member 0's third return
    assert B.out_gates(Part(sc, frozenset({(2, 2)}))) == frozenset({M02}) and B.out_gates(Part(sc, frozenset({(2, 4)}))) == frozenset({M11})
    assert B.out_gates(Part(sc, frozenset({(2, 1)}))) == frozenset({off[2] + 1})  # M[0][1]: read by wrap2 column 1's add
    assert B.out_gates(Part(sc, frozenset({(3, 0)}))) == frozenset({init0})  # never read by a gate; designated via every carry
    assert B.out_gates(Part(sc, frozenset({(4, 1)}))) == frozenset({xs1})  # xs[1]: g3 through iteration 1's y[1]
    assert B.in_gates(Part(sc, frozenset({(6, 0)}))) == frozenset({X1, sh1})  # g1 reads two Input gates through two pass-throughs
    assert M02 in B.in_gates(Part(sc, frozenset({(7, 0)}))) and xs1 in B.in_gates(Part(sc, frozenset({(8, 0)})))
    m = sc.child(5, 0)
    nodes = m.spec.body.nodes
    run_family("  QbMidPT body (batch members, scan iterations)", oracle,
               body_parts(m) + [merged(m, {(0, 0), (1, 0)}), merged(m, {(2, 0), (2, 1)}), merged(m, {(2, 2), (0, 1)})])
    js = [j for j, n in enumerate(nodes) if n.form == "scan"][0]
    jw, jv = [j for j, n in enumerate(nodes) if n.form == "batch"]
    assert B._ret_canon(nodes[js].fn).passthrough and B._ret_canon(nodes[jw].fn).passthrough and B._ret_canon(nodes[jv].fn).passthrough
    for mm in range(2):
        run_family("    wrap member body", oracle, body_parts(m.child(jw, mm)))
    for mm in range(3):
        run_family("    wrap2 column body", oracle, body_parts(m.child(jv, mm)))
        it = m.child(js, mm)
        run_family("    step iteration body", oracle, body_parts(it))
        run_family("      inner body inside the step", oracle, body_parts(it.child(0, 0)) + [merged(it.child(0, 0), {(0, 0), (1, 0)})])
    # the mul of wrap member 1 is dead (r[0][1][0] is returned by the root: designated, so it is out after all); member 0's is dead
    assert B.w_out(Part(m.child(jw, 0), frozenset({(0, 0)}))) == 0 and B.w_out(Part(m.child(jw, 1), frozenset({(0, 0)}))) == 32
    assert B.used_param_leaves(nodes[js].fn) == frozenset({(0, 1), (1, 0)}) and B.used_param_leaves(nodes[jw].fn) == frozenset({(0, 0), (1, 0)})


def test_dead_input_gates_and_partial_reads():
    def body(Bd, S, x):  # reads x[0], x[1] only; x[2], x[3] are dead Input gates
        return Bd.call(P.F32Add, x[0], x[1])
    P_ = Program(comp("QbDeadInput", [("x", Array(4, F32))], F32, body))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    run_family("dead Input gates", oracle, body_parts(sc) + [merged(sc, {(0, m) for m in range(4)})])
    for m in (2, 3):
        p = Part(sc, frozenset({(0, m)}))
        assert B.out_gates(p) == frozenset() and B.in_gates(p) == frozenset() and B.w_out(p) == 0
        assert B.w_out_bound(p) == 0  # unread and not returned: the bound has nothing to charge
    assert B.w_out(merged(sc, {(0, m) for m in range(4)})) == 64


def test_axis1_and_strided_batch_arguments():
    row = comp("QbRow", [("x", Array(2, F32))], F32, lambda Bd, S, x: Bd.call(P.F32Add, x[0], x[1]))

    def body(Bd, S, M):  # M: [2][3]; axis-1 batch (3 members read M[0][m], M[1][m]) and a strided axis-0 view (leaves 0,1 and 4,5)
        y = Bd.batch(row, M, axes=(1,))
        view = Coll(Array(2, Array(2, F32)), strided_view(M, 0, 2, 4, 1, 4))
        z = Bd.batch(row, view, axes=(0,))
        return tuple_of(y, z)
    P_ = Program(comp("QbAxes", [("M", Array(2, Array(3, F32)))], Tuple(Array(3, F32), Array(2, F32)), body))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    run_family("axis-1 / strided batch arguments", oracle, body_parts(sc) + [merged(sc, {(1, 0), (2, 0)}), merged(sc, {(0, 0), (0, 2), (1, 0)})])
    assert B.in_gates(Part(sc, frozenset({(1, 1)}))) == frozenset({1, 4})  # M[0][1], M[1][1]
    assert B.in_gates(Part(sc, frozenset({(2, 1)}))) == frozenset({4, 5})  # the strided view's second row
    ix = B.consumer_index(sc.spec)
    assert any(isinstance(d, B._Aff) and d.stride == 4 for _, d in ix.consumers(("n", 0)))
    assert any(isinstance(d, B._One) for _, d in ix.consumers(("n", 0)))
    for m in range(2):
        run_family("  Row body (strided member)", oracle, body_parts(sc.child(2, m)))


def test_two_dimensional_strided_member_reads():
    """Strided axis-0 arguments whose (member, i) decomposition is unique are read analytically (`_Aff2`), both over
    node outputs (root level: Input nodes) and over parameters (nested: the used-leaf intervals): columns of a table
    (i-major, unit member stride: the Embedding gather), i-major with a wider member stride, m-major with a gap
    between the elements of a row, and a zero-stride broadcast (every member reads the same row)."""
    row3 = comp("QbRow3", [("x", Array(3, F32))], F32, lambda Bd, S, x: Bd.call(P.F32Add, x[0], x[2]))  # x[1] unread
    col4 = comp("QbCol4", [("x", Array(4, F32))], F32, lambda Bd, S, x: Bd.call(P.F32Add, x[1], x[3]))  # x[0], x[2] unread
    row6 = comp("QbRow6", [("x", Array(6, F32))], F32, lambda Bd, S, x: Bd.call(P.F32Add, x[0], x[5]))

    def views(Bd, S, M):  # M: [4][6] = 24 leaves
        gap = Bd.batch(row3, Coll(Array(4, Array(3, F32)), strided_view(M, 0, 3, 6, 2, 12)), axes=(0,))  # m-major: row m, elements 0,2,4
        cols = Bd.batch(col4, Coll(Array(3, Array(4, F32)), strided_view(M, 0, 4, 2, 6, 12)), axes=(0,))  # i-major, m_stride 2: columns 0,2,4
        bc = Bd.batch(row6, Coll(Array(2, Array(6, F32)), strided_view(M, 6, 6, 0, 1, 12)), axes=(0,))  # broadcast: both read row 1
        return tuple_of(gap, cols, bc)
    ret_t = Tuple(Array(4, F32), Array(3, F32), Array(2, F32))
    mid = comp("QbViewsMid", [("M", Array(4, Array(6, F32)))], ret_t, views)
    for label, body in (("root", views), ("nested", lambda Bd, S, M: Bd.call(mid, M))):
        P_ = Program(comp("QbViews" + label, [("M", Array(4, Array(6, F32)))], ret_t, body))
        oracle = Oracle(P_)
        sc = P_.circuit.root_scope
        scope = sc if label == "root" else sc.child(1, 0)
        ix = B.consumer_index(scope.spec)
        kinds = {type(d).__name__ for v in ix.edges.values() for _, d in v}
        if label == "root":
            assert "_Aff2" in kinds and "_All" in kinds and "_One" not in kinds, kinds
            majors = {d.major for v in ix.edges.values() for _, d in v if isinstance(d, B._Aff2)}
            assert majors == {"m", "i"}
        else:  # the views read a parameter: only the used-leaf intervals are recorded
            assert B.used_param_leaves(scope.spec) == frozenset((0, l) for l in {0, 4, 6, 10, 12, 16, 18, 22}  # gap: elements 0, 4 of each row
                                                                 | {6, 8, 10, 18, 20, 22}  # cols: rows 1 and 3 of columns 0, 2, 4
                                                                 | {6, 11})  # row6 on row 1
            check_used_against_liveness(scope.spec)
        o = 1 if label == "root" else 0  # node index of the gap batch (the root body has the M Input node first)
        parts = body_parts(sc) + body_parts(scope) + [merged(scope, {(o, 0), (o, 3)}), merged(scope, {(o + 1, 1), (o + 2, 0), (o + 2, 1)})]
        run_family(f"two-dimensional strided reads ({label})", oracle, parts)
        off = sc.spec.body.offsets[0]
        assert B.out_gates(Part(sc, frozenset({(0, 1)}))) == frozenset()  # M[0][1]: in no view's used leaves
        assert B.out_gates(Part(sc, frozenset({(0, 6)}))) == frozenset({off + 6})  # M[1][0]: gap row 1, column 0, broadcast
        assert B.out_gates(Part(sc, frozenset({(0, 12)}))) == frozenset({off + 12}) and B.out_gates(Part(sc, frozenset({(0, 13)}))) == frozenset()
        assert B.in_gates(Part(scope, frozenset({(o, 2)}))) == frozenset({off + 12, off + 16})  # gap row 2: elements 0 and 4 (leaf 14 unread)
        assert B.in_gates(Part(scope, frozenset({(o + 1, 1)}))) == frozenset({off + 8, off + 20})  # column 2: rows 1 and 3
        assert B.in_gates(Part(scope, frozenset({(o + 2, 1)}))) == frozenset({off + 6, off + 11})  # broadcast member 1: row 1
        with pytest.raises(ValueError):
            B.out_gates(Part(scope, frozenset({(o + 1, 3)})))  # cols has 3 members: an invalid item is a caller error


def test_embedding_gather_is_analytic():
    """Embedding_v1's gather reads the table's columns through a Strided view with inner_stride H: per-member listing
    would cost V*H intervals; the analytic form makes the index O(1) and the answers stay exact."""
    P_ = Program(bind(b1.Embedding, V=16, H=8))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    ix = B.consumer_index(sc.spec)
    assert any(isinstance(d, B._Aff2) and d.major == "i" and d.m_stride == 1 and d.i_stride == 8 for v in ix.edges.values() for _, d in v)
    run_family("Embedding_v1{V=16,H=8} root nodes/members", oracle, body_parts(sc) + [merged(sc, {(1, m) for m in range(0, 128, 8)}), merged(sc, {(2, 0), (2, 7)})])
    assert B.in_gates(Part(sc, frozenset({(2, 3)}))) == frozenset({sc.spec.body.offsets[0]}) | frozenset(sc.spec.body.offsets[1] + 3 + 8 * v for v in range(16))
    big = bind(b1.Embedding, V=4096, H=512)
    assert B.used_param_intervals(big) == (((0, 1),), ((0, 4096 * 512),))
    assert B.consumer_index(big).used_exact


# --------------------------------------------------------------------------------------------------------------
# named Unsupported cases (produced by tightening the module caps; the natural fixtures have none)
# --------------------------------------------------------------------------------------------------------------


def test_unsupported_unresolved_consumer_members(monkeypatch):
    """A consumer whose members read non-affine references beyond the per-member cap is `_Lazy`: exact answers stay
    exact while the consumer is wholly outside the part (its union is exact), become `Unsupported` when the part
    splits it, and the bound stays sound either way."""
    monkeypatch.setattr(B, "_MEMBER_CAP", 1)
    row = comp("QbRowL", [("x", Array(2, F32))], F32, lambda Bd, S, x: Bd.call(P.F32Add, x[0], x[1]))
    ab = comp("QbAB", [("m", Array(2, Array(2, F32)))], Tuple(F32, F32),
              lambda Bd, S, m: tuple_of(Bd.call(P.F32Add, m[0][0], m[0][1]), Bd.call(P.F32Mul, m[1][0], m[1][1])))

    def body(Bd, S, M):
        r = Bd.call(ab, M)
        a, b = r[0], r[1]
        pairs = array_of([array_of([a, b]), array_of([b, a])])  # explicit, non-affine axis-0 references
        y = Bd.batch(row, pairs, axes=(0,))
        return y
    P_ = Program(comp("QbLazy", [("M", Array(2, Array(2, F32)))], Array(2, F32), body))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    ix = B.consumer_index(sc.spec)
    assert any(isinstance(d, B._Lazy) and d.exact for _, d in ix.consumers(("n", 1)))
    run_family("named Unsupported: unresolved consumer members", oracle,
               body_parts(sc) + [merged(sc, {(1, 0), (2, 0)}), merged(sc, {(0, 1), (2, 1)})], allow_unsupported=("unresolved_consumer",))
    split = merged(sc, {(1, 0), (2, 0)})
    assert B.out_gates(split) == Unsupported("unresolved_consumer", B.out_gates(split).detail)
    assert B.w_out_bound(split) >= sum(oracle.width[g] for g in oracle.out(part_gates(split)))
    assert B.out_gates(Part(sc, frozenset({(1, 0)}))) == oracle.out(part_gates(Part(sc, frozenset({(1, 0)}))))
    # inside the call: its returns are consumed by the lazy batch, whose union is exact -> the escape is exact
    inner = sc.child(1, 0)
    assert run_family("  QbAB body (returns read by an exact lazy consumer)", oracle, body_parts(inner))["unsupported"] == 0
    for m in range(2):
        assert run_family("  QbRowL member body (lazy consumer's interior)", oracle, body_parts(sc.child(2, m)))["unsupported"] == 0


def test_unsupported_index_incomplete(monkeypatch):
    """Pass-through canonicalisation beyond its cap leaves the index incomplete: every exact query on the
    specialization is `Unsupported("index_incomplete")`; the bound falls back to the declared widths."""
    monkeypatch.setattr(B, "_PASSTHROUGH_CAP", 0)
    wrap = comp("QbWrapI", [("x", F32)], Tuple(F32, F32), lambda Bd, S, x: tuple_of(Bd.call(P.F32Mul, x, x), x))

    def body(Bd, S, x):
        w = Bd.batch(wrap, x, axes=(0,))
        return Bd.call(P.F32Add, w[0][1], w[1][0])
    P_ = Program(comp("QbIncomplete", [("x", Array(2, F32))], F32, body))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    assert B.consumer_index(sc.spec).incomplete
    c = run_family("named Unsupported: index incomplete (pass-through cap)", oracle, body_parts(sc), allow_unsupported=("index_incomplete",))
    assert c["unsupported"] == c["parts"]
    for p in body_parts(sc):
        assert B.out_gates(p).reason == "index_incomplete" and isinstance(B.in_gates(p), Unsupported)
    # the members' interiors: their own index is fine, but the returned leaves cannot be resolved in the incomplete parent
    for m in range(2):
        c = run_family("  wrap member body under an incomplete parent", oracle, body_parts(sc.child(1, m)), allow_unsupported=("index_incomplete",))
        assert c["unsupported"] == c["parts"]
        assert B.in_gates(Part(sc.child(1, m), frozenset({(0, 0)}))) == frozenset({m})  # in(S) needs no parent walk


XT, MT = Array(3, Array(2, F32)), Array(2, Array(3, F32))


def _nested_partial_program(deep: bool = False):
    """`mid` batches `sel` (reads leaf 0 of its Array(2) argument only) over a parameter along axis 0 and another along
    axis 1: half of each parameter is declared but never read.  The root also batches `sel` along axis 1 of an Input
    node directly.  `deep`: an `outer` layer batches `mid` over two activations and returns the result twice."""
    sel = comp("QbSelP", [("x", Array(2, F32))], F32, lambda Bd, S, x: Bd.call(P.F32Add, x[0], x[0]))
    mid = comp("QbMidP", [("X", XT), ("M", MT)], Tuple(Array(3, F32), Array(3, F32)),
               lambda Bd, S, X, M: tuple_of(Bd.batch(sel, X, axes=(0,)), Bd.batch(sel, M, axes=(1,))))
    if not deep:
        def root(Bd, S, X, M):
            r = Bd.call(mid, X, M)
            y = Bd.batch(sel, M, axes=(1,))  # columns of an Input node
            z = Bd.batch(sel, array_of([r[0], r[1]]), axes=(1,))  # columns of mid's two returns: reads r[0][m] only
            return tuple_of(r[0], r[1], y, z)
        return Program(comp("QbRootP", [("X", XT), ("M", MT)], Tuple(Array(3, F32), Array(3, F32), Array(3, F32), Array(3, F32)), root))
    mid_t = Array(2, Tuple(Array(3, F32), Array(3, F32)))
    outer = comp("QbOuterP", [("XX", Array(2, XT)), ("MM", Array(2, MT))], Tuple(mid_t, mid_t),
                 lambda Bd, S, XX, MM: (lambda r: tuple_of(r, r))(Bd.batch(mid, XX, MM, axes=(0, 0))))
    return Program(comp("QbRootPD", [("XX", Array(2, XT)), ("MM", Array(2, MT))], Tuple(mid_t, mid_t),
                        lambda Bd, S, XX, MM: Bd.call(outer, XX, MM)))


def test_partial_member_reads_of_nested_parameters():
    P_ = _nested_partial_program()
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    mid = sc.child(2, 0)
    run_family("nested partial parameter reads (root + mid body)", oracle, body_parts(sc) + body_parts(mid) + [merged(mid, {(0, 1), (1, 1)})])
    assert B.used_param_leaves(mid.spec) == frozenset({(0, 0), (0, 2), (0, 4), (1, 0), (1, 1), (1, 2)})
    check_used_against_liveness(mid.spec)
    for m in (1, 3, 5):  # X[.][1] are dead Input gates
        assert B.out_gates(Part(sc, frozenset({(0, m)}))) == frozenset() and B.w_out_bound(Part(sc, frozenset({(0, m)}))) == 0
    for m in (3, 4, 5):  # M[1][.] too
        assert B.out_gates(Part(sc, frozenset({(1, m)}))) == frozenset()
    assert B.in_gates(Part(mid, frozenset({(1, 2)}))) == frozenset({sc.spec.body.offsets[1] + 2})  # column 2 reads M[0][2] only
    P_ = _nested_partial_program(deep=True)
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    outer = sc.child(2, 0)
    parts = body_parts(sc) + body_parts(outer) + [merged(outer, {(0, 0), (0, 1)})]
    for m in range(2):
        parts += body_parts(outer.child(0, m))
    run_family("nested partial parameter reads (deep: outer batch of mid, returned twice)", oracle, parts)
    assert B.used_param_leaves(outer.spec) == frozenset({(0, l) for l in (0, 2, 4, 6, 8, 10)} | {(1, l) for l in (0, 1, 2, 6, 7, 8)})
    check_used_against_liveness(outer.spec)


def test_unsupported_inexact_nested_reads_propagate(monkeypatch):
    """When a cap leaves a nested specialization's parameter reads conservative, the enclosing bodies must not treat
    the over-approximation as exact: every producer wired into that parameter becomes an unresolved consumer edge
    (`Unsupported("unresolved_consumer")`), the other parameter stays exact, and the bounds stay sound."""
    monkeypatch.setattr(B, "_MEMBER_CAP", 1)
    P_ = _nested_partial_program()
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    mid = sc.child(2, 0)
    ix = B.consumer_index(mid.spec)
    assert ix.incomplete is None and ix.used_exact_params == (False, False) and not ix.used_exact
    with pytest.raises(B.BoundaryLimit, match="used_params"):
        B.used_param_leaves(mid.spec)
    assert any(isinstance(d, B._Lazy) and not d.exact for _, d in B.consumer_index(sc.spec).consumers(("n", 0)))
    c = run_family("named Unsupported: inexact nested parameter reads", oracle, body_parts(sc) + body_parts(mid),
                   allow_unsupported=("unresolved_consumer", "used_params"))
    assert c["unsupported"] > 0
    for m in range(6):  # the root cannot say whether X[.] / M[.] escape: Unsupported, never the (wrong) conservative answer
        assert B.out_gates(Part(sc, frozenset({(0, m)}))).reason == "unresolved_consumer"
        assert B.out_gates(Part(sc, frozenset({(1, m)}))).reason == "unresolved_consumer"
        assert B.w_out_bound(Part(sc, frozenset({(0, m)}))) == 32
    assert B.in_gates(Part(sc, frozenset({(2, 0)}))).reason == "used_params"
    # the call's outputs feed the root's column batch `z` (3 columns > cap, partial reads): unresolved there too
    assert B.out_gates(Part(sc, frozenset({(2, 0)}))).reason == "unresolved_consumer"
    for j in range(2):  # inside mid the batches read their (exact) callee normally; their returns' escape is undetermined
        for m in range(3):
            p = Part(mid, frozenset({(j, m)}))
            assert B.in_gates(p) == oracle.in_(part_gates(p)) and B.out_gates(p).reason == "unresolved_consumer"
    assert B.out_gates(Part(sc, frozenset({(4, 1)}))) == oracle.out(part_gates(Part(sc, frozenset({(4, 1)}))))  # z's members: exact
    # the root's own axis-1 batch over the M Input node (3 columns > cap, partial reads): a `_Lazy(exact=False)` edge
    assert any(isinstance(d, B._Lazy) and not d.exact and jp == 3 for jp, d in B.consumer_index(sc.spec).consumers(("n", 1)))
    # deep: the inexactness crosses two levels (outer's batch of mid over its parameters, then the root's call)
    P_ = _nested_partial_program(deep=True)
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    outer = sc.child(2, 0)
    assert B.consumer_index(outer.spec).used_exact_params == (False, False) and B.consumer_index(outer.spec).incomplete is None
    parts = body_parts(sc) + body_parts(outer) + body_parts(outer.child(0, 1))
    c = run_family("named Unsupported: inexact nested parameter reads (deep)", oracle, parts, allow_unsupported=("unresolved_consumer", "used_params"))
    assert 0 < c["unsupported"] < c["parts"]
    for m in range(12):
        assert B.out_gates(Part(sc, frozenset({(0, m)}))).reason == "unresolved_consumer"


def test_unsupported_boundary_too_large(monkeypatch):
    """`Limits.max_intervals` caps the boundary leaves an exact query enumerates: beyond it `out_gates` / `in_gates` /
    `w_out` / `explain` are `Unsupported("boundary_too_large")`; `w_out_bound` falls back to the declared output widths
    (batch members and scan carries + ys), which stays >= the oracle."""
    step = comp("QbStepL", [("carry", F32), ("x", F32)], Tuple(F32, F32),
                lambda Bd, S, carry, x: (lambda s: tuple_of(s, s))(Bd.call(P.F32Add, carry, x)))

    def body(Bd, S, xs):
        sq = Bd.batch(P.F32Mul, xs, xs, axes=(0, 0))
        res = Bd.scan(step, sq[0], xs=(sq,))
        return tuple_of(res[0], res[1])
    P_ = Program(comp("QbLarge", [("xs", Array(8, F32))], Tuple(F32, Array(8, F32)), body))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    run_family("boundary budget (uncapped)", oracle, body_parts(sc) + [merged(sc, {(1, m) for m in range(8)}), merged(sc, {(2, m) for m in range(8)})])
    monkeypatch.setattr(Limits, "max_intervals", 3)
    sq_all, scan_all = merged(sc, {(1, m) for m in range(8)}), merged(sc, {(2, m) for m in range(8)})
    for p in (sq_all, scan_all, merged(sc, {(0, m) for m in range(8)})):
        assert B.out_gates(p).reason == "boundary_too_large" and B.w_out(p).reason == "boundary_too_large"
        assert B.explain(p).reason == "boundary_too_large"
        assert B.w_out_bound(p) >= sum(oracle.width[g] for g in oracle.out(part_gates(p)))
    assert B.in_gates(sq_all).reason == "boundary_too_large"
    assert B.w_out_bound(sq_all) == 8 * 32 and B.w_out_bound(scan_all) == 8 * (32 + 32)  # declared: members; per iteration carry + y
    small = Part(sc, frozenset({(1, 3)}))  # within budget: still exact
    assert B.out_gates(small) == oracle.out(part_gates(small)) and B.in_gates(small) == oracle.in_(part_gates(small))


def test_unsupported_too_many_items(monkeypatch, gemm):
    monkeypatch.setattr(Limits, "max_items_per_part", 1)
    P_, oracle = gemm
    sc = P_.circuit.root_scope
    p = merged(sc, {(0, 0), (0, 1)})
    assert B.out_gates(p).reason == "too_many_items" and B.in_gates(p).reason == "too_many_items"
    assert B.w_out(p).reason == "too_many_items"
    assert B.w_out_bound(p) >= sum(oracle.width[g] for g in oracle.out(part_gates(p)))
    assert B.out_gates(Part(sc, frozenset({(0, 0)}))) == oracle.out(part_gates(Part(sc, frozenset({(0, 0)}))))


def test_partial_reads_shared_argument_and_scan_xs():
    """A batch member reading part of a shared (axis None) argument, and a scan step reading part of each xs element:
    the unread leaves are dead Input gates, not wires."""
    sel = comp("QbSel", [("shared", Array(3, F32)), ("x", F32)], F32,
               lambda Bd, S, shared, x: Bd.call(P.F32Add, Bd.call(P.F32Add, shared[0], shared[2]), x))
    step = comp("QbStepX", [("carry", F32), ("x", Array(2, F32))], Tuple(F32, F32),
                lambda Bd, S, carry, x: (lambda s: tuple_of(s, s))(Bd.call(P.F32Add, carry, x[0])))

    def body(Bd, S, shared, X, xs):
        y = Bd.batch(sel, shared, X, axes=(None, 0))
        res = Bd.scan(step, y[0], xs=(xs,))
        return tuple_of(y, res[0], res[1])
    P_ = Program(comp("QbPartial", [("shared", Array(3, F32)), ("X", Array(2, F32)), ("xs", Array(3, Array(2, F32)))],
                      Tuple(Array(2, F32), F32, Array(3, F32)), body))
    oracle = Oracle(P_)
    sc = P_.circuit.root_scope
    nodes = sc.spec.body.nodes
    run_family("partial reads: shared argument / scan xs", oracle, body_parts(sc) + [merged(sc, {(3, 0), (4, 0)}), merged(sc, {(4, 1), (4, 2)})])
    assert B.used_param_leaves(nodes[3].fn) == frozenset({(0, 0), (0, 2), (1, 0)})
    assert B.used_param_leaves(nodes[4].fn) == frozenset({(0, 0), (1, 0)})
    assert B.out_gates(Part(sc, frozenset({(0, 1)}))) == frozenset()  # shared[1] is never read
    assert B.in_gates(Part(sc, frozenset({(3, 1)}))) == frozenset({0, 2, 4})  # shared[0], shared[2], X[1]
    xs0 = sc.spec.body.offsets[2]
    assert B.in_gates(Part(sc, frozenset({(4, 1)}))) == frozenset({xs0 + 2}) | B.out_gates(Part(sc, frozenset({(4, 0)})))
    for m in range(3):  # xs[m][1] is a dead Input gate
        assert B.out_gates(Part(sc, frozenset({(2, 2 * m + 1)}))) == frozenset()
    for spec in (bind(b1.SiluMul, I=8), bind(b1.AttentionHeadV3, T=40, D=16, BN=16), bind(b1.AttentionV3, T=8, NH=2, KVH=1, D=16, BN=16)):
        check_used_against_liveness(spec)
    from verity_vllm.program.quarantine.ov_moe import moe as Q
    from verity.ml.definitions.registry import sampling as G
    for spec in (bind(Q.MoeRouterTopK, E=8, TOPK=2, VPT=8), bind(Q.MoeExpertGemm, E=4, K=16, N=3), bind(G.GumbelArgmaxBf16, V=8, SEED=12345, POS=3)):
        check_used_against_liveness(spec)


def test_unsupported_nested_scope_without_parent(gemm):
    """A parameterised body handed over as a Scope with no parent is a nested activation without its context: the
    returned leaves cannot be resolved exactly (Unsupported), the bound still charges them."""
    P_, oracle = gemm
    sc = P_.circuit.root_scope
    jb = [j for j, n in enumerate(sc.spec.body.nodes) if n.form == "batch" and not isinstance(n.fn, PrimitiveDefinition)][0]
    real = sc.child(jb, 1)
    orphan = Scope(real.spec, real.offset, None, -1, 0)
    for j in range(len(real.spec.body.nodes)):
        p_real, p_orphan = Part(real, frozenset({(j, 0)})), Part(orphan, frozenset({(j, 0)}))
        assert B.out_gates(p_orphan).reason == "no_parent_scope" and B.in_gates(p_orphan).reason == "no_parent_scope"
        assert B.w_out_bound(p_orphan) >= B.w_out(p_real) == sum(oracle.width[g] for g in oracle.out(part_gates(p_real)))


def test_query_cost_independent_of_member_count():
    """Per-part queries must not scale with the body size: a member of Gemm{K=256,N=2048} (2048 coordinates, 524k weight
    Input gates) is answered exactly from its own boundary, at a size where listing the body per query is impractical (the
    test checks the answers, not a time); the whole-node part enumerates only its boundary."""
    P_ = Program(bind(b1.Gemm, K=256, N=2048))
    sc = P_.circuit.root_scope
    jb = [j for j, n in enumerate(sc.spec.body.nodes) if n.form == "batch" and not isinstance(n.fn, PrimitiveDefinition)][0]
    for m in (0, 1000, 2047):
        p = Part(sc, frozenset({(jb, m)}))
        assert B.w_out(p) == 16 and B.w_out_bound(p) == 16
        assert len(B.in_gates(p)) == 256 + 256
    inner = sc.child(jb, 7).child(0, 0)  # DotBf16{K=256}: 1 + 16 nodes
    assert B.w_out(Part(inner, frozenset({(16, 0)}))) == 32 and B.w_out(Part(inner, frozenset({(3, 0)}))) == 32
    whole = Part(sc, frozenset({(jb, m) for m in range(2048)}))
    assert B.w_out(whole) == 2048 * 16 and len(B.out_gates(whole)) == 2048
    weights = Part(sc, frozenset({(1, m) for m in range(0, 2048 * 256, 97)}))  # a spread of weight Input gates
    assert B.w_out(weights) == 16 * len(weights.items) and B.in_gates(weights) == frozenset()
    # the summaries are intrinsic: the same specialization's index is shared by every activation
    assert B.consumer_index(sc.spec.body.nodes[jb].fn) is B.consumer_index(sc.child(jb, 5).spec)


@pytest.mark.slow
def test_serve_v4_tiny_program():
    """A whole (tiny) serving program: Serve_v4 over a 2-layer, H=32 config (~37k gates, 40 root nodes: Embedding,
    LayerPre/LayerPost per position and layer, Attention_v3{T=1..5} with Strided KV-cache views, Final, Input gates).
    Root nodes/members, one activation of every composite at root and one level below, and merged root pairs, all
    against the oracle."""
    tiny = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
            "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 16, "BIAS": True,
            "tie_word_embeddings": True, "profile": {}}
    P_ = Program(bind(b1.ServeV4, C=tiny, LP=3, STEPS=2, BN=16))
    oracle = Oracle(P_)
    check_widths(P_, oracle)
    sc = P_.circuit.root_scope
    root = run_family("Serve_v4 tiny root nodes/members", oracle, body_parts(sc))
    assert root["unsupported"] == 0 and root["parts"] > 10_000
    composites = [j for j, n in enumerate(sc.spec.body.nodes) if not isinstance(n.fn, PrimitiveDefinition)]
    seen: set[str] = set()
    nested_parts: list[Part] = []
    for j in composites:
        sub = sc.child(j, 0)
        nested_parts += body_parts(sub)
        for jj, nn in enumerate(sub.spec.body.nodes):
            if isinstance(nn.fn, PrimitiveDefinition) or nn.fn.id in seen:
                continue
            seen.add(nn.fn.id)
            nested_parts += body_parts(sub.child(jj, 0))
    nested = run_family("Serve_v4 tiny nested activations", oracle, nested_parts)
    assert nested["unsupported"] == 0 and nested["parts"] > 200
    pairs = [merged(sc, [(a, 0), (b, 0)]) for a, b in zip(composites, composites[1:])]
    pairs.append(merged(sc, [(j, 0) for j in composites]))
    assert run_family("Serve_v4 tiny merged root composites", oracle, pairs)["unsupported"] == 0
    check_used_against_liveness(sc.spec)
    for j in composites[:6]:
        check_used_against_liveness(sc.spec.body.nodes[j].fn)


def test_zz_report_counts(capsys):
    """Per-family exact / bound / unsupported counts (printed; copied into the owner's status file)."""
    total = {"parts": 0, "exact": 0, "unsupported": 0}
    with capsys.disabled():
        print("\n| family | parts | exact | bound == exact | bound > exact | unsupported |\n|---|---|---|---|---|---|")
        for fam, c in Tally.families.items():
            print(f"| {fam} | {c['parts']} | {c['exact']} | {c['bound_tight']} | {c['bound_loose']} | {c['unsupported']} |")
            for k in total:
                total[k] += c[k]
        print(f"| **total** | {total['parts']} | {total['exact']} | | | {total['unsupported']} |")
    assert total["exact"] + total["unsupported"] == total["parts"]
