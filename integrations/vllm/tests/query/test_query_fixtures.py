"""Fixture sanity: every fixture Program builds, its circuit expands, and the explicit-oracle helpers give the answers
the fixture was designed around (dead Input, constant gate, fan-out counted once, dead return, dead parameter, scan carry,
64-bit gates).  These tests depend on nothing from the three core query modules -- they pin the ORACLE the modules are
measured against."""
from __future__ import annotations

import pytest

from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.parts import Part, n_members
from tests.program import fixtures_query as FX


@pytest.mark.parametrize("name", FX.FIXTURE_NAMES)
def test_fixture_builds_and_expands(name):
    P = FX.fixture(name)
    C = P.circuit
    assert C.size > 0
    assert C.size <= 200_000, f"{name}: {C.size} gates is too large for an explicit-oracle fixture"
    # every gate is addressable and every root output is a gate of the circuit
    C.gate(0)
    C.gate(C.size - 1)
    for g in C.root_outputs():
        assert 0 <= g < C.size
    # Input gates are ordinary gates at the head of the root body
    assert P.input_gates >= 0
    for j, node in enumerate(P.root.body.nodes):
        if FX.is_input_node(node):
            assert P.root.body.offsets[j] < P.input_gates


@pytest.mark.parametrize("name", FX.FIXTURE_NAMES)
def test_finest_parts_tile_the_circuit(name):
    """The finest hierarchy parts of the root body are a partition of [0, size) -- the induction base every partition
    validator relies on."""
    P = FX.fixture(name)
    parts = FX.body_parts(FX.root_scope(P))
    ivs = sorted(iv for p in parts for iv in p.intervals())
    pos = 0
    for lo, hi in ivs:
        assert lo == pos, f"{name}: gap or overlap at {pos} (next interval starts at {lo})"
        pos = hi
    assert pos == P.circuit.size
    assert sum(p.size() for p in parts) == P.circuit.size


def test_tiny_io_oracle():
    P = FX.tiny_io_program()
    sc = FX.root_scope(P)
    nodes = sc.spec.body.nodes
    assert FX.is_input_node(nodes[0]) and nodes[0].n == 4
    x = [sc.offset + sc.spec.body.offsets[0] + m for m in range(4)]
    s, c, t, r = (sc.spec.body.offsets[j] for j in (1, 2, 3, 4))
    assert FX.is_const_node(nodes[2])
    assert FX.root_outputs(P) == {r, s}
    # designated output that is also consumed inside: out counts it once, as itself
    assert FX.oracle_out(P, [s]) == {s}
    assert FX.oracle_w_out(P, [s]) == 32
    # the dead Input: nobody reads it and it is not designated
    assert FX.oracle_out(P, [x[3]]) == frozenset()
    assert FX.oracle_out(P, FX.item_part(sc, 0, 3)) == frozenset()
    # Input gates are ordinary source gates: in({x0}) = {}, out({x0}) = {x0}
    assert FX.oracle_in(P, [x[0]]) == frozenset()
    assert FX.oracle_out(P, [x[0]]) == {x[0]}
    # the constant gate is a gate: out({c}) = {c}; in({t}) = {x2, c}
    assert FX.oracle_out(P, [c]) == {c}
    assert FX.oracle_in(P, [t]) == {x[2], c}
    # the whole circuit: out = designated outputs, in = {}
    every = range(P.circuit.size)
    assert FX.oracle_out(P, every) == {r, s}
    assert FX.oracle_in(P, every) == frozenset()
    # a set missing the constant: {t} alone has c on its in-boundary (the "gap" a validator must report)
    assert c in FX.oracle_in(P, [t])


def test_fan_out_counted_once():
    P = FX.fan_out_program()
    sc = FX.root_scope(P)
    off = sc.spec.body.offsets
    x, y = off[0], off[1]
    z = [off[2], off[3], off[4]]
    # y feeds three gates through five operand slots: out({y}) is the one gate, 32 bits (not 5 x 32)
    consumers = [g for g in range(P.circuit.size) if y in P.circuit.gate(g).operands]
    assert consumers == z
    assert sum(list(P.circuit.gate(g).operands).count(y) for g in z) == 5
    assert FX.oracle_out(P, [y]) == {y}
    assert FX.oracle_w_out(P, [y]) == 32
    # the Input x fans out to y, z0, z2 -- likewise once
    assert FX.oracle_out(P, [x]) == {x}
    # in({z0, z1, z2}) = {x, y}: producers, distinct
    assert FX.oracle_in(P, z) == {x, y}


def test_dead_return_exact_vs_interface_bound():
    P = FX.dead_return_program()
    sc = FX.root_scope(P)
    jcall = FX.find_nodes(sc, form="call")[0]
    call_part = FX.item_part(sc, jcall, 0)
    inner = FX.child_scope(sc, jcall)
    a = inner.offset + inner.spec.body.offsets[0]
    b = inner.offset + inner.spec.body.offsets[1]
    # the call returns (a, b); the parent reads only a
    assert FX.oracle_out(P, call_part) == {a}
    assert FX.oracle_w_out(P, call_part) == 32
    assert FX.interface_w_out_bound(call_part) == 64  # what the declared interface charges
    # inside the callee: b is produced and dead
    assert FX.oracle_out(P, [b]) == frozenset()


def test_reexport_both_designated():
    P = FX.reexport_program()
    sc = FX.root_scope(P)
    jcall = FX.find_nodes(sc, form="call")[0]
    inner = FX.child_scope(sc, jcall)
    a = inner.offset + inner.spec.body.offsets[0]
    b = inner.offset + inner.spec.body.offsets[1]
    assert FX.root_outputs(P) == {a, b}
    assert FX.oracle_out(P, FX.item_part(sc, jcall, 0)) == {a, b}
    assert FX.oracle_w_out(P, FX.item_part(sc, jcall, 0)) == 64


def test_dead_param_synthetic():
    P = FX.dead_param_program()
    sc = FX.root_scope(P)
    body = sc.spec.body
    x = body.offsets[0]           # Input x (batch of one Input32)
    y = body.offsets[1]           # Input y: dead
    jc = FX.find_nodes(sc, callee="Const")[0]
    jcall = FX.find_nodes(sc, form="call")[0]
    c = body.offsets[jc]
    assert FX.oracle_out(P, [c]) == frozenset(), "a constant read only by a dead parameter is not an out gate"
    assert FX.oracle_out(P, [y]) == frozenset()
    assert FX.oracle_in(P, FX.item_part(sc, jcall, 0)) == {x}, "args superset of in: the unread parameter is not an edge"
    assert FX.interface_w_out_bound(FX.item_part(sc, jc, 0)) == 32 and FX.oracle_w_out(P, [c]) == 0


def test_attention_head_dead_parameter_case():
    """Design doc §5: `AttentionHead_v3` passes `Const32[0]` as m_prev/l_prev/o_prev to the FIRST=True block, whose body
    never reads them -> out({Const32[0]}) = {} (interface-based counting said 32)."""
    P = FX.attention_head_program()
    sc = FX.root_scope(P)
    body = sc.spec.body
    blocks = FX.find_nodes(sc, form="call", callee="AttnBlock_v3")
    assert len(blocks) == 3  # T=40, BN=16: ceil(40/16) = 3 blocks visited last -> first
    first = [j for j in blocks if body.nodes[j].fn.bindings["FIRST"]]
    assert len(first) == 1
    zero32 = [j for j in FX.find_nodes(sc, callee="Const32[0x00000000]") if j < first[0]]
    assert zero32, "the FIRST block's zero state constant precedes the block call"
    z = body.offsets[zero32[-1]]
    assert FX.oracle_out(P, [z]) == frozenset()
    # the FIRST block call reads q, k, v but not the state constant
    first_part = FX.item_part(sc, first[0], 0)
    assert z not in FX.oracle_in(P, first_part)
    # every block call has a 672-bit declared interface (32 + 4*32 + 16*32); the exact out is <= that and > 32
    for j in blocks:
        part = FX.item_part(sc, j, 0)
        assert FX.interface_w_out_bound(part) == 672
        w = FX.oracle_w_out(P, part)
        assert 32 < w <= 672
    # the last-visited block (b = 0, the highest node index) returns m_new that nobody reads: 672 - 32 = 640
    assert FX.oracle_w_out(P, FX.item_part(sc, max(blocks), 0)) == 640
    assert FX.oracle_w_out(P, FX.item_part(sc, min(blocks), 0)) == 672


def test_scan_carry_boundary():
    P = FX.scan_carry_program()
    sc = FX.root_scope(P)
    js = FX.find_nodes(sc, form="scan")[0]
    node = sc.spec.body.nodes[js]
    assert node.n == FX.SCAN_N
    for m in range(node.n):
        it = FX.item_part(sc, js, m)
        out = FX.oracle_out(P, it)
        # each iteration: its carry (consumed by the next iteration, or designated for the last) and its y (designated)
        assert len(out) == 2 and FX.oracle_w_out(P, it) == 64
        if m > 0:
            prev_carry = FX.child_scope(sc, js, m - 1).offset + 0  # node 0 of the step body is c2
            assert prev_carry in FX.oracle_in(P, it)
    whole = FX.node_part(sc, js)
    assert FX.oracle_out(P, whole) == FX.root_outputs(P)


def test_scan_carry_only_boundary():
    P = FX.scan_carry_only_program()
    sc = FX.root_scope(P)
    js = FX.find_nodes(sc, form="scan")[0]
    node = sc.spec.body.nodes[js]
    for m in range(node.n):
        it = FX.item_part(sc, js, m)
        c2 = FX.child_scope(sc, js, m).offset + 0
        assert FX.oracle_out(P, it) == {c2}, f"iteration {m}: only the carry escapes (ys are dead)"
        assert FX.oracle_w_out(P, it) == 32
    assert FX.oracle_out(P, FX.node_part(sc, js)) == FX.root_outputs(P)


def test_gumbel_has_64_bit_gates_and_nothing_else_does():
    wide = FX.wide_gates(FX.gumbel_program())
    assert len(wide) == 2 and all(w == 64 and pid.startswith("Const64[") for _, pid, w in wide)
    for name in FX.FIXTURE_NAMES:
        if name in FX.WIDE_GATE_FIXTURES:
            continue
        assert FX.wide_gates(FX.fixture(name)) == [], name


def test_moe_router_scan_carry_is_96_bits():
    P = FX.moe_router_program()
    sc = FX.root_scope(P)
    scans = FX.find_nodes(sc, form="scan")
    assert len(scans) == 2  # TOPK rounds
    js = scans[0]
    node = sc.spec.body.nodes[js]
    for m in range(node.n - 1):
        # (best: F32, best_i: I32, i: I32) carried into the next iteration = 96 bits; y (= ni) is dead
        assert FX.oracle_w_out(P, FX.item_part(sc, js, m)) == 96


def test_gemm_members_out_is_one_bf16():
    P = FX.gemm_program()
    sc = FX.root_scope(P)
    jb = FX.find_nodes(sc, form="batch", primitive=False)[0]
    node = sc.spec.body.nodes[jb]
    assert node.n == 6
    for m in range(node.n):
        assert FX.oracle_w_out(P, FX.item_part(sc, jb, m)) == 16
    inner = FX.child_scope(sc, jb, 2)
    dot = FX.first_composite_child(inner, "DotBf16")
    # the accumulator chain: every step's out is the 32-bit accumulator
    for p in FX.body_parts(dot):
        assert FX.oracle_w_out(P, p) == 32


def test_gateset_kind_drops_T():
    P40, P72 = FX.attention_head_program(T=40), FX.attention_head_program(T=72)
    k40 = {FX.gateset_kind(p) for p in FX.body_parts(FX.root_scope(P40))}
    k72 = {FX.gateset_kind(p) for p in FX.body_parts(FX.root_scope(P72))}
    assert k40 == k72


# ---------------------------------------------------------------------------------------------------------
# the canonical VU shapes (design doc §6) evaluated BY HAND on the fixtures and checked with the oracle: this pins the
# design (every set of the canonical query has w_out <= 32 and the sets tile the body) independently of the evaluator.
# ---------------------------------------------------------------------------------------------------------


def _zip_sets(sc, node_idxs):
    nodes = sc.spec.body.nodes
    ns = {nodes[j].n for j in node_idxs}
    assert len(ns) == 1, f"zip needs equal n, got {ns}"
    return [FX.items_part(sc, [(j, m) for j in node_idxs]) for m in range(ns.pop())]


def _canonical_sets(P, sc) -> list[Part]:
    """The canonical query's sets for the body `sc` activates (registry families only), as Parts; a body with no case
    falls to default(32): one set per member for nodes whose declared return width <= 32, else descended."""
    body = sc.spec.body
    nodes = body.nodes
    sid = sc.spec.id
    sets: list[Part] = []
    covered: set[int] = set()
    if sid.startswith(("RMSNormTriton_v1", "RMSNormFusedCuda_v")):
        muls = FX.find_nodes(sc, form="batch", callee="F32Mul_v1")
        cvts = FX.find_nodes(sc, form="batch", callee="F2fpBf16_v1")
        assert len(muls) == 2 and len(cvts) == 1
        sets += _zip_sets(sc, muls + cvts)
        covered |= set(muls + cvts)
    elif sid.startswith("AttnBlock_v"):
        pv = FX.find_nodes(sc, form="batch", callee="BF16TcDot16_v2") or FX.find_nodes(sc, form="batch", callee="WgmmaDot16_v1")
        assert pv, "the PV k-step batches"
        sets += _zip_sets(sc, pv)
        dots = FX.find_nodes(sc, form="call", callee="DotBf16_v")
        assert dots
        sets += [FX.item_part(sc, j, 0) for j in dots]
        covered |= set(pv) | set(dots)
    elif sid.startswith("MoeRouterTopK_v1"):
        ex = FX.find_nodes(sc, form="call", callee="NvExpf_v1")
        assert ex
        sets += [FX.item_part(sc, j, 0) for j in ex]
        covered |= set(ex)
        for js in FX.find_nodes(sc, form="scan"):
            for m in range(nodes[js].n):
                sets += _canonical_sets(P, FX.child_scope(sc, js, m))   # each(scan, default(32)) per iteration
            covered.add(js)
    elif sid.startswith(("Gemm_v", "SiluMul_v1", "MoeExpertGemm")):
        jb = FX.find_nodes(sc, form="batch")
        assert len(jb) == 1 and len(nodes) == 1
    # rest() / default(32) for everything uncovered (heads: blocks descend; everything else one set per member)
    for j, node in enumerate(nodes):
        if j in covered:
            continue
        if sid.startswith("AttentionHead_v") and node.form == "call" and node.fn.id.startswith("AttnBlock_v"):
            sets += _canonical_sets(P, FX.child_scope(sc, j, 0))
            continue
        descend = (not isinstance(node.fn, PrimitiveDefinition)) and sum(FX._type_widths(node.fn.body.ret_type)) > 32
        for m in range(n_members(node)):
            if descend:
                sets += _canonical_sets(P, FX.child_scope(sc, j, m))
            else:
                sets.append(FX.item_part(sc, j, m))
    return sets


@pytest.mark.parametrize("name", FX.VLLM_FAMILY_FIXTURES)
@pytest.mark.slow
def test_canonical_shapes_by_oracle(name):
    P = FX.fixture(name)
    sets = _canonical_sets(P, FX.root_scope(P))
    # tiles the circuit
    ivs = sorted(iv for s in sets for iv in s.intervals())
    pos = 0
    for lo, hi in ivs:
        assert lo == pos, f"{name}: canonical sets leave a gap / overlap at {pos}"
        pos = hi
    assert pos == P.circuit.size
    # w_out <= 32 for every set (all of them when few, a stride otherwise), the family's headline widths as designed
    step = max(1, len(sets) // 600)
    widths = [FX.oracle_w_out(P, s) for s in sets[::step]]
    assert max(widths) <= 32, f"{name}: max w_out {max(widths)}"
    if name.startswith("gemm") or name == "silu_mul" or name == "moe_expert_gemm":
        sc = FX.root_scope(P)
        jb = FX.find_nodes(sc, form="batch", primitive=name == "silu_mul")[-1]
        assert all(FX.oracle_w_out(P, FX.item_part(sc, jb, m)) == 16 for m in range(sc.spec.body.nodes[jb].n))


def test_canonical_norm_zip_is_16_bits():
    for name in ("rmsnorm_triton", "rmsnorm_fused"):
        P = FX.fixture(name)
        sc = FX.root_scope(P)
        muls = FX.find_nodes(sc, form="batch", callee="F32Mul_v1")
        cvts = FX.find_nodes(sc, form="batch", callee="F2fpBf16_v1")
        for s in _zip_sets(sc, muls + cvts):
            assert FX.oracle_w_out(P, s) == 16, name
        # without the zip, the F32Mul members are 32-bit sets (still valid, just more VUs)
        assert all(FX.oracle_w_out(P, FX.item_part(sc, muls[0], m)) == 32 for m in range(3))


@pytest.mark.parametrize("name", ["attention_head", "attention_head_t72", "attention_head_v1", "attention_head_v2"])
def test_canonical_attention_pv_chain_is_32_bits(name):
    P = FX.fixture(name)
    sc = FX.root_scope(P)
    blocks = FX.find_nodes(sc, form="call", callee="AttnBlock_v")
    n_ksteps = set()
    for jb in blocks:
        sb = FX.child_scope(sc, jb)
        pv = FX.find_nodes(sb, form="batch", callee="BF16TcDot16_v2")
        n_ksteps.add(len(pv))
        for s in _zip_sets(sb, pv):
            assert FX.oracle_w_out(P, s) == 32
        # a single k-step member alone is also 32 bits, but the chain is the designed VU
        for j in FX.find_nodes(sb, form="call", callee="DotBf16_v"):
            assert FX.oracle_w_out(P, FX.item_part(sb, j, 0)) == 32
    if name == "attention_head_v1":
        assert n_ksteps == {1, 4}, "BN = 64: the full block has four PV k-steps, the 8-column FIRST block one"


def test_canonical_attention_kinds_do_not_grow_with_T_by_hand():
    P40, P72 = FX.fixture("attention_head"), FX.fixture("attention_head_t72")
    s40, s72 = _canonical_sets(P40, FX.root_scope(P40)), _canonical_sets(P72, FX.root_scope(P72))
    assert len(s72) > len(s40)
    k40 = {FX.gateset_kind(s) for s in s40}
    k72 = {FX.gateset_kind(s) for s in s72}
    assert k40 == k72, (sorted(map(str, k72 - k40))[:3], sorted(map(str, k40 - k72))[:3])
    assert not any(cid.startswith("AttnBlock_v") for k in k72 for _, callees in k for cid, _ in callees)
