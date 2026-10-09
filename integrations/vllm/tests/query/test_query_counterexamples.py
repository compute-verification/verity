"""Counterexamples for the query track (the 2026-09-17 query-interface freeze): families with an overlap,
with a gap (a forgotten Input gate, a forgotten constant gate), a set whose interface bound exceeds 32 while the exact
w_out is 32 (dead return), sets whose exact w_out exceeds 32 (AttnBlock calls: 640 / 672 bits), scan-carry boundaries,
fan-out counted once, dead parameters, 64-bit gates -- and the RA rule that `Unsupported` is never a pass.

The three core modules are other owners' (`verity.ml.circuits.query_ast`, `verity.ml.circuits.boundary`, `verity.ml.circuits.partition`): every test
`importorskip`s what it needs, so this file SKIPS (never fails) until the branches are merged.  The oracle side of every
assertion is `fixtures_query` (explicit `GateSelection` on the expanded circuit)."""
from __future__ import annotations

import pytest

from verity.ml.circuits.parts import Limits, Part, Unsupported
from tests.program import fixtures_query as FX

# ---------------------------------------------------------------------------------------------------------
# module access (skip, never fail, while a module or a constructor is missing)
# ---------------------------------------------------------------------------------------------------------

_OWNER = {"gateset": "qapi", "boundary": "qbound", "partition": "qpart"}


_MODULES = {"gateset": "verity.ml.circuits.query_ast", "boundary": "verity.ml.circuits.boundary", "partition": "verity.ml.circuits.partition"}


def _mod(name: str):
    return pytest.importorskip(_MODULES[name], reason=f"{_MODULES[name]} not merged yet (owner {_OWNER[name]})")


@pytest.fixture(scope="module")
def GS():
    return _mod("gateset")


@pytest.fixture(scope="module")
def BD():
    return _mod("boundary")


@pytest.fixture(scope="module")
def PT():
    return _mod("partition")


@pytest.fixture(scope="module")
def PTW():
    """partition with widths: `validate_width` / `validate_vu_partition` call `boundary` at run time."""
    _mod("boundary")
    return _mod("partition")


def need(mod, *names):
    """First attribute of `mod` among `names`, else skip with an issue line the owner can grep."""
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    owner = _OWNER.get(mod.__name__.rsplit(".", 1)[-1], "?")
    pytest.skip(f"ISSUE FOR {owner}: {mod.__name__} lacks {' / '.join(names)}")


PASS_WORDS = {"pass", "passed", "ok", "exact", "within", "certified", "valid"}
FAIL_WORDS = {"fail", "failed", "violation", "violations", "exceeded", "invalid", "overlap", "gap"}


def is_pass(status) -> bool:
    return str(status).lower() in PASS_WORDS


def is_fail(status) -> bool:
    return str(status).lower() in FAIL_WORDS


def is_unsupported(x) -> bool:
    return isinstance(x, Unsupported) or str(getattr(x, "status", x)).lower() == "unsupported"


def gateset(GS, *parts: Part):
    return GS.GateSet(tuple(parts))


def finest_sets(GS, program, skip_input: bool = False):
    return [gateset(GS, p) for p in FX.body_parts(FX.root_scope(program), skip_input=skip_input)]


def limits(**kw) -> Limits:
    lim = Limits()
    for k, v in kw.items():
        setattr(lim, k, v)
    return lim


def check_part(BD, program, part: Part, *, exact_required: bool) -> tuple[bool, bool]:
    """Compare boundary.{out_gates,in_gates,w_out,w_out_bound} on `part` with the explicit oracle.  Returns
    (out_exact, in_exact); Unsupported is tolerated only when `exact_required` is False; a WRONG exact answer never is."""
    o_out, o_in = FX.oracle_out(program, part), FX.oracle_in(program, part)
    o_w = FX.oracle_w_out(program, part)
    out = BD.out_gates(part)
    inn = BD.in_gates(part)
    w = BD.w_out(part)
    bound = BD.w_out_bound(part)
    assert isinstance(bound, int) and not isinstance(bound, bool), "w_out_bound is never Unsupported"
    assert bound >= o_w, f"w_out_bound {bound} below the exact w_out {o_w} for {sorted(part.items)[:3]} in {part.spec.id}"
    out_exact = not isinstance(out, Unsupported)
    in_exact = not isinstance(inn, Unsupported)
    if out_exact:
        assert frozenset(out) == o_out, f"out_gates mismatch in {part.spec.id} items {sorted(part.items)[:3]}"
    if in_exact:
        assert frozenset(inn) == o_in, f"in_gates mismatch in {part.spec.id} items {sorted(part.items)[:3]}"
    if not isinstance(w, Unsupported):
        assert w == o_w, f"w_out {w} != oracle {o_w} in {part.spec.id}"
        assert bound >= w
    else:
        assert not out_exact or True  # w_out may be Unsupported independently; never a wrong number
    if exact_required:
        assert out_exact and in_exact and not isinstance(w, Unsupported), \
            f"root-body part {sorted(part.items)[:3]} of {part.spec.id} must be exact (got Unsupported)"
    return out_exact, in_exact


# ---------------------------------------------------------------------------------------------------------
# boundary counterexamples (qbound)
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", FX.FIXTURE_NAMES)
@pytest.mark.slow
def test_boundary_root_parts_match_oracle(BD, name):
    P = FX.fixture(name)
    parts = FX.body_parts(FX.root_scope(P))
    if len(parts) > 300:  # RMSNormTriton has ~1000 root items; sample the head, the tail and a stride
        parts = parts[:100] + parts[-100:] + parts[100:-100:7]
    for part in parts:
        check_part(BD, P, part, exact_required=True)


@pytest.mark.parametrize("name", ["gemm", "attention_head", "moe_expert_gemm", "dead_return", "reexport", "scan_carry", "token_select"])
def test_boundary_nested_parts_never_wrong(BD, name):
    """One level down (GemmCoordinate members, AttnBlock activations, scan iterations): exact or Unsupported, never wrong."""
    P = FX.fixture(name)
    sc = FX.root_scope(P)
    n_exact = n_total = 0
    for j in FX.find_nodes(sc, primitive=False):
        node = sc.spec.body.nodes[j]
        for m in range(min(node.n if node.form in ("batch", "scan") else 1, 3)):
            inner = FX.child_scope(sc, j, m)
            for part in FX.body_parts(inner)[:60]:
                oe, ie = check_part(BD, P, part, exact_required=False)
                n_total += 1
                n_exact += oe and ie
    assert n_total > 0
    assert n_exact > 0, f"{name}: every nested part came back Unsupported -- the exact path never engaged"


def test_fan_out_counted_once(BD):
    P = FX.fan_out_program()
    sc = FX.root_scope(P)
    y = sc.spec.body.offsets[1]
    part = FX.item_part(sc, 1, 0)
    assert frozenset(BD.out_gates(part)) == {y}
    assert BD.w_out(part) == 32, "three consumers through five operand slots: the producer counts once"
    assert BD.w_out_bound(part) >= 32
    # the Input gate x fans out to three gates as well
    xin = FX.item_part(sc, 0, 0)
    assert BD.w_out(xin) == 32 and BD.gate_width(P, sc.spec.body.offsets[0]) == 32


def test_dead_return_exact_le_32_even_if_bound_exceeds(BD):
    """`Pair` returns (a, b); the parent reads only a.  The interface bound charges both leaves (64); the exact answer is
    32.  The exact path must walk the parent's consumer index (freeze doc: one level up)."""
    P = FX.dead_return_program()
    sc = FX.root_scope(P)
    jcall = FX.find_nodes(sc, form="call")[0]
    part = FX.item_part(sc, jcall, 0)
    inner = FX.child_scope(sc, jcall)
    a = inner.offset + inner.spec.body.offsets[0]
    assert frozenset(BD.out_gates(part)) == {a}
    assert BD.w_out(part) == 32
    bound = BD.w_out_bound(part)
    assert bound >= 32
    assert FX.interface_w_out_bound(part) == 64
    # the dead leaf, as a part of the callee body: out = {} and w_out = 0 (its return is re-consumed by nobody)
    dead = FX.item_part(inner, 1, 0)
    r = BD.out_gates(dead)
    if not isinstance(r, Unsupported):
        assert frozenset(r) == frozenset() and BD.w_out(dead) == 0
    # re-returned by the root (designated): both leaves escape
    P2 = FX.reexport_program()
    sc2 = FX.root_scope(P2)
    j2 = FX.find_nodes(sc2, form="call")[0]
    assert BD.w_out(FX.item_part(sc2, j2, 0)) == 64


def test_dead_parameter_is_not_an_edge(BD):
    # synthetic: the constant read only by an unread parameter
    P = FX.dead_param_program()
    sc = FX.root_scope(P)
    jc = FX.find_nodes(sc, callee="Const")[0]
    jcall = FX.find_nodes(sc, form="call")[0]
    x = sc.spec.body.offsets[0]
    assert frozenset(BD.out_gates(FX.item_part(sc, jc, 0))) == frozenset()
    assert BD.w_out(FX.item_part(sc, jc, 0)) == 0
    assert frozenset(BD.in_gates(FX.item_part(sc, jcall, 0))) == {x}
    # dead Input y: out = {}
    assert BD.w_out(FX.item_part(sc, 1, 0)) == 0
    # the registry case: AttentionHead_v3 passes Const32[0] to the FIRST block's unread m_prev / l_prev / o_prev
    P3 = FX.attention_head_program()
    sc3 = FX.root_scope(P3)
    body = sc3.spec.body
    blocks = FX.find_nodes(sc3, form="call", callee="AttnBlock_v3")
    first = [j for j in blocks if body.nodes[j].fn.bindings["FIRST"]][0]
    jz = [j for j in FX.find_nodes(sc3, callee="Const32[0x00000000]") if j < first][-1]
    z = body.offsets[jz]
    assert frozenset(BD.out_gates(FX.item_part(sc3, jz, 0))) == frozenset()
    assert BD.w_out(FX.item_part(sc3, jz, 0)) == 0
    assert z not in frozenset(BD.in_gates(FX.item_part(sc3, first, 0)))


def test_exact_w_out_above_32_attn_block_calls(BD):
    P = FX.attention_head_program()
    sc = FX.root_scope(P)
    blocks = FX.find_nodes(sc, form="call", callee="AttnBlock_v3")
    ws = {}
    for j in blocks:
        part = FX.item_part(sc, j, 0)
        w = BD.w_out(part)
        assert not isinstance(w, Unsupported)
        assert w == FX.oracle_w_out(P, part) and w > 32
        assert BD.w_out_bound(part) >= w
        ws[j] = w
    assert ws[min(blocks)] == 672 and ws[max(blocks)] == 640, "first-visited block: full interface; last: m_new unread"


def test_scan_carry_boundary(BD):
    P = FX.scan_carry_only_program()
    sc = FX.root_scope(P)
    js = FX.find_nodes(sc, form="scan")[0]
    node = sc.spec.body.nodes[js]
    for m in range(node.n):
        it = FX.item_part(sc, js, m)
        c2 = FX.child_scope(sc, js, m).offset
        assert frozenset(BD.out_gates(it)) == {c2}, f"iteration {m}: exactly the carry escapes"
        assert BD.w_out(it) == 32
        if m > 0:
            assert FX.child_scope(sc, js, m - 1).offset in frozenset(BD.in_gates(it)), "iteration m reads iteration m-1's carry"
    P2 = FX.scan_carry_program()
    sc2 = FX.root_scope(P2)
    js2 = FX.find_nodes(sc2, form="scan")[0]
    for m in range(sc2.spec.body.nodes[js2].n):
        assert BD.w_out(FX.item_part(sc2, js2, m)) == 64  # carry + designated y
    # the whole scan node as one part: out = the designated outputs only
    assert frozenset(BD.out_gates(FX.node_part(sc2, js2))) == FX.root_outputs(P2)
    # registry: MoeRouterTopK's argmax scan carries (F32, I32, I32) = 96 bits between iterations
    P4 = FX.moe_router_program()
    sc4 = FX.root_scope(P4)
    js4 = FX.find_nodes(sc4, form="scan")[0]
    assert BD.w_out(FX.item_part(sc4, js4, 0)) == 96


def test_input_and_constant_gates_are_gates(BD):
    P = FX.tiny_io_program()
    sc = FX.root_scope(P)
    body = sc.spec.body
    x0 = body.offsets[0]
    assert BD.gate_width(P, x0) == 32
    assert frozenset(BD.out_gates(FX.item_part(sc, 0, 0))) == {x0}
    assert frozenset(BD.in_gates(FX.item_part(sc, 0, 0))) == frozenset()
    assert BD.w_out(FX.item_part(sc, 0, 3)) == 0, "the dead Input"
    jc = FX.find_nodes(sc, callee="Const")[0]
    c = body.offsets[jc]
    assert frozenset(BD.out_gates(FX.item_part(sc, jc, 0))) == {c}
    # s: consumed inside and designated -> once
    s = body.offsets[1]
    assert frozenset(BD.out_gates(FX.item_part(sc, 1, 0))) == {s} and BD.w_out(FX.item_part(sc, 1, 0)) == 32
    # whole root body as one part: out = designated outputs; in = {}
    whole = FX.items_part(sc, [(j, m) for p in FX.body_parts(sc) for (j, m) in p.items])
    assert frozenset(BD.out_gates(whole)) == FX.root_outputs(P)
    assert frozenset(BD.in_gates(whole)) == frozenset()


def test_64_bit_gates_have_width_64(BD):
    P = FX.gumbel_program()
    wide = FX.wide_gates(P)
    assert len(wide) == 2
    for g, pid, w in wide:
        assert BD.gate_width(P, g) == 64
    sc = FX.root_scope(P)
    for j in FX.find_nodes(sc, callee="Const64"):
        assert BD.w_out(FX.item_part(sc, j, 0)) == 64


# ---------------------------------------------------------------------------------------------------------
# partition / width counterexamples (qpart), hand-made families through GateSet (qapi)
# ---------------------------------------------------------------------------------------------------------


def test_finest_parts_are_exact(GS, PT):
    validate = need(PT, "validate_partition")
    for name in FX.FIXTURE_NAMES:
        P = FX.fixture(name)
        v = validate(P, FX.ListFamily(finest_sets(GS, P)))
        assert v.status == "exact", f"{name}: {v.status} {v.codes}"


def test_overlap_is_reported(GS, PT):
    validate = need(PT, "validate_partition")
    P = FX.tiny_io_program()
    sets = finest_sets(GS, P)
    v = validate(P, FX.ListFamily(sets + [sets[1]]))  # the Add gate twice
    assert v.status == "overlap", (v.status, v.codes)
    # overlapping by algebra: the union of two adjacent sets beside the sets themselves
    v2 = validate(P, FX.ListFamily(sets + [sets[1].union(sets[2])]))
    assert v2.status == "overlap"


def test_gap_missing_input_gate(GS, PT):
    validate = need(PT, "validate_partition")
    P = FX.tiny_io_program()
    sc = FX.root_scope(P)
    parts = FX.body_parts(sc)
    dead_input = FX.item_part(sc, 0, 3)
    sets = [gateset(GS, p) for p in parts if p != dead_input]
    v = validate(P, FX.ListFamily(sets))
    assert v.status == "gap", "coverage means ALL gates, a dead Input gate included"
    # skipping every Input node (today's vu_query convention) is a gap under the new semantics
    v2 = validate(P, FX.ListFamily(finest_sets(GS, P, skip_input=True)))
    assert v2.status == "gap"


def test_gap_missing_constant_gate(GS, PT):
    validate = need(PT, "validate_partition")
    P = FX.tiny_io_program()
    sc = FX.root_scope(P)
    jc = FX.find_nodes(sc, callee="Const")[0]
    sets = [gateset(GS, p) for p in FX.body_parts(sc) if (jc, 0) not in p.items]
    v = validate(P, FX.ListFamily(sets))
    assert v.status == "gap"


def test_partition_unsupported_is_not_a_verdict(GS, PT):
    validate = need(PT, "validate_partition")
    P = FX.rmsnorm_triton_program()
    fam = FX.ListFamily(finest_sets(GS, P))
    v = validate(P, fam, limits(max_intervals=4))
    assert v.status == "unsupported", f"the sweep over {fam.count()} intervals must give up under max_intervals=4, got {v.status}"
    v_ok = validate(P, fam)
    assert v_ok.status == "exact"


def _loose_bound_part(BD):
    """The whole body of `Pair` as ONE nested Part: both returned leaves are sourced in the part, so any bound that does not
    consult the parent's consumption charges both (64 > 32) while the exact w_out is 32 (b is dead at the caller)."""
    P = FX.dead_return_program()
    sc = FX.root_scope(P)
    inner = FX.child_scope(sc, FX.find_nodes(sc, form="call")[0])
    part = FX.items_part(inner, [it for q in FX.body_parts(inner) for it in q.items])
    assert FX.oracle_w_out(P, part) == 32
    assert FX.interface_w_out_bound(part) == 64
    bound = BD.w_out_bound(part)
    assert bound >= 32, "w_out_bound must be an upper bound"
    if bound <= 32:
        pytest.skip("qbound's bound is already exact for the nested dead return; no bound > 32 > exact case in this fixture set")
    return P, part


def test_width_dead_return_certified_by_exact_path(GS, PTW, BD):
    """bound > 32 but exact <= 32: the validator must not report a violation on the bound alone -- it must fall through to
    the exact path (tier 3) and certify the set."""
    PT = PTW
    validate_width = need(PT, "validate_width")
    P, part = _loose_bound_part(BD)
    assert BD.w_out(part) == 32
    v = validate_width(P, FX.ListFamily([gateset(GS, part)]), 32)
    assert not v.violators, f"exact w_out of the Pair body is 32 (bound {BD.w_out_bound(part)}); violators {v.violators}"
    assert not v.unsupported
    assert v.max_seen == 32
    assert is_pass(v.status), v.status
    # the same set with a 64-bit budget is certified by the bound alone, and with a 31-bit budget it is a true violation
    assert is_pass(validate_width(P, FX.ListFamily([gateset(GS, part)]), 64).status)
    v31 = validate_width(P, FX.ListFamily([gateset(GS, part)]), 31)
    assert v31.violators == [0] or list(v31.violators) == [0], v31.violators
    assert is_fail(v31.status)


def test_width_attn_block_calls_violate(GS, PTW):
    PT = PTW
    validate_width = need(PT, "validate_width")
    P = FX.attention_head_program()
    sc = FX.root_scope(P)
    parts = FX.body_parts(sc)
    blocks = set(FX.find_nodes(sc, form="call", callee="AttnBlock_v3"))
    idx = [i for i, p in enumerate(parts) if any(j in blocks for j, _ in p.items)]
    v = validate_width(P, FX.ListFamily([gateset(GS, p) for p in parts]), 32)
    assert set(v.violators) == set(idx), (v.violators, idx)
    assert v.max_seen == 672
    assert not is_pass(v.status)
    assert v.w_max == 32


def test_width_unsupported_never_passes(GS, PTW, BD):
    """bound > 32 and the exact path starved (`Limits.max_members_exact=0`): the member is Unsupported -- the verdict must
    not pass and must not call it a violator either."""
    PT = PTW
    validate_width = need(PT, "validate_width")
    P, part = _loose_bound_part(BD)
    v = validate_width(P, FX.ListFamily([gateset(GS, part)]), 32, limits(max_members_exact=0))
    if not v.unsupported:
        pytest.skip("ISSUE FOR qpart: Limits.max_members_exact=0 did not make the exact path Unsupported; cannot exercise the RA rule here")
    assert not is_pass(v.status), "an Unsupported member must never make the verdict pass"
    assert not v.violators, "an Unsupported member is not a violator either"
    assert is_unsupported(v)


def test_readiness_32bit_reports_the_64_bit_constants(PT):
    readiness = need(PT, "readiness_32bit")
    r = readiness(FX.gumbel_program())
    viol = getattr(r, "violators", None)
    assert viol, "GumbelArgmaxBf16_v1 has two Const64 gates"
    text = repr(viol)
    assert "Const64" in text
    r0 = readiness(FX.tiny_io_program())
    assert not getattr(r0, "violators", None)


def test_vu_partition_default32_is_exact_and_narrow(GS, PTW):
    """`default(32)` is a partition by construction and every member's w_out <= 32 (out of a single call / member is a
    subset of its declared returns), except where a primitive gate itself is wider than 32."""
    PT = PTW
    validate_vu = need(PT, "validate_vu_partition")
    default = need(GS, "default")
    for name in FX.VLLM_FAMILY_FIXTURES + ("tiny_io", "fan_out", "dead_return", "dead_param", "scan_carry"):
        P = FX.fixture(name)
        v = validate_vu(P, default(32), 32)
        assert v.partition.status == "exact", f"{name}: {v.partition.status} {v.partition.codes}"
        assert not v.width.violators, f"{name}: width violators {v.width.violators}"
        assert not v.width.unsupported, f"{name}: width unsupported {v.width.unsupported}"
        assert v.count > 0
    # the 64-bit case: partition exact, width NOT passed, readiness lists the constants, overall status not pass
    v = validate_vu(FX.gumbel_program(), default(32), 32)
    assert v.partition.status == "exact"
    assert v.width.violators and v.width.max_seen == 64
    assert getattr(v.readiness_32, "violators", None)
    assert not is_pass(v.status)


# ---------------------------------------------------------------------------------------------------------
# query-built counterexamples (qapi constructors through the qpart validators)
# ---------------------------------------------------------------------------------------------------------


def test_query_concat_overlap_and_instances_gap(GS, PT):
    validate = need(PT, "validate_partition")
    evaluate, instances, default, concat = (need(GS, "evaluate"), need(GS, "instances"), need(GS, "default"), need(GS, "concat"))
    P = FX.wrapped_fixture("gemm")
    # instances of DotBf16 alone: misses Inputs, the cvt gates, the zero accumulators -> gap
    fam = evaluate(P, instances("//DotBf16_v1"))
    assert fam.count() == 6
    v = validate(P, fam)
    assert v.status == "gap", (v.status, v.codes)
    # the same instances beside a complete family: overlap
    v2 = validate(P, evaluate(P, concat(instances("//DotBf16_v1"), default(32))))
    assert v2.status == "overlap", (v2.status, v2.codes)


def test_query_members_gemm(GS, PT, BD):
    evaluate, members = need(GS, "evaluate"), need(GS, "members")
    validate = need(PT, "validate_partition")
    P = FX.wrapped_fixture("gemm")
    fam = evaluate(P, members("//Gemm_v1{K=*,N=*}", "@batch:GemmCoordinate_v1"))
    assert fam.count() == 6
    for i in range(fam.count()):
        s = fam.by_index(i)
        assert fam.index_of(s) == i
        # one member of the GemmCoordinate batch = one coordinate's gates (K=32 -> 4 gates: dot + ftz conversions)
        gemm_scope = FX.child_scope(FX.root_scope(P), FX.find_nodes(FX.root_scope(P), form="call")[0])
        jc = FX.find_nodes(gemm_scope, form="batch", callee="GemmCoordinate")[0]
        assert s.size() == gemm_scope.spec.body.nodes[jc].fn.gates
        assert FX.oracle_w_out(P, s) == 16
    # members alone leave the Input gates uncovered: a gap (Input gates are inside the partition)
    assert validate(P, fam).status == "gap"


def test_query_json_roundtrip_and_id(GS):
    to_json, from_json, query_id = need(GS, "to_json"), need(GS, "from_json"), need(GS, "query_id")
    override, default, members, instances = need(GS, "override"), need(GS, "default"), need(GS, "members"), need(GS, "instances")
    q = override(default(32), {"//Gemm_v1{*}": members("//Gemm_v1{*}", "@batch:GemmCoordinate_v1")})
    doc = to_json(q)
    assert isinstance(doc, dict)
    q2 = from_json(doc)
    assert to_json(q2) == doc
    assert query_id(q) == query_id(q2)
    assert len(query_id(q)) == 128
    assert query_id(q) != query_id(default(32))
    assert query_id(instances("//DotBf16_v1")) != query_id(instances("//Gemm_v1"))


def test_query_gatesets_from_different_activations_union(GS):
    evaluate, instances = need(GS, "evaluate"), need(GS, "instances")
    P = FX.wrapped_fixture("gemm")
    fam = evaluate(P, instances("//DotBf16_v1"))
    a, b = fam.by_index(0), fam.by_index(1)
    u = a.union(b)
    assert u.size() == a.size() + b.size()
    assert u.minus(a).size() == b.size()
    assert u.inter(a).size() == a.size()
    assert all(u.contains(g) for g in FX.gates_of(a)) and all(u.contains(g) for g in FX.gates_of(b))
    assert not a.inter(b).size()
    assert FX.oracle_w_out(P, u) == 64  # two independent accumulators
