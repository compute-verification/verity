"""`validate_partition` fast path (tier A, per-body induction over `Family.tiling()`) against the interval sweep on
the same materialised sets; `validate_width` (bound-only certification vs exact, budget, Unsupported); readiness on
Programs with 64-bit gates; the `VuVerdict` composition."""
from __future__ import annotations

import pytest

from verity.ml.circuits import partition
from verity.ml.circuits.parts import Limits, Part
from verity.ml.circuits.partition import (ReadinessReport, VuVerdict, WidthVerdict, gate_width_readiness, readiness_32bit,
                                 validate_partition, validate_vu_family, validate_width, vu_status)
from tests.query.test_partition_stubs import (OracleBoundary, StubFamily, StubGateSet, body_tiling, family_of,
                                                  gemm_program,                                                   materialise, members_k, ranges_of, synth_program, tiling,
                                                  twice_program, whole_body, wide_program)


@pytest.fixture
def gemm():
    return gemm_program()


@pytest.fixture
def synth():
    return synth_program(4)


def gemm_vu_tiling(P_):
    """The VU tiling of the Gemm program: every Input gate its own set, every GemmCoordinate one set."""
    return tiling(whole_body(P_.root, {0: members_k(), 1: members_k(), 2: members_k()}))


def gemm_deep_tiling(P_):
    """Descend into every coordinate: DotBf16's three gates and the F2fpBf16 each their own set."""
    coord = P_.root.body.nodes[2].fn
    dot = coord.body.nodes[0].fn
    dot_t = whole_body(dot)
    coord_t = body_tiling(coord, {0: ("descend", dot_t), 1: ("whole",)})
    return tiling(whole_body(P_.root, {2: ("descend", coord_t)}))


def both_paths(P_, t):
    """(fast-path verdict, sweep verdict) on the family the tiling induces."""
    fast = validate_partition(P_, family_of(P_, t, with_tiling=True))
    sweep = validate_partition(P_, family_of(P_, t, with_tiling=False))
    assert fast.detail["path"] == "tiling" and sweep.detail["path"] == "sweep"
    return fast, sweep


# -- fast path vs sweep: agreement ---------------------------------------------------------------------------


def test_whole_root_nodes_exact(gemm):
    fast, sweep = both_paths(gemm, tiling(whole_body(gemm.root)))
    assert fast.status == sweep.status == "exact"
    assert fast.codes == () and fast.detail["sets"] == 3 == sweep.detail["sets"]
    assert fast.detail["bodies"] == 1 and fast.detail["gates"] == 248


def test_vu_tiling_exact(gemm):
    fast, sweep = both_paths(gemm, gemm_vu_tiling(gemm))
    assert fast.status == sweep.status == "exact"
    assert fast.detail["sets"] == 32 + 192 + 6 == sweep.detail["sets"]


def test_deep_tiling_exact_count_by_induction(gemm):
    fast, sweep = both_paths(gemm, gemm_deep_tiling(gemm))
    assert fast.status == sweep.status == "exact"
    assert fast.detail["sets"] == 2 + 6 * (3 + 1) == sweep.detail["sets"]
    assert fast.detail["bodies"] == 3


def test_members_k_ranges_with_partial_last_block(gemm):
    t = tiling(whole_body(gemm.root, {1: members_k(50), 2: members_k(4)}))   # 192 = 3*50 + 42; 6 = 4 + 2
    fast, sweep = both_paths(gemm, t)
    assert fast.status == sweep.status == "exact" and fast.detail["sets"] == 1 + 4 + 2


def test_unassigned_node_is_gap(gemm):
    rules = {0: ("whole",), 2: members_k()}  # root node 1 (the weights) missing
    fast, sweep = both_paths(gemm, tiling(body_tiling(gemm.root, rules)))
    assert fast.status == sweep.status == "gap"
    assert fast.codes == ("tiling-unassigned",)
    [p] = fast.detail["problems"]
    assert p["code"] == "tiling-unassigned" and p["spec"] == gemm.root.id and p["node"] == 1 and p["fn"] == "Input16_v1"
    assert sweep.detail["gaps"] == [{"gates": [32, 224]}]


def test_unassigned_nested_node_is_gap(gemm):
    coord = gemm.root.body.nodes[2].fn
    coord_t = body_tiling(coord, {0: ("whole",)})  # the F2fpBf16 of every coordinate uncovered
    fast, sweep = both_paths(gemm, tiling(whole_body(gemm.root, {2: ("descend", coord_t)})))
    assert fast.status == sweep.status == "gap"
    assert fast.detail["problems"][0]["spec"] == coord.id and fast.detail["problems"][0]["node"] == 1
    assert sweep.detail["gap_count"] == 6


def test_double_assignment_is_overlap(gemm):
    rules = [(0, ("whole",)), (1, ("whole",)), (2, members_k()), (2, ("whole",))]  # node 2 in two entries, full ranges both
    fast, sweep = both_paths(gemm, tiling(body_tiling(gemm.root, rules)))
    assert fast.status == sweep.status == "overlap"
    assert fast.codes == ("tiling-members-overlap",)
    assert fast.detail["problems"][0]["node"] == 2 and fast.detail["sets"] == 2 + 6 + 1 == sweep.detail["sets"]
    # two entries whose ranges complement each other are fine: atomic over [0, 3), whole over [3, 6)
    rules = {0: ("whole",), 1: ("whole",), 2: ("multi", ("atomic_ranges", 1, ((0, 3),)), ("ranges", ((3, 6),)))}
    fast, sweep = both_paths(gemm, tiling(body_tiling(gemm.root, rules)))
    assert fast.status == sweep.status == "exact" and fast.detail["sets"] == 2 + 3 + 1
    # a partial descent: members [0, 4) of the coordinates descended, [4, 6) atomic
    coord = gemm.root.body.nodes[2].fn
    rules = {0: ("whole",), 1: ("whole",), 2: ("multi", ("descend_partial", whole_body(coord), ((0, 4),)), ("atomic_ranges", 1, ((4, 6),)))}
    fast, sweep = both_paths(gemm, tiling(body_tiling(gemm.root, rules)))
    assert fast.status == sweep.status == "exact" and fast.detail["sets"] == 2 + 4 * 2 + 2


def test_member_ranges_gap_and_overlap(gemm):
    fast, sweep = both_paths(gemm, tiling(whole_body(gemm.root, {2: ranges_of((0, 3))})))
    assert fast.status == sweep.status == "gap" and fast.codes == ("tiling-members-gap",)
    assert fast.detail["problems"][0]["members"] == [3, 6]
    fast, sweep = both_paths(gemm, tiling(whole_body(gemm.root, {2: ranges_of((0, 4), (3, 6))})))
    assert fast.status == sweep.status == "overlap" and fast.codes == ("tiling-members-overlap",)
    # gap and overlap together: overlap wins, both codes listed
    fast, _ = both_paths(gemm, tiling(whole_body(gemm.root, {2: ranges_of((0, 4), (3, 5))})))
    assert fast.status == "overlap" and fast.codes == ("tiling-members-gap", "tiling-members-overlap")


def test_zip_group_exact_and_unequal_n(synth):
    t = tiling(whole_body(synth.root, {0: members_k(), 1: ("zip", "g"), 2: ("zip", "g"), 4: members_k()}))
    fast, sweep = both_paths(synth, t)
    assert fast.status == sweep.status == "exact" and fast.detail["sets"] == 4 + 4 + 1 + 4
    # zip of a batch (n=4) with the constant (n=1): the sets such a group induces are not defined -> the description
    # is malformed and the sweep decides on whatever the family materialised (here: the covering reading)
    t = tiling(whole_body(synth.root, {2: ("zip", "g"), 3: ("zip", "g")}))
    v = validate_partition(synth, family_of(synth, t, with_tiling=True))
    assert v.detail["path"] == "sweep" and v.codes[0] == "tiling-malformed"
    assert "tiling-zip-unequal-n" in v.detail["tiling_malformed"]


def test_descend_into_scan_iterations(synth):
    step = synth.root.body.nodes[4].fn
    t = tiling(whole_body(synth.root, {4: ("descend", whole_body(step))}))
    fast, sweep = both_paths(synth, t)
    assert fast.status == sweep.status == "exact" and fast.detail["sets"] == 4 + 4 * 2


def test_shared_callee_from_two_parents_two_plans():
    """Gemm called twice from the root; each call descended through its own plan into ONE shared coordinate plan,
    which descends into the dot plan: multiplicities 2 x 32 coordinates, 64 dots.  Discovery order is not
    topological here (the coordinate plan is found under the first Gemm plan before the second Gemm plan exists)."""
    P_ = twice_program()
    gemm_spec = P_.root.body.nodes[2].fn
    assert P_.root.body.nodes[3].fn is gemm_spec
    coord = gemm_spec.body.nodes[0].fn
    dot = coord.body.nodes[0].fn
    dot_t = whole_body(dot, key="dot")
    coord_t = body_tiling(coord, {0: ("descend", dot_t), 1: ("whole",)}, key="coord")
    g1 = whole_body(gemm_spec, {0: ("descend", coord_t)}, key="gemm-a")
    g2 = whole_body(gemm_spec, {0: ("descend", coord_t)}, key="gemm-b")
    t = tiling(whole_body(P_.root, {2: ("descend", g1), 3: ("descend", g2)}))
    assert set(t.bodies) == {t.root, "gemm-a", "gemm-b", "coord", "dot"}
    fast, sweep = both_paths(P_, t)
    assert fast.status == sweep.status == "exact"
    assert fast.detail["sets"] == 2 + 2 * 32 * (3 + 1) == sweep.detail["sets"] and fast.detail["bodies"] == 5
    # width: the tiling path must count 64 coordinate activations (2 parents x 32 members) and 64 dot activations
    B = OracleBoundary(P_)
    import verity.ml.circuits.partition as pm
    saved = pm._BOUNDARY
    pm._BOUNDARY = B
    try:
        v = validate_width(P_, family_of(P_, t), 32)
    finally:
        pm._BOUNDARY = saved
    assert v.detail["sets"] == fast.detail["sets"] and v.detail["path"] == "tiling"
    # the two Input batches as whole sets fail; the rest (32-bit accumulator chain gates, 16-bit cvt) pass
    assert v.status == "violation" and v.detail["violation_count"] == 2 and v.max_seen == 1024 * 16
    assert v.detail["exact_evaluated"] == 2                       # only the two Input sets needed the oracle
    # with the Inputs per member the whole family certifies, and the 64 activations of every nested class are counted
    t = tiling(whole_body(P_.root, {0: members_k(), 1: members_k(), 2: ("descend", g1), 3: ("descend", g2)}))
    pm._BOUNDARY = B
    try:
        v = validate_width(P_, family_of(P_, t), 32)
    finally:
        pm._BOUNDARY = saved
    assert v.status == "ok" and v.detail["sets"] == 32 + 1024 + 2 * 32 * 4


def test_scopes_of_enumerates_every_activation():
    P_ = twice_program()
    gemm_spec = P_.root.body.nodes[2].fn
    coord = gemm_spec.body.nodes[0].fn
    coord_t = whole_body(coord, key="coord")
    g = whole_body(gemm_spec, {0: ("descend", coord_t)}, key="gemm")
    t = tiling(whole_body(P_.root, {2: ("descend", g), 3: ("descend", g)}))
    from verity.ml.circuits.partition import CheckedTiling
    T = CheckedTiling(P_, t)
    assert T.instances == {(t.root,): 1, ("gemm",): 2, ("coord",): 64}
    scopes = list(T.scopes_of(("coord",)))
    assert len(scopes) == 64 and len({sc.offset for sc in scopes}) == 64
    assert all(sc.spec is coord for sc in scopes)
    assert [sc.path()[-1] for sc in scopes[:2]] == [f"{coord.id}#0", f"{coord.id}#1"]


# -- malformed descriptions fall back to the sweep --------------------------------------------------------


def test_descend_on_primitive_node_is_malformed_then_swept(gemm):
    t = tiling(whole_body(gemm.root, {0: ("descend", whole_body(gemm.root))}))
    fam = StubFamily(materialise(gemm, tiling(whole_body(gemm.root))), tiling=t)
    v = validate_partition(gemm, fam)
    assert v.status == "exact" and v.detail["path"] == "sweep"
    assert v.codes == ("tiling-malformed",) and "primitive" in v.detail["tiling_malformed"]


def test_descend_into_wrong_spec_is_malformed(gemm, synth):
    t = tiling(whole_body(gemm.root, {2: ("descend", whole_body(synth.root.body.nodes[4].fn))}))
    v = validate_partition(gemm, StubFamily(materialise(gemm, tiling(whole_body(gemm.root))), tiling=t))
    assert v.codes[0] == "tiling-malformed" and v.detail["path"] == "sweep"


def test_wrong_root_is_malformed(gemm, synth):
    v = validate_partition(gemm, StubFamily(materialise(gemm, tiling(whole_body(gemm.root))), tiling=tiling(whole_body(synth.root))))
    assert v.codes[0] == "tiling-malformed" and v.status == "exact"


def test_unknown_rule_kind_is_malformed(gemm):
    t = tiling(whole_body(gemm.root, {2: ("halve",)}))
    v = validate_partition(gemm, StubFamily(materialise(gemm, tiling(whole_body(gemm.root))), tiling=t))
    assert v.codes[0] == "tiling-malformed"


def test_count_mismatch_falls_back_to_sweep(gemm):
    t = gemm_vu_tiling(gemm)
    sets = materialise(gemm, t)
    v = validate_partition(gemm, StubFamily(sets, tiling=t, count=len(sets) + 1))
    assert v.codes[0] == "tiling-count-mismatch" and v.detail["path"] == "sweep" and v.status == "exact"
    assert v.detail["tiling_count"] == {"structural": 230, "family": 231}


def test_tiling_that_lies_about_its_sets_is_caught_by_the_count(gemm):
    """A family whose sets are NOT what its tiling says (one coordinate dropped): the structural count disagrees
    and the sweep, not the tiling, decides."""
    t = gemm_vu_tiling(gemm)
    sets = materialise(gemm, t)[:-1]
    v = validate_partition(gemm, StubFamily(sets, tiling=t))
    assert v.status == "gap" and v.detail["path"] == "sweep"


def test_fast_path_visits_no_gates(gemm, monkeypatch):
    calls = {"n": 0}
    orig = type(gemm.circuit).gate

    def counting(self, i):
        calls["n"] += 1
        return orig(self, i)

    monkeypatch.setattr(type(gemm.circuit), "gate", counting)
    v = validate_partition(gemm, family_of(gemm, gemm_deep_tiling(gemm)))
    assert v.status == "exact" and calls["n"] == 0


def test_unsupported_limit_applies_to_fallback_only(gemm):
    class L(Limits):
        max_intervals = 4

    t = gemm_vu_tiling(gemm)
    assert validate_partition(gemm, family_of(gemm, t, with_tiling=True), L()).status == "exact"
    assert validate_partition(gemm, family_of(gemm, t, with_tiling=False), L()).status == "unsupported"


# -- width -----------------------------------------------------------------------------------------------------


@pytest.fixture
def oracle(gemm, monkeypatch):
    def install(slack: int = 0, exact_unsupported: bool = False, program=None) -> OracleBoundary:
        B = OracleBoundary(program or gemm, slack=slack, exact_unsupported=exact_unsupported)
        monkeypatch.setattr(partition, "_BOUNDARY", B)
        return B
    return install


def test_width_bound_certifies_every_class(gemm, oracle):
    B = oracle()
    v = validate_width(gemm, family_of(gemm, gemm_vu_tiling(gemm)), 32)
    assert isinstance(v, WidthVerdict)
    assert v.status == "ok" and v.violators == () and v.unsupported == ()
    assert v.max_seen == 16 and v.w_max == 32
    d = v.detail
    assert d["path"] == "tiling" and d["sets"] == 230 and d["classes"] == 3
    assert d["bound_certified"] == 230 and d["exact_evaluated"] == 0
    # Input16 members: certified structurally (sum w(g) = 16), no boundary call; the 6 coordinates (sum w(g) = 112)
    # need the boundary bound, one call per member of the one root instance
    assert d["struct_certified"] == 224 and d["classes_struct_certified"] == 2
    assert B.calls["exact"] == 0 and B.calls["bound"] == 6


def test_width_bound_fails_exact_certifies(gemm, oracle):
    B = oracle(slack=20)                                             # boundary bound = 36 > 32 for every coordinate
    v = validate_width(gemm, family_of(gemm, gemm_vu_tiling(gemm)), 32)
    assert v.status == "ok" and v.max_seen == 16
    assert v.detail["bound_certified"] == 224 and v.detail["exact_evaluated"] == 6 == v.detail["exact_certified"]
    assert B.calls["exact"] == 6 and v.detail["classes_exact"] == 1


def test_width_bound_above_w_max_proves_nothing(gemm, oracle):
    oracle(slack=100, exact_unsupported=True)
    v = validate_width(gemm, family_of(gemm, gemm_vu_tiling(gemm)), 32)
    assert v.status == "unsupported" and v.violators == ()
    assert v.detail["unsupported_count"] == 6 and v.detail["violation_count"] == 0     # the 6 coordinates
    assert "boundary-unsupported" in v.detail["codes"]
    assert v.max_seen == 16                                                             # the Input members


def test_width_violation_exact(gemm, oracle):
    oracle()
    t = tiling(whole_body(gemm.root, {0: members_k(), 1: members_k()}))   # all six coordinates as one set: 96 bits
    v = validate_width(gemm, family_of(gemm, t), 32)
    assert v.status == "violation" and v.max_seen == 96
    assert v.detail["violation_count"] == 1 and v.detail["examples"][0]["value"] == {"w_out": 96}
    assert v.detail["examples"][0]["spec"] == gemm.root.id and v.detail["examples"][0]["node"] == 2
    # the same family without a tiling hint: per-set path, the violator is a family index
    v = validate_width(gemm, family_of(gemm, t, with_tiling=False), 32)
    assert v.status == "violation" and v.violators == (224,) and v.detail["path"] == "sets"
    # a wide enough w_max certifies it
    assert validate_width(gemm, family_of(gemm, t), 96).status == "ok"


def test_width_input_batch_as_one_set_violates(gemm, oracle):
    oracle()
    v = validate_width(gemm, family_of(gemm, tiling(whole_body(gemm.root))), 32)
    assert v.status == "violation" and v.detail["violation_count"] == 3     # x (512 bits), w (3072), coordinates (96)
    assert v.max_seen == 192 * 16
    # the cheap structural bound never certifies these (sum w(g) >= w_out), so the oracle was consulted exactly
    assert v.detail["exact_evaluated"] == 3


def test_width_exact_budget_exhausted_is_unsupported(gemm, oracle):
    class L(Limits):
        max_members_exact = 5

    oracle(slack=20)
    v = validate_width(gemm, family_of(gemm, gemm_vu_tiling(gemm)), 32, L())
    assert v.status == "unsupported" and "limit-members-exact" in v.detail["codes"]
    assert v.detail["exact_evaluated"] == 5 and v.detail["unsupported_count"] == 1


def test_width_multi_part_set_certifies_by_sum_never_refutes(gemm, oracle):
    oracle()
    sc = gemm.circuit.root_scope
    two = StubGateSet((Part(sc, frozenset({(0, 0)})), Part(sc.child(2, 0), frozenset({(1, 0)}))))   # x[0] + one cvt gate
    rest = [StubGateSet((Part(sc, frozenset({(0, m)})),)) for m in range(1, 32)]
    fam = StubFamily([two] + rest)
    assert validate_width(gemm, fam, 32).status == "ok"                  # 16 + 16 <= 32 certifies
    v = validate_width(gemm, fam, 16)
    assert v.status == "unsupported" and v.unsupported == (0,) and "multi-part-exact" in v.detail["codes"]
    assert v.violators == ()


def test_width_sets_path_limit(gemm, oracle):
    class L(Limits):
        max_intervals = 10

    oracle()
    v = validate_width(gemm, family_of(gemm, gemm_vu_tiling(gemm), with_tiling=False), 32, L())
    assert v.status == "unsupported" and "limit-sets" in v.detail["codes"] and v.detail["sets"] == 10
    assert v.detail["unsupported_count"] == 220


def test_width_zip_and_scan_classes(synth, oracle):
    oracle(program=synth)
    step = synth.root.body.nodes[4].fn
    t = tiling(whole_body(synth.root, {0: members_k(), 1: ("zip", "g"), 2: ("zip", "g"), 4: ("descend", whole_body(step))}))
    v = validate_width(synth, family_of(synth, t), 32)
    # zip member m: F32Mul[m] feeds only F32Add[m] (inside) -> out = {F32Add[m]} = 32; scan iteration gates: 32 each
    assert v.status == "ok" and v.max_seen == 32 and v.detail["classes"] == 5
    v = validate_width(synth, family_of(synth, t), 31)
    assert v.status == "violation"
    # whole nodes: the scan as one set returns carry + 4 ys = 160 bits; the Input and the two batches escape 128 each
    v = validate_width(synth, family_of(synth, tiling(whole_body(synth.root))), 64)
    assert v.status == "violation" and v.max_seen == 160 and v.detail["violation_count"] == 4


def test_width_tiling_with_problems_uses_per_set_path(gemm, oracle):
    oracle()
    t = tiling(body_tiling(gemm.root, {0: members_k(), 2: members_k()}))   # node 1 unassigned
    v = validate_width(gemm, family_of(gemm, t), 32)
    assert v.detail["path"] == "sets" and v.status == "ok"


# -- readiness ----------------------------------------------------------------------------------------------


def test_readiness_ok_on_gemm(gemm):
    r = readiness_32bit(gemm)
    assert isinstance(r, ReadinessReport) and r.ok and r.violators == () and r.max_w == 32 and r.specs == 3


def test_readiness_lists_64bit_input_and_primitive():
    P_ = wide_program()
    r = readiness_32bit(P_)
    assert not r.ok and r.max_w == 64 and r.gates_over == 2
    prims = sorted((v.prim, v.form, v.members, v.instances) for v in r.violators)
    assert prims == [("Input64_v1", "batch", 1, 1), ("QpartWide64_v1", "primitive", 1, 1)]
    assert all(v.spec == P_.root.id for v in r.violators)
    # 64 is a parameter: at w_max = 64 the same Program is ready
    assert gate_width_readiness(P_, 64).ok


def test_readiness_gumbel_const64():
    G = pytest.importorskip("verity.ml.definitions.registry.sampling")
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.program import Program
    P_ = Program(bind(G.GumbelArgmaxBf16, V=8, SEED=12345, POS=3))
    r = readiness_32bit(P_)
    assert not r.ok
    assert sorted(v.prim for v in r.violators) == ["Const64[0x0000000000000003]_v1", "Const64[0x0000000000003039]_v1"]
    assert r.gates_over == 2


def test_readiness_counts_instances_by_multiplicity(gemm):
    # every AmpereBF16TcDot16 (32 bits) sits in DotBf16, instantiated 6 times: at w_max = 16 the count is 6 * 2 + 6 (Const32) + 6 (F2fpBf16? no: 16 bits)
    r = gate_width_readiness(gemm, 16)
    over = {(v.prim.split("_")[0], v.instances) for v in r.violators}
    assert ("Const32[0x00000000]", 6) in over and ("AmpereBF16TcDot16", 6) in over
    assert r.gates_over == 6 * 3   # Const32 + 2 Ampere steps per coordinate; F2fpBf16 (16) and Input16 are fine


# -- composition -------------------------------------------------------------------------------------------


def test_vu_verdict_ok(gemm, oracle):
    oracle()
    v = validate_vu_family(gemm, family_of(gemm, gemm_vu_tiling(gemm)), 32)
    assert isinstance(v, VuVerdict) and v.status == "ok" and v.count == 230
    assert v.partition.status == "exact" and v.width.status == "ok" and v.readiness_32.ok
    j = v.to_json()
    assert j["status"] == "ok" and j["count"] == 230 and j["partition"]["status"] == "exact"
    import json
    json.dumps(j)  # JSON-serialisable


def test_vu_verdict_fail_and_not_ready_and_unsupported(gemm, oracle):
    oracle()
    t = tiling(whole_body(gemm.root, {0: members_k(), 1: members_k()}))
    assert validate_vu_family(gemm, family_of(gemm, t), 32).status == "fail"           # width violation
    t = tiling(body_tiling(gemm.root, {0: members_k(), 2: members_k()}))
    assert validate_vu_family(gemm, family_of(gemm, t), 32).status == "fail"           # gap
    W = wide_program()
    oracle(program=W)
    v = validate_vu_family(W, family_of(W, tiling(whole_body(W.root))), 64)
    assert v.status == "ok" and v.count == 5                                            # a 64-bit protocol
    v = validate_vu_family(W, family_of(W, tiling(whole_body(W.root))), 32)
    assert v.status == "fail" and not v.readiness_32.ok                                 # the returned 64-bit gate escapes
    # the same Program where the 64-bit gates do not escape any set: readiness alone objects
    fam = StubFamily([StubGateSet((Part(W.circuit.root_scope, frozenset({(0, 0), (1, 0)})),)),
                      StubGateSet((Part(W.circuit.root_scope, frozenset({(2, 0), (3, 0), (4, 0)})),))])
    v = validate_vu_family(W, fam, 64)
    assert v.status == "ok"
    v = validate_vu_family(W, fam, 96)
    assert v.status == "ok"
    from verity.ml.circuits.partition import gate_width_readiness
    assert not gate_width_readiness(W, 32).ok

    class L(Limits):
        max_intervals = 2

    oracle()
    v = validate_vu_family(gemm, family_of(gemm, gemm_vu_tiling(gemm), with_tiling=False), 32, L())
    assert v.status == "unsupported" and v.partition.status == "unsupported"


def test_vu_status_precedence():
    from verity.ml.circuits.partition import PartitionVerdict
    ok_p = PartitionVerdict("exact", (), {})
    gap_p = PartitionVerdict("gap", (), {})
    un_p = PartitionVerdict("unsupported", (), {})
    ok_w = WidthVerdict(32, 16, (), (), "ok")
    bad_w = WidthVerdict(32, 96, (0,), (), "violation")
    un_w = WidthVerdict(32, None, (), (0,), "unsupported")
    ready = ReadinessReport(32, True, (), 32, 1, 0)
    unready = ReadinessReport(32, False, (), 64, 1, 1)
    assert vu_status(ok_p, ok_w, ready) == "ok"
    assert vu_status(gap_p, ok_w, ready) == "fail"
    assert vu_status(ok_p, bad_w, unready) == "fail"
    assert vu_status(ok_p, ok_w, unready) == "not-ready"
    assert vu_status(un_p, ok_w, unready) == "not-ready"
    assert vu_status(un_p, ok_w, ready) == "unsupported"
    assert vu_status(ok_p, un_w, ready) == "unsupported"
