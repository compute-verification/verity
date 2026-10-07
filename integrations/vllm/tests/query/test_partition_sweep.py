"""`validate_partition` fallback (tier B): the interval sweep over materialised intervals -- overlap, gap, exact,
unsupported-by-limit on synthetic interval families; then the same sweep over real Programs' Parts."""
from __future__ import annotations

import pytest

from verity.ml.circuits.parts import Limits, Part
from verity.ml.circuits.partition import PartitionVerdict, validate_partition
from tests.query.test_partition_stubs import (FakeProgram, IntervalSet, StubFamily, StubGateSet, family_of,
                                                  gemm_program, members_k, synth_program, tiling, whole_body)


def fam(*sets: tuple[tuple[int, int], ...]) -> StubFamily:
    return StubFamily([IntervalSet(tuple(s)) for s in sets])


def test_exact_partition():
    v = validate_partition(FakeProgram(10), fam(((0, 3),), ((3, 10),)))
    assert isinstance(v, PartitionVerdict)
    assert v.status == "exact" and v.codes == ()
    assert v.detail["path"] == "sweep" and v.detail["sets"] == 2 and v.detail["intervals"] == 2 and v.detail["gates"] == 10


def test_exact_partition_out_of_order_and_multi_interval_sets():
    # zip-like sets: {[0,2),[4,6)} and {[2,4),[6,8)}, given in reverse order
    v = validate_partition(FakeProgram(8), fam(((6, 8), (2, 4)), ((4, 6), (0, 2))))
    assert v.status == "exact"


def test_overlap():
    v = validate_partition(FakeProgram(10), fam(((0, 5),), ((3, 10),)))
    assert v.status == "overlap" and v.codes == ("sweep-overlap",)
    assert v.detail["overlaps"] == [{"gates": [3, 5], "sets": [0, 1]}]
    assert v.detail["overlap_count"] == 1


def test_gap_interior_and_trailing():
    v = validate_partition(FakeProgram(10), fam(((0, 3),), ((5, 8),)))
    assert v.status == "gap" and v.codes == ("sweep-gap",)
    assert v.detail["gaps"] == [{"gates": [3, 5]}, {"gates": [8, 10]}]
    assert v.detail["gap_count"] == 2


def test_leading_gap():
    v = validate_partition(FakeProgram(4), fam(((1, 4),)))
    assert v.status == "gap" and v.detail["gaps"] == [{"gates": [0, 1]}]


def test_overlap_wins_over_gap_both_codes_listed():
    v = validate_partition(FakeProgram(10), fam(((0, 4),), ((2, 6),), ((8, 10),)))
    assert v.status == "overlap"
    assert v.codes == ("sweep-overlap", "sweep-gap")
    assert v.detail["gaps"] == [{"gates": [6, 8]}]


def test_duplicate_set_is_overlap():
    v = validate_partition(FakeProgram(4), fam(((0, 4),), ((0, 4),)))
    assert v.status == "overlap" and v.detail["overlaps"][0]["gates"] == [0, 4]


def test_nested_interval_overlap():
    v = validate_partition(FakeProgram(10), fam(((0, 10),), ((3, 5),)))
    assert v.status == "overlap" and v.detail["overlaps"] == [{"gates": [3, 5], "sets": [0, 1]}]


def test_empty_family_is_a_gap():
    v = validate_partition(FakeProgram(3), fam())
    assert v.status == "gap" and v.detail["gaps"] == [{"gates": [0, 3]}]


def test_zero_gate_program_with_empty_family_is_exact():
    assert validate_partition(FakeProgram(0), fam()).status == "exact"


def test_empty_set_is_reported_but_does_not_break_partitionhood():
    v = validate_partition(FakeProgram(4), fam(((0, 4),), ()))
    assert v.status == "exact" and v.codes == ("empty-set",)
    assert v.detail["empty_sets"] == [1] and v.detail["empty_count"] == 1


def test_unsupported_by_limit_is_never_a_verdict():
    class L(Limits):
        max_intervals = 3

    sets = [((i, i + 1),) for i in range(5)]
    v = validate_partition(FakeProgram(5), fam(*sets), L())
    assert v.status == "unsupported" and v.codes == ("limit-intervals",)
    assert v.detail["max_intervals"] == 3 and v.detail["intervals_seen"] == 4
    # the same family, under the default limits, is exact
    assert validate_partition(FakeProgram(5), fam(*sets)).status == "exact"
    # and an overlapping family beyond the limit is also unsupported, not "overlap"
    bad = [((0, 5),)] + sets
    assert validate_partition(FakeProgram(5), fam(*bad), L()).status == "unsupported"


def test_interval_outside_program_is_an_error_not_a_verdict():
    with pytest.raises(ValueError):
        validate_partition(FakeProgram(4), fam(((0, 5),)))
    with pytest.raises(ValueError):
        validate_partition(FakeProgram(4), fam(((-1, 2),)))


def test_many_intervals_sweep_is_fast_enough():
    n = 20_000
    sets = [((2 * i, 2 * i + 1), (2 * i + 1, 2 * i + 2)) for i in range(n)]
    v = validate_partition(FakeProgram(2 * n), fam(*sets))
    assert v.status == "exact" and v.detail["intervals"] == 2 * n


# -- the sweep over real Parts -------------------------------------------------------------------------


def test_sweep_over_gemm_parts_exact_and_gap():
    P_ = gemm_program()
    sc = P_.circuit.root_scope
    sets = [StubGateSet((Part(sc, frozenset({(0, m)})),)) for m in range(32)]          # every Input16 x gate
    sets += [StubGateSet((Part(sc, frozenset({(1, m) for m in range(192)})),))]        # w as one set
    sets += [StubGateSet((Part(sc, frozenset({(2, m)})),)) for m in range(6)]          # each GemmCoordinate
    v = validate_partition(P_, StubFamily(sets))
    assert v.status == "exact" and v.detail["sets"] == 39 and v.detail["intervals"] == 39
    # drop one coordinate -> a 4-gate gap at that member's interval
    v = validate_partition(P_, StubFamily(sets[:-1]))
    assert v.status == "gap" and v.detail["gaps"] == [{"gates": [244, 248]}]


def test_sweep_over_nested_parts_and_inner_overlap():
    P_ = gemm_program()
    sc = P_.circuit.root_scope
    inner = sc.child(2, 3)                                        # GemmCoordinate#3
    dot = inner.child(0, 0)                                       # its DotBf16 call
    sets = [StubGateSet((Part(sc, frozenset({(0, m) for m in range(32)})),)),
            StubGateSet((Part(sc, frozenset({(1, m) for m in range(192)})),))]
    sets += [StubGateSet((Part(sc, frozenset({(2, m)})),)) for m in range(6) if m != 3]
    sets += [StubGateSet((Part(dot, frozenset({(j, 0)})),)) for j in range(3)]         # Const32, 2 Ampere steps
    sets += [StubGateSet((Part(inner, frozenset({(1, 0)})),))]                        # F2fpBf16
    assert validate_partition(P_, StubFamily(sets)).status == "exact"
    # the whole coordinate again on top of its pieces -> overlap inside the nested body
    v = validate_partition(P_, StubFamily(sets + [StubGateSet((Part(sc, frozenset({(2, 3)})),))]))
    assert v.status == "overlap" and v.detail["overlap_count"] >= 1


def test_sweep_over_synth_zip_and_scan_iterations():
    P_ = synth_program(4)
    sc = P_.circuit.root_scope
    sets = [StubGateSet((Part(sc, frozenset({(0, m)})),)) for m in range(4)]
    sets += [StubGateSet((Part(sc, frozenset({(1, m), (2, m)})),)) for m in range(4)]  # zip of the two batches
    sets += [StubGateSet((Part(sc, frozenset({(3, 0)})),))]
    sets += [StubGateSet((Part(sc, frozenset({(4, m)})),)) for m in range(4)]          # scan iterations
    v = validate_partition(P_, StubFamily(sets))
    assert v.status == "exact" and v.detail["intervals"] == 4 + 8 + 1 + 4


def test_sweep_matches_materialised_tiling_without_tiling_hint():
    P_ = gemm_program()
    t = tiling(whole_body(P_.root, {0: members_k(), 2: members_k()}))
    v = validate_partition(P_, family_of(P_, t, with_tiling=False))
    assert v.status == "exact" and v.detail["path"] == "sweep" and v.detail["sets"] == 32 + 1 + 6
