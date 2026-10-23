"""`verity_flock.struct_lower`: the structural template is ordered and unrolls to a unit computing what the flat lowering
computes."""
from __future__ import annotations

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import gemm as GB
from verity.ml.circuits.ops.boolean import scalar as SB
from verity.ml.circuits.refs import array_of
from verity.ml.circuits.types import Array
from verity_flock import struct_lower as S
from verity_flock import tmpl as T


def test_a_scan_is_one_run_and_unrolls_to_the_flat_lowering():
    t = bind(GB.GemmAccumulatorS8, K=128, DOT=GB.BlackwellS8ImmaDot32)
    assert S.check(t)["mismatched_bits"] == 0
    tm, _lw, _top = S.lower(t)
    assert T.ordered(tm)["ok"]
    assert max(c.count for d in tm.defs for c in d.calls) == 3


def _repeats(B, _S, acc, x):
    return array_of([B.call(GB.BlackwellS8ImmaDot32, acc, x, array_of([x[j]] * 32)) for j in (0, 1)])


REPEATS = CompositeDefinition("StructLowerRepeats", 1, (), lambda _S: ((("acc", GB.F32), ("x", GB.A32_8)), Array(2, GB.F32)),
                              _repeats, register=False)


def test_calls_repeating_different_arguments_are_different_specializations():
    """Both calls pass the same distinct bits in the same order; only which of x's leaves fills b differs."""
    assert S.check(REPEATS)["mismatched_bits"] == 0
    _tm, lw, _top = S.lower(REPEATS)
    assert sum(k[0].startswith("BlackwellS8ImmaDot32") for k in lw.specs) == 2


def _constants(B, _S, x):
    return array_of([B.call(GB.BlackwellS8ImmaDot32, B.call(SB.const_word(32, v)), x, x) for v in (7, 0, 1, 0x3F800000)])


CONSTANTS = CompositeDefinition("StructLowerConstants", 1, (), lambda _S: ((("x", GB.A32_8),), Array(4, GB.F32)),
                                _constants, register=False)


def test_calls_past_the_fold_cap_share_one_specialization_with_constants_as_inputs():
    """The first call's constant accumulator is folded in; the other three bind its bits to the unit's constant and a
    zero column, the second and third side by side, so a run must not step a bit from the zero column to the constant."""
    fold = {"fold_cap": 1, "fold_gates": 0}
    assert S.check(CONSTANTS, **fold)["mismatched_bits"] == 0
    tm, lw, _top = S.lower(CONSTANTS, **fold)
    assert T.ordered(tm)["ok"]
    assert sum(k[0].startswith("BlackwellS8ImmaDot32") for k in lw.specs) == 2
    _tm, lw, _top = S.lower(CONSTANTS, fold_cap=4, fold_gates=0)
    assert sum(k[0].startswith("BlackwellS8ImmaDot32") for k in lw.specs) == 4


def test_a_template_evaluates_to_its_unrolled_rows_outputs():
    """`tmpl.evaluate`, which never unrolls: on 192 random instances, the scan's output bits are those `evaluate` gives on
    its unrolled rows, and the same under a budget small enough to split the lanes and step a run's copies in chunks."""
    import numpy as np

    tm, _lw, _top = S.lower(bind(GB.GemmAccumulatorS8, K=128, DOT=GB.BlackwellS8ImmaDot32))
    flat = T.unroll(tm)
    nin = sum(hi - lo for lo, hi in T._ports(tm)[1])
    lanes = np.random.default_rng(1).integers(0, 1 << 64, size=(nin, 3), dtype=np.uint64)
    outs = np.concatenate([np.arange(lo, hi) for lo, hi in T.out_ports(tm)])
    want = np.array([[S.evaluate(flat, [int(x) for x in lanes[:, w]], nin)[c] for w in range(3)] for c in outs], np.uint64)
    assert np.array_equal(T.evaluate(tm, lanes), want)
    assert np.array_equal(T.evaluate(tm, lanes, budget=1 << 12), want)
    bits = np.random.default_rng(2).integers(0, 2, size=(130, 7), dtype=np.uint8)
    assert np.array_equal(T.unslice_lanes(T.slice_lanes(bits), 130), bits)


def test_a_definition_two_calls_read_different_columns_of_is_evaluated_for_each():
    """Two calls of one two-column Definition (an AND and an XOR of its inputs), the first read for its AND, the second
    for its XOR: each call gets the columns it reads, as `evaluate` on the unrolled rows gives them."""
    import numpy as np

    K = 385
    rows = [(i, (i,), (i,)) for i in range(4)] + [(256, (128,), (K,)), (257, (131,), (K,))]
    calls = [T.Call(0, 1, 128, 2, ((2, 0, 0, 1, 0),)), T.Call(0, 1, 130, 2, ((2, 2, 0, 1, 0),))]
    t = T.Tmpl("t", 1, [str(x) for x in (1, 0, 1, 1, 4, 1, 2, 1, 1, 2)],
               [T.Def(2, 2, ((0, (2,), (3,)), (1, (2, 3), (4,))), ()), T.Def(K, 0, tuple(rows), tuple(calls))])
    lanes = np.random.default_rng(3).integers(0, 1 << 64, size=(4, 2), dtype=np.uint64)
    flat = T.unroll(t)
    want = [[S.evaluate(flat, [int(x) for x in lanes[:, w]], 4)[c] for w in range(2)] for c in (256, 257)]
    assert np.array_equal(T.evaluate(t, lanes), np.array(want, np.uint64))
    assert want[0][0] == int(lanes[0, 0] & lanes[1, 0]) and want[1][0] == int(lanes[2, 0] ^ lanes[3, 0])
