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
