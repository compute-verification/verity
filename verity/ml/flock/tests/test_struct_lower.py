"""`verity_flock.struct_lower`: the structural template is ordered and unrolls to a unit computing what the flat lowering
computes."""
from __future__ import annotations

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import gemm as GB
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
