"""Thin lanes on property/v1's pass (`thin_pass.py`): a step on run-time bits is thin lane by lane, a step with a
registered or wide multiplicand isn't, and the honest exact-int8 Definitions have none, so the repricing leaves their
class counts as property/v1 has them."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cases as C  # noqa: E402
import thin_pass as T  # noqa: E402
from verity.ml.circuits import stacking as P  # noqa: E402
from verity.ml.circuits.ops import prims  # noqa: E402
from verity.ml.circuits.refs import array_of  # noqa: E402

BITS, BYTE = (0, 1), (-128, 127)


def step(pad: int = 0):
    """`IMMA(0, a, b)` over two parameters of 32 s8 lanes, the last `pad` lanes of `a` replaced by the constant 0."""
    def body(B, _S, a, b):
        k = C._Consts(B)
        lanes = [a[i] for i in range(32 - pad)] + [k.word(8, 0)] * pad
        return B.call(C.IMMA, B.call(prims.ZERO32), array_of(lanes) if pad else a, b)
    return C._fn(f"ThinStep{pad}", [("a", C.A32), ("b", C.A32)], C.S32, body)


@pytest.mark.parametrize("dyn, ranges, want", [
    ((True, True), (BITS, BYTE), 32),   # an AND-like select: a two-valued run-time lane times a run-time byte
    ((True, True), (BITS, BITS), 32),   # an AND of two run-time bits
    ((True, True), ((5, 6), BYTE), 32),  # two values anywhere, not only {0, 1}
    ((True, False), (BITS, BYTE), 0),   # a registered multiplicand: an affine map of the other
    ((False, True), (BYTE, BITS), 0),
    ((True, True), (BYTE, BYTE), 0),    # an honest s8 x s8 lane
    ((True, True), ((0, 2), BYTE), 0),  # three values
])
def test_a_step_on_two_valued_run_time_lanes_is_thin(dyn, ranges, want):
    c = T.census(step(), dyn, ranges)
    assert c["thin"] == want and c["family"] == 1 and c["macs"] == 32
    assert c["units"] == {k: P.count(step(), dyn, ranges).units.get(k, 0) for k in P.BUDGETED}


def test_thin_is_read_lane_by_lane():
    """A constant lane is not run-time, so it is not thin even though it holds one value."""
    assert T.census(step(pad=5), (True, True), (BITS, BYTE))["thin"] == 27
    assert T.census(step(pad=5), (True, True), (BYTE, BYTE))["thin"] == 0


def test_honest_int8_has_no_thin_lane():
    """`Int8Linear_v1`, `Int8KvQuant_v1` and `Int8LutAttentionRow_v1` at `SHAPES`: no thin lane, nothing refused, and
    property/v1's own class counts. The linear and the row's MACs are one a byte of what they stream per token row: the
    linear's `K·N` weights, the row's `S·D` of K′ and of V′."""
    got = T.int8_census()
    assert len(got) == 3
    for name, c in got.items():
        assert c["thin"] == 0 and not c["refused"], name
        assert c["units"] == c["property_v1"], name
    (lin, kv, row), shapes = got.values(), [s for _, s, _ in T.SHAPES]
    K, N = shapes[0]["K"], shapes[0]["N"]
    D, S = shapes[2]["D"], shapes[2]["S"]
    assert lin["macs"] == K * N and lin["family"] == K * N // 32
    assert kv["macs"] == 0
    assert S % 32 == 0 and row["macs"] == 2 * S * D
