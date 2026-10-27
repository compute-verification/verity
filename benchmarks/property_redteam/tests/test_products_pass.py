"""Data products on property/v1's pass (`products_pass.py`): a step on two run-time multiplicands is a data product lane
by lane, whatever their range, a step with a registered or constant multiplicand isn't, and of the honest exact-int8
Definitions only the attention row has them, its `Q′K′ᵀ` and `P V′`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cases as C  # noqa: E402
import products_pass as T  # noqa: E402
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
    return C._fn(f"ProductsStep{pad}", [("a", C.A32), ("b", C.A32)], C.S32, body)


@pytest.mark.parametrize("dyn, ranges, want", [
    ((True, True), (BITS, BYTE), 32),    # an AND-like select: a two-valued run-time lane times a run-time byte
    ((True, True), (BITS, BITS), 32),    # an AND of two run-time bits
    ((True, True), (BYTE, BYTE), 32),    # an s8 x s8 lane on two data values: priced whatever its range
    ((True, True), ((0, 2), BYTE), 32),
    ((True, True), (None, None), 32),    # any word
    ((True, False), (BITS, BYTE), 0),    # a registered multiplicand: a linear map of the other
    ((False, True), (BYTE, BITS), 0),
])
def test_a_step_on_two_run_time_multiplicands_is_a_product_a_lane(dyn, ranges, want):
    c = T.census(step(), dyn, ranges)
    assert c["products"] == want and c["family"] == 1 and c["macs"] == 32
    assert c["units"] == {k: P.count(step(), dyn, ranges).units.get(k, 0) for k in P.BUDGETED}
    assert c["nonlinear"] == sum(c["units"].values()) + want


def test_products_are_read_lane_by_lane():
    """A constant lane is not run-time, so it is free; the rest of the step is priced whatever its range."""
    assert T.census(step(pad=5), (True, True), (BITS, BYTE))["products"] == 27
    assert T.census(step(pad=5), (True, True), (BYTE, BYTE))["products"] == 27
    assert T.census(step(pad=5), (True, False), (BYTE, BYTE))["products"] == 0


def test_honest_int8_products():
    """`Int8Linear_v1`, `Int8KvQuant_v1` and `Int8LutAttentionRow_v1` at `SHAPES`: nothing refused and property/v1's own
    class counts. The linear's MACs are activations by registered weights, a linear map, so it has no data product;
    the KV quantizers have no family step; the attention row's `Q′K′ᵀ` and `P V′` are `S·D` data products each."""
    got = T.int8_census()
    assert len(got) == 3
    for name, c in got.items():
        assert not c["refused"], name
        assert c["units"] == c["property_v1"], name
        assert c["nonlinear"] == sum(c["units"].values()) + c["products"], name
    (lin, kv, row), shapes = got.values(), [s for _, s, _ in T.SHAPES]
    K, N = shapes[0]["K"], shapes[0]["N"]
    D, S = shapes[2]["D"], shapes[2]["S"]
    assert lin["macs"] == K * N and lin["family"] == K * N // 32 and lin["products"] == 0
    assert kv["macs"] == 0 and kv["products"] == 0
    assert S % 32 == 0 and row["macs"] == 2 * S * D and row["products"] == 2 * S * D
