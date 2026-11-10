"""`pouw_h1_records.reason` reads in full only the row blocks that hold a served row and accepts exactly what the whole read
(`_reason`) accepts; every records it refuses get the whole read's reason, over calls of several row blocks, honest and
forged on the served blocks and on the empty ones."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
E = pytest.importorskip("verity_vllm.engine.int8_exact")
ZR = pytest.importorskip("verity_vllm.protocols.pouw.h1.records")

K, N, ROWS = 64, 96, 128
HALF = 4 * 32 * 32


def _call(m: int, seed: int = 0, s_w=None):
    """(records, y bits [m, N], s_a [m], s_w [N]) of an honest call of m rows over ROWS slots."""
    g = torch.Generator().manual_seed(seed)
    xq = torch.randint(-127, 128, (m, K), dtype=torch.int8, generator=g)
    w = torch.randint(-127, 128, (N, K), dtype=torch.int8, generator=g)
    s_a = torch.rand(m, generator=g) * 0.02
    s_w = torch.rand(N, generator=g) * 0.01 if s_w is None else torch.as_tensor(s_w, dtype=torch.float32)
    y, zy = E.scaled_mm_zy(xq, s_a, w, s_w, ROWS)
    return zy.numpy().copy(), y.view(torch.int16).numpy().copy(), s_a.numpy(), s_w.numpy()


def _both(records, y, s_a, s_w):
    got, want = ZR.reason(records, ROWS, N, y, s_a, s_w), ZR._reason(records, ROWS, N, y, s_a, s_w)
    assert got == want
    assert want is None or not ZR._holds(records, ROWS, N, y, s_a, s_w)
    return got


@pytest.mark.parametrize("m", [0, 1, 2, 31, 32, 33, 64, 100, ROWS])
def test_honest_records_hold_on_every_row_block_count(m):
    call = _call(m)
    assert ZR._holds(call[0], ROWS, N, *call[1:])
    assert _both(*call) is None


def _units(m: int) -> tuple[list[int], list[int]]:
    """The records of the row blocks holding a served row, and the rest."""
    cols, served = N // 32, -(-m // 32)
    return list(range(served * cols)), list(range(served * cols, ROWS // 32 * cols))


@pytest.mark.parametrize("m", [0, 1, 33, 100])
@pytest.mark.parametrize("half", ["z", "y"])
@pytest.mark.parametrize("where", ["served", "empty"])
def test_a_flipped_bit_anywhere_gets_the_whole_reads_verdict(m, half, where):
    """A flipped bit is refused wherever it is, except a served slot's z bit whose dq rounds to the same y, which no
    value check sees: there the verdict is still the whole read's."""
    rng = np.random.default_rng(m * 7 + len(half + where))
    units = _units(m)[0 if where == "served" else 1]
    if not units:
        pytest.skip("no row block of that kind at this m")
    for _ in range(6):
        records, y, s_a, s_w = _call(m, seed=int(rng.integers(1 << 30)))
        u = int(rng.choice(units))
        off = int(rng.integers(HALF)) if half == "z" else HALF + int(rng.integers(ZR.ZY_BYTES - HALF))
        records[u, off] ^= np.uint8(1 << int(rng.integers(8)))
        got = _both(records, y, s_a, s_w)
        assert got is not None or (where, half) == ("served", "z")


@pytest.mark.parametrize("m", [1, 33])
def test_an_empty_slot_with_a_z_whose_y_is_redone_is_refused(m):
    """An empty slot's z set and its y redone as dq of it at scale 0: every y is dq of its z, so only the empty-slot rule
    refuses it, on a served block and on an empty one."""
    for row in (m, ROWS - 1):
        records, y, s_a, s_w = _call(m)
        z, yr = E.zy_split(torch.from_numpy(records), ROWS, N)
        z[row, 5] = -(1 << 20)
        sa = torch.zeros(ROWS)
        sa[:m] = torch.from_numpy(s_a)
        records = E.zy_records(z, E.dequantize(z, sa, torch.from_numpy(s_w))).numpy()
        assert "empty slot has a z" in _both(records, y, s_a, s_w)


def test_a_served_y_that_is_not_the_records_is_refused():
    records, y, s_a, s_w = _call(3)
    y[2, 7] ^= 1
    assert "the served y is not" in _both(records, y, s_a, s_w)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_weight_scales_take_the_whole_read(bad):
    s_w = np.full(N, 0.003, np.float32)
    s_w[40] = bad
    records, y, s_a, s_w = _call(2, s_w=s_w)
    assert not ZR._holds(records, ROWS, N, y, s_a, s_w)
    _both(records, y, s_a, s_w)


def test_negative_and_zero_weight_scales_hold():
    s_w = np.full(N, 0.003, np.float32)
    s_w[::3], s_w[1::7] = -0.004, 0.0
    call = _call(5, s_w=s_w)
    assert ZR._holds(call[0], ROWS, N, *call[1:])
    assert _both(*call) is None


def test_shapes_that_are_not_a_calls_take_the_whole_read():
    records, y, s_a, s_w = _call(4)
    assert not ZR._holds(records[:-1], ROWS, N, y, s_a, s_w)
    assert not ZR._holds(records, ROWS, N, y[:3], s_a, s_w)
    with pytest.raises(ValueError):
        ZR.reason(records, ROWS, N, y[:3], s_a, s_w)
