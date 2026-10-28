"""The vectorised twins of the block-scaled FP8 families (`derived_rows.tc_dot32_e4m3`, `fp8_group_quant_row`,
`scaled_mm_fp8_block_row`) are word-for-word the registered Definitions (`verity/ml/definitions/registry/fp8.py`) and the saved H100 kernel
words (`data/fp8_block_operands.npz`: `_C.per_token_group_fp8_quant` + blockwise `_C.cutlass_scaled_mm`, on an H100).
Python >= 3.12 (`veritor.core.silicon_total_fp8`, the scalar QGMMA step)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 12):  # pragma: no cover
    pytest.skip("veritor.core needs Python 3.12", allow_module_level=True)

from verity.ml.circuits import Program, bind
from verity.ml.definitions.numerics import derived_rows as D
from verity.ml.definitions.registry import fp8 as F

from tests.program.test_fp8 import _evaluate_returns

DATA = Path(__file__).parent / "data"


def _rand_e4m3(rng, n: int, p_zero: float = 0.1, p_sub: float = 0.1) -> np.ndarray:
    b = rng.integers(0, 256, n, dtype=np.int64)
    b = np.where((b & 0x7F) == 0x7F, b & ~0x40, b)                       # never a NaN byte
    b = np.where(rng.random(n) < p_zero, b & 0x80, b)
    b = np.where(rng.random(n) < p_sub, b & 0x87, b)                      # exponent 0: subnormal / zero
    return b.astype(np.uint8)


# ------------------------------------------------------------------------------------------------------------ the k32 step

def test_tc_dot32_e4m3_matches_the_total_model_on_random_operands_and_accumulators() -> None:
    from verity.ml.circuits.silicon.total_fp8 import HOPPER_E4M3_WGMMA_K32 as PIPE, tc_dot_total_e4m3

    rng = np.random.default_rng(7)
    n = 4000
    a, b = _rand_e4m3(rng, n * 32).reshape(n, 32), _rand_e4m3(rng, n * 32).reshape(n, 32)
    e = rng.integers(-140, 30, n)
    acc = ((rng.random(n) * 2 - 1) * np.exp2(e.astype(np.float64))).astype(np.float32)
    acc[rng.random(n) < 0.05] = 0.0
    acc_w = acc.view(np.uint32)
    got, nf = D.tc_dot32_e4m3(acc_w, a, b)
    assert not nf.any()
    bad = [i for i in range(n) if tc_dot_total_e4m3(PIPE, int(acc_w[i]), a[i].tolist(), b[i].tolist()) != int(got[i])]
    assert bad == []


def test_tc_dot32_e4m3_cancellation_and_subnormal_accumulators() -> None:
    """The 14-bit adder's truncation of the ACCUMULATOR (`acc + 0 != acc`) and the denormalisation path below 2^-126."""
    from verity.ml.circuits.silicon.total_fp8 import HOPPER_E4M3_WGMMA_K32 as PIPE, tc_dot_total_e4m3

    rng = np.random.default_rng(3)
    n = 2000
    a, b = _rand_e4m3(rng, n * 32).reshape(n, 32), _rand_e4m3(rng, n * 32).reshape(n, 32)
    first, _ = D.tc_dot32_e4m3(np.zeros(n, dtype=np.uint32), a, b)
    neg = first ^ np.uint32(0x80000000)                                    # acc = -(first step): the cancellation path
    got, _ = D.tc_dot32_e4m3(neg, a, b)
    assert all(tc_dot_total_e4m3(PIPE, int(neg[i]), a[i].tolist(), b[i].tolist()) == int(got[i]) for i in range(n))
    sub = rng.integers(1, 0x7FFFFF, n).astype(np.uint32) | (rng.integers(0, 2, n).astype(np.uint32) << 31)
    got, _ = D.tc_dot32_e4m3(sub, a, b)
    assert all(tc_dot_total_e4m3(PIPE, int(sub[i]), a[i].tolist(), b[i].tolist()) == int(got[i]) for i in range(n))
    zero_step, _ = D.tc_dot32_e4m3(np.array([0x3F800001], dtype=np.uint32), np.zeros((1, 32), np.uint8), np.zeros((1, 32), np.uint8))
    assert int(zero_step[0]) == 0x3F800000                                # 1 + 2^-23 through the 14-bit adder: the low bits are gone


def test_tc_dot32_e4m3_flags_nan_bytes_and_nonfinite_accumulators() -> None:
    a = np.zeros((3, 32), dtype=np.uint8)
    b = np.zeros((3, 32), dtype=np.uint8)
    a[1, 5] = 0x7F                                                          # e4m3 NaN
    acc = np.array([0, 0, 0x7F800000], dtype=np.uint32)                     # lane 2: +inf accumulator
    _got, nf = D.tc_dot32_e4m3(acc, a, b)
    assert nf.tolist() == [False, True, True]


# ------------------------------------------------------------------------------------------------- the families vs the kernel words

@pytest.mark.parametrize("tag", ["p", "c"])
def test_fp8_group_quant_row_matches_the_kernel_words(tag: str) -> None:
    d = np.load(DATA / "fp8_block_operands.npz")
    x, xq, xs = d[f"{tag}_x"], d[f"{tag}_xq"], d[f"{tag}_xs"]
    for m in range(x.shape[0]):
        q, s = D.fp8_group_quant_row(x[m], G=F.FP8_BLOCK)
        assert np.array_equal(q, xq[m]) and np.array_equal(s, xs[m]), (tag, m)


@pytest.mark.parametrize("tag", ["p", "c"])
def test_scaled_mm_fp8_block_row_matches_the_kernel_words(tag: str) -> None:
    d = np.load(DATA / "fp8_block_operands.npz")
    xq, xs, w, ws, y = d[f"{tag}_xq"], d[f"{tag}_xs"], d[f"{tag}_w"], d[f"{tag}_ws"], d[f"{tag}_y"]
    for m in range(xq.shape[0]):
        out, nf = D.scaled_mm_fp8_block_row(xq[m], xs[m], w, ws, G=F.FP8_BLOCK)
        assert np.array_equal(out, y[m]) and not nf.any(), (tag, m, int((out != y[m]).sum()))
        # chained through the quant twin from the committed bf16 row (the replay's composition of the FP8 linear)
        q, s = D.fp8_group_quant_row(d[f"{tag}_x"][m], G=F.FP8_BLOCK)
        out2, _ = D.scaled_mm_fp8_block_row(q, s, w, ws, G=F.FP8_BLOCK)
        assert np.array_equal(out2, y[m]), (tag, m)


def test_scaled_mm_fp8_block_row_wrong_scale_moves_only_its_column_block() -> None:
    d = np.load(DATA / "fp8_block_operands.npz")
    xq, xs, w, ws, y = d["p_xq"], d["p_xs"], d["p_w"], d["p_ws"], d["p_y"]
    ws2 = ws.copy()
    ws2[0, 0] = np.uint32(int(ws2[0, 0]) ^ 0x00080000)
    out, _ = D.scaled_mm_fp8_block_row(xq[0], xs[0], w, ws2, G=F.FP8_BLOCK)
    assert not np.array_equal(out[:128], y[0][:128]) and np.array_equal(out[128:], y[0][128:])


# ---------------------------------------------------------------------------------------------- the families vs the Definitions

def test_twins_match_the_definitions_gate_by_gate() -> None:
    rng = np.random.default_rng(11)
    K, N, G = 256, 128, 128
    xf = (rng.standard_normal(K) * np.exp2(rng.integers(-8, 6, K))).astype(np.float32)
    xf[3], xf[100] = 0.0, -0.0
    x = (xf.view(np.uint32) >> 16).astype(np.uint16)
    x[7] = 0x7FC0                                                             # a bf16 NaN: fmaxf drops the quotient -> -448 -> 0xFE
    ref = _evaluate_returns(Program(bind(F.Fp8GroupQuant, K=K, G=G)), [x])
    xq_ref, sx_ref = np.array(ref[:K], dtype=np.uint8), np.array(ref[K:], dtype=np.uint32)
    q, s = D.fp8_group_quant_row(x, G=G)
    assert np.array_equal(q, xq_ref) and np.array_equal(s, sx_ref)
    assert int(q[7]) == 0xFE
    w = _rand_e4m3(rng, N * K).reshape(N, K)
    ws = np.exp2(rng.uniform(-12, -2, (N // G, K // G))).astype(np.float32).view(np.uint32)
    y_ref = np.array(_evaluate_returns(Program(bind(F.ScaledMmFp8Block, K=K, N=N, G=G)), [xq_ref, sx_ref, w, ws]), dtype=np.uint16)
    y, nf = D.scaled_mm_fp8_block_row(xq_ref, sx_ref, w, ws, G=G)
    assert np.array_equal(y, y_ref) and not nf.any()


def test_promotion_is_ffma_ascending_in_the_twin() -> None:
    """The adversarial case of test_fp8.test_fp8_block_promotion_is_ffma_ascending, through the vectorised row."""
    one, onehalf = 0x38, 0x3C
    K, G = 256, 128
    xq = np.zeros(K, dtype=np.uint8)
    xq[0] = xq[G] = one
    w = np.zeros((G, K), dtype=np.uint8)                                      # N = G = 128 columns, column 0 carries the case
    w[0, 0], w[0, G] = one, onehalf
    sx = np.array([0x3F800000] * 2, dtype=np.uint32)
    mid = int(np.float32(1.0 + 2.0 ** -8).view(np.uint32))
    s1 = int(np.float32(2.0 ** -25).view(np.uint32)) | 0x2AAAAB
    ws = np.array([[mid, s1]], dtype=np.uint32)
    y, _ = D.scaled_mm_fp8_block_row(xq, sx, w, ws, G=G)
    assert int(y[0]) == 0x3F81
