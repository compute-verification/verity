"""`derived_rows_torch` equals `derived_rows` (the numpy twin already validated gate for gate against the
registered semantics) on random rows, for every derived class and for the scalar kernels.  Runs on CPU here;
the same assertions run on the pod GPU via `verity_vllm_adapter.derived_b0_check --backend torch`."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "vllm-poc") not in sys.path:
    sys.path.insert(0, str(REPO / "vllm-poc"))

torch = pytest.importorskip("torch")


def _mods():
    from verity_vllm.registry import derived_rows as D
    from verity_vllm.registry import derived_rows_torch as DT
    try:
        D._fa2_tables()
        D._rms_tables()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"MUFU tables unavailable: {e}")
    return D, DT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(DEVICE)


def _n(t):
    return t.cpu().numpy()


def _rand_words(rng, n):
    return rng.integers(0, 2 ** 32, n, dtype=np.uint64).astype(np.uint32)


def _rand_bf16(rng, shape, scale=1.0):
    return ((rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def test_scalar_kernels_match_numpy():
    D, DT = _mods()
    rng = np.random.default_rng(5)
    x = np.concatenate([_rand_words(rng, 30000),
                        np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x00000001,
                                  0x1D6F2968, 0x807FFFFF, 0x7F7FFFFF], dtype=np.uint32)])
    y = np.roll(x, 7)
    for name in ("mufu_ex2_ftz", "mufu_rcp_ftz", "fa2_inv_sum", "mufu_sqrt_ftz", "rsqrt_approx",
                 "div_full_rcp", "ftz", "guard_neg_inf_zero"):
        got = _n(getattr(DT, name)(_t(x)))
        want = getattr(D, name)(x).astype(np.int64)
        assert np.array_equal(got, want), name
    is_nan = lambda w: ((w & 0x7F800000) == 0x7F800000) & ((w & 0x007FFFFF) != 0)  # noqa: E731
    fin = ~(is_nan(x) | is_nan(y))  # NaN payload propagation in plain add/sub/mul is host-defined (see
    for name in ("max_fmnmx", "div_full_scale_a"):  # test_derived_rows.test_binary_kernels_match_prims)
        got = _n(getattr(DT, name)(_t(x), _t(y)))
        want = getattr(D, name)(x, y).astype(np.int64)
        assert np.array_equal(got, want), name
    for name in ("add_ftz", "sub_ftz", "mul_ftz", "add_rn", "mul_rn"):
        got = _n(getattr(DT, name)(_t(x[fin]), _t(y[fin])))
        want = getattr(D, name)(x[fin], y[fin]).astype(np.int64)
        assert np.array_equal(got, want), name
    z = np.roll(x, 313)
    got = _n(DT.fma_rn(_t(x), _t(y), _t(z)))
    assert np.array_equal(got, D.fma_rn(x, y, z).astype(np.int64))
    got = _n(DT.fma_sub_ftz(_t(x), _t(y), _t(z)))
    assert np.array_equal(got, D.fma_sub_ftz(x, y, z).astype(np.int64))


def test_tc_dot16_matches_numpy():
    D, DT = _mods()
    rng = np.random.default_rng(6)
    n = 4000
    w = rng.integers(0, 2 ** 16, (2 * n, 16), dtype=np.int64).astype(np.uint16)
    w = np.where(((w >> 7) & 0xFF) == 0xFF, w & 0x7F7F, w)
    a, b = w[:n], w[n:]
    acc = _rand_words(rng, n)
    acc = np.where((acc & 0x7F800000) == 0x7F800000, acc & 0xFF7FFFFF, acc)
    want, wf = D.tc_dot16(acc, a, b)
    got, gf = DT.tc_dot16(_t(acc), _t(a), _t(b))
    assert np.array_equal(_n(got), want.astype(np.int64))
    assert np.array_equal(_n(gf), wf)


def test_row_evaluators_match_numpy():
    D, DT = _mods()
    rng = np.random.default_rng(8)
    # triton norm
    x = _rand_bf16(rng, (7, 576), 3.0)
    w = _rand_bf16(rng, 576)
    rn, rt = D.rmsnorm_triton_rows(x, w, 1e-5), DT.rmsnorm_triton_rows(_t(x), _t(w), 1e-5)
    for k in ("out", "sum", "mean_eps", "rms", "inv"):
        assert np.array_equal(_n(rt[k]), rn[k].astype(np.int64)), k
    # fused v2
    res = _rand_bf16(rng, (7, 576), 3.0)
    rn, rt = D.rmsnorm_fused_v2_rows(x, res, w, 1e-5), DT.rmsnorm_fused_v2_rows(_t(x), _t(res), _t(w), 1e-5)
    for k in ("normed", "resid", "var", "s"):
        assert np.array_equal(_n(rt[k]), rn[k].astype(np.int64)), k
    # silu
    gu = _rand_bf16(rng, (5, 3072), 4.0)
    rn, rt = D.silu_mul_rows(gu), DT.silu_mul_rows(_t(gu))
    for k in ("mid", "out"):
        assert np.array_equal(_n(rt[k]), rn[k].astype(np.int64)), k
    # attention (two blocks + partial)
    R, T, Dh = 4, 100, 64
    q = _rand_bf16(rng, (R, Dh))
    k = _rand_bf16(rng, (R, T, Dh))
    v = _rand_bf16(rng, (R, T, Dh))
    rn, rt = D.attention_head_states(q, k, v), DT.attention_head_states(_t(q), _t(k), _t(v))
    assert rn["blocks"] == rt["blocks"]
    for kk in ("m", "l", "o", "out", "total", "inv"):
        assert np.array_equal(_n(rt[kk]), rn[kk].astype(np.int64)), kk
    assert np.array_equal(_n(rt["nonfinite"]), rn["nonfinite"])
