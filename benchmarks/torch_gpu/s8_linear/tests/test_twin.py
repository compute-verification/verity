"""The twin (`fixture.epilogue` over `fixture.sums`) word for word against `ScaledMmS8Coordinate_v1{K, BlackwellS8ImmaDot32_v1}`'s
evaluator, on the inputs the GPU gate uses: random and structured code rows, every special scale against every other and
against spread ones, and the s32 edge at K = 131040."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops import gemm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fixture as FX  # noqa: E402


def _words(a: np.ndarray) -> list[int]:
    return [int(v) & 0xFF for v in np.asarray(a, dtype=np.int8).reshape(-1)]


def _definition_rows(x: np.ndarray, sa: np.ndarray, w: np.ndarray, sw: np.ndarray) -> np.ndarray:
    k, n = x.shape[1], w.shape[0]
    defn = bind(gemm.ScaledMmS8, K=k, N=n, **gemm.GEMM_SM120_S8)
    wv, swv = _words(w), [int(v) for v in sw.view(np.uint32)]
    return np.array([evaluate(defn, _words(x[i]), int(sa.view(np.uint32)[i]), wv, swv) for i in range(x.shape[0])],
                    dtype=np.uint16)


def _codes(rng: np.random.Generator, rows: int, k: int, structured: dict[int, str]) -> np.ndarray:
    x = rng.integers(-128, 128, (rows, k), dtype=np.int8)
    for i, kind in structured.items():
        if i < rows:
            x[i] = FX.structured(kind, k)
    return x


@pytest.mark.parametrize("k", [32, 64, 256])
def test_the_twin_is_the_definition_on_the_gates_inputs(k):
    rng = np.random.default_rng(k)
    m = FX.FIRST_SPECIAL + len(FX.SPECIAL_SA) + 4
    n = FX.FIRST_SPECIAL + len(FX.SPECIAL_SW) + 4
    x = _codes(rng, m, k, FX.STRUCT_ROWS)
    w = _codes(rng, n, k, FX.STRUCT_COLS)
    sa, sw = FX.scales(m, n, seed=k)
    got = FX.epilogue(FX.sums(x, w), sa, sw)
    want = _definition_rows(x, sa, w, sw)
    assert np.array_equal(got, want), np.argwhere(got != want)[:8].tolist()


def test_every_special_scale_meets_every_other_at_every_structured_sum():
    """Each structured row against each structured column (z = 0, +-16384 K, -16256 K, 16129 K ...) and a small random
    one, under every (s_a, s_w) pair of the special words and two spread ones."""
    k = 32
    rng = np.random.default_rng(7)
    x = np.stack([FX.structured(kind, k) for kind in FX.STRUCTURED] + [rng.integers(-128, 128, k, dtype=np.int8)])
    w = np.stack([FX.structured(kind, k) for kind in FX.STRUCTURED] + [rng.integers(-8, 8, k, dtype=np.int8)])
    sas = np.array(FX.SPECIAL_SA + (FX.f32w(3.7e-3), FX.f32w(1.5)), dtype=np.uint32).view(np.float32)
    sws = np.array(FX.SPECIAL_SW + (FX.f32w(2.2e-4), FX.f32w(0.75)), dtype=np.uint32).view(np.float32)
    xr = np.repeat(x, len(sas), axis=0)
    sa = np.tile(sas, len(x))
    wr = np.repeat(w, len(sws), axis=0)
    sw = np.tile(sws, len(w))
    got = FX.epilogue(FX.sums(xr, wr), sa, sw)
    want = _definition_rows(xr, sa, wr, sw)
    assert np.array_equal(got, want), np.argwhere(got != want)[:8].tolist()


def test_the_s32_edge():
    """K = 131040 with every code -128: z = 2,146,959,360, the largest s8 sum the Definitions take, and its negative
    neighbour; S32ToF32Rn rounds it (0x4EFFF000) before the scales."""
    k = FX.EDGE_K
    x = np.stack([FX.structured("min", k), FX.structured("max", k)])
    w = np.stack([FX.structured("min", k), FX.structured("max", k), FX.structured("alt", k)])
    z = FX.sums(x, w)
    assert z[0, 0] == 128 * 128 * k and z[0, 1] == -128 * 127 * k
    sa = np.array([1.0, 3e-3], dtype=np.float32)
    sw = np.array([1.0, 2.0**-20, 1e30], dtype=np.float32)
    got = FX.epilogue(z, sa, sw)
    defn = bind(gemm.ScaledMmS8Coordinate, K=k, **gemm.GEMM_SM120_S8)
    for i in range(2):
        for j in range(3):
            want = evaluate(defn, _words(x[i]), int(sa.view(np.uint32)[i]), _words(w[j]), int(sw.view(np.uint32)[j]))[0]
            assert got[i, j] == want, (i, j, hex(got[i, j]), hex(want))


def test_the_twin_refuses_a_sum_past_its_k():
    with pytest.raises(ValueError):
        FX.sums(np.zeros((1, 2**17), dtype=np.int8), np.zeros((1, 2**17), dtype=np.int8))


def test_a_flushing_epilogue_would_differ():
    """The specials catch a flush: the same words with each multiply's subnormal results flushed differ somewhere."""
    m = FX.FIRST_SPECIAL + len(FX.SPECIAL_SA)
    n = FX.FIRST_SPECIAL + len(FX.SPECIAL_SW)
    rng = np.random.default_rng(3)
    z = rng.integers(-500_000, 500_000, (m, n), dtype=np.int32)
    sa, sw = FX.scales(m, n, seed=3)
    exact = FX.epilogue(z, sa, sw)

    def ftz(a):
        a = a.copy()
        a[np.abs(a) < np.float32(2.0**-126)] = 0
        return a
    with np.errstate(all="ignore"):
        t = ftz(ftz(z.astype(np.float32) * ftz(sa)[:, None]) * ftz(sw)[None, :])
    from verity.ml.kernels.tc import f2fp_bf16_batch
    flushed = f2fp_bf16_batch(t.view(np.uint32))[0]
    assert (flushed != exact).sum() > 0


def test_the_shapes():
    got = {name: (n, k) for name, n, k in FX.shapes()}
    assert got["d8192:qkv"] == (24576, 8192) and got["d12288:down"] == (12288, 49152) and got["d8192:head"] == (256, 8192)
    assert all(k % 32 == 0 and k < 2**17 for _, _, k in FX.shapes())
