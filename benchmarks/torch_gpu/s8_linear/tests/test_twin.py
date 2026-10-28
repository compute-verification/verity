"""The twin (`fixture.epilogue` over `fixture.sums`) word for word against `ScaledMmS8Coordinate_v1{K, DOT}`'s evaluator, DOT
the H200's step (`GEMM_SM90_S8`) and the RTX PRO 6000's (`GEMM_SM120_S8`), on the inputs the GPU gate uses: random and
structured code rows, every special scale against every other and against spread ones, and the s32 edge at K = 131040."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops import gemm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fixture as FX


def _words(a: np.ndarray) -> list[int]:
    return [int(v) & 0xFF for v in np.asarray(a, dtype=np.int8).reshape(-1)]


STEPS = ("GEMM_SM90_S8", "GEMM_SM120_S8")


def _definition_rows(x: np.ndarray, sa: np.ndarray, w: np.ndarray, sw: np.ndarray, step: str = "GEMM_SM90_S8") -> np.ndarray:
    k, n = x.shape[1], w.shape[0]
    defn = bind(gemm.ScaledMmS8, K=k, N=n, **getattr(gemm, step))
    wv, swv = _words(w), [int(v) for v in sw.view(np.uint32)]
    return np.array([evaluate(defn, _words(x[i]), int(sa.view(np.uint32)[i]), wv, swv) for i in range(x.shape[0])],
                    dtype=np.uint16)


def _codes(rng: np.random.Generator, rows: int, k: int, structured: dict[int, str]) -> np.ndarray:
    x = rng.integers(-128, 128, (rows, k), dtype=np.int8)
    for i, kind in structured.items():
        if i < rows:
            x[i] = FX.structured(kind, k)
    return x


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("k", [32, 64, 256])
def test_the_twin_is_the_definition_on_the_gates_inputs(k, step):
    rng = np.random.default_rng(k)
    m = FX.FIRST_SPECIAL + len(FX.SPECIAL_SA) + 4
    n = FX.FIRST_SPECIAL + len(FX.SPECIAL_SW) + 4
    x = _codes(rng, m, k, FX.STRUCT_ROWS)
    w = _codes(rng, n, k, FX.STRUCT_COLS)
    sa, sw = FX.scales(m, n, seed=k)
    got = FX.epilogue(FX.sums(x, w), sa, sw)
    want = _definition_rows(x, sa, w, sw, step)
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


@pytest.mark.parametrize("step", STEPS)
def test_the_s32_edge(step):
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
    defn = bind(gemm.ScaledMmS8Coordinate, K=k, **getattr(gemm, step))
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


def test_the_twin_takes_the_definitions_order_where_the_other_orders_round_differently():
    """Scales searched so that s_w first (CUTLASS c2x's order) or s_a * s_w first rounds to another bf16 word: the twin is
    the Definition there and both other orders differ from it, so a gate through the twin fails a reordered epilogue.
    Random scales rarely tell the orders apart."""
    from verity.ml.kernels.tc import f2fp_bf16_batch

    rng = np.random.default_rng(7)
    r, k = 64, 64
    x, w = (rng.integers(-128, 128, (r, k), dtype=np.int8) for _ in range(2))
    f = np.diag(FX.sums(x, w)).astype(np.float32)[:, None]
    ca = np.exp2(rng.uniform(-24, 2, (r, 1 << 16))).astype(np.float32)
    cw = np.exp2(rng.uniform(-20, -2, (r, 1 << 16))).astype(np.float32)

    def bf(t):
        return f2fp_bf16_batch(t.view(np.uint32))[0].reshape(t.shape)

    with np.errstate(all="ignore"):
        want = bf((f * ca) * cw)
        apart = (want != bf((f * cw) * ca)) & (want != bf(f * (ca * cw)))
        rows = np.flatnonzero(apart.any(1))
        assert len(rows) >= 4
        pick = apart[rows].argmax(1)
        sa, sw = ca[rows, pick], cw[rows, pick]
        fr = f[rows]
        others = {"sw-first": bf((fr * sw[:, None]) * sa[:, None])[:, 0], "scales-first": bf(fr * (sa * sw)[:, None])[:, 0]}
    twin = np.diag(FX.epilogue(FX.sums(x[rows], w[rows]), sa, sw))
    want = np.array([_definition_rows(x[i:i + 1], sa[j:j + 1], w[i:i + 1], sw[j:j + 1])[0, 0] for j, i in enumerate(rows)])
    assert np.array_equal(twin, want)
    for order, got in others.items():
        assert (got != want).all(), order


def test_the_shapes():
    """The models' shapes: the H200's (the default; o is q's shape) and the RTX PRO 6000's two."""
    h200 = {name: (n, k) for name, n, k in FX.shapes()}
    assert h200 == {"mqa16k52:q": (16384, 16384), "mqa16k52:kv": (128, 16384), "mqa16k52:up": (65536, 16384),
                    "mqa16k52:down": (16384, 65536), "mqa16k52:head": (256, 16384)}
    md = FX.MODELS["mqa16k52"]
    assert (md.layers, md.context, md.sequences) == (52, 32, 20480)
    assert md.layers * sum(n * k for _, n, k, _ in md.linears) == 139_695_489_024  # the seeded weights' 52 layers of s8
    got = {name: (n, k) for name, n, k in FX.shapes(("mqa16k", "mha8k"))}
    assert got["mqa16k:q"] == (16384, 16384) and got["mqa16k:kv"] == (128, 16384) and got["mqa16k:down"] == (16384, 65536)
    assert got["mha8k:qkv"] == (24576, 8192) and got["mha8k:down"] == (8192, 32768) and got["mha8k:head"] == (256, 8192)
    assert got["mha8k:o"] == (8192, 8192) and got["mqa16k:head"] == (256, 16384)
    assert all(k % 32 == 0 and k < 2**17 for _, _, k in FX.shapes())
    assert FX.linears(12288)["down"] == (12288, 49152) and FX.model("d12288").layers == 44
    assert abs(FX.MODELS["mqa16k"].weight_bytes / 2**30 - 87.58) < 0.05
    assert abs(FX.MODELS["mha8k"].weight_bytes / 2**30 - 52.5) < 0.05


def test_the_committed_bytes_a_token_row():
    """One layer's committed bytes a token row (codes, without the kernels' 4-byte words): mha8k's fused set is 256 KiB
    against the estimate's 254 KB, every op 360 KiB against its 365 KB (its scores at the mean context)."""
    per = {s: sum(k for _, k, _ in FX.tables(FX.MODELS["mha8k"], s)) for s in FX.TABLE_SETS}
    assert per == {"quantize": 7 * 8192, "fused": 262144, "every": 368640}
    ops = [op for op, _, _ in FX.tables(FX.MODELS["mqa16k"], "every")]
    assert len(ops) == len(set(ops)) == 17 and all(k % 2 == 0 for _, k, _ in FX.tables(FX.MODELS["mqa16k"], "every"))
