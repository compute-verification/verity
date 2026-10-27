"""`verity.ml.definitions.registry.torch_optim`: the foreach optimizer Definitions equal a torch CPU reference composed in the
CUDA kernels' order (each torch f32 op one IEEE rounding, as the GPU's ``.rn`` instructions; each ``std::fma`` and each
contraction exact rational arithmetic with one rounding, `ops.fp32.f32_fma_bits`; the square root numpy's, since torch CPU's
f32 `sqrt` on an AVX-512 build is not correctly rounded: 0x2A893F0E gives 0x35048AE4, sqrt.rn 0x35048AE5) on random words, on special words, at
the lerp's branch edge, and at the scalars a step of training's recipe produces. NaN payloads are compared as the GPU's
(every f32 NaN 0x7FFFFFFF, every bf16 NaN 0x7FFF)."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import fp32 as W
from verity.ml.definitions.registry import torch_optim as TO

torch = pytest.importorskip("torch")

U32 = np.uint32
NAN, BF16_NAN = 0x7FFFFFFF, 0x7FFF
SPECIALS = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0xBF800000, 0x3F000000, 0x3EFFFFFF,
                     0xBF000000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FFFFFFF,
                     0x33D6BF95, 0x3DCCCCCD, 0x3F666666, 0x3F733333], U32)


def _f(u) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u, U32).view(np.float32))


def _u(t: torch.Tensor) -> np.ndarray:
    return t.numpy().view(U32)


def _canon(u) -> np.ndarray:
    u = np.asarray(u, U32)
    return np.where((u & 0x7FFFFFFF) > 0x7F800000, U32(NAN), u)


def _canon_bf16(u) -> np.ndarray:
    u = np.asarray(u, np.uint32)
    return np.where((u & 0x7FFF) > 0x7F80, BF16_NAN, u)


def _fma(a, b, c) -> np.ndarray:
    return np.array([W.f32_fma_bits(int(x), int(y), int(z)) for x, y, z in zip(a, b, c)], U32)


def ref_adamw(p, g, m, v, k, fused: bool):
    """(p2, m1, v2, bf16(p2)) of one element a row, `TO.AdamWForeach`'s steps on torch CPU f32 tensors."""
    a, w, b2, c2, s2, e, st = (np.ascontiguousarray(k[:, i]) for i in range(7))
    P, G, M, V = map(_f, (p, g, m, v))
    p1 = _u(P * _f(a))
    d = G - M
    om = torch.ones_like(G) - _f(w)
    small = (w & 0x7FFFFFFF) < 0x3F000000
    if fused:
        m1 = np.where(small, _fma(w, _u(d), m), _fma(_u(-d), _u(om), g))
    else:
        m1 = np.where(small, _u(M + _f(w) * d), _u(G - d * om))
    v1 = _u(V * _f(b2))
    v2 = np.where(c2 == 0x3F800000, _fma(g, g, v1), _fma(c2, _u(G * G), v1))
    dd = torch.from_numpy(np.sqrt(_f(v2).numpy())) / _f(s2) + _f(e)
    q = _u(_f(m1) / dd)
    p2 = _fma(st, q, p1)
    return _canon(p2), _canon(m1), _canon(v2), _canon_bf16(_u(_f(p2).to(torch.bfloat16).view(torch.int16).to(torch.int32)) & 0xFFFF)


def ref_sgd(p, g, nlr, fused: bool):
    return _canon(_fma(nlr, g, p) if fused else _u(_f(p) + _f(nlr) * _f(g)))


def torch_scalars(lr: float, beta1: float, beta2: float, eps: float, wd: float, step: float) -> np.ndarray:
    """The seven f32 scalars of `TO.SCALARS` as `_multi_tensor_adam`'s host code makes them at step count ``step`` (after its
    increment), each ``Scalar::to<float>()`` (round to nearest)."""
    bc1, bc2 = 1 - beta1 ** step, 1 - beta2 ** step
    vals = [1 - lr * wd if wd != 0 else 1.0, 1 - beta1, beta2, 1 - beta2, bc2 ** 0.5, eps, (lr / bc1) * -1]
    return np.array(vals, np.float32).view(U32)


def _words(rng, n: int, scale: float = 1.0) -> np.ndarray:
    """A mix: normal values over 2^-40 .. 2^10 times ``scale``, raw 32-bit words, and `SPECIALS`."""
    normal = (rng.standard_normal(n) * scale * np.exp2(rng.integers(-40, 10, n))).astype(np.float32).view(U32)
    raw = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)
    spec = SPECIALS[rng.integers(0, len(SPECIALS), n)]
    pick = rng.integers(0, 10, n)
    return np.where(pick < 6, normal, np.where(pick < 8, raw, spec)).astype(U32)


def _recipe_k(rng, n: int) -> np.ndarray:
    """Scalars of training's AdamW (lr 3e-4, betas (0.9, 0.95), eps 1e-8, weight decay 0.1) and of perturbed recipes, at steps
    1 .. 2^24, mixed with raw and special scalar words."""
    rows = []
    for _ in range(n):
        if rng.random() < 0.5:
            lr, b1, b2, eps, wd = 3e-4, 0.9, 0.95, 1e-8, 0.1
        else:
            lr, b1, b2 = 10 ** rng.uniform(-6, 0), rng.uniform(0, 1), rng.choice([0.0, rng.uniform(0, 1), 0.999])
            eps, wd = rng.choice([0.0, 1e-8, 10 ** rng.uniform(-12, -3)]), rng.choice([0.0, 0.1, rng.uniform(0, 1)])
        step = float(rng.choice([1, 2, 3, 10, 1000, int(rng.integers(1, 1 << 24))]))
        rows.append(torch_scalars(lr, b1, b2, eps, wd, step))
    k = np.array(rows, U32)
    wild = rng.random(k.shape) < 0.1
    return np.where(wild, _words(rng, k.size).reshape(k.shape), k).astype(U32)


@pytest.mark.parametrize("fused", [0, 1])
def test_adamw_element_matches_reference(fused: int) -> None:
    rng = np.random.default_rng(117 + fused)
    n = 3000
    p, g, m = _words(rng, n, 0.04), _words(rng, n, 1e-3), _words(rng, n, 1e-3)
    v = np.where(rng.random(n) < 0.8, np.abs(_words(rng, n, 1e-3).view(np.float32) ** 2).view(U32), _words(rng, n)).astype(U32)
    k = _recipe_k(rng, n)
    got = BS.evaluate_words(bind(TO.AdamWForeach, N=1, LERP_FMA=fused, BF16=1), p[:, None], g[:, None], m[:, None], v[:, None], k)
    want = ref_adamw(p, g, m, v, k, bool(fused))
    for j, name in enumerate(("p2", "m1", "v2", "bf16")):
        bad = np.flatnonzero(got[:, j] != want[j])
        assert not bad.size, (f"{name}: {bad.size} of {n} differ, first at {bad[0]}: p {p[bad[0]]:#x} g {g[bad[0]]:#x} m {m[bad[0]]:#x} "
                              f"v {v[bad[0]]:#x} k {[hex(x) for x in k[bad[0]]]}: {got[bad[0], j]:#x} != {want[j][bad[0]]:#x}")


@pytest.mark.slow
def test_adamw_lerp_branch_edges() -> None:
    """The lerp's branch at |w| = 0.5 and its neighbours, NaN and infinite weights, for each contraction."""
    rng = np.random.default_rng(7)
    ws = np.array([0x3F000000, 0x3EFFFFFF, 0x3F000001, 0xBF000000, 0xBEFFFFFF, 0x3F800000, 0, 0x80000000, 0x7FC00000, 0x7F800000,
                   0xFF800000, 0x3DCCCCCD, 0x3F7FFFFF], U32)
    n = len(ws) * 40
    w = np.repeat(ws, 40)
    k = np.tile(torch_scalars(3e-4, 0.9, 0.95, 1e-8, 0.1, 5.0), (n, 1))
    k[:, 1] = w
    p, g, m, v = _words(rng, n, 0.04), _words(rng, n, 1e-3), _words(rng, n, 1e-3), np.abs(_words(rng, n, 1e-6).view(np.float32)).view(U32)
    for fused in (0, 1):
        got = BS.evaluate_words(bind(TO.AdamWForeach, N=1, LERP_FMA=fused, BF16=0), p[:, None], g[:, None], m[:, None], v[:, None], k)
        want = ref_adamw(p, g, m, v, k, bool(fused))
        for j in range(3):
            assert np.array_equal(got[:, j], want[j]), (fused, j, np.flatnonzero(got[:, j] != want[j])[:5])


@pytest.mark.slow
def test_adamw_block_shares_its_scalars() -> None:
    """At N = 4 each element is the N = 1 step under the one k, and the outputs are (p2, m1, v2, bf16) runs of N."""
    rng = np.random.default_rng(11)
    n, N = 200, 4
    p, g, m = (_words(rng, n * N, s).reshape(n, N) for s in (0.04, 1e-3, 1e-3))
    v = np.abs(_words(rng, n * N, 1e-3).view(np.float32) ** 2).view(U32).reshape(n, N)
    k = _recipe_k(rng, n)
    got = BS.evaluate_words(bind(TO.AdamWForeach, N=N, LERP_FMA=1, BF16=1), p, g, m, v, k).reshape(n, 4, N)
    for i in range(N):
        want = ref_adamw(p[:, i], g[:, i], m[:, i], v[:, i], k, True)
        for j in range(4):
            assert np.array_equal(got[:, j, i], want[j]), (i, j)


@pytest.mark.parametrize("fused", [0, 1])
def test_sgd_element_matches_reference(fused: int) -> None:
    rng = np.random.default_rng(23 + fused)
    n = 4000
    p, g = _words(rng, n, 0.04), _words(rng, n, 1e-3)
    nlr = np.where(rng.random(n) < 0.7, np.float32(-1e-2).view(U32), _words(rng, n)).astype(U32)
    got = BS.evaluate_words(bind(TO.SgdForeach, N=1, ALPHA_FMA=fused), p[:, None], g[:, None], nlr[:, None])
    assert np.array_equal(got[:, 0], ref_sgd(p, g, nlr, bool(fused)))


def test_scalars_follow_torch_host_code() -> None:
    """`torch_scalars` is `_multi_tensor_adam`'s host arithmetic: the step size and the root of the second bias correction
    torch 2.14 hands its foreach kernels at training's recipe."""
    k = torch_scalars(3e-4, 0.9, 0.95, 1e-8, 0.1, 1.0)
    assert k[TO.SCALARS.index("st")] == np.float32(-(3e-4 / (1 - 0.9))).view(U32)
    assert k[TO.SCALARS.index("s2")] == np.float32((1 - 0.95) ** 0.5).view(U32)
    assert k[TO.SCALARS.index("a")] == np.float32(1 - 3e-4 * 0.1).view(U32)
    assert torch_scalars(3e-4, 0.9, 0.95, 1e-8, 0.0, 3.0)[0] == 0x3F800000
