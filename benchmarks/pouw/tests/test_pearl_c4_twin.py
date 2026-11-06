"""Pearl-C4's numpy twin (benchmarks/pouw/pearl_c4/twin.py) against the scalar reference `verity.ml.protocols.pouw.schemes.pearl_c4`: the
vectorized casts on grids through every rounding case, and whole tiles (forming, C̃, U, the peel factors, the leaf and the
tickets) for both formats."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile
from verity.ml.protocols.pouw.schemes import pearl_c4 as F

# Loaded under its own name: pearl_c's twin.py is `twin` to the tests beside this one.
_spec = importlib.util.spec_from_file_location("pearl_c4_twin", Path(__file__).resolve().parents[1] / "pearl_c4" / "twin.py")
W = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = W
_spec.loader.exec_module(W)


def grid() -> np.ndarray:
    """FP32 words around every E2M1, UE4M3 and power-of-two boundary, subnormals and zero included."""
    rng = np.random.default_rng(7)
    anchors = np.float32([0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, 448, 464, 2.0 ** -9, 2.0 ** -10, 2.0 ** -6, 1, 3])
    words = anchors.view(np.uint32)
    near = (words[:, None].astype(np.int64) + np.arange(-3, 4)[None, :]).clip(0, 0x7F7FFFFF).ravel()
    rand = rng.integers(0, 0x7F7FFFFF, 4000)
    sub = rng.integers(0, 1 << 23, 500)
    return np.unique(np.concatenate([near, rand, sub]).astype(np.uint32))


def test_casts_equal_the_scalar_reference():
    w = grid()
    assert W.e2m1(w).tolist() == [F.e2m1(int(x)) for x in w]
    assert W.e2m1(w | np.uint32(0x80000000)).tolist() == [F.e2m1(int(x) | 0x80000000) for x in w]
    assert W.ue4m3(w).tolist() == [F.ue4m3(int(x)) for x in w]
    assert W.ue8m0_up(w).tolist() == [F.ue8m0_up(int(x)) for x in w]
    for fmt in (F.NVFP4, F.MXFP4):
        rows = w[: len(w) // fmt.block * fmt.block].reshape(-1, fmt.block)
        s, r = W.block_scales(fmt, rows)
        assert list(zip(s.ravel().tolist(), r.ravel().tolist())) == [F.block_scale(fmt, max(int(x) for x in row))
                                                                     for row in rows]


def bf16_rows(rng: np.random.Generator, n: int, k: int, scale: float) -> np.ndarray:
    x = (rng.standard_normal((n, k)) * scale).astype(np.float32).view(np.uint32)
    return W.rne_bf16(x)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_tile_equals_the_scalar_reference(fmt):
    k, n = 128, 16
    s = F.PearlC4(fmt, k_min=128, tile=(n, n))
    rng = np.random.default_rng(3 if fmt == "nvfp4" else 4)
    a, b = bf16_rows(rng, n, k, 1.0), bf16_rows(rng, n, k, 0.02)
    ctx = Context(b"s" * 32, Matmul(1, Shape(n, k, n), 2), b"a" * 32, b"b" * 32)
    noise = s.noise(ctx)
    tile = Tile(tuple(range(n)), tuple(range(n)))
    rows = lambda m: [struct.pack(f"<{k}H", *r.tolist()) for r in m]  # noqa: E731
    ops = s.encode(ctx, noise, rows(a), rows(b), tile)
    checked = s.checked(ctx, ops)
    jb = W.job(s.fmt, noise.seed_a, noise.seed_b, k)
    got = W.tile(jb, a, b, np.arange(n), np.arange(n))
    assert got.a.codes.tolist() == ops.a.codes and got.b.scales.tolist() == ops.b.scales
    assert got.a.alpha.tolist() == ops.a.alpha and got.b.ae.tolist() == ops.b.ae
    assert got.pa.tolist() == ops.pa and got.pb.tolist() == ops.pb
    assert got.c.tolist() == checked.c and got.u.tolist() == checked.u
    assert got.leaf == s.digest(ctx, noise, checked) and got.tickets == s.tickets(noise, checked)
    assert jb.g.tolist() == [list(r) for r in F.gram(s.fmt, noise.seed_b, k)]
