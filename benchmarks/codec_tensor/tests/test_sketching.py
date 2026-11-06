"""`sketching`'s arithmetic: the host and device sketches are S · e_v exactly, and a sketch tile at the drivers' T = 8
is sketch-sha512's 4 KiB block."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sketch_receipt as SR
import sketching as SK


def _case(rows: int, n: int, k: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    vs = rng.integers(-128, 128, (rows, n), dtype=np.int64).astype(np.int8)
    bits = rng.integers(0, 2, (k, n), dtype=np.uint8)
    vs[0] = np.where(bits[0] == 1, -128, 127)                        # σ_0[0] at its largest
    return vs, bits


@pytest.mark.parametrize("k", [16, 64])
def test_sketch_is_exact(k: int):
    vs, bits = _case(40, SR.BLOCK, k)
    want = vs.astype(np.int64) @ (1 - 2 * bits.astype(np.int64)).T
    assert np.array_equal(SK.sketch(vs, bits), want)
    assert want[0, 0] == int(np.where(bits[0] == 1, 128, 127).sum())


def test_torch_sketch_is_exact():
    torch = pytest.importorskip("torch")
    vs, bits = _case(40, SR.BLOCK, 16, seed=4)
    assert np.array_equal(SK.sketch_torch(torch, torch.from_numpy(vs), bits).numpy(), SK.sketch(vs, bits))


def test_vector_is_a_block():
    assert SK.vector_bytes(8192, 8) == SR.BLOCK and SK.steps(8256) == 17
