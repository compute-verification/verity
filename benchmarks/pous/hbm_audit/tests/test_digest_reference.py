"""CPU-only invariants of the HBM-audit digest's wire format: the interleaved layout the Triton kernel uses equals the
plain polynomial, and the digest depends on every label (universality witness). The device kernel is checked bit-exactly
against this reference at run time by `driver.py validate`, which aborts the job on a mismatch; these tests pin the
reference itself so a change to the format is caught without a GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import digest as D  # noqa: E402


def _blocks(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-(2**31), 2**31, size=(n, D.WORDS), dtype=np.int64).astype(np.int32)


@pytest.mark.parametrize("key", [0xABCDEF, 0x1234, 0x55AA55AA, 0xB01C])
def test_interleaved_layout_equals_plain_horner(key: int):
    host = _blocks(6, seed=key & 0xFF).view(np.uint32)
    for lane in (0, 1):
        g, _ = D.lane_points(key, lane)
        for r in range(host.shape[0]):
            assert D.reference_block_hash(host[r], g) == D.reference_block_hash_interleaved(host[r], g)


def test_lane_points_in_range():
    for key in (0, 1, 0xB01C, 2**31 - 1):
        for lane in range(4):
            g, z = D.lane_points(key, lane)
            assert 2 <= g <= D.MOD - 1 and 2 <= z <= D.MOD - 1


def test_digest_depends_on_every_block():
    """Changing one word of any challenged label changes the digest (the universality the audit relies on)."""
    host = _blocks(8, seed=7).view(np.uint32)
    idx = np.array([0, 3, 5, 1, 7, 2], dtype=np.int64)
    base = D.reference_digest(host, idx, key=0xBEEF, lane=0)
    for r in idx:
        pert = host.copy()
        pert[int(r), 0] ^= np.uint32(1)
        assert D.reference_digest(pert, idx, key=0xBEEF, lane=0) != base


def test_reorder_changes_digest():
    host = _blocks(8, seed=9).view(np.uint32)
    idx = np.array([0, 1, 2, 3], dtype=np.int64)
    assert D.reference_digest(host, idx, 0x11, 0) != D.reference_digest(host, idx[::-1].copy(), 0x11, 0)
