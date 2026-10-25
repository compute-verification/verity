"""The batched pinned H1 tile (`h1batch.tiles`) is `h1tile.reference`, word for word, on random tiles and on the codes'
extremes (X = A + b wrapping at A = 127, W' = B + c + U wrapping at either end, all-zero rows)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from verity.core.protocols.pouw.circuit import h1tile as Ht
from verity_torch_loop import h1batch as HB

SALT = hashlib.sha512(b"h1batch").digest()


def _tiles(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a, b = rng.integers(-128, 128, (n, 4, 64)), rng.integers(-128, 128, (n, 8, 64))
    if n > 5:
        for t, (va, vb) in enumerate([(127, 127), (-128, -128), (127, -128), (-128, 127), (0, 0)]):
            a[t], b[t] = va, vb
        a[5, :, ::2], b[5, :, 1::2] = 127, -128
    ra, rb = (rng.integers(0, 256, (n, 64), dtype=np.uint8) for _ in range(2))
    return a, b, [int(p) for p in rng.integers(0, 1 << 62, n)], ra, rb


@pytest.mark.parametrize("seed", [0, 1])
def test_the_batch_is_the_reference(seed):
    a, b, pos, ra, rb = _tiles(24, seed)
    z, rc = HB.tiles(a, b, SALT, pos, ra, rb)
    assert z.dtype == rc.dtype == np.uint32 and z.shape == (24, 32) and rc.shape == (24, HB.RECEIPT_WORDS)
    for t in range(24):
        zr, rr = Ht.reference((a[t] & 0xFF).tolist(), (b[t] & 0xFF).tolist(), SALT, pos[t], (bytes(ra[t]), bytes(rb[t])))
        assert z[t].tolist() == [w & 0xFFFFFFFF for w in zr] and rc[t].tolist() == rr


def test_another_salt_position_or_root_is_another_receipt_and_the_same_z():
    a, b, pos, ra, rb = _tiles(4, 2)
    z, rc = HB.tiles(a, b, SALT, pos, ra, rb)
    for args in ((hashlib.sha512(b"other").digest(), pos, ra, rb), (SALT, [p + 1 for p in pos], ra, rb),
                 (SALT, pos, ra ^ np.uint8(1), rb), (SALT, pos, ra, rb ^ np.uint8(1))):
        z2, rc2 = HB.tiles(a, b, *args)
        assert np.array_equal(z2, z) and (rc2 != rc).any(axis=1).all()


def test_shapes_off_the_pinned_tile_are_refused():
    a, b, pos, ra, rb = _tiles(2, 3)
    with pytest.raises(ValueError, match="take a"):
        HB.tiles(a[:, :, :32], b[:, :, :32], SALT, pos, ra, rb)
