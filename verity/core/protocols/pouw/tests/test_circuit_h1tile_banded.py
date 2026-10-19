"""`circuit.h1tile`'s banded noise (J10): at a band of one tile it is today's per-tile noise and keys; along a band the row
noise is one, across bands and row blocks it is fresh, and the column noise and fold keys stay each tile's own. The pinned
digests are the vectors a kernel's banded derivation is held to."""
from __future__ import annotations

import hashlib
import random

import pytest

from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import seeds as SD

pytestmark = pytest.mark.skipif(H.SG is None, reason="H1's receipts (circuit.segment, E2E-120) are not on this tree")
SALT = hashlib.sha512(b"salt").digest()
ROOTS = (hashlib.sha512(b"A0").digest(), hashlib.sha512(b"B0").digest())
LAY = H.Layout(32, 32, 64)
#: a call of 2 row blocks by 5 column tiles, unit positions from 7 in row-major order
POS = [[7 + 5 * i + j for j in range(5)] for i in range(2)]


def _digest(rows, cols, keys) -> str:
    h = hashlib.sha256()
    for dither, digits in rows:
        h.update(bytes(dither))
        for d in digits:
            h.update(bytes(d))
    for coins, u in cols:
        h.update(bytes(coins))
        h.update(bytes(u))
    for ks in keys:
        for k in ks:
            h.update(k.to_bytes(4, "big"))
    return h.hexdigest()


def _codes(rng, n, k):
    return [[rng.randrange(256) for _ in range(k)] for _ in range(n)]


def test_band_of_one_is_per_tile_noise():
    for I in range(2):
        for J in range(5):
            rows, cols = H.noise(SD.seed(SALT, POS[I][J], *ROOTS), LAY)
            keys = H.fold_keys(SD.receipt_seed(SALT, POS[I][J], *ROOTS), LAY)
            assert H.banded_noise(SALT, POS, ROOTS, LAY, I, J, 1) == (rows, cols, keys)


def test_band_of_one_reference():
    rng = random.Random(10)
    a, b = _codes(rng, 32, 64), _codes(rng, 32, 64)
    assert H.banded_reference(a, b, SALT, POS, ROOTS, 1, 3, 1) == H.reference(a, b, SALT, POS[1][3], ROOTS)


def test_rows_shared_along_a_band_and_fresh_across():
    g = 2
    tiles = {(I, J): H.banded_noise(SALT, POS, ROOTS, LAY, I, J, g) for I in range(2) for J in range(5)}
    for (I, J), (rows, cols, keys) in tiles.items():
        J0 = H.band_start(J, g)
        assert rows == H.row_noise(SD.seed(SALT, POS[I][J0], *ROOTS), LAY)
        assert cols == H.col_noise(SD.seed(SALT, POS[I][J], *ROOTS), LAY)
        assert keys == H.fold_keys(SD.receipt_seed(SALT, POS[I][J], *ROOTS), LAY)
        for (I2, J2), (rows2, cols2, keys2) in tiles.items():
            if (I2, J2) == (I, J):
                continue
            assert (rows == rows2) == (I == I2 and J0 == H.band_start(J2, g))
            assert cols != cols2 and keys != keys2
    # the last band (column tile 4) is short and its own
    assert tiles[(0, 4)][0] != tiles[(0, 3)][0]


def test_banded_reference_reads_the_band_rows():
    rng = random.Random(11)
    a, b = _codes(rng, 32, 64), _codes(rng, 32, 64)
    z1, r1 = H.banded_reference(a, b, SALT, POS, ROOTS, 0, 1, 1)
    z2, r2 = H.banded_reference(a, b, SALT, POS, ROOTS, 0, 1, 2)
    assert z1 == z2 and r1 != r2
    rows, cols, keys = H.banded_noise(SALT, POS, ROOTS, LAY, 0, 1, 2)
    assert (z2, r2) == H._run(a, b, rows, cols, keys, None)


def test_band_start():
    assert [H.band_start(J, 64) for J in (0, 63, 64, 127, 128)] == [0, 0, 64, 64, 128]
    assert [H.band_start(J, 3) for J in range(7)] == [0, 0, 0, 3, 3, 3, 6]
    for g in (0, 65):
        with pytest.raises(ValueError):
            H.band_start(0, g)
    with pytest.raises(ValueError):
        H.band_start(-1, 2)
    with pytest.raises(ValueError):
        H.banded_noise(SALT, POS, ROOTS, LAY, 2, 0, 2)


#: (I, J, g) -> SHA-256 of `_digest`'s serialization of `banded_noise` at LAY on POS
PINNED = {
    (0, 0, 1): "b4ee8284a90ade0e6e33261a9974ad4ce6ed2907df0460be9bee685a44819b57",
    (1, 3, 2): "ef2573de8d080c700b342b13022c3fdf8f20c763eac6e1e65ac41e1d58ddcd8a",
    (1, 4, 2): "41fc4e418239bda12afe0ee705a9189efc0c9da1be833f3466bc5ca2ee613061",
    (0, 2, 64): "5c09654151ad96e3e686c6b00c0dcfa91eaaa758b207fd762a10d9e7290fdaa2",
}
#: tile (1, 3) at g = 2: its first row's first 32 dither bits and its first column's first 4 U encodings, its first lane's
#: first two fold keys, and the receipt of `banded_reference` on random.Random(12)'s codes
PINNED_WORDS = {"dither0": 985616073, "u0": [217, 26, 27, 63], "keys0": [631428560, 422950301],
                "receipt": [2230468465, 3978679094, 1078801284, 2015474290, 1128322151, 2495750162, 3810385992, 3094862096]}


def test_pinned_vectors():
    for (I, J, g), want in PINNED.items():
        assert _digest(*H.banded_noise(SALT, POS, ROOTS, LAY, I, J, g)) == want, (I, J, g)
    rows, cols, keys = H.banded_noise(SALT, POS, ROOTS, LAY, 1, 3, 2)
    rng = random.Random(12)
    a, b = _codes(rng, 32, 64), _codes(rng, 32, 64)
    got = {"dither0": H._value(rows[0][0][:32]), "u0": cols[0][1][:4], "keys0": keys[0][:2],
           "receipt": H.banded_reference(a, b, SALT, POS, ROOTS, 1, 3, 2)[1]}
    assert got == PINNED_WORDS
