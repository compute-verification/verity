"""RT4-INT's item 3, the served form under ASK-71 A (INT-TIE's `Accounting.Served` on `cursor/int-tie-ea24` at 8f161917ab,
kernels' `dither-a` arm on `cursor/kernels-dither-cut-15ad` at f878a05970).

Every served arm forms a call's A rows after the salt its noise reads: dither-a's seeds read the window's salt (with
`-ls`, the round before's), and step s's A table goes under step s's round root between steps, after its receipts.
`Served.Tie` orders the other way (A under round r's root before coin r, receipts reading coin r), and `EpochTie`'s text
says a call's noise key reads "that salt and its call index only". The code's order is sound only because the key
reads root(A): each unit's seed is `seeds.seed(salt, position, root_a, root_b)`. The noise on A is one dither bit an
entry, X = A + b, so a key that reads no root(A) lets an A written after its noise be −b: X = 0, and the unit's receipt
is the same for every B, no noised work on the weights, while its z = A Bᵀ is still a product (RT4-INT-12).
"""
from __future__ import annotations

import hashlib
import random

from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.circuit import seeds as SD

LAY = HT.Layout(4, 8, 64)
SALT = hashlib.sha512(b"rt4-int salt").digest()
POSITION = 7
ROOT_B = hashlib.sha512(b"rt4-int root(B)").digest()


def _weights(seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(256) for _ in range(LAY.k)] for _ in range(LAY.c)]


def _cancelling(key_root: bytes) -> list[list[int]]:
    """A's rows as −b under the noise `key_root` keys, int8 encodings: X = A + b is zero in every cell."""
    rows, _ = HT.noise(SD.seed(SALT, POSITION, key_root, ROOT_B), LAY)
    return [[(-bit) & 0xFF for bit in rows[i][0]] for i in range(LAY.r)]


def test_a_noise_key_that_reads_no_a_root_lets_an_a_written_after_its_noise_cancel_it():
    fixed = hashlib.sha512(b"a key that reads no root(A)").digest()
    a = _cancelling(fixed)
    b1, b2 = _weights(1), _weights(2)
    z1, r1 = HT.reference(a, b1, SALT, POSITION, (fixed, ROOT_B))
    z2, r2 = HT.reference(a, b2, SALT, POSITION, (fixed, ROOT_B))
    assert z1 != z2
    assert r1 == r2


def test_the_served_key_reads_root_a_so_the_same_late_a_meets_other_bits():
    a = _cancelling(hashlib.sha512(b"a key that reads no root(A)").digest())
    root_a = hashlib.sha512(bytes(v for row in a for v in row)).digest()
    rows, _ = HT.noise(SD.seed(SALT, POSITION, root_a, ROOT_B), LAY)
    nonzero = sum((a[i][t] + rows[i][0][t]) & 0xFF != 0 for i in range(LAY.r) for t in range(LAY.k))
    assert nonzero > LAY.r * LAY.k // 4
    r1 = HT.reference(a, _weights(1), SALT, POSITION, (root_a, ROOT_B))[1]
    r2 = HT.reference(a, _weights(2), SALT, POSITION, (root_a, ROOT_B))[1]
    assert r1 != r2
