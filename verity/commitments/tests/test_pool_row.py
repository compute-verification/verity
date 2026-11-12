"""`pool_row`: the carried pool's row decodes to itself, its widths are NW = bitlen(NB − 1) per block number and 512 bits
a key, the root's packing is `rowleaf.bit_row` of its bits, the empty pool's row is every block free in order, and a
malformed row is refused."""
from __future__ import annotations

import random

import pytest

from verity.commitments import rowleaf as RL

from verity.commitments import pool_row as R
from verity.definitions.commitments.gates import block_key as BK

KA, KB = BK.block_key(BK.ROOT, [1, 2, 3, 4], 4), BK.block_key(BK.ROOT, [5, 6, 7, 8], 4)
ROW = R.PoolRow(3, 4, prev=(3, 5, 0, 2, 1), next=(5, 4, 1, 0, 2), cprev=(0, 5, 0, 0, 0), cnext=(0, 0, 0, 0, 2),
                mlast=(1, 0, 0, 0, 2), key=(KA, KB, R.NOKEY, R.NOKEY, KB))


def test_a_row_decodes_to_itself_at_its_widths() -> None:
    bits = R.encode(ROW)
    assert R.fields(6) == [("head", 1, 3), ("tail", 1, 3), ("prev", 5, 3), ("next", 5, 3), ("cprev", 5, 3),
                           ("cnext", 5, 3), ("mlast", 5, 3), ("key", 5, 512)]
    assert len(bits) == 2 * 3 + 5 * 5 * 3 + 5 * 512 and bits[:6] == [1, 1, 0, 0, 0, 1]
    assert R.decode(6, bits) == ROW
    assert len(R.root(ROW)) == 64 and R.root(ROW) != R.root(ROW._replace(head=1))


@pytest.mark.parametrize("nb", [2, 3, 6, 9, 17, 100])
def test_the_root_packs_the_row_as_bit_row_packs_its_bits(nb) -> None:
    rng = random.Random(nb)
    NW, n = R.width(nb), nb - 1
    row = R.PoolRow(rng.randrange(1 << NW), rng.randrange(1 << NW),
                    *(tuple(rng.randrange(1 << NW) for _ in range(n)) for _ in range(5)),
                    tuple(rng.randbytes(64) if rng.random() < 0.7 else R.NOKEY for _ in range(n)))
    assert R.row_bytes(row) == RL.bit_row((R.encode(row), 1))
    assert R.root(row) == RL.sha512_row_seg_digest(*RL.bit_row((R.encode(row), 1)))


@pytest.mark.parametrize("nb, nw", [(2, 1), (3, 2), (4, 2), (5, 3), (8, 3), (9, 4), (6286, 13)])
def test_a_block_number_has_nw_bits(nb, nw) -> None:
    assert R.width(nb) == nw


def test_the_empty_pool_is_every_block_free_in_order() -> None:
    e = R.empty(6)
    assert (e.head, e.tail, e.prev, e.next) == (1, 5, (0, 1, 2, 3, 4), (2, 3, 4, 5, 0))
    assert e.key == (R.NOKEY,) * 5 and R.genesis(6) == R.root(e) != R.genesis(7)


def test_a_malformed_row_is_refused() -> None:
    with pytest.raises(ValueError, match="one element a block"):
        R.encode(ROW._replace(mlast=(0,) * 4))
    with pytest.raises(ValueError, match="doesn't fit 3 bits"):
        R.encode(ROW._replace(head=8))
    for bad in (ROW._replace(head=8), ROW._replace(cnext=(0, 0, 0, 0, 8)), ROW._replace(prev=(-1, 5, 0, 2, 1))):
        with pytest.raises(ValueError, match="doesn't fit 3 bits"):
            R.row_bytes(bad)
    with pytest.raises(ValueError, match="one element a block"):
        R.row_bytes(ROW._replace(key=(KA,) * 4))
    with pytest.raises(ValueError, match="holds 2641 bits, not 2640"):
        R.decode(6, R.encode(ROW)[:-1])
    with pytest.raises(ValueError, match="has none to cache in"):
        R.empty(1)


@pytest.mark.parametrize("pack", [R.encode, R.row_bytes, R.root])
@pytest.mark.parametrize("key", [b"", KA[:63], KA + b"\x01"], ids=["0 bytes", "63 bytes", "65 bytes"])
def test_a_key_that_is_not_64_bytes_is_refused(pack, key) -> None:
    """`row_bytes` and `root` refuse the keys `encode` refuses. A 0-byte key would pack as 512 zero bits, `NOKEY`'s, and
    a 65-byte key as its first 64 bytes, so each would share a root with another row."""
    for at in range(5):
        with pytest.raises(ValueError, match=f"a digest is 64 bytes, not {len(key)}$"):
            pack(ROW._replace(key=ROW.key[:at] + (key,) + ROW.key[at + 1:]))


@pytest.mark.parametrize("pack", [R.encode, R.row_bytes, R.root])
@pytest.mark.parametrize("key", [KA[:63], KA + b"\x01"], ids=["63 bytes", "65 bytes"])
def test_a_zero_byte_key_hides_no_later_key(pack, key) -> None:
    with pytest.raises(ValueError, match="a digest is 64 bytes, not 0$"):
        pack(ROW._replace(key=(b"", key, R.NOKEY, R.NOKEY, KB)))
