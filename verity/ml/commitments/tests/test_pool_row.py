"""`pool_row`: the carried pool's row decodes to itself, its widths are NW = bitlen(NB − 1) per block number and 512 bits
a key, the empty pool's row is every block free in order, and a malformed row is refused."""
from __future__ import annotations

import pytest

from verity.ml.commitments import pool_row as R
from verity.ml.commitments.gates import block_key as BK

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
    with pytest.raises(ValueError, match="holds 2641 bits, not 2640"):
        R.decode(6, R.encode(ROW)[:-1])
    with pytest.raises(ValueError, match="has none to cache in"):
        R.empty(1)
