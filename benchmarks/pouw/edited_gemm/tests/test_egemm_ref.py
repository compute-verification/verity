"""egemm_ref.py against PoUW's receipt reference: receipt_conform.py's egemm_ref check (the r9 keys' published vectors,
and one whole tile's XOR-in-13 and duo receipts, every thread, against receipt_folds' schedules) passes, refuses a module
with one receipt word wrong by naming its thread, and reads exactly the names `__all__` offers."""
from __future__ import annotations

from types import SimpleNamespace

import egemm_ref as E
import pytest

from verity.protocols.pouw.receipts import receipt_conform as C


def test_the_receipts_agree_with_the_reference():
    """Two checks a thread, and the published key vectors beyond them."""
    assert C.egemm_ref(E) > 2 * E.NCT


@pytest.mark.parametrize("name", ["xin13_receipts", "tile_receipts"])
def test_one_wrong_receipt_word_is_refused(name):
    def wrong(seed, Ts, tile):
        r = getattr(E, name)(seed, Ts, tile)
        r[7, 2] ^= 1
        return r

    er = SimpleNamespace(**{n: getattr(E, n) for n in E.__all__})
    setattr(er, name, wrong)
    with pytest.raises(AssertionError, match=f"{name} thread 7"):
        C.egemm_ref(er)


def test_the_check_reads_what_the_module_offers():
    read = set()

    class Recorded:
        def __getattr__(self, name):
            read.add(name)
            return getattr(E, name)

    C.egemm_ref(Recorded())
    assert read == set(E.__all__), "egemm_ref.__all__ is what receipt_conform's egemm_ref check reads: list those names"
