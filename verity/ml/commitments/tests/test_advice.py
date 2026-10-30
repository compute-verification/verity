"""`verity.ml.commitments.advice`: hiding rows in one frame-v3-sha512 tree.  Each leaf is the hm96-sha512/row/v2 leaf the
frame vectors pin, every row opens against the root, and a changed row, salt, index, binding or count does not."""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from verity.ml.commitments import advice as A
from verity.ml.commitments import hm96
from verity.ml.commitments.frame_v3 import BitRowLeaf
from verity.ml.commitments.frame_v3.vectors import expand
from verity.ml.commitments.rowleaf import row_leaf

BINDING = hashlib.sha512(b"test/advice").digest()[:32]
N_BITS = 1024


def _rows(n: int) -> list[bytes]:
    return [expand(f"test/advice/row/{i}", N_BITS // 8) for i in range(n)]


def _salts(n: int) -> list[bytes]:
    return [expand(f"test/advice/salt/{i}", A.INST.salt_bytes) for i in range(n)]


def test_a_leaf_is_the_hiding_row_leaf():
    rows, salts = _rows(3), _salts(3)
    reg = A.register(rows, N_BITS, BINDING, salts)
    d, inst, kind = A.domain(BINDING, 3), hm96.SHA512, BitRowLeaf(N_BITS)
    for i in range(3):
        bc = inst.commit_string(inst.default_key, kind.layout().digest(rows[i]), salts[i])
        assert reg.leaves[i] == row_leaf(d, i, inst.tree_leaf(inst.default_key, bc), "hm96-sha512/row/v2")
    assert d.hash == "sha512" and d.owner == -1 and len(reg.root) == 64


@pytest.mark.parametrize("n", [1, 2, 5, 16, 33])
def test_every_row_opens_and_nothing_else_does(n):
    reg = A.register(_rows(n), N_BITS, BINDING, _salts(n))
    for i in range(n):
        o = reg.open(i)
        assert A.verify(reg.root, n, N_BITS, BINDING, o)
        assert not A.verify(reg.root, n, N_BITS, BINDING, dataclasses.replace(o, row=bytes(N_BITS // 8)))
        assert not A.verify(reg.root, n, N_BITS, BINDING, dataclasses.replace(o, salt=bytes(A.INST.salt_bytes)))
        assert not A.verify(reg.root, n, N_BITS, hashlib.sha512(b"other").digest()[:32], o)
        assert not A.verify(reg.root, n + 1, N_BITS, BINDING, o)
        if n > 1:
            assert not A.verify(reg.root, n, N_BITS, BINDING, dataclasses.replace(o, index=(i + 1) % n))
    assert not A.verify(reg.root, n, N_BITS, BINDING, dataclasses.replace(reg.open(0), row=b"short"))


def test_fresh_salts_hide_equal_rows():
    rows = [bytes(N_BITS // 8)] * 4
    a, b = A.register(rows, N_BITS, BINDING), A.register(rows, N_BITS, BINDING)
    assert a.root != b.root and len(set(a.leaves)) == 4 and all(len(s) == A.INST.salt_bytes for s in a.salts)


def test_malformed_registrations_are_refused():
    with pytest.raises(ValueError):
        A.register([], N_BITS, BINDING)
    with pytest.raises(ValueError):
        A.register([b"\0" * 3], N_BITS, BINDING)
    with pytest.raises(ValueError):
        A.register(_rows(2), N_BITS, BINDING, _salts(1))
    with pytest.raises(ValueError):
        A.register(_rows(1), N_BITS, b"short")
