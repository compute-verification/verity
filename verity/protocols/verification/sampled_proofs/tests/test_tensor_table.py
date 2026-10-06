"""W committed by weight row and its tensor table (`one_stage.tensor_table`, `verity/tensor-table/v1`): the table's entries
and their encoding, the well-formedness W1-W4 a statement proves about the hidden table, a row's place in W and W's byte
stream, and the two trees' leaves."""
from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from verity.primitives.commitments import hm96
from verity.protocols.verification.sampled_proofs.one_stage import registered as G, tensor_table as TT

DEPLOYMENT = "deployment/test-model@2026-10-06"
#: a 4-element BF16 row tensor of two rows, a 12-element one of three (24-byte rows), a 1-row 32-bit tensor
TENSORS = (TT.Tensor("embed", 2, 64), TT.Tensor("layers.0.mlp.down", 3, 192), TT.Tensor("norm", 1, 64, 32))
CAP = 5
LEAVES, BYTES = 6, 2 * 8 + 3 * 24 + 8
#: the roots of that table (salts 01 00.., 01 01.., ...) and of W over `_rows()` (salts 02 00.., ...)
TABLE_ROOT = ("3e85926c2e955bc35705db8465e9b8b25a893f72e785bd5fcc2e5157a7002e1c"
              "c67f634ccb6a4af1b2d41f6c912adb0f7d72be054c750b8c780a2481cca9af9f")
WEIGHTS_ROOT = ("2f5460cf1349f70a9319f4c56078d4869d9f73a695becded7f9218985996e7d0"
                "e9cecd0c835683949721faa929e1a5d275b69c63660ca5a016b33c81e338102b")


def _rows() -> list[bytes]:
    return [bytes((17 * i + j) % 256 for j in range(n)) for i, n in enumerate([8, 8, 24, 24, 24, 8])]


def _salts(n: int, tag: int) -> list[bytes]:
    return [bytes([tag, i]) * (hm96.SHA512.salt_bytes // 2) for i in range(n)]


def _row_v2(row: bytes, n_bits: int) -> bytes:
    return hashlib.sha512((b"verity/sha512-row/v2\0" + n_bits.to_bytes(8, "big")).ljust(128, b"\0") + row).digest()


def test_the_table_tiles_w_and_pads_to_its_public_size():
    t = TT.table(TENSORS, CAP)
    assert [(e.first, e.rows, e.row_bits, e.elem_bits, e.offset) for e in t] == [
        (0, 2, 64, 16, 0), (2, 3, 192, 16, 16), (5, 1, 64, 32, 88), (6, 0, 0, 0, 96), (6, 0, 0, 0, 96)]
    assert t[1].name == hashlib.sha512(b"layers.0.mlp.down").digest() and t[3].padding and not t[2].padding
    TT.check(t, LEAVES, BYTES, CAP)
    with pytest.raises(ValueError, match="do not fit"):
        TT.table(TENSORS, 2)
    for bad in (TT.Tensor("odd", 1, 12), TT.Tensor("empty", 0, 64), TT.Tensor("ragged", 1, 64, 24)):
        with pytest.raises(ValueError, match="no tensor of whole-byte rows"):
            TT.table([bad], CAP)


def test_an_entry_is_52_u16_words_little_endian():
    e = TT.table(TENSORS, CAP)[1]
    w = TT.entry_words(e)
    assert len(w) == TT.ENTRY_WORDS == 52 and TT.ENTRY_BITS == 832
    assert w[:20] == [2, 0, 0, 0, 3, 0, 0, 0, 192, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0]
    assert b"".join(x.to_bytes(2, "little") for x in w[20:]) == e.name
    assert TT.decode_entry(w) == e
    with pytest.raises(ValueError, match="52 u16 words"):
        TT.decode_entry(w[:-1])


@pytest.mark.parametrize("edit, want", [
    (lambda t: [replace(t[0], first=1), *t[1:]], "W1"),
    (lambda t: [t[0], replace(t[1], first=3), *t[2:]], "W2"),
    (lambda t: [t[0], replace(t[1], offset=17), *t[2:]], "W2"),
    (lambda t: [t[0], t[1], t[3], replace(t[2], first=5, offset=88), t[4]], "W2"),
    (lambda t: [t[0], replace(t[1], row_bits=190), *t[2:]], "W3"),
    (lambda t: [t[0], replace(t[1], elem_bits=7), *t[2:]], "W3"),
    (lambda t: [*t[:3], replace(t[3], name=b"\1" * 64), t[4]], "W3"),
    (lambda t: [*t[:4], replace(t[4], rows=1, row_bits=64, elem_bits=16)], "W3"),
])
def test_each_rule_refuses_a_table_that_breaks_it(edit, want):
    with pytest.raises(ValueError, match=want):
        TT.check(edit(TT.table(TENSORS, CAP)), LEAVES, BYTES, CAP)


def test_the_table_ends_w_and_has_its_public_size():
    t = TT.table(TENSORS, CAP)
    for leaves, total in ((LEAVES + 1, BYTES), (LEAVES, BYTES - 8)):
        with pytest.raises(ValueError, match="W4"):
            TT.check(t, leaves, total, CAP)
    with pytest.raises(ValueError, match="public size is 6"):
        TT.check(t, LEAVES, BYTES, CAP + 1)
    with pytest.raises(ValueError, match="W3: entry 0"):
        TT.check(t, LEAVES, BYTES, CAP, min_row_bits=128)


def test_a_row_of_w_and_the_rows_a_byte_range_falls_in():
    t = TT.table(TENSORS, CAP)
    assert [TT.locate(t, i) for i in range(LEAVES)] == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0)]
    with pytest.raises(ValueError, match="not a row of W"):
        TT.locate(t, LEAVES)
    stream, rows = b"".join(_rows()), _rows()
    assert TT.byte_rows(t, 10, 30) == [(1, 2, 8), (2, 0, 14)]
    assert TT.byte_rows(t, 16, 40) == [(2, 0, 24)]
    for lo in range(BYTES):
        for hi in range(lo + 1, BYTES + 1):
            got = TT.byte_rows(t, lo, hi)
            assert b"".join(rows[i][a:b] for i, a, b in got) == stream[lo:hi]
            if hi - lo - 1 <= 8:
                assert len(got) <= 2, "every row is at least 8 bytes: a range of at most 9 bytes touches two rows"
    with pytest.raises(ValueError, match="not bytes of W"):
        TT.byte_rows(t, BYTES - 1, BYTES + 1)


def test_w_and_its_table_are_registered_rows_and_w_binds_the_table():
    t = TT.table(TENSORS, CAP)
    tab = TT.commit_table(DEPLOYMENT, t, _salts(CAP, 1))
    for k, e in enumerate(t):
        x = _row_v2(b"".join(w.to_bytes(2, "little") for w in TT.entry_words(e)), 832)
        assert tab.bcs[k] == hm96.SHA512.commit_string(hm96.SHA512.default_key, x, tab.salts[k])
        assert TT.verify_entry(tab.domain, tab.root, k, tab.bcs[k], tab.path(k))
    w = TT.commit_weights(DEPLOYMENT, t, _rows(), tab.root, _salts(LEAVES, 2))
    for i, row in enumerate(_rows()):
        bits = t[TT.locate(t, i)[0]].row_bits
        assert w.bcs[i] == hm96.SHA512.commit_string(hm96.SHA512.default_key, _row_v2(row, bits), w.salts[i])
        assert G.verify_read(w.domain, w.root, i, w.bcs[i], w.path(i))
    assert w.domain == TT.weights_domain(DEPLOYMENT, LEAVES, BYTES, tab.root) and w.domain.count == LEAVES
    other = TT.commit_table(DEPLOYMENT, t, _salts(CAP, 3))
    assert TT.weights_domain(DEPLOYMENT, LEAVES, BYTES, other.root) != w.domain, "W's domain binds its table's root"
    assert TT.table_domain("another", CAP) != tab.domain
    m = TT.manifest(DEPLOYMENT, CAP, LEAVES, BYTES, tab.root, w.root)
    assert m["format"] == TT.FORMAT and m["schema"] == "hm96-sha512/row/v2" and m["table"] == tab.root.hex()
    assert (tab.root.hex(), w.root.hex()) == (TABLE_ROOT, WEIGHTS_ROOT), "the pinned format: entries, domains and leaves"


def test_a_row_committed_at_another_length_opens_no_read_of_its_own():
    """The digest's prefix carries the row's bits: tensor 2's row read as 48 bits (its first 6 bytes, as a forged table
    entry would name it) is not the committed 64-bit row."""
    t = TT.table(TENSORS, CAP)
    tab = TT.commit_table(DEPLOYMENT, t, _salts(CAP, 1))
    w = TT.commit_weights(DEPLOYMENT, t, _rows(), tab.root, _salts(LEAVES, 2))
    row = _rows()[5]
    short = hm96.SHA512.commit_string(hm96.SHA512.default_key, _row_v2(row[:6], 48), w.salts[5])
    assert not G.verify_read(w.domain, w.root, 5, short, w.path(5))
    with pytest.raises(ValueError, match="its tensor's rows are 8"):
        TT.commit_weights(DEPLOYMENT, t, [*_rows()[:5], row[:6]], tab.root)
    with pytest.raises(ValueError, match="names 6 rows of W"):
        TT.commit_weights(DEPLOYMENT, t, _rows()[:5], tab.root)
