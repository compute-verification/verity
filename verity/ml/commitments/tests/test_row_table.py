"""Row tables (`row_table`): the root is frame-v3-sha512's over `row_leaf` of hm96-sha512's tree leaf of each s8 row's
`sha512/row-seg/v1` commit string, an opening of any set of ranks verifies from public data, and a tampered row, salt, string,
rank, root, domain or proof does not."""

from __future__ import annotations

import random

import pytest

from verity.errors import ProtocolError
from verity.ml.commitments import hm96
from verity.ml.commitments import row_table as RT
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, row_leaf, sha512_row_seg_digest


def domain(n: int, binding: bytes = b"\x07" * 64) -> CommitmentDomain:
    return CommitmentDomain(binding, -1, RangeIndexedDomain(0, n), hash="sha512")


def table(n: int, k: int = 64, seed: int = 1) -> tuple[RT.RowTable, list[tuple[bytes, int]], list[bytes]]:
    rng = random.Random(seed)
    rows = [RT.s8_row([rng.randrange(-128, 128) for _ in range(k)], rng.getrandbits(32)) for _ in range(n)]
    salts = [rng.randbytes(192) for _ in range(n)]
    return RT.commit(domain(n), rows, salts), rows, salts


def test_an_s8_row_is_its_codes_then_its_scale():
    row, bits = RT.s8_row([-1, 0, 127, -128], 0x3F000000)
    assert (row, bits) == (bytes([0xFF, 0, 0x7F, 0x80, 0, 0, 0, 0x3F]), 64)
    assert RT.s8_row(bytes([0xFF, 0, 0x7F, 0x80]), 0x3F000000) == (row, bits)
    assert RT.s8_row([0] * 4096, 0)[1] == 8 * 4096 + 32
    for bad in ([128], [-129]):
        with pytest.raises(ValueError, match="s8 code"):
            RT.s8_row(bad, 0)
    with pytest.raises(ValueError, match="bit pattern"):
        RT.s8_row([0], 1 << 32)


@pytest.mark.parametrize("n,k", [(1, 64), (5, 64), (32, 64), (33, 64), (3, 4096)])
def test_the_root_is_frame_v3_sha512s_over_hm96_row_seg_leaves(n, k):
    t, rows, salts = table(n, k)
    inst, key, dom = hm96.SHA512, hm96.SHA512.default_key, domain(n)
    bcs = [inst.commit_string(key, sha512_row_seg_digest(r, b), y) for (r, b), y in zip(rows, salts)]
    leaves = [row_leaf(dom, i, inst.tree_leaf(key, bc), SCHEMA_HM96_SHA512_ROW_SEG) for i, bc in enumerate(bcs)]
    assert list(t.strings) == bcs and t.count == n
    assert [RT.leaf(dom.domain_id, i, bc) for i, bc in enumerate(bcs)] == leaves
    assert t.root == tree_levels(dom, leaves)[-1][0]
    assert RT.RowTable(domain(n, b"\x08" * 64), bcs).root != t.root


@pytest.mark.parametrize("ranks", [[0], [4], [0, 1, 2, 3, 4], [1, 3], [4, 0, 4]])
def test_an_opening_verifies_against_the_root(ranks):
    t, rows, salts = table(5)
    o = t.open(ranks)
    assert o.ranks == tuple(sorted(set(ranks))) and o.strings == tuple(t.strings[r] for r in o.ranks)
    assert RT.verify(t.root, t.domain.domain_id, 5, o)
    assert RT.verify_rows(t.root, t.domain.domain_id, 5, o, {r: (*rows[r], salts[r]) for r in o.ranks})
    assert RT.RowOpening.of_json(o.to_json()) == o
    assert o.nbytes == 128 * len(o.ranks) + 64 * len(o.proof)


def test_a_tampered_row_salt_string_rank_root_domain_or_proof_fails():
    t, rows, salts = table(6)
    root, did, o = t.root, t.domain.domain_id, t.open([1, 4])
    honest = {r: (*rows[r], salts[r]) for r in (1, 4)}
    assert RT.verify_rows(root, did, 6, o, honest)
    row, bits = rows[4]
    flipped = bytes([row[0] ^ 1]) + row[1:]
    assert not RT.verify_rows(root, did, 6, o, {**honest, 4: (flipped, bits, salts[4])})
    assert not RT.verify_rows(root, did, 6, o, {**honest, 4: (row, bits, salts[3])})
    assert not RT.verify_rows(root, did, 6, o, {**honest, 4: (row[:-1], bits, salts[4])})
    assert not RT.verify_rows(root, did, 6, o, {1: honest[1]})
    bad = o.strings[1][:5] + bytes([o.strings[1][5] ^ 1]) + o.strings[1][6:]
    assert not RT.verify(root, did, 6, RT.RowOpening(o.ranks, (o.strings[0], bad), o.proof))
    assert not RT.verify(root, did, 6, RT.RowOpening((1, 5), o.strings, o.proof))
    assert not RT.verify(root, did, 6, RT.RowOpening(o.ranks, o.strings[::-1], o.proof))
    assert not RT.verify(bytes(64), did, 6, o)
    assert not RT.verify(root, domain(6, b"\x08" * 64).domain_id, 6, o)
    assert not RT.verify(root, did, 9, o) and not RT.verify(root, domain(7).domain_id, 7, o)
    assert not RT.verify(root, did, 6, RT.RowOpening(o.ranks, o.strings, o.proof[:-1]))
    assert not RT.verify(root, did, 6, RT.RowOpening(o.ranks, o.strings, o.proof + (bytes(64),)))


def test_a_malformed_opening_or_table_is_refused():
    t, _rows, _salts = table(3)
    o = t.open([0, 2])
    with pytest.raises(ProtocolError, match="ascending"):
        RT.RowOpening((2, 0), o.strings, o.proof)
    with pytest.raises(ProtocolError, match="128 bytes"):
        RT.RowOpening(o.ranks, (o.strings[0], o.strings[1][:-1]), o.proof)
    with pytest.raises(ProtocolError, match="frame-v3-sha512"):
        RT.RowTable(CommitmentDomain(b"\x07" * 64, -1, RangeIndexedDomain(0, 3)), t.strings)
    with pytest.raises(ProtocolError, match="3 rows, 2 salts"):
        RT.commit(domain(3), [RT.s8_row([0], 0)] * 3, [bytes(192)] * 2)
    with pytest.raises(ProtocolError, match="outside"):
        t.open([3])
