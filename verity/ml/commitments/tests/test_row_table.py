"""Row tables (`row_table`): the root is frame-v3-sha512's over `row_leaf` of hm96-sha512's tree leaf of each s8 row's
`sha512/row-seg/v1` commit string, an opening of any set of ranks verifies from public data, and a tampered row, salt, string,
rank, root, domain or proof does not.  Steps: a step's root is the tree over its tables' roots and its subtree of the window's
table, the window's root is the tree over its steps' roots, and a step opening verifies against its step's root while a
tampered row, table root, index, context, step or count does not."""

from __future__ import annotations

import random

import pytest

from verity.errors import ProtocolError
from verity.ml.commitments import hm96
from verity.ml.commitments import row_table as RT
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth, tree_levels
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


@pytest.mark.parametrize("n", [1, 2, 5, 33])
def test_a_rows_path_climbs_from_its_leaf_to_the_root_and_from_no_other(n):
    """`path` is the single path a registered read opens (Lean's `Registered.opens`): siblings leaf level first, the frame-v3
    node at each level, over the tree padded to a power of two."""
    t, _rows, _salts = table(n)
    dom = t.domain
    for r in range(n):
        path, depth = t.path(r), merkle_depth(n)
        assert len(path) == 64 * depth

        def climb(rank: int, bc: bytes) -> bytes:
            digest, i = RT.leaf(dom.domain_id, rank, bc), rank
            for level in range(depth):
                sib = path[64 * level:64 * level + 64]
                digest = dom.node(level, i >> 1, *((digest, sib) if i % 2 == 0 else (sib, digest)))
                i >>= 1
            return digest

        assert climb(r, t.strings[r]) == t.root
        if n > 1:
            assert climb(r ^ 1 if r ^ 1 < n else r - 1, t.strings[r]) != t.root
            assert climb(r, t.strings[r - 1]) != t.root


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


CTX = bytes(range(64))
#: rows per table of each step: a prefill step of 3 tokens with 3 linears, two decode steps of 2 requests with 3, and one with 1
STEPS = [(3, 3, 3), (2, 2, 2), (2, 2, 2), (1,)]


def steps(ctx: bytes = CTX, seed: int = 5) -> tuple[list[RT.Step], list[list[tuple[list, list]]]]:
    rng = random.Random(seed)
    data = [[([RT.s8_row([rng.randrange(-128, 128) for _ in range(64)], rng.getrandbits(32)) for _ in range(n)],
              [rng.randbytes(192) for _ in range(n)]) for n in ns] for ns in STEPS]
    return [RT.commit_step(ctx, s, d) for s, d in enumerate(data)], data


def window_domain(n: int) -> CommitmentDomain:
    return CommitmentDomain(b"\x09" * 64, -1, RangeIndexedDomain(0, n), hash="sha512")


def fold(dom: CommitmentDomain, roots: list[bytes]) -> bytes:
    """frame-v3-sha512's root over ``roots`` as ROOTS_SCHEMA leaves, by merkle's own leaf and tree."""
    return tree_levels(dom, [dom.leaf(i, i, RT.ROOTS_SCHEMA, r) for i, r in enumerate(roots)])[-1][0]


def test_a_step_root_is_the_tree_over_its_tables_and_its_subtree_of_the_window_table():
    ss, data = steps()
    win = RT.RootTree(window_domain(len(ss)), [s.root for s in ss])
    for s, (step, d) in zip(ss, enumerate(data)):
        tables = [RT.commit(RT.table_domain(CTX, step, i, len(rows)), rows, salts) for i, (rows, salts) in enumerate(d)]
        assert [t.root for t in s.tables] == [t.root for t in tables]
        assert s.root == fold(RT.step_domain(CTX, step, len(d)), [t.root for t in tables]) == win.roots[step]
        o = win.open([step])
        assert RT.verify_roots(win.root, win.domain.domain_id, len(ss), o) and o.roots == (s.root,)
    assert [RT.root_leaf(win.domain.domain_id, i, r) for i, r in enumerate(win.roots)] == \
        [win.domain.leaf(i, i, RT.ROOTS_SCHEMA, r) for i, r in enumerate(win.roots)]


def test_the_window_root_is_the_tree_over_the_step_roots():
    ss, _ = steps()
    roots = [s.root for s in ss]
    win = RT.RootTree(window_domain(len(ss)), roots)
    assert win.root == fold(window_domain(len(ss)), roots) and win.count == len(ss)
    assert RT.RootTree(window_domain(len(ss)), roots[::-1]).root != win.root
    assert RT.RootTree(window_domain(len(ss) - 1), roots[:-1]).root != win.root
    o = win.open([3, 0, 2])
    assert RT.verify_roots(win.root, win.domain.domain_id, 4, o) and RT.RootOpening.of_json(o.to_json()) == o
    assert o.nbytes == 64 * (3 + len(o.proof))
    assert not RT.verify_roots(win.root, win.domain.domain_id, 4, RT.RootOpening(o.indices, (roots[1], *o.roots[1:]), o.proof))
    assert not RT.verify_roots(win.root, window_domain(5).domain_id, 5, o)


def test_a_step_opening_verifies_against_its_step_root_and_a_tampered_one_does_not():
    ss, data = steps()
    s, (rows, salts) = ss[1], data[1][2]
    o = s.open(2, [1, 0])
    honest = {r: (*rows[r], salts[r]) for r in (0, 1)}
    assert (o.index, o.root, o.rows.ranks) == (2, s.tables[2].root, (0, 1))
    assert RT.verify_step(s.root, CTX, 1, 3, 2, o) and RT.verify_step(s.root, CTX, 1, 3, 2, o, honest)
    assert RT.StepOpening.of_json(o.to_json()) == o and o.nbytes == o.table.nbytes + o.rows.nbytes
    row, bits = rows[1]
    assert not RT.verify_step(s.root, CTX, 1, 3, 2, o, {**honest, 1: (bytes([row[0] ^ 1]) + row[1:], bits, salts[1])})
    assert not RT.verify_step(s.root, CTX, 1, 3, 2, o, {**honest, 1: (row, bits, salts[0])})
    other = ss[2].open(2, [0, 1])                                      # the same shape, another step's table
    assert not RT.verify_step(s.root, CTX, 1, 3, 2, RT.StepOpening(o.table, other.rows))
    assert not RT.verify_step(s.root, CTX, 1, 3, 2, RT.StepOpening(other.table, o.rows))
    for bad in [(ss[2].root, CTX, 1, 3, 2), (s.root, bytes(64), 1, 3, 2), (s.root, CTX, 2, 3, 2), (s.root, CTX, 1, 4, 2),
                (s.root, CTX, 1, 3, 3)]:
        assert not RT.verify_step(*bad, o)
    swapped = s.open(1, [0, 1])
    assert not RT.verify_step(s.root, CTX, 1, 3, 2, RT.StepOpening(swapped.table, o.rows))
    assert not RT.verify_step(s.root, CTX, 1, 3, 2, RT.StepOpening(RT.RootOpening((1,), o.table.roots, o.table.proof), o.rows))


def test_serve_time_domains_bind_context_step_index_and_count_and_a_step_refuses_a_table_outside_them():
    ids = {RT.table_domain(c, s, i, n).domain_id for c in (CTX, bytes(64)) for s in (0, 1) for i in (0, 1) for n in (1, 2)}
    assert len(ids) == 16 and RT.step_domain(CTX, 0, 2).domain_id not in ids
    assert steps(bytes(64))[0][0].root != steps()[0][0].root
    with pytest.raises(ProtocolError, match="64 bytes"):
        RT.table_domain(CTX[:32], 0, 0, 1)
    with pytest.raises(ProtocolError, match="nonnegative"):
        RT.step_domain(CTX, -1, 1)
    ss, _ = steps()
    with pytest.raises(ProtocolError, match="table 1 is not under its table domain"):
        RT.Step(CTX, 0, [ss[0].tables[0], ss[0].tables[2]])
    with pytest.raises(ProtocolError, match="one table's root"):
        RT.StepOpening(ss[0].tree.open([0, 1]), ss[0].tables[0].open([0]))
    with pytest.raises(ProtocolError, match="64 bytes"):
        RT.RootTree(window_domain(1), [bytes(32)])
