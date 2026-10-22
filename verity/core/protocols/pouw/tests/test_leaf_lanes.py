"""`pouw.leaf_lanes`: the stated lane geometry and partial word order held to the circuits (`h1tile`, `h1zy`), and the
statement set's reads and ``--registered`` files."""
from __future__ import annotations

import random

import pytest

from verity.core.protocols.pouw import leaf_lanes as LL
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import h1zy as Z
from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.pouw.circuit import segment as SG
from verity.core.protocols.sampled_proofs.one_stage import produced as PR
from verity.core.protocols.sampled_proofs.one_stage import registered as G

SALT, POSITION, ROOTS = bytes(range(64)), 11, (b"\x01" * 64, b"\x02" * 64)


@pytest.mark.parametrize("rc", [(32, 32), (8, 16), (16, 32), (4, 8)])
def test_the_lane_geometry_is_the_tiles(rc):
    lay = H.Layout(*rc, 64)
    assert LL.lanes(*rc) == lay.lanes
    for ln in range(lay.lanes):
        assert LL.lane_rows(*rc, ln) == tuple(lay.rows(ln))
        assert LL.lane_cols(*rc, ln) == tuple(lay.cols(ln))
        assert LL.key_block(ln) == lay.lane(ln)
    cells = sorted((i, j) for ln in range(lay.lanes) for i in LL.lane_rows(*rc, ln) for j in LL.lane_cols(*rc, ln))
    assert cells == [(i, j) for i in range(rc[0]) for j in range(rc[1])], "the lanes hold the leaf's cells once each"


def test_a_32_by_32_leafs_lane_is_4g_plus_q():
    for ln in range(32):
        g, q = ln >> 2, ln & 3
        assert LL.lane_rows(32, 32, ln) == tuple(g + 8 * s for s in range(4))
        assert LL.lane_cols(32, 32, ln) == tuple(2 * q + 8 * (t >> 1) + (t & 1) for t in range(8))
    with pytest.raises(ValueError, match="lanes"):
        LL.lane_rows(32, 32, 32)
    with pytest.raises(ValueError, match="4 rows and 8 columns"):
        LL.lanes(30, 32)


def test_the_word_count_is_the_definitions():
    assert all(LL.n_digits(ks) == SG.n_digits(ks) for ks in range(1, 3000))
    for k in (64, 384, 4096, 14336):
        assert LL.words(k) == 2 * H.Layout(32, 32, k).sets
    assert LL.words(4096) == 260
    assert LL.word_order(64) == ["T0.y", "T0.z", "T1.y", "T1.z", "O'.y", "O'.z", "D1.y", "D1.z"]
    assert LL.word_order(4096)[-4:] == ["O'.y", "O'.z", "D1.y", "D1.z"]


def test_a_lanes_partial_is_its_chains_halves_after_each_set_in_the_stated_order():
    rng = random.Random(7)
    rc, k = (8, 16), 64
    lay = H.Layout(*rc, k)
    a = [[rng.randrange(256) for _ in range(k)] for _ in range(rc[0])]
    b = [[rng.randrange(256) for _ in range(k)] for _ in range(rc[1])]
    rows, cols = H.noise(SD.seed(SALT, POSITION, *ROOTS), lay)
    keys = H.fold_keys(SD.receipt_seed(SALT, POSITION, *ROOTS), lay)
    names = LL.word_order(k)
    parts = []
    for ln in range(lay.lanes):
        got = Z.lane_partial([a[i] for i in LL.lane_rows(*rc, ln)], [b[j] for j in LL.lane_cols(*rc, ln)], [1] * rc[0],
                             SALT, POSITION, ROOTS, ln, rc)
        st, want = list(keys[LL.key_block(ln)]), {}
        for s, p in zip([n[:-2] for n in names[::2]], H.lane_sets(a, b, rows, cols, lay, ln), strict=True):
            st = SG.absorb(H.FOLD, st, p)
            want[f"{s}.y"], want[f"{s}.z"] = H._xor_words(st[0:8]), H._xor_words(st[8:16])
        assert got == [want[n] for n in names]
        parts.append(got)
    receipt = [0] * len(names)
    for p in parts:
        receipt = [x ^ v for x, v in zip(receipt, p)]
    assert receipt == H.reference(a, b, SALT, POSITION, ROOTS)[1], "the rows' XOR is the leaf's receipt"
    assert Z.lane_partial([a[i] for i in LL.lane_rows(*rc, 0)], [b[j] for j in LL.lane_cols(*rc, 0)], [0] * rc[0], SALT,
                          POSITION, ROOTS, 0, rc) == [0] * len(names), "a dead leaf's partials are zero"


def test_the_statement_set_and_its_reads():
    sts = LL.statements(32, 32)
    assert sts == [("lane", ln) for ln in range(32)] + [("combine", None)]
    r7 = LL.reads(32, 32, "lane", 7)
    assert r7[LL.PARTIALS] == ("partials", 7) and r7["admit"] == ("admit", 0) and r7["zy"] == ("zy", 0)
    assert [r7[f"a{s}"][1] for s in range(4)] == [1, 9, 17, 25]
    assert [r7[f"b{t}"][1] for t in range(8)] == [6, 7, 14, 15, 22, 23, 30, 31]
    rc = LL.reads(32, 32, "combine")
    assert rc == {**{f"p{x}": ("partials", x) for x in range(32)}, "receipt": ("receipt", 0)}
    with pytest.raises(ValueError, match="lane with its index"):
        LL.reads(32, 32, "combine", 3)
    with pytest.raises(ValueError, match="lane with its index"):
        LL.reads(32, 32, "seeds")


def test_every_statements_partials_are_registered_reads_and_an_unregistered_row_does_not_open():
    """A port the verifier's registered reads (``own``) don't name is free in the checked relation (RT4-LOGIC-29c), so in
    each of the 33 statements every port reading the partials is a registered read of the after-draw tree at its row."""
    for kind, ln in LL.statements(32, 32):
        got = {p: r for p, (v, r) in LL.reads(32, 32, kind, ln).items() if v == LL.PARTIALS}
        assert got == ({LL.PARTIALS: ln} if kind == LL.LANE else {f"p{x}": x for x in range(32)})
    reg, pd = "ab" * 64, "12" * 64
    rows = [[ln, 7] for ln in range(4)]
    mine = PR.commit(pd, LL.PARTIALS, rows, reg, schema=G.SCHEMA_V2)
    resalted = PR.commit(pd, LL.PARTIALS, rows, reg, schema=G.SCHEMA_V2)
    elsewhere = PR.commit(pd, LL.PARTIALS, rows, "cd" * 64, schema=G.SCHEMA_V2)

    def opens(t, position, row):
        return G.verify_read(mine.domain, mine.root, position, t.bcs[row], t.path(position), G.SCHEMA_V2)

    assert all(opens(mine, ln, ln) for ln in range(4))
    assert not opens(mine, 3, 2), "another row of the tree at row 3"
    assert not opens(resalted, 3, 3), "row 3's words committed outside the registered tree"
    assert not opens(elsewhere, 3, 3), "row 3 of a tree under another registration"


def test_a_statements_registered_file_reads_each_value_at_its_row_under_its_root():
    reg, pd = "ab" * 64, "12" * 64
    parts = PR.commit(pd, LL.PARTIALS, [[ln, 0] for ln in range(4)], reg, schema=G.SCHEMA_V2)
    entries = {LL.PARTIALS: PR.entry(pd, LL.PARTIALS, 4, 2, reg), LL.RECEIPT: {**parts.entry(), "domain": "00" * 64}}
    roots = {LL.PARTIALS: parts.root.hex(), LL.RECEIPT: "ff" * 64}
    f = LL.registered_file(LL.reads(8, 16, "combine"), entries, roots, {LL.PARTIALS: G.SCHEMA_V2})
    assert f["format"] == G.READS and sorted(f["ports"]) == ["p0", "p1", "p2", "p3", "receipt"]
    assert [f["ports"][f"p{x}"]["reads"] for x in range(4)] == [[0], [1], [2], [3]]
    assert all(f["ports"][f"p{x}"]["domain"] == parts.domain.domain_id.hex() for x in range(4))
    assert f["ports"]["receipt"]["root"] == "ff" * 64
    with pytest.raises(ValueError, match="does not hold"):
        LL.registered_file(LL.reads(8, 16, "lane", 0), entries, roots)
