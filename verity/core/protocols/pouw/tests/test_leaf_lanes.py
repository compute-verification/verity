"""`pouw.leaf_lanes`: the stated lane geometry and partial word order held to the circuits (`h1tile`, `h1zy`), the
statement set's reads, known rows and ``--registered`` files."""
from __future__ import annotations

import itertools
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
    assert r7[LL.PARTIALS] == ("partials", 7) and r7["zy"] == ("zy", 0) and "admit" not in r7
    assert [r7[f"a{s}"][1] for s in range(4)] == [1, 9, 17, 25]
    assert [r7[f"b{t}"][1] for t in range(8)] == [6, 7, 14, 15, 22, 23, 30, 31]
    rc = LL.reads(32, 32, "combine")
    assert rc == {**{f"p{x}": ("partials", x) for x in range(32)}, "receipt": ("receipt", 0)}
    with pytest.raises(ValueError, match="lane with its index"):
        LL.reads(32, 32, "combine", 3)
    with pytest.raises(ValueError, match="lane with its index"):
        LL.reads(32, 32, "seeds")


def test_every_statements_partials_are_registered_reads_and_an_unregistered_row_does_not_open():
    """A port the verifier's registered reads (``own``) don't name is free in the checked relation, so in
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


def test_every_port_of_every_statement_is_the_verifiers_own():
    """A port the verifier doesn't register is free in the checked relation, so each statement's every port, its output
    row included, is a read or a known port of ``own``, never one the public file states; lane l's ``pub`` and every
    statement's ``o0`` are values of its own."""
    lane = {*(f"a{s}" for s in range(4)), *(f"b{t}" for t in range(8)), "admit", "zy", "partials", "pub", "o0"}
    combine = {*(f"p{x}" for x in range(32)), "receipt", "o0"}
    oks = []
    for i, (kind, ln) in enumerate(LL.statements(32, 32)):
        assert LL.own(32, 32, kind, ln) == (lane if kind == LL.LANE else combine)
        k = LL.known(32, 32, kind, ln)
        assert not set(k) & set(LL.reads(32, 32, kind, ln)) and k[LL.OK] == f"ok{i}"
        assert kind == LL.COMBINE or (k[LL.PUB], k[LL.ADMIT]) == (f"pub{ln}", LL.ADMIT)
        oks.append(k[LL.OK])
    assert len(set(oks)) == 33


def _bits_n(words: list[int], lo: int, n: int) -> int:
    """Bits ``[lo, lo + n)`` of a row as a number, least significant first (`Accounting.TS.Lanes.bitsN`)."""
    return sum(((words[(lo + i) // 16] >> ((lo + i) % 16)) & 1) << i for i in range(n))


def test_a_lanes_pub_row_is_its_seeds_leaves_then_its_index():
    """Lane l's ``pub`` row is the salt's, position's and roots' leaves as the seed reads them, 100 words, then l, and its
    fields read least significant bit first (`decLane`) are the salt and roots with their big-endian halves least
    significant first, the position with its halves swapped, and l."""
    salt, position, roots = bytes(range(64)), 0x0123456789ABCDEF, (bytes(range(64, 128)), bytes(range(128, 192)))
    w = LL.pub_words(salt, position, roots, 7)
    assert len(w) == 101 == sum(LL.PUBLIC_BITS) // 16 and w[100] == 7
    assert w[:4] == [0x0203, 0x0001, 0x0607, 0x0405], "the salt's half 00 01 02 03 is its low 32 bits"
    assert w[32:36] == [0x4567, 0x0123, 0xCDEF, 0x89AB], "the position's high half, then its low"
    bits = SD.leaves(salt, True) + SD.position_leaves(position, True) + SD.leaves(roots[0], True) + SD.leaves(roots[1], True)
    assert [(w[t // 16] >> (t % 16)) & 1 for t in range(1600)] == bits
    at = dict(zip(LL.PUBLIC, zip(itertools.accumulate((0,) + LL.PUBLIC_BITS), LL.PUBLIC_BITS)))
    assert at == {"salt": (0, 512), "position": (512, 64), "roots": (576, 1024), "lane": (1600, 16)}
    got = {name: _bits_n(w, *at[name]) for name in LL.PUBLIC}
    assert got["salt"] == sum(h << 32 * i for i, h in enumerate(SD.halves(salt)))
    assert got["position"] == 0x89ABCDEF01234567
    assert got["roots"] == sum(h << 32 * i for i, h in enumerate(SD.halves(roots[0] + roots[1])))
    assert got["lane"] == 7
    assert LL.pub_words(salt, position, roots, 0xFFFF)[100] == 0xFFFF
    for bad in ((salt[:63], position, roots, 7), (salt, position, roots[:1], 7), (salt, position, roots, 1 << 16),
                (salt, 1 << 64, roots, 7), (salt, position, roots, -1)):
        with pytest.raises(ValueError):
            LL.pub_words(*bad)


def test_each_statements_known_rows_are_its_known_ports():
    """Each statement's `known_rows` are its `known` ports: lane l's ``pub`` (`pub_words` at l) and ``admit`` (bit i the
    admit bit i, whole words, of any length, as Lean's `PouwWindow.knownRows` takes them), and every ``o0`` the row 1."""
    admit = [int(i % 3 == 0) for i in range(32)]
    data = {"admit": admit, "salt": SALT, "position": POSITION, "roots": ROOTS}
    for kind, ln in LL.statements(32, 32):
        got = LL.known_rows(32, 32, kind, ln, **data)
        assert {p: v for p, (v, _) in got.items()} == LL.known(32, 32, kind, ln) and got[LL.OK][1] == [1]
        if kind == LL.LANE:
            assert got[LL.PUB][1] == LL.pub_words(SALT, POSITION, ROOTS, ln) and got[LL.ADMIT][1] == [0x9249, 0x4924]
    assert LL.known_rows(32, 32, LL.COMBINE) == {LL.OK: ("ok32", [1])}
    assert LL.known_rows(8, 16, LL.LANE, 3, **{**data, "admit": admit[:8]})[LL.ADMIT][1] == [0x49]
    assert LL.known_rows(32, 32, LL.LANE, 3, **{**data, "admit": admit[:17]})[LL.ADMIT][1] == [0x9249, 0]
    assert LL.bits_words([1] * 17) == [0xFFFF, 1] and LL.bits_words([]) == []
    with pytest.raises(ValueError, match="admit bits"):
        LL.known_rows(32, 32, LL.LANE, 3, salt=SALT, position=POSITION, roots=ROOTS)
    with pytest.raises(ValueError, match="lanes"):
        LL.known_rows(32, 32, LL.LANE, 32, **data)
    with pytest.raises(ValueError, match="0 or 1"):
        LL.bits_words([0, 2])


def test_a_known_port_opens_only_its_content():
    """The verifier's ``pub`` of lane 3 (`produced.known_port`) opens the row with index 3 under its salt, and neither the
    row with index 4 nor index 3's row under another salt."""
    reg, pd, salt = "ab" * 64, "12" * 64, bytes(range(192))
    port, (_, bc) = PR.known_port(pd, "pub3", [7, 3], salt, reg)
    dom = PR.domain(pd, "pub3", 2, 1, 16, reg)
    root = bytes.fromhex(port["root"])

    def opens(row, s):
        t = PR.commit(pd, "pub3", [row], reg, [s], schema=G.SCHEMA_V2)
        return G.verify_read(dom, root, 0, t.bcs[0], t.path(0), G.SCHEMA_V2)

    assert opens([7, 3], salt) and PR.commit(pd, "pub3", [[7, 3]], reg, [salt], schema=G.SCHEMA_V2).bcs[0].hex() == bc
    assert not opens([7, 4], salt), "a lane whose index isn't the verifier's"
    assert not opens([7, 3], bytes(192)), "the verifier's content under another salt"
    f = LL.registered_file({}, {}, {}, known_ports={LL.PUB: port})
    assert f["ports"] == {LL.PUB: port} and port["reads"] == [0]
    assert set(f) == {"format", "ports"}, "the verifier holds the circuit's every port to the file's, with no switch"
    with pytest.raises(ValueError, match="both reads and known"):
        LL.registered_file({LL.PUB: ("x", 0)}, {"x": port}, {"x": port["root"]}, known_ports={LL.PUB: port})


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


def test_a_statements_row_ports_and_only_those_are_read_in_full():
    """Lane l's ``a<s>``, ``b<t>`` and ``partials`` and the combine's ``p<x>`` are marked ``read_in_full``, which
    flock-verify holds to one row of the registered row's words (`KnownRead.fullRows`, proofs'
    `PouwWindow.Lanes.readFull`); ``zy``, which each lane reads only at its own cells, and ``receipt`` are not."""
    reg, pd = "ab" * 64, "12" * 64
    vals = [LL.PARTIALS, LL.RECEIPT, LL.A, LL.B, LL.ZY]
    entries = {v: PR.entry(pd, v, 32, 2, reg) for v in vals}
    roots = {v: f"{i:02x}" * 64 for i, v in enumerate(vals)}
    for kind, ln in LL.statements(32, 32):
        ports = LL.reads(32, 32, kind, ln)
        f = LL.registered_file(ports, entries, roots)
        full = sorted(p for p, e in f["ports"].items() if e.get(LL.READ_IN_FULL) is True)
        assert full == sorted(p for p, (v, _) in ports.items() if v in LL.FULL), (kind, ln)
        assert full == (sorted([*(f"a{s}" for s in range(4)), *(f"b{t}" for t in range(8)), "partials"])
                        if kind == LL.LANE else sorted(f"p{x}" for x in range(32)))
        assert all(LL.READ_IN_FULL not in e for p, e in f["ports"].items() if p not in full)


def test_the_verifiers_entries_are_the_registrations_and_the_produced_records():
    """Each registered value's entry and root are the registration's first of its value; the partials' are the
    verifier's own entry under the registration's digest at the produced record's root, over any registered entry of
    that name (Lean's `PouwWindow.Lanes.roots?`)."""
    from verity.core.protocols.sampled_proofs.one_stage import registration as R

    e = {v: {"value": v, "root": f"{i:02x}" * 64, "leaves": 4, "words": 2, "domain": f"{i + 8:02x}" * 64}
         for i, v in enumerate((LL.A, LL.B, LL.PARTIALS))}
    rec = {"format": "verity/registration/v2", "program": "12" * 64,
           "registered": [e[LL.A], e[LL.B], {**e[LL.A], "root": "ee" * 64}, e[LL.PARTIALS]]}
    produced = {"produced": [{"value": "x", "root": "aa" * 64}, {"value": LL.PARTIALS, "root": "bb" * 64}]}
    ent, roots = LL.verifier_entries(rec, produced, 32, 6)
    assert roots == {LL.A: e[LL.A]["root"], LL.B: e[LL.B]["root"], LL.PARTIALS: "bb" * 64}
    assert ent[LL.A] is rec["registered"][0] and ent[LL.PARTIALS] == PR.entry("12" * 64, LL.PARTIALS, 32, 6, R.digest(rec))
    with pytest.raises(ValueError, match="no partials"):
        LL.verifier_entries(rec, {"produced": [produced["produced"][0]]}, 32, 6)


# -- a J10-R leaf ---------------------------------------------------------------------------------------------------------------

def _j10r_definition(t, n: int, st: tuple):
    return Z.combine(t) if st[0] == LL.COMBINE else Z.lane(t) if st[0] == LL.LANE else Z.lane_chunk(t, n, st[2])


@pytest.mark.parametrize("k", [448, 8192, 12288])
def test_a_j10r_leafs_statements_and_every_port_are_its_v2_definitions(k):
    """A J10-R leaf's statements are `h1zy.lane_statements` lane by lane, then the combine, and each one's registered
    ports (`own`) are exactly its Definition's C-Flock ports (`h1zy.statement_ports`), ``o0`` among them; statement i's
    ``o0`` reads row i of ``oks`` and the combine's ``ok<i>`` the same row."""
    from verity.core.protocols.pouw.circuit import h1window as HW

    t, n = HW.h1_tile(k, (32, 32), H.PouwH1Tile5), LL.n_chunks(k)
    sts = LL.statements(32, 32, j10r=k)
    assert sts[-1] == (LL.COMBINE, None) and len(sts) == 32 * n + 1
    assert [d.bindings.get("CHUNK") for d in Z.lane_statements(t)] == ([None] if n == 1 else list(range(n)))
    assert sts[:-1] == [(LL.LANE, ln) if n == 1 else (LL.CHUNK, ln, ch) for ln in range(32) for ch in range(n)]
    ports = {}
    for i, st in enumerate(sts):
        d = _j10r_definition(t, n, st)
        key = (d.defn.id, d.bindings.get("CHUNK"))
        ports.setdefault(key, {p for p, _f in Z.statement_ports(d)})
        assert LL.own(32, 32, *st, j10r=k) == ports[key], st
        rd, kn = LL.reads(32, 32, *st, j10r=k), LL.known(32, 32, *st, j10r=k)
        assert not set(rd) & set(kn)
        if st[0] == LL.COMBINE:
            assert kn == {LL.OK: f"ok{32 * n}"} and {p: rd[p] for p in rd if p.startswith("ok")} == {
                f"ok{j}": (LL.OKS, j) for j in range(32 * n)}
        else:
            assert rd[LL.OK] == (LL.OKS, i) and kn[LL.PUB] == f"pub{st[1]}"
            assert (LL.ADMIT in kn) == (LL.ZY in rd) == (st[0] == LL.LANE or st[2] == n - 1)


def test_a_j10r_chunks_rows_carries_and_partials():
    """Chunk c of lane l in n reads its A and B rows' chunk c, leaf n i + c of a table at K (`h1zy.chunk_leaves`), what
    chunk c - 1 handed on and what it hands on (row l of ``carry<c - 1>`` and ``carry<c>``, `h1zy.CARRY`'s widths), and
    its last chunk the record and row l of the partials."""
    k, n = 12288, 3
    for ch in range(n):
        r = LL.reads(32, 32, LL.CHUNK, 7, ch, j10r=k)
        assert [r[f"a{s}"] for s in range(4)] == [(LL.A, n * i + ch) for i in LL.lane_rows(32, 32, 7)]
        assert [r[f"b{t}"] for t in range(8)] == [(LL.B, n * j + ch) for j in LL.lane_cols(32, 32, 7)]
        assert r.get("carry_in") == (None if ch == 0 else (LL.carry(ch - 1), 7))
        assert r.get("carry_out") == (None if ch == n - 1 else (LL.carry(ch), 7))
        assert (r.get(LL.ZY), r.get(LL.PARTIALS)) == (((LL.ZY, 0), (LL.PARTIALS, 7)) if ch == n - 1 else (None, None))
    for ch in range(n - 1):
        assert 16 * LL.carry_words(k, n, ch) == sum(Z.CARRY(H.Layout(32, 32, k), n, ch).leaf_widths())
    assert LL.produced_values(32, 32, k, j10r=True) == {
        LL.PARTIALS: (32, 2 * LL.words(k)), "carry0": (32, LL.carry_words(k, n, 0)), "carry1": (32, LL.carry_words(k, n, 1)),
        LL.OKS: (96, 1)}
    assert LL.produced_values(32, 32, k) == {LL.PARTIALS: (32, 2 * LL.words(k))}
    for bad in ((LL.LANE, 7, None), (LL.CHUNK, 7, 3), (LL.CHUNK, 32, 0), (LL.COMBINE, None, 0)):
        with pytest.raises(ValueError, match="lane"):
            LL.reads(32, 32, *bad, j10r=k)
    with pytest.raises(ValueError, match="hands on no carry"):
        LL.carry_words(k, n, n - 1)
    with pytest.raises(ValueError, match="lane with its index"):
        LL.reads(32, 32, LL.CHUNK, 7, 0)


def test_a_j10r_lanes_pub_row_is_v1s_then_its_call_and_blocks_then_its_index():
    """A J10-R lane's ``pub`` row is v1's 100 words, then its call's and its two blocks' leaves as the unit reads them
    (`seeds.position_leaves`), 12 words, then the index, at the fields `h1zy.statement_ports` states."""
    from verity.core.protocols.pouw.circuit import h1window as HW

    call, blocks = 0x0123456789ABCDEF, (5, 1 << 40)
    w = LL.pub_words(SALT, POSITION, ROOTS, 9, call=call, blocks=blocks)
    assert len(w) == 113 == sum(LL.PUBLIC_J10R_BITS) // 16 and w[:100] == LL.pub_words(SALT, POSITION, ROOTS, 9)[:100]
    assert w[112] == 9
    want = SD.position_leaves(call, True) + SD.position_leaves(blocks[0], True) + SD.position_leaves(blocks[1], True)
    assert [(w[t // 16] >> (t % 16)) & 1 for t in range(1600, 1792)] == want
    pub = dict(Z.statement_ports(Z.lane_chunk(HW.h1_tile(8192, (32, 32), H.PouwH1Tile5), 2, 0)))["pub"]
    at = dict(zip(LL.PUBLIC_J10R, zip(itertools.accumulate((0,) + LL.PUBLIC_J10R_BITS), LL.PUBLIC_J10R_BITS)))
    assert [(p, lo, n) for p, lo, n in pub] == [(p, *at[p]) for p in LL.PUBLIC_J10R]
    assert _bits_n(w, *at["call"]) == 0x89ABCDEF01234567, "a u64's high half, then its low, as the position's"
    for bad in ({"call": call}, {"blocks": blocks}, {"call": call, "blocks": blocks[:1]}):
        with pytest.raises(ValueError, match="call"):
            LL.pub_words(SALT, POSITION, ROOTS, 9, **bad)


def test_a_j10r_leafs_known_rows_and_registered_files():
    """A J10-R leaf's known rows are each chunk's ``pub`` (with the call and blocks), its last chunk's ``admit`` and only
    the combine's ``o0``; its files mark every carry and ``oks`` port read in full, and no ``o0``: the output row binds
    only the ok bit the circuit returns, and the combine holds each ok row's other bits to 0."""
    k, n = 8192, 2
    data = {"admit": [1] * 32, "salt": SALT, "position": POSITION, "roots": ROOTS, "j10r": k, "call": 64,
            "blocks": (3, 4)}
    first, last = (LL.known_rows(32, 32, LL.CHUNK, 5, ch, **data) for ch in range(n))
    pub = LL.pub_words(SALT, POSITION, ROOTS, 5, call=64, blocks=(3, 4))
    assert first == {LL.PUB: ("pub5", pub)} and last == {LL.PUB: ("pub5", pub), LL.ADMIT: (LL.ADMIT, [0xFFFF, 0xFFFF])}
    assert LL.known_rows(32, 32, LL.COMBINE, j10r=k) == {LL.OK: ("ok64", [1])}
    with pytest.raises(ValueError, match="call and blocks"):
        LL.known_rows(32, 32, LL.CHUNK, 5, 0, **{**data, "call": None})
    with pytest.raises(ValueError, match="call and blocks"):
        LL.known_rows(32, 32, LL.LANE, 5, **{**data, "j10r": None})
    reg, pd = "ab" * 64, "12" * 64
    vals = [LL.A, LL.B, LL.ZY, LL.RECEIPT, *LL.produced_values(32, 32, k, j10r=True)]
    entries = {v: PR.entry(pd, v, 64, 2, reg) for v in vals}
    roots = {v: f"{i:02x}" * 64 for i, v in enumerate(vals)}
    for st in LL.statements(32, 32, j10r=k):
        ports = LL.reads(32, 32, *st, j10r=k)
        f = LL.registered_file(ports, entries, roots)
        full = {p for p, e in f["ports"].items() if e.get(LL.READ_IN_FULL) is True}
        assert full == {p for p, (v, _) in ports.items() if v not in (LL.ZY, LL.RECEIPT) and p != LL.OK}, st
        assert LL.OK not in full and all(LL.read_in_full(ports[p][0]) for p in full)
    assert not LL.read_in_full("carryx") and LL.read_in_full("carry12") and not LL.read_in_full(LL.ZY)


def test_the_verifiers_entries_of_a_j10r_leafs_produced_values():
    """With ``values`` (`produced_values`) every after-draw value's entry is the verifier's own under the registration's
    digest at the produced record's root, and a value the record lacks is refused by name."""
    from verity.core.protocols.sampled_proofs.one_stage import registration as R

    rec = {"format": "verity/registration/v2", "program": "12" * 64, "registered": []}
    vals = LL.produced_values(32, 32, 8192, j10r=True)
    produced = {"produced": [{"value": v, "root": f"{i:02x}" * 64} for i, v in enumerate(vals)]}
    ent, roots = LL.verifier_entries(rec, produced, 32, 0, values=vals)
    assert roots == {v: f"{i:02x}" * 64 for i, v in enumerate(vals)}
    assert all(ent[v] == PR.entry("12" * 64, v, rows, w, R.digest(rec)) for v, (rows, w) in vals.items())
    with pytest.raises(ValueError, match="no oks"):
        LL.verifier_entries(rec, {"produced": produced["produced"][:-1]}, 32, 0, values=vals)
