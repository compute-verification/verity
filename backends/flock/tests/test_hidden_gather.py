"""Reads at a private index of a registered table (``verity_flock.hidden_gather``): the registered tree is #730's byte for byte,
each gadget is frame-v3's and hm96's hash at every length of a private integer, an opening rejects a forged entry, row, salt,
path, root, domain or position, and a registered read gives today's public read; the measured MUFU tables stay lookups."""
from __future__ import annotations

import hashlib
import math
import os
import random

import numpy as np
import pytest

from verity.commitments.rowleaf import row_bytes
from verity_flock import gf2
from verity_flock import hidden_gather as HG
from verity_flock import ir_lower as IL
from verity_flock import tail_pieces as TP
from verity_flock.boolean_export import prims as _registered

#: #730's ``Registered.commit`` (``cursor/registered-values-95d4`` at 31f689486) over 512 rows of 64 random u16 words, rows and
#: salts from ``random.Random(730)``, program SHA-512("verity_flock/hidden-gather/test-program"), value "table": its domain id,
#: root, and the SHA-512 of its paths at positions 0, 255, 256 and 511 (siblings concatenated, leaf level first)
PIN_730 = {"domain": "bdb6921c6265b979a03941cfc009fe83a89d4a93e06ca1f75db1b16cc62de06eb91fc05292d997a3f7ae04571811be5df685de69a35a89c4e96c214ac19a7466",
           "root": "6957f7245c57cd9888c8de9d00fee88c52c225c7ce1b9a6deaf7a74d067ac5342e95b76bf5e86ab9b8f0fa95b00058d531e97cf38d103855975964ab5eeb587c",
           "paths": "73812e5ef007922ed32f98cdcdb46769f87ae35c8417f93c381afbf8ecc27de9788a5edb6387a106627b59ff6a6b941284d093a626831948b5aebfbda59c25d9"}
PROGRAM = hashlib.sha512(b"verity_flock/hidden-gather/test-program").hexdigest()


def _lane_bytes(sliced, nbytes: int, T: int) -> list[bytes]:
    out = [bytearray(nbytes) for _ in range(T)]
    for t, v in enumerate(sliced):
        for lane in range(T):
            if (v >> lane) & 1:
                out[lane][t >> 3] |= 1 << (t & 7)
    return [bytes(o) for o in out]


def _ints(values, T: int) -> list[int]:
    return [int(x) for x in IL.unslice_words(values, T)]


def _seeded(n: int, seed: int) -> list[bytes]:
    rng = random.Random(seed)
    return [rng.randbytes(HG.H5.salt_bytes) for _ in range(n)]


def test_the_registered_tables_are_the_program_tables():
    """``gelu_tanh_bf16`` and ``tanh_rn`` are the tables a program brings (``tail_pieces._DERIVED``), each read with the index
    and value widths of today's read, in rows of 64 u16 words; no measured MUFU table is one."""
    assert set(HG.TABLES) == {"gelu_tanh_bf16", "tanh_rn"} <= set(TP.TABLE_SHA256)
    assert not set(HG.TABLES) & set(IL.TABLES)
    for name, t in HG.TABLES.items():
        assert (t.index_bits, t.value_bits) == (TP.TABLE_BITS[name], TP.TABLE_VALUE_BITS.get(name, TP.VALUE_BITS))
    g, t = HG.TABLES["gelu_tanh_bf16"], HG.TABLES["tanh_rn"]
    assert (g.per_row, g.rows, g.depth) == (64, 1024, 10) and (t.per_row, t.rows, t.depth) == (32, 1 << 22, 22)


def test_the_tree_is_730s_registered_value():
    """:class:`HG.Tree` commits a table as #730 commits a registered value: the same domain id, root and paths, at positions of
    one and two bytes."""
    rng = random.Random(730)
    rows = [[rng.randrange(1 << 16) for _ in range(64)] for _ in range(512)]
    salts = [rng.randbytes(192) for _ in range(512)]
    tree = HG.Tree.commit(HG.Table("t", 15, 1, 16), HG.domain(PROGRAM, "table", 512), rows, salts)
    assert tree.domain.domain_id.hex() == PIN_730["domain"] and tree.root.hex() == PIN_730["root"]
    paths = b"".join(b"".join(tree.path(p)) for p in (0, 255, 256, 511))
    assert hashlib.sha512(paths).hexdigest() == PIN_730["paths"]
    assert all(HG.verify(tree.domain, tree.root, p, tree.bc(p), tree.path(p)) for p in (0, 255, 256, 511))
    assert not HG.verify(tree.domain, tree.root, 256, tree.bc(255), tree.path(256))


def test_a_sparse_tree_opens_its_rows():
    """A sparse tree's rows open its root, a row of another position or a forged row does not."""
    t = HG.TABLES["tanh_rn"]
    rng = random.Random(2)
    rows = {p: tuple(rng.randrange(1 << 16) for _ in range(64)) for p in (0, 1, 255, 256, 65536, (1 << 22) - 1)}
    tree = HG.Tree.sparse(t, HG.domain(PROGRAM, "tanh_rn", t.rows), rows, b"seed", dict(zip(rows, _seeded(len(rows), 3))))
    assert all(HG.verify(tree.domain, tree.root, p, tree.bc(p), tree.path(p)) for p in rows)
    assert not HG.verify(tree.domain, tree.root, 1, tree.bc(0), tree.path(1))
    assert not HG.verify(tree.domain, tree.root, 0, HG.commit_string(rows[1], tree.salts[0]), tree.path(0))


@pytest.mark.slow
def test_the_leaf_and_node_gadgets_are_frame_v3s():
    """The frame leaf and node at a private integer of 1, 2 and 3 bytes (``merkle._uint``'s lengths, every later byte moved),
    the domain id a wire for the leaf and a constant for the nodes, against ``CommitmentDomain.leaf`` and ``.node``."""
    rng = random.Random(5)
    dom = HG.domain(PROGRAM, "tanh_rn", 1 << 22)
    T = 8
    C = gf2.Circuit()
    pos, val, D = C.input("pos", 22), C.input("v", 512), C.input("d", 512)
    C.output("leaf", HG.frame_leaf_gadget(C, D, pos, val))
    rs = [0, 255, 256, 65535, 65536, (1 << 22) - 1, rng.randrange(1 << 22), rng.randrange(256)]
    vs = [rng.randbytes(64) for _ in range(T)]
    o, fail = C.evaluate({"pos": IL.slice_words(rs, 22), "v": HG.slice_bytes(vs),
                          "d": [((1 << T) - 1) * x for x in HG.slice_bytes([dom.domain_id])]}, T)
    assert not fail and _lane_bytes(o["leaf"], 64, T) == [dom.leaf(r, r, HG.SCHEMA, v) for r, v in zip(rs, vs)]
    for level in (0, 13, 21):
        nb = 21 - level
        C = gf2.Circuit()
        ib, left, right = C.input("i", nb), C.input("l", 512), C.input("r", 512)
        C.output("n", HG.node_gadget(C, HG._const(dom.domain_id), level, ib, left, right))
        idx = [x for x in (0, 255, 256, 65535, 65536, (1 << nb) - 1, rng.randrange(1 << nb) if nb else 0) if x < 1 << nb][:T]
        idx += [0] * (T - len(idx))
        ls, rs_ = [rng.randbytes(64) for _ in range(T)], [rng.randbytes(64) for _ in range(T)]
        o, _ = C.evaluate({"i": IL.slice_words(idx, nb), "l": HG.slice_bytes(ls), "r": HG.slice_bytes(rs_)}, T)
        assert _lane_bytes(o["n"], 64, T) == [dom.node(level, i, a, b) for i, a, b in zip(idx, ls, rs_)]


#: a registered table small enough to open in a test: 512 entries of one word, 8 rows, depth 3
SMALL = HG.Table("small", 9, 1, 16)


@pytest.mark.slow
def test_an_opening_reads_its_entry_and_rejects_every_forgery(monkeypatch):
    """:func:`HG.read` (root and domain id public inputs) on honest lanes gives the table's entries, positions of every row
    among them; a lane with a forged entry, another word of the row, a salt bit, a sibling bit, the root, the domain id or the
    row of another position fails an assertion, and only that lane."""
    monkeypatch.setitem(HG.TABLES, SMALL.name, SMALL)
    rng = random.Random(9)
    entries = [rng.randrange(1 << 16) for _ in range(512)]
    tree = HG.Tree.commit(SMALL, HG.domain(PROGRAM, "small", SMALL.rows), HG.table_rows(SMALL, entries), _seeded(SMALL.rows, 10))
    C = gf2.Circuit()
    HG.register(C, "small")
    idx = C.input("idx", 9)
    C.output("v", HG.read(C, "small", idx))
    forged = ["entry", "word", "salt", "sibling", "root", "domain", "position"]
    honest = [0, 63, 64, 200, 511, 448, 300, 7]
    lanes = honest + [rng.randrange(512) for _ in forged]
    T = len(lanes)
    ins = {"idx": IL.slice_words(lanes, 9)}
    w = HG.witness(C, ins, T, {"small": tree})
    pre = "registered/small/0/"
    for k, kind in enumerate(forged):
        lane, i = len(honest) + k, lanes[len(honest) + k]
        flip = {"entry": (pre + "row", 16 * (i % 64) + 3), "word": (pre + "row", 16 * ((i + 1) % 64)), "salt": (pre + "salt", 777),
                "sibling": (pre + "path1", 100), "root": ("registered/small/root", 5), "domain": ("registered/small/domain", 9)}
        if kind == "position":
            other = HG.witness(C, {"idx": IL.slice_words([(i + 64) % 512] * T, 9)}, T, {"small": tree})
            for name in (pre + "row", pre + "salt", *(f"{pre}path{lv}" for lv in range(SMALL.depth))):
                w[name] = [(x & ~(1 << lane)) | (y & (1 << lane)) for x, y in zip(w[name], other[name])]
        else:
            name, bit = flip[kind]
            w[name] = list(w[name])
            w[name][bit] ^= 1 << lane
    out, fail = C.evaluate({**ins, **w}, T)
    assert fail == ((1 << T) - 1) ^ ((1 << len(honest)) - 1)
    assert _ints(out["v"], T)[:len(honest)] == [entries[i] for i in honest]


def _gelu_tree() -> HG.Tree:
    t = HG.TABLES["gelu_tanh_bf16"]
    words = np.asarray(gf2.table("gelu_tanh_bf16"), dtype=np.uint64)
    return HG.Tree.commit(t, HG.domain(PROGRAM, "gelu_tanh_bf16", t.rows), HG.table_rows(t, words), _seeded(t.rows, 11))


@pytest.mark.slow
def test_a_registered_gelu_read_is_todays_read():
    """``GeluTanhMulBf16_v1`` in a circuit that registers ``gelu_tanh_bf16`` (its tree over the pinned table, root and domain id
    public inputs): its read gives the pinned table's word, and the piece today's piece and the IR's, at the table's edges,
    every row's first and last entry among them, and random gate words. Builds one 2.2M-AND opening (about 45 s, 5 GB)."""
    tree = _gelu_tree()
    rng = random.Random(12)
    gs = [0, 1, 63, 64, 0x3F80, 0x7F80, 0x7FC0, 0x8000, 0xBF80, 0xFF80, 0xFFFF, 1023 * 64]
    gs += [64 * r + rng.choice((0, 63)) for r in rng.sample(range(1024), 20)] + [rng.randrange(1 << 16) for _ in range(32)]
    us = [rng.randrange(1 << 16) for _ in gs]
    T = len(gs)
    C = gf2.Circuit()
    HG.register(C, "gelu_tanh_bf16")
    g, u = C.input("g", 16), C.input("u", 16)
    C.output("o", IL.PIECES["GeluTanhMulBf16_v1"](C, g, u))
    (_, _, act), = C.registered["gelu_tanh_bf16"].reads
    C.output("act", act)
    out, fail = HG.evaluate(C, {"g": IL.slice_words(gs, 16), "u": IL.slice_words(us, 16)}, T, {"gelu_tanh_bf16": tree})
    table = gf2.table("gelu_tanh_bf16")
    ref = _registered()["GeluTanhMulBf16_v1"].evaluate
    assert not fail and C.lookups == []
    assert _ints(out["act"], T) == [int(table[x]) for x in gs]
    assert _ints(out["o"], T) == [ref(x, y) for x, y in zip(gs, us)]


def _tanh_rn_row(position: int) -> tuple[int, ...]:
    """``tail_pieces._tanh_rn_table``'s row at ``position`` (32 entries, each its low word first), from RN32(math.tanh) of the
    one table-region word whose low 27 bits index it (0 where none)."""
    lo, hi = TP.TANH_BOUNDS["tanh_rn"]
    out = []
    for i in range(32 * position, 32 * position + 32):
        ws = [w for w in ((k << 27) | i for k in range(16)) if lo <= w < hi]
        v = 0
        if ws:
            x = float(np.uint32(ws[0]).view(np.float32))
            v = int(np.float32(math.tanh(x)).view(np.uint32))
        out += [v & 0xFFFF, v >> 16]
    return tuple(out)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("FLOCK_HIDDEN_GATHER_TANH"), reason="builds a 4.1M-AND opening of depth 22 (about 9 GB "
                    "and 90 s): set FLOCK_HIDDEN_GATHER_TANH=1")
def test_a_registered_tanh_rn_read_is_the_ir_primitive():
    """``TanhF32Rn_v1`` with ``tanh_rn`` registered (a sparse tree of depth 22 over the rows its words read): the IR's result on
    words across the table's region and the regions around it, both signs. What it adds to the gelu test: the depth-22
    climb, whose node indices take three bytes, and a two-word entry."""
    t = HG.TABLES["tanh_rn"]
    lo, hi = TP.TANH_BOUNDS["tanh_rn"]
    rng = random.Random(13)
    words = [lo, lo + 1, hi - 1, hi, 0, 0x3F800000, 0x7FC00000] + [rng.randrange(lo, hi) for _ in range(25)]
    xs = [w | (rng.getrandbits(1) << 31) for w in words]
    rows = {((x & 0x7FFFFFFF) & 0x7FFFFFF) >> t.entry_log: None for x in xs}
    tree = HG.Tree.sparse(t, HG.domain(PROGRAM, "tanh_rn", t.rows), {p: _tanh_rn_row(p) for p in rows}, b"tanh")
    T = len(xs)
    C = gf2.Circuit()
    HG.register(C, "tanh_rn", root=tree.root, domain_id=tree.domain.domain_id)
    x = C.input("x", 32)
    C.output("o", IL.PIECES["TanhF32Rn_v1"](C, x))
    out, fail = HG.evaluate(C, {"x": IL.slice_words(xs, 32)}, T, {"tanh_rn": tree})
    ref = _registered()["TanhF32Rn_v1"].evaluate
    assert not fail and _ints(out["o"], T) == [ref(v) for v in xs]


def test_the_mufu_tables_stay_lookups():
    """No measured MUFU table registers, a circuit that registers gelu still reads rcp by a lookup, a table registers once,
    and an unregistered program table is today's lookup."""
    C = gf2.Circuit()
    for name in ("rsq", "sqrt", "rcp", "ex2", "tanh_mufu"):
        with pytest.raises(ValueError, match="no registered table"):
            HG.register(C, name)
    HG.register(C, "gelu_tanh_bf16")
    with pytest.raises(ValueError, match="already"):
        HG.register(C, "gelu_tanh_bf16")
    IL.PIECES["MufuRcpFtz_v1"](C, C.input("x", 32))
    assert [t for t, _i, _w in C.lookups] == ["rcp"] and C.registered["gelu_tanh_bf16"].reads == []
    D = gf2.Circuit()
    TP.table_read(D, "gelu_tanh_bf16", D.input("g", 16))
    assert [t for t, _i, _w in D.lookups] == ["gelu_tanh_bf16"] and not hasattr(D, "registered")


def test_slice_bytes_is_byte_string_order():
    data = [bytes([1, 0x80]), bytes([0, 1])]
    assert HG.slice_bytes(data) == [1] + [0] * 7 + [2] + [0] * 6 + [1]
    assert row_bytes((0x1234,), 16) == b"\x34\x12"
