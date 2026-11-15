"""``ir_lower.layout`` writes the rows the layout it replaced wrote (`_reference`, that code, kept as the oracle): each AND's
rows straight from its forms' columns, and the parity path for a form over a library read's constant-wired value bit, on
random circuits with constant forms, assertions, split port groups and inline table reads, plain and compact
(`class_statement._UnitCircuit`)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity_flock import class_statement as CS
from verity_flock import gf2
from verity_flock import ir_lower as IL
from verity_flock.fp import ceil_pow2


def _reference(C, in_names, out_names, in_split=None, out_split=None) -> IL.LoweredUnit:
    live = C.live()
    col: dict[int, int] = {}
    rows_a: list = []
    rows_b: list = []

    def zero():
        rows_a.append([])
        rows_b.append([])

    def open_group(nbits):
        words = ceil_pow2(max(1, -(-nbits // IL.WORD)))
        while len(rows_a) % (IL.WORD * words):
            zero()
        return len(rows_a) // IL.WORD, words

    def close_group(c, words):
        while len(rows_a) < (c + words) * IL.WORD:
            zero()

    in_groups = []
    for g in IL._groups(in_names, in_split):
        bits = [len(C.inputs[nm]) for nm in g]
        c, words = open_group(sum(bits))
        for nm in g:
            for w in C.inputs[nm]:
                (v,) = tuple(w.s)
                col[v] = len(rows_a)
                rows_a.append([len(rows_a)])
                rows_b.append([len(rows_b)])
        close_group(c, words)
        in_groups.append((c, words, bits))
    ands, reads, const_bit = [], {}, {}
    value_bit = {C.data[v]: v for v in live if C.kind[v] == "lookup"}
    for v in range(len(C.kind)):
        if v not in live or C.kind[v] == "in":
            continue
        if C.kind[v] == "and":
            ands.append(v)
            col[v] = len(rows_a)
            rows_a.append(None)
            rows_b.append(None)
            continue
        g, j = C.data[v]
        if g not in reads:
            reads[g] = r = IL._Read(C, g, sorted(jj for (gg, jj) in value_bit if gg == g), len(rows_a))
            rows_a.extend([None] * r.rows)
            rows_b.extend([None] * r.rows)
        if j in reads[g].const_bits:
            const_bit[v] = reads[g].const_bits[j]
        else:
            col[v] = reads[g].out_col[j]

    def form(Lf):
        s = [col[v] for v in Lf.s if v not in const_bit]
        if (Lf.c + sum(const_bit.get(v, 0) for v in Lf.s)) & 1:
            s.append("C")
        return s

    for v in ands:
        a, b = C.data[v]
        rows_a[col[v]], rows_b[col[v]] = form(a), form(b)
    for r in reads.values():
        r.fill(rows_a, rows_b, form)
    for _, Lf in C.asserts:
        j = len(rows_a)
        rows_a.append(form(Lf) + [j])
        rows_b.append(["C"])
    out_groups = []
    for g in IL._groups(out_names, out_split):
        bits = [len(C.outputs[nm]) for nm in g]
        c, words = open_group(sum(bits))
        for nm in g:
            for w in C.outputs[nm]:
                rows_a.append(form(w))
                rows_b.append(["C"])
        close_group(c, words)
        out_groups.append((c, words, bits))
    const = len(rows_a)
    rows_a.append([const])
    rows_b.append([const])

    def fix(r):
        if isinstance(r, np.ndarray):
            return r
        par: dict[int, int] = {}
        for c in r:
            c = const if c == "C" else c
            par[c] = par.get(c, 0) ^ 1
        return sorted(c for c, p in par.items() if p)

    for rows in (rows_a, rows_b):
        for i, r in enumerate(rows):
            rows[i] = fix(r)
    lay = IL.LoweredUnit(rows_a, rows_b, const, in_groups, out_groups, {})
    read_rows = sum(r.rows for r in reads.values())
    lay.counts = {"ands": len(ands) + read_rows, "asserts": len(C.asserts), "rows": len(rows_a), "in_words": lay.in_words,
                  "out_words": lay.out_words, "nnz_a": sum(map(len, rows_a)), "nnz_b": sum(map(len, rows_b)),
                  "reads": len(reads), "read_rows": read_rows, "read_nnz_a": sum(r.nnz_a for r in reads.values())}
    return lay


TABLE = "test-layout-rows"


@pytest.fixture
def table(monkeypatch):
    """A 6-bit-index table of 8-bit words whose bit 3 is always set and bit 5 always clear."""
    rng = np.random.default_rng(5)
    words = (rng.integers(0, 256, size=64, dtype=np.uint32) | 0x08) & ~np.uint32(0x20)
    monkeypatch.setitem(gf2.TABLES, TABLE, lambda: words)
    monkeypatch.setitem(gf2._TABLE_WORDS, TABLE, words)
    return words


def _circuit(seed: int, read: bool, cls=gf2.Circuit) -> gf2.Circuit:
    rng = random.Random(seed)
    C = cls()
    pool = C.input("in0", 24) + C.input("in1", 9) + C.input("in2", 40)

    def form():
        f = gf2.ONE if rng.random() < 0.5 else gf2.ZERO
        for w in rng.sample(pool, rng.randint(1, 7)):
            f = f ^ w
        return f

    if read:
        pool += C.lookup(TABLE, [form() for _ in range(6)], 8)
    for k in range(400):
        pool.append(C.AND(form(), form()))
        if k % 97 == 0:
            C.assert_zero(f"a{k}", form())
    C.output("out0", pool[-16:])
    C.output("out1", [form() for _ in range(20)])
    C.output("out2", [gf2.ONE, gf2.ZERO, form(), pool[30]])
    return C


def _rows(rows) -> list[list[int]]:
    return [IL._cols_of(r) for r in rows]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("read", ["none", "program", "library"])
@pytest.mark.parametrize("cls", [gf2.Circuit, CS._UnitCircuit], ids=["plain", "unit"])
def test_layout_is_the_reference_layout(seed, read, cls, table, monkeypatch):
    """On a plain circuit and on `class_statement.unit_lowering`'s (`_UnitCircuit`, an AND's operand bits a tuple), the
    reference's rows of the plain circuit."""
    if read == "library":
        monkeypatch.setattr(IL, "_library_table", lambda name: name == TABLE)
    C, P = _circuit(seed, read != "none", cls), _circuit(seed, read != "none")
    names = (["in0", "in1", "in2"], ["out0", "out1", "out2"])
    got, want = IL.layout(C, *names, [1], [2]), _reference(P, *names, [1], [2])
    assert (got.const, got.in_groups, got.out_groups, got.counts) == (want.const, want.in_groups, want.out_groups, want.counts)
    assert _rows(got.rows_a) == _rows(want.rows_a) and _rows(got.rows_b) == _rows(want.rows_b)
    assert IL.netlist(got, "t") == IL.netlist(want, "t")
    ands = [v for v in P.live() if P.kind[v] == "and"]
    assert all(isinstance(C.data[v][0].s, tuple if cls is CS._UnitCircuit else frozenset) for v in ands)
    if read == "library":
        r = IL._Read(P, 0, list(range(8)), 0)
        assert r.const_bits == {3: 1, 5: 0}
        wired = {v for v in range(len(P.kind)) if P.kind[v] == "lookup" and P.data[v][1] in (3, 5)}
        assert any(wired & P.data[v][0].s or wired & P.data[v][1].s for v in ands)


def _reference_netlist(lay, name, cut=None, leaves=None) -> str:
    head = [IL.FORMAT, name, lay.useful, lay.const]
    for groups in (lay.in_groups, lay.out_groups):
        head.append(len(groups))
        for c, words, bits in groups:
            head += [c, words, len(bits), *bits]
    lines = [" ".join(map(str, head))]
    for a, b in zip(lay.rows_a, lay.rows_b):
        lines.append(" ".join(map(str, [len(a), *IL._cols_of(a), len(b), *IL._cols_of(b)])))
    return "\n".join(lines) + "\n" + IL.netlist_pins(cut, leaves)


class _Whole(str):
    """A text that refuses to be encoded whole."""

    def encode(self, *a):
        raise AssertionError("encoded whole")


@pytest.mark.parametrize("rows", [1, 3, 64, 1 << 16])
def test_the_netlist_and_its_digest_go_in_pieces(rows, table, monkeypatch):
    """`netlist` joins its rows :data:`ir_lower.NETLIST_ROWS` at a time (with an inline read's array rows and the pins) into
    the text the one-piece netlist wrote, and `digest` encodes it :data:`ir_lower.HASH_CHUNK` characters at a time."""
    import hashlib

    monkeypatch.setattr(IL, "NETLIST_ROWS", rows)
    monkeypatch.setattr(IL, "HASH_CHUNK", 7 * rows)
    lay = IL.layout(_circuit(rows, True), ["in0", "in1", "in2"], ["out0", "out1", "out2"])
    assert any(isinstance(r, np.ndarray) for r in lay.rows_a)
    for pins in ((None, None), ({"words": 2}, None), (None, {"in": [[0, 1]]}), ({"words": 1}, {"in": []})):
        text = IL.netlist(lay, "t", *pins)
        assert text == _reference_netlist(lay, "t", *pins)
        assert IL.digest(_Whole(text)) == hashlib.sha256(text.encode()).hexdigest()
