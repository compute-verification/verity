"""Row units cut along k (`verity.core.protocols.pouw.circuit.rowk`, ``Shape.row_k``): on every row, admitted or refused, a row's head, forms
and tail in sequence give the whole row unit's outputs, for Pearl-C on the H100, sm_120 and sm_120 unpromoted and for
Pearl-C4, rows of A and of B; the cut roots give today's roots' outputs with the same computing gates, and their partition
computes every value in exactly one unit; each piece's ports are committed rows of `leaves`' schemas and public input
regions, a form's its segment of F_A and F_B under the slice rule, and the head commits the words its readers read; and
with ``row_k`` unset nothing is the cut's."""
from __future__ import annotations

import json
import math
import struct
from collections import Counter
from dataclasses import replace

import pytest

import circuit_fixtures as CF
import pearl_c4_vectors as V4
import pearl_c_vectors as V8
from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X, bit_row, row_bytes
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.cut import CallGraph
from verity.ml.circuits.defs import CompositeDefinition
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Value
from verity.ml.circuits.ops.scalar import BitAnd, SelectBf16
from verity.core.protocols.pouw.circuit import leaves as L
from verity.core.protocols.pouw.circuit import pc4, pc8, plan, rowk
from verity.core.protocols.pouw.circuit.anchors import ANCHORED, Derivation, TileAnchors, public_ports
from verity.core.protocols.pouw.protocol import DomainError, Tile
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_c4 as F
from verity.core.protocols.pouw.schemes import pearl_kw as P
from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES

SEED = bytes(range(32))
#: (device, k, row_k): three segments each, so a form carries a carry it was given
CUTS8 = (("h100", 384, 128), ("sm120", 384, 128), ("sm120-unpromoted", 96, 32))
K4, KS4 = 256, 64


def _flat(lines) -> list[int]:
    return [c for ln in lines for c in ln]


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _split(out, types_) -> list:
    """A piece's flat output leaves by field: an int for a scalar, a list for an array."""
    fields, at = [], 0
    for t in types_:
        n = len(t.leaf_widths())
        fields.append(out[at] if isinstance(t, Value) else list(out[at:at + n]))
        at += n
    assert at == len(out)
    return fields


def _head_row(head, head_row: tuple, x, before) -> tuple[list, tuple]:
    """The head's output leaves on one row: (its row's fields, ``head_row``'s, as the forms and tail read them; its
    other outputs' leaves)."""
    out = evaluate(head, x, *before)
    n = sum(len(t.leaf_widths()) for _n, t in head_row)
    return _split(out[:n], [t for _n, t in head_row]), out[n:]


def _in_sequence(pieces, ks: int, head_row: tuple, n_codes: int, x, before, after, own, other, extra):
    """One row through its pieces as separate instances: the head, each segment's form on its slice of the bases with the
    carry the form before it returned, then the tail; (the head's other outputs' leaves, the tail's output leaves)."""
    head, first, forms, tail = pieces
    h, rest = _head_row(head, head_row, x, before)
    segs, carry = [], []
    for j, fn in enumerate((first, *forms)):
        lo, hi = len(own) * j * ks // len(x), len(own) * (j + 1) * ks // len(x)
        out = _split(evaluate(fn, x, *h, *carry, *after, own[lo:hi], other[lo:hi]), fn.signature()[1].fields)
        segs.append(out[:n_codes])
        carry = out[n_codes:]
    cs = [[v for s in segs for v in s[f]] for f in range(n_codes)]
    return rest, evaluate(tail, *h, *cs, *carry, *after, *extra)


# ---- the pieces against the whole row ---------------------------------------------------------------------------------

def _spikes(k: int, count: int, value: float) -> list[int]:
    """A row of ones with ``count`` unsampled positions (not multiples of 8) at ``value``, spread over every segment."""
    row = [_bits(1.0)] * k
    for i in range(count):
        row[(i * k // count + 8 * i + 3) % k | 1] = _bits(value)
    assert sum(w == _bits(value) for w in row) == count
    return row


def _rows8(k: int) -> dict[str, list[int]]:
    """Honest rows and rows the scheme refuses, their words placed in later segments: non-finite words sampled and not,
    an overflowing sum of squares, zero and subnormal rows, the largest word, a spike under the noise floor, and dead
    counts of k/64 (admitted) and k/64 + 1 (refused) with the dead words in every segment."""
    base = [C.parse_f32_row(r, k) for r in V8.rows_f32("pc8-rowk", "A", 3, k)]
    at = lambda w, updates: [updates.get(t, v) for t, v in enumerate(w)]  # noqa: E731
    mid, last = (k // 2) // 8 * 8, k - 3
    return {"honest-0": base[0], "honest-1": base[1], "negated": [w ^ 0x80000000 for w in base[2]],
            "inf-unsampled": at(base[0], {last: 0x7F800000}), "nan-sampled": at(base[0], {mid: 0xFFC00001}),
            "squares-overflow": at(base[0], {p: _bits(2.0 ** 126) for p in range(0, k, 8)}), "zero": [0] * k,
            "subnormal": [1] * k, "max-unsampled": at(base[0], {mid + 1: 0x7F7FFFFF}),
            "under-floor": at(base[1], {last: _bits(2.0 ** 100)}), "dead-limit": _spikes(k, k // 64, 9.0),
            "dead-over": _spikes(k, k // 64 + 1, 9.0)}


@pytest.mark.parametrize("dev, k, ks", CUTS8, ids=[c[0] for c in CUTS8])
@pytest.mark.parametrize("side", ["a", "b"])
def test_pearl_c_pieces_give_the_whole_rows_outputs(side, dev, k, ks):
    """Pearl-C's head, forms and tail of a row give `Pc8RowA`'s (`Pc8RowB`'s) codes, P and ok on every row, on each device:
    a segment is whole windows and 32-position blocks, so each form's dead count and promoted chains carried from the one
    before are the whole row's."""
    fa, fb = _flat(C._basis(SEED, 0, k, C.LINE_NORM_V1)), _flat(C._basis(SEED, 1, k, C.LINE_NORM_V1))
    own, other = (fa, fb) if side == "a" else (fb, fa)
    extra = [] if side == "a" else [_flat(C.gram(SEED, k, C.LINE_NORM_V1, DEVICES[dev]))]
    whole = pc8.templates(k, pc8.TILE, dev)[f"row_{side}"]
    e = P.sample_line(SEED, 0 if side == "a" else 1, 0, 5)
    pieces = rowk.pieces8(k, ks, dev, side)
    assert len(pieces[2]) == k // ks - 1
    verdicts = {}
    for name, x in _rows8(k).items():
        want = evaluate(whole, x, e, own, other, *extra)
        _rest, got = _in_sequence(pieces, ks, rowk.HEAD8, 1, x, (e,), (), own, other, extra)
        assert got == want, name
        verdicts[name] = want[-1]
    assert {n for n, v in verdicts.items() if v} >= {"honest-0", "honest-1", "negated", "dead-limit"}
    assert not any(verdicts[n] for n in ("inf-unsampled", "nan-sampled", "squares-overflow", "zero", "dead-over"))


def _rows4(k: int) -> dict[str, list[int]]:
    """`test_circuit_pc4`'s rows at depth k, their words placed in later segments."""
    base = [V4.row_codes("pc4-rowk", "A", i, k) for i in range(3)]
    at = lambda w, updates: [updates.get(t, v) for t, v in enumerate(w)]  # noqa: E731
    mid, last = (k // 2) // 8 * 8, k - 3
    spike = lambda v: at(base[0], {last: P.f32_to_bf16(v)})  # noqa: E731
    return {"honest-0": base[0], "honest-1": base[1], "negated": [w ^ 0x8000 for w in base[2]],
            "inf-unsampled": at(base[0], {last: 0x7F80}), "minus-nan-sampled": at(base[0], {mid: 0xFFC1}),
            "spike-2^16": spike(2.0 ** 16), "spike-256": spike(256.0), "max-unsampled": at(base[0], {mid + 1: 0x7F7F}),
            "zero": [0] * k, "tiny": [0x0001] * k, "huge": [0x7F7F] * k}


@pytest.mark.parametrize("side", ["a", "b"])
def test_pearl_c4_pieces_give_the_whole_rows_outputs(side):
    """Pearl-C4's head (its admission whole: `pc4._stats`), forms and tail of a row give `Pc4RowA`'s (`Pc4RowB`'s) codes,
    scale bytes, P and ok on every row: a segment is whole 64-position atoms, so each form's chains carried from the one
    before are the whole row's."""
    fa, fb = _flat(F.basis(SEED, F.ROLE_FA, K4)), _flat(F.basis(SEED, F.ROLE_FB, K4))
    own, other = (fa, fb) if side == "a" else (fb, fa)
    extra = [] if side == "a" else [_flat(F.gram(F.NVFP4, SEED, K4))]
    whole = pc4.templates(K4)[f"row_{side}"]
    pieces = rowk.pieces4(K4, KS4, side)
    verdicts = {}
    for i, (name, x) in enumerate(_rows4(K4).items()):
        e = F.line(SEED, F.ROLE_EA if side == "a" else F.ROLE_EB, i)
        want = evaluate(whole, x, e, own, other, *extra)
        ok, got = _in_sequence(pieces, KS4, rowk.HEAD4, 2, x, (), (e,), own, other, extra)
        assert got + ok == want, name
        verdicts[name] = want[-1]
    assert {n for n, v in verdicts.items() if v} == {"honest-0", "honest-1", "negated", "spike-256"}


# ---- the roots --------------------------------------------------------------------------------------------------------

def _inputs8(dev: str, k: int, m: int, n: int):
    s = C.PearlC(k_min=32, tile=(16, 16), forming="v1", device=DEVICES[dev])
    ctx = V8.ctx_of(f"pc8-rowk-{dev}", (m, k, n), 3, 9)
    a, b = V8.rows_f32(f"pc8-rowk-{dev}", "A", m, k), V8.rows_f32(f"pc8-rowk-{dev}", "B", n, k)
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, a, b, Tile(tuple(range(m)), tuple(range(n))))
    fa, fb = C._basis(noise.seed_b, 0, k, C.LINE_NORM_V1), C._basis(noise.seed_b, 1, k, C.LINE_NORM_V1)
    words = lambda rows: [w for r in rows for w in C.parse_f32_row(r, k)]  # noqa: E731
    return (words(a), _flat(ops.e_a), words(b), _flat(ops.e_b), _flat(fa), _flat(fb),
            _flat(C.gram(noise.seed_b, k, C.LINE_NORM_V1, s.device)))


def _inputs4(k: int, m: int, n: int):
    a = [w for i in range(m) for w in V4.row_codes("pc4-rowk", "A", i, k)]
    b = [w for j in range(n) for w in V4.row_codes("pc4-rowk", "B", j, k)]
    ea, eb = (_flat(F.line(SEED, role, i) for i in range(c)) for role, c in ((F.ROLE_EA, m), (F.ROLE_EB, n)))
    return a, ea, b, eb, _flat(F.basis(SEED, F.ROLE_FA, k)), _flat(F.basis(SEED, F.ROLE_FB, k)), _flat(F.gram(F.NVFP4, SEED, k))


def _shapes():
    out = [pc8.Shape(2, k, 2, tile=(1, 1), dev=dev, row_k=ks) for dev, k, ks in CUTS8]
    return out + [pc4.Shape(2, K4, 2, tile=(1, 1), row_k=KS4)]


SHAPES = _shapes()
IDS = [getattr(s, "dev", "pc4") for s in SHAPES]


def _mod(shape):
    return pc8 if isinstance(shape, pc8.Shape) else pc4


@pytest.mark.parametrize("hidden", [False, True], ids=["plain", "hidden"])
@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_the_cut_roots_give_todays_outputs(shape, hidden):
    """The cut's `weight`, `call` and `check` give today's roots' outputs on a scheme's honest block (the call reading the
    weight's committed rows), plain and hiding."""
    mod, cut = _mod(shape), replace(shape, hidden=hidden)
    today = replace(cut, row_k=None)
    a, ea, b, eb, fa, fb, g = _inputs8(shape.dev, shape.k, 2, 2) if mod is pc8 else _inputs4(shape.k, 2, 2)
    w = evaluate(mod.weight(today), b, eb, fa, fb, g)
    assert evaluate(mod.weight(cut), b, eb, fa, fb, g) == w
    n, k = shape.n, shape.k
    committed = (w[:n * k], w[n * k:-n]) if mod is pc8 else (w[:n * k], w[n * k:n * k + n * k // 16], w[n * k + n * k // 16:-n])
    assert evaluate(mod.call(cut), a, ea, fa, fb, *committed) == evaluate(mod.call(today), a, ea, fa, fb, *committed)
    assert evaluate(mod.check(cut), a, ea, b, eb, fa, fb, g) == evaluate(mod.check(today), a, ea, b, eb, fa, fb, g)
    assert w[-n:] == (1,) * n


@pytest.mark.parametrize("hidden", [False, True], ids=["plain", "hidden"])
@pytest.mark.parametrize("side", ["check", "weight", "call"])
@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_the_cut_computes_every_value_in_exactly_one_unit(shape, side, hidden):
    """Per row a head, k/row_k forms and a tail, and a unit per tile; `validate_unit_cut` certifies every gate once with no
    recompute, the partition object verifies, and the cut computes today's gates (pc4's forms decode their segment's BF16
    words again: `Bf16ToF32`, wiring)."""
    mod, cut = _mod(shape), replace(shape, hidden=hidden)
    build = {"check": mod.check, "weight": mod.weight, "call": mod.call}[side]
    fn, segs = build(cut), shape.k // shape.row_k
    prog, q = Program(fn), mod.query(cut, side)
    us = mod.units(fn)
    rows = {"row_b": shape.n, "row_a": shape.m}
    per = {"head": 1, "first": 1, "form": segs - 1, "tail": 1}
    full = {f"{r}_{p}": rows[r] * c for r in rows for p, c in per.items()} | {"tile": 4}
    assert Counter(u.role for u in us) == {r: full[r] for r in rowk.ROLES[side]}
    assert [u.where for u in us if u.role in rowk.FORMS] == [
        (i, j) for r in ("row_b", "row_a") if any(u.role.startswith(r) for u in us)
        for j in range(segs) for i in range(rows[r])]
    part = PO.evaluate(prog, q)
    assert [t for _k, t, _a, n in part.ranges for _ in range(n)] == [u.template for u in us]
    assert PO.verify(PO.build(prog, q), prog).ok
    g, _owner, verdict = mod.unit_cut(fn)
    assert verdict.ok, verdict.codes
    assert verdict.detail["certified"] == verdict.detail["computed_gates"]
    whole = CallGraph(build(replace(cut, row_k=None)))
    computing = lambda gr: Counter(gr.prim_names[p] for p in gr.prim)  # noqa: E731
    assert computing(g) == computing(whole)
    extra = g.free_by_prim - whole.free_by_prim
    assert set(extra) <= ({"Bf16ToF32_v1"} if mod is pc4 else set()) | {p for p in extra if "Const" in p}, extra


def test_each_piece_is_one_static_circuit_per_segment():
    """A row's forms are distinct templates (J is static), the first form takes no carry, and only the forms read DEV:
    the head and tail are the same on every Pearl-C device."""
    heads = {rowk.pieces8(384, 128, d, "a")[0].id for d in ("h100", "sm120")}
    tails = {rowk.pieces8(384, 128, d, "a")[3].id for d in ("h100", "sm120")}
    assert len(heads) == len(tails) == 1
    head, first, forms, _tail = rowk.pieces8(384, 128, "sm120", "a")
    assert [rowk.segment(f) for f in (first, *forms)] == [0, 1, 2] and len({first.id, *(f.id for f in forms)}) == 3
    carried = [n for n, _t in forms[0].signature()[0]]
    assert [n for n, _t in first.signature()[0]] == [n for n in carried if n not in dict(rowk.CARRY8)]


def test_the_layout_proves_each_rows_pieces_with_its_tiles():
    """`plan.tile_layout` on a cut shape: a tile's closure is the tile and its rows of A's units (head, forms, tail, row by
    row), and its weight rows are its rows of B's in the weight's Program."""
    shape = pc4.Shape(4, K4, 6, tile=(2, 3), row_k=KS4)
    lay = plan.tile_layout(shape, 1)
    lay.check()
    us, ws = pc4.units(pc4.call(shape)), pc4.units(pc4.weight(shape))
    for t in (u for u in us if u.role == "tile"):
        rs, cs = t.where
        assert lay.closure[t.index][0] == t.index
        assert sorted(us[i].where[0] for i in lay.closure[t.index][1:]) == sorted(
            [rs * 2 + i for i in range(2)] * (K4 // KS4 + 2))
        assert all(us[i].role.startswith("row_a") for i in lay.closure[t.index][1:])
        assert sorted(ws[j].where[0] for j in lay.b_rows[t.index]) == sorted([cs * 3 + j for j in range(3)] * (K4 // KS4 + 2))


# ---- the ports and the anchors ----------------------------------------------------------------------------------------

def _pieces_of(shape) -> dict[str, list]:
    pieces = (lambda s: rowk.pieces8(shape.k, shape.row_k, shape.dev, s)) if isinstance(shape, pc8.Shape) else (
        lambda s: rowk.pieces4(shape.k, shape.row_k, s))
    out = {}
    for side in ("a", "b"):
        head, first, forms, tail = pieces(side)
        out |= {f"row_{side}_head": [head], f"row_{side}_first": [first], f"row_{side}_form": list(forms),
                f"row_{side}_tail": [tail]}
    return out


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_the_ports_are_each_pieces_committed_rows(shape):
    """Each piece's ports (`rowk.PORTS8`, `PORTS4`) list every parameter and output once: input rows read whole, the head,
    carry and codes rows (a segment's codes rows at depth row_k, the tail reading one per segment), the data and ok rows,
    and the anchors as public input regions; the tile's are today's."""
    ports = shape.ports
    assert ports["tile"] == _mod(shape).PORTS["tile"]
    assert set(ports) == set(rowk.ROLES["check"])
    segs = shape.k // shape.row_k
    codes = "fp8-codes" if isinstance(shape, pc8.Shape) else "nvfp4-codes"
    for role, fns in _pieces_of(shape).items():
        for fn in fns:
            CF.check_ports(fn, ports[role], shape.derivations, shape.k, {"cs": segs, "ss": segs}, {codes: shape.row_k})


def _words(types_, values) -> list[int]:
    """Values of ``types_`` (an int per scalar, a list per array or tuple) as u16 words: their leaves' bits run together
    little-endian, zero-padded to a whole word."""
    bits, at = 0, 0
    for t, v in zip(types_, values, strict=True):
        for w, leaf in zip(t.leaf_widths(), v if isinstance(v, list) else [v], strict=True):
            bits |= (leaf & ((1 << w) - 1)) << at
            at += w
    return [(bits >> (16 * j)) & 0xFFFF for j in range(-(-at // 16))]


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("shape", [SHAPES[1], SHAPES[-1]], ids=["sm120", "pc4"])
def test_the_head_commits_the_row_its_readers_read(shape, side):
    """Flock commits each output port zero-padded to whole u16 words and reads an input row as its parameters' bits
    run together (`class_statement.unit_lowering`).  So on every row, admitted or refused (Pearl-C's at every pair of
    finite and floor bits), the head's pc8-head (pc4-head) ports, each padded, are `leaves.words_pc8_head`
    (`words_pc4_head`) of its values: the very words the first form, each form and the tail read from that row."""
    pc8_ = isinstance(shape, pc8.Shape)
    head_row, schema = (rowk.HEAD8, "pc8-head") if pc8_ else (rowk.HEAD4, "pc4-head")
    ports, pieces = shape.ports, _pieces_of(shape)
    (head,) = pieces[f"row_{side}_head"]
    fields = head.signature()[1].fields
    (outs,) = [ps for s, _r, ps in ports[f"row_{side}_head"]["rows"] if s == schema]
    readers = [(role, fn) for role in (f"row_{side}_{p}" for p in ("first", "form", "tail")) for fn in pieces[role]]
    assert len(readers) == shape.k // shape.row_k + 1
    rows = _rows8(shape.k) if pc8_ else _rows4(shape.k)
    seen = set()
    for i, (name, x) in enumerate(rows.items()):
        before = (P.sample_line(SEED, 0 if side == "a" else 1, i, 5),) if pc8_ else ()
        vals = _split(evaluate(head, x, *before), fields)
        committed = [w for p in outs for w in _words([fields[p]], [vals[p]])]
        h = dict(zip([n for n, _t in head_row], _head_row(head, head_row, x, before)[0]))
        want = (L.words_pc8_head(h["alpha"], h["ep"], h["th"], h["finite"], h["floor"]) if pc8_
                else L.words_pc4_head(h["alpha"], h["ae"]))
        assert len(want) == L.ROW_WORDS[schema](shape.k)
        assert committed == want, name
        for role, fn in readers:
            (names,) = [ps for s, _r, ps in ports[role]["rows"] if s == schema]
            types_ = dict(fn.signature()[0])
            assert _words([types_[n] for n in names], [h[n] for n in names]) == want, (name, fn.id)
        seen.add(want[-1])
    assert seen == {0, 1, 2, 3} if pc8_ else len(seen) > 2, seen


def test_each_row_is_its_elements_bits():
    """The cut's rows' words (`leaves.words_pc8_head` …) are `rowleaf.bit_row`'s of their elements in their pieces'
    output order, of the schema's word count, and the head's pad, pc8-carry's zero fill and the ok row are constants in
    gates."""
    ep, tot, c8 = list(range(1, 33)), [0x3F800000 + i for i in range(32)], [(7 * i) & 0xFF for i in range(128)]
    c4, sc = [i & 0xF for i in range(128)], [(3 * i) & 0xFF for i in range(8)]
    rows = {"pc8-head": (L.words_pc8_head(0x40490FDB, ep, 1 << 63 | 5, 1, 0),
                         bit_row(([0x40490FDB], 32), (ep, 8), ([1 << 63 | 5], 64), ([1], 1), ([0], 1), ([0], 14))),
            "pc8-carry": (L.words_pc8_carry(17, tot), bit_row(([17], 32), (tot, 32), ([0] * rowk.FILL8, 16))),
            "fp8-codes": (L.words_fp8_codes(c8), bit_row((c8, 8))),
            "pc4-head": (L.words_pc4_head(0x3E000000, 0x38), bit_row(([0x3E000000], 32), ([0x38], 8), ([0], 8))),
            "pc4-carry": (L.words_pc4_carry(tot), bit_row((tot, 32))),
            "nvfp4-codes": (L.words_nvfp4_codes(c4, sc), bit_row((c4, 4), (sc, 8)))}
    for row, (words, (memory, n_bits)) in rows.items():
        assert (row_bytes(words, 16), 16 * len(words)) == (memory, n_bits), row
        assert len(words) == L.ROW_WORDS[row](128), row
        assert all(L.ROW_SCHEMA[row, role] == L.ROW_SCHEMA["ok", role] for role in (ROLE_X, ROLE_W)), row
    for fn in (rowk.pieces8(256, 128, "sm120", "a")[3], rowk.pieces8(256, 128, "h100", "b")[3],
               rowk.pieces4(128, 64, "a")[0], rowk.pieces4(128, 64, "b")[0]):
        b = fn.body
        gate = b.nodes[b.ret[len(b.ret) - 1][1]]
        assert gate.fn.id == SelectBf16.id and b.nodes[gate.args[0].refs[0][1]].fn.id == BitAnd.id, fn.id
    head = rowk.pieces8(256, 128, "sm120", "a")[0].body
    pad = head.nodes[head.ret[len(head.ret) - 1][1]]
    assert (pad.fn.family, pad.fn.evaluate()) == ("Const", 0)
    _head, first, forms, _tail = rowk.pieces8(256, 128, "h100", "b")
    for form in (first, *forms):
        b = form.body
        fill = [b.ret[i][1] for i in range(len(b.ret) - rowk.FILL8, len(b.ret))]
        assert {(b.nodes[n].fn.family, b.nodes[n].fn.evaluate()) for n in fill} == {("Const", 0)}


def test_each_cut_row_is_short_or_whole_blocks():
    """At the deployed row_k each cut row is one SHA-512 block with its padding (at most 55 words) or whole blocks (64 |
    words), the v2 rows Flock's widened scope takes."""
    for row in ("pc8-head", "pc8-carry", "fp8-codes", "pc4-head", "pc4-carry", "nvfp4-codes"):
        n = L.ROW_WORDS[row](rowk.ROW_K)
        assert 2 * n + 17 <= 128 or n % 64 == 0, (row, n)


def _anchors(shape):
    if isinstance(shape, pc8.Shape):
        scheme = C.PearlC(forming="v1", device=DEVICES[shape.dev])
    else:
        scheme = F.PearlC4("nvfp4")
    ctx = V8.ctx_of("pc-rowk-anchors", (shape.m, shape.k, shape.n), 3, 9)
    return TileAnchors.of(scheme, ctx, shape), TileAnchors.of(scheme, ctx, replace(shape, row_k=None))


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_a_form_reads_its_segment_of_the_bases(shape):
    """A form's F_A and F_B regions are its segment's lines of today's whole regions, under the slice rule; its E line
    and the tail's G are today's; a segment outside the row is refused."""
    cut, today = _anchors(shape)
    segs, ks = shape.k // shape.row_k, shape.row_k
    for side, i in (("a", 1), ("b", 0)):
        whole = today.regions(f"row_{side}", i)
        for j in range(segs):
            role = f"row_{side}_first" if j == 0 else f"row_{side}_form"
            got = cut.regions(role, i, j)
            for port in ("fa", "fb"):
                assert got[port] == whole[port][32 * ks * j:32 * ks * (j + 1)], (role, j, port)
            if "e" in got:
                assert got["e"] == whole["e"]
        tail = cut.regions(f"row_{side}_tail", i)
        assert all(tail[p] == whole[p] for p in tail)
        with pytest.raises(DomainError):
            cut.regions(f"row_{side}_form", i, segs)
        own = cut.public_inputs(f"row_{side}_form", (0, 1), 1)
        cut.check_public_inputs(f"row_{side}_form", (0, 1), own, 1)
        with pytest.raises(DomainError):
            cut.check_public_inputs(f"row_{side}_form", (0, 1), own, segs - 1 if segs > 2 else 0)


SLICE_RULES = {"pc8": {f"{a}-slice": f"verity/pouw/anchor/pearl-c/{a}/slice/v1" for a in ("fa", "fb")},
               "pc4": {f"{a}-slice": f"verity/pouw/anchor/pearl-c4/{a}/slice/v2" for a in ("fa", "fb")}}


def test_slice_rules_exist_only_under_the_cut():
    """The cut's shapes name today's five rules and the two slice rules; the class and a shape without ``row_k`` name
    today's five only, and a slice rule is refused unless derived per segment (and the reverse)."""
    for name, mod, shape in (("pc8", pc8, SHAPES[0]), ("pc4", pc4, SHAPES[-1])):
        today = {a: mod.DERIVATIONS[a].rule for a in ANCHORED}
        assert {a: d.rule for a, d in mod.Shape.derivations.items()} == today
        assert {a: d.rule for a, d in replace(shape, row_k=None).derivations.items()} == today
        assert {a: d.rule for a, d in shape.derivations.items()} == today | SLICE_RULES[name]
        assert mod.Shape.ports is mod.PORTS and replace(shape, row_k=None).ports is mod.PORTS
        names = {p["derivation"] for role in rowk.ROLES["check"] if role != "tile" and shape.ports[role]["regions"]
                 for p in public_ports(shape, role).values()}
        assert names >= set(SLICE_RULES[name].values())
    d = rowk.DERIVATIONS8["fa-slice"]
    assert d.per_segment and not d.per_row
    for bad in (dict(per_segment=False), dict(rule="verity/pouw/anchor/pearl-c/fa/v1"), dict(per_row=True)):
        with pytest.raises(ValueError, match="slice"):
            Derivation(**{"rule": d.rule, "seed": d.seed, "domain": d.domain, "fn": d.fn, "bits": d.bits,
                          "per_segment": True} | bad)
    with pytest.raises(ValueError, match="versioned id"):
        replace(d, rule="verity/pouw/anchor/pearl-c/fa/slice")


def test_shapes_off_the_cut_are_refused():
    """row_k is a positive multiple of the device's window and of 32 (Pearl-C) or of 64 (Pearl-C4) that divides k, and
    `ROW_K` is one on every device."""
    for d in DEVICES.values():
        assert rowk.ROW_K % math.lcm(32, d.window) == 0
    assert rowk.ROW_K % pc4.ATOM == 0
    pc8.Shape(1, 4096, 1, tile=(1, 1), row_k=rowk.ROW_K).check()
    pc4.Shape(1, 4096, 1, tile=(1, 1), row_k=rowk.ROW_K).check()
    for shape in (pc8.Shape(1, 256, 1, tile=(1, 1), row_k=64), pc8.Shape(1, 384, 1, tile=(1, 1), row_k=256),
                  pc8.Shape(1, 256, 1, tile=(1, 1), row_k=0), pc8.Shape(1, 256, 1, tile=(1, 1), row_k=128.0),
                  pc8.Shape(1, 96, 1, tile=(1, 1), dev="sm120-unpromoted", row_k=48),
                  pc4.Shape(1, 256, 1, tile=(1, 1), row_k=32), pc4.Shape(1, 192, 1, tile=(1, 1), row_k=128),
                  pc4.Shape(1, 256, 1, tile=(1, 1), row_k=True)):
        for build in (_mod(shape).check, _mod(shape).weight, _mod(shape).call):
            with pytest.raises(DomainError):
                build(shape)


def test_off_the_cut_nothing_is_the_cuts():
    """With ``row_k`` unset the roots, queries and units are today's: no root, template or role of the cut's."""
    cut_ids = {id(d) for d in vars(rowk).values() if isinstance(d, CompositeDefinition)}
    assert len(cut_ids) == 16 + 10
    for mod, shape in ((pc8, pc8.Shape(4, 128, 6, tile=(2, 3))), (pc4, pc4.Shape(4, 64, 6, tile=(2, 3)))):
        for side in ("weight", "call", "check"):
            fn = getattr(mod, side)(shape)
            assert id(fn.defn) not in cut_ids and all(id(nd.fn.defn) not in cut_ids for nd in fn.body.nodes)
            assert {u.role for u in mod.units(fn)} <= {"row_a", "row_b", "tile"}
            assert json.dumps(mod.query(shape, side)) == json.dumps(
                PO.template_instances_query([mod.templates(shape.k, shape.tile, *(
                    (shape.dev,) if mod is pc8 else ()))[r] for r in mod.ROLES[side]]))
