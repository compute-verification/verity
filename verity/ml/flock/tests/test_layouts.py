"""Layouts (`verity_flock.layouts`): own-region sizes by derive's v2 rule, the export rule, the unit's block slots, reads
inline and placed, and `validate` refusing each broken rule."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

from verity.ml.circuits import library
from verity_flock import circuit_types as CT
from verity_flock import gf2
from verity_flock import layouts as LY

sys.path.insert(0, str(Path(__file__).parent))
from test_circuit_types import _adder, _full_adder  # noqa: E402

rng = random.Random(5)
#: 64 words of 8 value bits: bits 0-4 vary (bit 3 is set in every word, but bit 4 varies above it), bits 5-7 are 0
TOY = [rng.getrandbits(5) | 0b1000 for _ in range(64)]
TOY_SHA = hashlib.sha512(b"".join(w.to_bytes(4, "little") for w in TOY)).hexdigest()
INFO = {TOY_SHA: LY.table_info(TOY)}


def info(sha):
    return INFO.get(sha)


def _read_type() -> CT.CircuitType:
    C = gf2.Circuit()
    x = C.input("x", 6)
    gf2.TABLES["toy_layout"] = lambda: TOY
    v = C.lookup("toy_layout", x, 8)
    return CT.from_gf2(C, x, [v[j] ^ x[0] for j in range(8)], ((1, 6),), ((1, 8),), library.PROGRAM, {"toy_layout": (TOY_SHA, 6, 8)})


def _wrapper(add4: CT.CircuitType) -> CT.CircuitType:
    """One call of the 4-bit adder on its own 8 inputs, returning the sum."""
    return CT.CircuitType(library.PROGRAM, ((8, 1),), ((5, 1),), (add4.digest(),), (),
                          ((CT.CALL, 0, tuple(((i,), 0) for i in range(8))),), tuple(((8 + i,), 0) for i in range(5)))


@pytest.fixture(scope="module")
def adders():
    fa = _full_adder()
    add4 = _adder(fa)
    w = _wrapper(add4)
    types = {t.digest(): t for t in (fa, add4, w)}
    for t in types.values():
        CT.validate(t, types.__getitem__)
    return types, fa, add4, w


def _lay(t, rl, calls=(), reads=(), own=("derive",), ins=(), outs=()):
    return LY.BlockLayout(t.digest() if t is not None else None, rl, tuple(ins), tuple(outs), own, tuple(calls), tuple(reads))


def test_own_sizes_follow_the_v2_rule(adders):
    types, fa, add4, _ = adders
    L_fa = _lay(fa, 9)
    # the full adder: an input word (3 bits), 2 ANDs, an output word (2 bits, aligned), the constant
    assert LY.own_size(types, [L_fa], 0, info) == 128 + 128 + 128 + 1
    placed = _lay(add4, 12, calls=[(i, 0, i + 1) for i in range(4)])
    inline = _lay(add4, 9, calls=[(i, 0, None) for i in range(4)])
    # placed: all 8 inputs export (each a single wire read once), so the input group is one forced-zero word
    assert LY.exported(add4, placed) == set(range(8)) and LY.exported(add4, inline) == set()
    assert LY.own_size(types, [L_fa, placed], 1, info) == 128 + 128 + 1
    # inline: the calls' 8 ANDs sit in the own region after the input word; the 5-bit output word aligns to 256
    assert LY.own_size(types, [L_fa, inline], 1, info) == 256 + 128 + 1


def test_valid_statements_pass(adders):
    types, fa, add4, w = adders
    L_fa = _lay(fa, 9)
    assert LY.validate(types, [L_fa, _lay(add4, None, calls=[(i, 0, i) for i in range(4)])], 1, info) == [385, 257]
    assert LY.validate(types, [L_fa, _lay(add4, None, calls=[(i, 0, None) for i in range(4)])], 1, info) == [385, 385]
    nested = [L_fa, _lay(add4, 12, calls=[(i, 0, i + 1) for i in range(4)]), _lay(w, None, calls=[(0, 1, 0)])]
    assert LY.validate(types, nested, 2, info) == [385, 257, 257]
    flat = [L_fa, _lay(add4, 9, calls=[(i, 0, None) for i in range(4)]), _lay(w, None, calls=[(0, 1, None)])]
    assert LY.validate(types, flat, 2, info) == [385, 385, 385]


def test_records_round_trip_and_the_digest_is_checked(adders):
    types, fa, add4, _ = adders
    for lay in (_lay(fa, 9), _lay(add4, None, calls=[(i, 0, i) for i in range(4)]), _lay(None, 9, own=("gen", TOY_SHA, 6, 8, 3))):
        d = lay.to_json()
        assert LY.from_json(json.loads(json.dumps(d))) == lay
        assert d["sha512"] == hashlib.sha512(b"verity/layout/v1\0" + json.dumps(lay.record(), sort_keys=True,
                                                                                  separators=(",", ":")).encode()).hexdigest()
        with pytest.raises(LY.InvalidLayout, match="sha512"):
            LY.from_json({**d, "sha512": "0" * 128})


def test_reads_inline_and_placed():
    rt = _read_type()
    types = {rt.digest(): rt}
    CT.validate(rt, types.__getitem__)
    # a program table keeps every value bit: k is its 8 value bits, though only bits 0-4 vary
    k = LY.k_of(TOY_SHA, 8, INFO[TOY_SHA])
    assert k == 8 and LY.live_bits(TOY, 8) == 5 and LY.read_rows_inline(6, 3, k) == 12 + 12 + 8 * 8 + 8
    # an input word, the read's 96 rows, the 8-bit output word aligned to 256, the constant
    assert LY.validate(types, [_lay(rt, None, reads=[(0, None, 3)])], 0, info) == [256 + 128 + 1]
    gen = _lay(None, 9, own=("gen", TOY_SHA, 6, 8, 3))
    assert LY.validate(types, [gen, _lay(rt, None, reads=[(0, 0, 0)])], 1, info)[0] == LY.read_rows_placed(6, 3, k) == 385
    assert LY.read_rows_placed(23, 14, 24) == 29_953 <= 1 << 15


def test_a_library_table_drops_only_its_constant_bits():
    ex2 = library.table("ex2")
    ti = LY.TableInfo(1 << 23, (1 << 23) - 1)  # ex2's words vary in bits 0-22, and agree above
    assert LY.k_of(ex2.sha512, 31, ti) == 23 and LY.k_of("0" * 128, 31, ti) == 31


def _refused(types, layouts, unit, match, inf=info):
    with pytest.raises(LY.InvalidLayout, match=match):
        LY.validate(types, layouts, unit, inf)


def test_validate_refuses_each_broken_rule(adders):
    types, fa, add4, w = adders
    L_fa = _lay(fa, 9)
    unit4 = lambda calls, **kw: _lay(add4, None, calls=calls, **kw)  # noqa: E731
    _refused(types, [L_fa, _lay(add4, 12, calls=[(i, 0, i + 1) for i in range(4)])], 1, "range_log")
    _refused(types, [_lay(fa, None), unit4([(i, 0, i) for i in range(4)])], 1, "range_log")
    _refused(types, [L_fa, unit4([(i, 0, i) for i in range(4)]), _lay(fa, 9)], 1, "does not reach")
    _refused(types, [unit4([(i, 1, i) for i in range(4)]), L_fa], 0, "listed before|does not reach")
    _refused(types, [L_fa, unit4([(i + 1, 0, i) for i in range(4)])], 1, "wrong item")
    _refused(types, [L_fa, unit4([(i, 0, 0) for i in range(4)])], 1, "slots 0, 1")
    _refused(types, [L_fa, unit4([(i, 0, i) for i in range(3)])], 1, "one entry per")
    _refused(types, [L_fa, unit4([(i, 0, i) for i in range(4)], ins=[2])], 1, "port groups")
    _refused(types, [_lay(None, 9, own=("gen", TOY_SHA, 6, 8, 3)), unit4([(i, 0, i) for i in range(4)])], 1, "not of the callee")
    _refused(types, [_lay(fa, 8), unit4([(i, 0, i) for i in range(4)])], 1, "exceeds its range")
    nested = lambda calls, rl=12: [L_fa, _lay(add4, rl, calls=calls), _lay(w, None, calls=[(0, 1, 0)])]  # noqa: E731
    _refused(types, nested([(i, 0, i) for i in range(4)]), 2, "off its own region")
    _refused(types, nested([(0, 0, 1), (1, 0, 1), (2, 0, 2), (3, 0, 3)]), 2, "overlap")
    _refused(types, nested([(i, 0, i + 5) for i in range(4)]), 2, "out of its range")
    _refused(types, nested([(i, 0, i + 1) for i in range(4)], rl=11), 2, "out of its range")
    places = [L_fa, _lay(add4, 12, calls=[(i, 0, i + 1) for i in range(4)]), _lay(w, None, calls=[(0, 1, None)])]
    _refused(types, places, 2, "places something")


def test_reads_are_refused_when_their_table_is_off():
    rt = _read_type()
    types = {rt.digest(): rt}
    _refused(types, [_lay(rt, None, reads=[(0, None, 6)])], 0, "split")
    _refused(types, [_lay(None, 9, own=("gen", TOY_SHA, 7, 8, 3)), _lay(rt, None, reads=[(0, 0, 0)])], 1, "index width")
    _refused(types, [_lay(None, 7, own=("gen", TOY_SHA, 6, 8, 3)), _lay(rt, None, reads=[(0, 0, 0)])], 1, "exceed its range")
    _refused(types, [_lay(rt, None, reads=[(0, None, 3)])], 0, "not held", inf=lambda s: None)


def _reader(sha: str, ib: int, vb: int) -> CT.CircuitType:
    """A type that reads the table (sha, ib, vb) once at its ib input bits and returns the value."""
    return CT.CircuitType(library.PROGRAM, ((ib, 1),), ((vb, 1),), (), ((sha, ib, vb),),
                          ((CT.READ, 0, tuple(((b,), 0) for b in range(ib))),), tuple(((ib + b,), 0) for b in range(vb)))


def test_a_read_is_tied_to_its_types_table_entry():
    """The red team's ib_mismatch and shape_mismatch cases (review of #200): the held table must have 2^index_bits words
    for the type's entry, and a placed read's generated layout its index and value widths."""
    ws16 = [(i * 7 + 3) & 0xF for i in range(16)]
    sha16 = hashlib.sha512(b"".join(w.to_bytes(4, "little") for w in ws16)).hexdigest()
    inf = {sha16: LY.table_info(ws16)}.get
    three = _reader(sha16, 3, 4)  # declares a 3-bit index for a 16-word table
    types = {three.digest(): three}
    CT.validate(three, types.__getitem__)
    _refused(types, [_lay(None, 10, own=("gen", sha16, 4, 4, 2)), _lay(three, None, reads=[(0, 0, 0)])], 1, r"2\^index_bits", inf)
    _refused(types, [_lay(three, None, reads=[(0, None, 2)])], 0, r"2\^index_bits", inf)
    four = _reader(sha16, 4, 4)
    types = {four.digest(): four}
    assert LY.validate(types, [_lay(None, 10, own=("gen", sha16, 4, 4, 2)), _lay(four, None, reads=[(0, 0, 0)])], 1, inf)
    _refused(types, [_lay(None, 10, own=("gen", sha16, 4, 2, 2)), _lay(four, None, reads=[(0, 0, 0)])], 1, "value width", inf)
    ws12 = ws16[:12]
    sha12 = hashlib.sha512(b"".join(w.to_bytes(4, "little") for w in ws12)).hexdigest()
    odd = _reader(sha12, 4, 4)
    _refused({odd.digest(): odd}, [_lay(odd, None, reads=[(0, None, 2)])], 0, r"2\^index_bits", {sha12: LY.table_info(ws12)}.get)


def _fanout(m: int, r: int):
    """The red team's reach_fanout: a unit calling one small type m times and reading a 16-word table r times through
    one placed generated read, over 3 layouts."""
    callee = CT.CircuitType(library.PROGRAM, ((1, 1),), ((1, 1),), (), (), ((CT.AND, ((0,), 0), ((0,), 1)),), (((1,), 0),))
    ws = [(i * 7 + 3) & 0xF for i in range(16)]
    sha = hashlib.sha512(b"".join(w.to_bytes(4, "little") for w in ws)).hexdigest()
    items = [(CT.CALL, 0, (((i,), 0),)) for i in range(m)] + [(CT.READ, 0, tuple(((i % m,), 0) for _ in range(4))) for i in range(r)]
    outs = [((m + i,), 0) for i in range(m)] + [((2 * m + 4 * i + b,), 0) for i in range(r) for b in range(4)]
    unit = CT.CircuitType(library.PROGRAM, ((m, 1),), ((len(outs), 1),), (callee.digest(),), ((sha, 4, 4),), tuple(items), tuple(outs))
    types = {callee.digest(): callee, unit.digest(): unit}
    layouts = [_lay(callee, 10), _lay(None, 10, own=("gen", sha, 4, 4, 2)),
               _lay(unit, None, calls=[(i, 0, i) for i in range(m)], reads=[(m + i, 1, i) for i in range(r)])]
    return types, layouts, {sha: LY.table_info(ws)}.get


@pytest.mark.parametrize("m,r", [(2, 2), (20, 20), (1092, 131)])
def test_a_wide_unit_is_reached(m, r):
    """Attention's shape (1,092 calls, 131 reads) is accepted: reachability is one downward sweep, with no fuel."""
    types, layouts, inf = _fanout(m, r)
    assert len(LY.validate(types, layouts, 2, inf)) == 3
