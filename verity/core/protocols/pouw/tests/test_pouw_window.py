"""PoUW's audit window (`pouw.window`): the grid a window's tile count is padded to and the q it gives the leakage bound,
L3's K′, the entry row and side trees of the layout for gates, the map from a drawn index to a tile, the window statement
on the entry rows, and the tile instances' public inputs."""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import struct
from dataclasses import replace
from fractions import Fraction

import pytest

import window_vectors as V
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import sha512_row_v2_digest
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.protocol import Shape, grid_tiles


def test_the_grid_is_the_numbers_of_at_most_s_plus_1_significant_bits():
    pts = [n for n in range(1, 1 << 12) if Wd.on_grid(n)]
    assert pts[:16] == list(range(1, 17)) and pts[16:20] == [18, 20, 22, 24]
    assert [Wd.level(g) for g in pts] == list(range(len(pts)))
    assert [Wd.point(i) for i in range(len(pts))] == pts
    for n in range(1, 1 << 12):
        g = Wd.floor_point(n)
        assert g <= n < Wd.point(Wd.level(g) + 1) and Wd.on_grid(g)
    assert all(b / a <= Fraction(9, 8) for a, b in zip(pts[15:], pts[16:]))
    for bad in (0, -3, True, 1.0):
        assert not Wd.on_grid(bad)
    with pytest.raises(ValueError):
        Wd.level(17)


def test_q_over_the_declared_ranges():
    """The window registers one level: log2 q bits.  304 levels over [1, 2^40], 161 over [2^10, 2^30]."""
    assert Wd.levels(1, 1 << 40) == 304 and Wd.levels(1 << 10, 1 << 30) == 161
    assert round(math.log2(304), 2) == 8.25


def _bucket(real: int) -> Wd.Work:
    """The work of a window of N = pad(real + 1) tiles of which `real` are credited."""
    return Wd.credit(real, Wd.pad(real + 1))


def _at(ell: int) -> Wd.Work:
    """Level ell's work in a window of the next grid point's tiles."""
    return Wd.Work.at(ell, Wd.point(ell + 1))


def test_credit_is_the_floor_point_of_the_masked_sum_of_the_shapes_tiles():
    """The window keeps its N, the shape's, and is credited floor_point(real) of them; real is 1 to N − 1 (the filler is
    never credited)."""
    for real in (1, 7, 16, 17, 24, 25, 1000, 10**7, 2**33 + 1):
        for n in (real + 1, 2 * real + 3):
            w = Wd.credit(real, n)
            assert w.credited == Wd.floor_point(real) <= real < w.n == n and Wd.Work.at(w.level, n) == w
            if real > 16:
                assert Fraction(w.credited, real) > Fraction(8, 9)
    assert Wd.credit(24, 129).to_json(4096) == {"level": 19, "n": 129, "credited_tiles": 24, "credited": 24 * 4096}
    for real, n in ((0, 5), (5, 5), (6, 5), (True, 5), (3, 2.0)):
        with pytest.raises(ValueError):
            Wd.credit(real, n)


def test_pad_is_the_least_grid_point_at_or_above_n():
    for n in range(1, 3000):
        g = Wd.pad(n)
        assert Wd.on_grid(g) and g >= n and (g == n or Wd.floor_point(n) < n < g)
    assert [Wd.pad(n) for n in (16, 17, 18, 19, 8192, 8193)] == [16, 18, 18, 20, 8192, 9216]


def test_k_prime_is_the_smallest_k_whose_miss_probability_is_at_most_delta():
    eps, delta = Fraction(1, 1000), Fraction(1, 2**40)
    full = Wd.k_prime(eps, delta, 10**7, 10**7, 1)
    assert full == 27713
    w = _bucket(10**7)
    k = Wd.k_prime(eps, delta, w.credited, w.n, 1)
    p = float(eps * w.credited / w.n)
    assert (1 - p) ** k <= float(delta) < (1 - p) ** (k - 1) and full <= k <= math.ceil(full * 9 / 8)
    assert Wd.k_prime(Fraction(1, 2), Fraction(1, 10), _bucket(8192).credited, _bucket(8192).n, 1) == 4
    assert Wd.k_prime(Fraction(1), Fraction(1, 10), 5, 5, 1) == 1
    assert Wd.k_prime(Fraction(1, 2), Fraction(1, 2**40), 2, 3, 1) == 3
    with pytest.raises(ValueError):
        Wd.k_prime(Fraction(0), Fraction(1, 2), 1, 1, 1)
    s = Wd.sampling(Fraction(1, 2), Fraction(1, 10), _bucket(8192), 4096)
    assert s == {"eps": "1/2", "delta": "1/10", "grid_bits": 3, "law": {"law": "subset", "k": 4}}


def _k_naive(eps, delta, credited, n, w):
    q = 1 - eps * credited / (n * w)
    k = 1
    while k < n and q ** k > delta:
        k += 1
    return k


def test_k_prime_is_exact_in_rationals():
    """(1 − p)^K = δ exactly takes the K that reaches it: at level 6 (7 tiles credited, 8 padded) with ε = 1, (1/8)^3 = 2^-9,
    where binary64 logarithms gave 4."""
    w = _at(6)
    assert (w.credited, w.n) == (7, 8)
    assert Wd.k_prime(Fraction(1), Fraction(1, 512), w.credited * 4096, w.n, 4096) == 3
    assert Wd.k_prime(Fraction(1, 2), Fraction(1, 8), 1, 1, 1) == 1 and Wd.k_prime(Fraction(1, 2), Fraction(1, 8), 1, 4, 1) == 4
    rng = random.Random(7)
    for _ in range(3000):
        n, w = rng.randint(1, 60), rng.randint(1, 4)
        credited, eps = rng.randint(1, n * w), Fraction(rng.randint(1, 9), 9)
        q = 1 - eps * credited / (n * w)
        delta = q ** rng.randint(1, 12) if rng.random() < 0.5 and 0 < q < 1 else Fraction(rng.randint(1, 99), 100)
        if not 0 < delta < 1:
            continue
        assert Wd.k_prime(eps, delta, credited, n, w) == _k_naive(eps, delta, credited, n, w)


def test_miss_at_most_decides_by_bounds_as_the_integers_do():
    """Past `EXACT_BITS` the comparison is decided from rounded bounds; on sizes the integers still reach, it agrees."""
    rng = random.Random(11)
    for _ in range(300):
        b = rng.getrandbits(rng.randint(8, 40)) + 2
        q = Fraction(rng.randint(max(1, b - b // 50 - 1), b - 1), b)
        k = Wd.EXACT_BITS // max(1, b.bit_length()) + rng.randint(1, 3000)
        delta = Fraction(1, 2 ** rng.randint(1, 40)) if rng.random() < 0.5 else q ** k * Fraction(rng.choice((999, 1001)), 1000)
        if not 0 < delta:
            continue
        assert Wd.miss_at_most(q, k, delta) == (q.numerator ** k * delta.denominator <= delta.numerator * q.denominator ** k)


def test_k_prime_at_the_cap_and_at_2_to_the_60():
    """With n up to 2^60 the answer is still the least K: (1 − p)^K <= δ < (1 − p)^(K − 1), or n when no K <= n reaches δ."""
    eps, delta = Fraction(1, 2), Fraction(1, 2 ** 40)
    for n, credited, w in ((1 << 60, 1, 1), (1 << 40, (1 << 40) - 5, 1), ((1 << 60) + 1, 3 << 50, 1 << 10),
                           (1 << 40, 1 << 20, 1)):
        k = Wd.k_prime(eps, delta, credited, n, w)
        q = 1 - eps * credited / (n * w)
        assert 1 <= k <= n and not (k > 1 and Wd.miss_at_most(q, k - 1, delta))
        assert k == n or Wd.miss_at_most(q, k, delta)
    assert Wd.k_prime(eps, delta, 1, 1 << 60, 1) == 1 << 60


def _d(n: int) -> bytes:
    return bytes([n]) * 64


#: the pass's -h3 tile root every call's entry names
H3 = _d(60)


#: two calls of 16 slots and 4 weight rows, at pass tiles 0 and 1: N = 129
SHAPE = (Wd.Call(16, 4, 0), Wd.Call(16, 4, 1))


def _entries(shape=SHAPE, bound=5):
    """The shape's entries with their roots: each call's sides, the pass's -h3 root and a record digest; the filler's."""
    out = []
    for c, e in enumerate(Wd.shape_entries(shape, bound)):
        if e.kind == "call":
            e = replace(e, root_a=_d(2 * c + 1), root_b=_d(2 * c + 2), root_h3=H3, record=_d(100 + c))
        elif e.kind == "filler":
            e = replace(e, root_a=_d(50), root_b=_d(51))
        out.append(e)
    return out


def _fields(shape=SHAPE, real=(3, 2), refused=((), ())):
    """Each call's side rows' (index, ok, admit, blank): its first real[c] slots admitted, those in refused[c] with ok 0,
    the rest canonical empty rows; every weight row admitted and formed; the filler's one empty row each side."""
    out = []
    for c, call in enumerate(shape):
        a = [[i, int(i not in refused[c]), 1, 0] if i < real[c] else [i, 0, 0, 1] for i in range(call.m)]
        out.append({"a": a, "b": [[j, 1, 1, 0] for j in range(call.n)]})
    return out + [{"a": [[0, 0, 0, 1]], "b": [[0, 0, 0, 1]]}]


def _rows(entries):
    return [e.row() for e in entries]


def test_an_entry_is_one_fixed_width_row():
    """An entry row is 336 bytes (168 u16 words): kind, lo, hi, n_a, n_b and a zero word as u64 LE, then root_a, root_b and
    root_h3, then tile0, cols, m and a zero word as u64 LE, then the record's SHA-512, each digest on a 128-bit word; it
    round-trips through its bytes, its words and its JSON, and a row of another length, an unknown kind or a nonzero pad is
    no entry."""
    e = Wd.Entry("call", 5, 17, 3, 4, _d(1), _d(2), _d(3), _d(4), tile0=7, cols=2, m=80)
    row = e.row()
    assert len(row) == Wd.ENTRY_BYTES == 336 and len(e.words()) == Wd.ENTRY_WORDS == 168 and Wd.ENTRY_BITS == 2688
    assert struct.unpack_from("<6Q", row) == (1, 5, 17, 3, 4, 0) and struct.unpack_from("<4Q", row, 240) == (7, 2, 80, 0)
    assert (row[48:112], row[112:176], row[176:240], row[272:336]) == (_d(1), _d(2), _d(3), _d(4))
    assert all(row.index(_d(n)) % 16 == 0 for n in (1, 2, 3, 4)) and Wd.ENTRY_BYTES % 16 == 0
    assert Wd.Entry.from_row(row) == e == Wd.Entry.from_words(e.words()) == Wd.Entry.from_json(e.to_json())
    assert Wd.Entry("none").row() == bytes(336) and Wd.Entry("filler").row()[0] == 2
    assert Wd.entry_digest(e) == sha512_row_v2_digest(row, 2688)
    for bad, why in ((row[:-1], "336 bytes"), (b"\3" + row[1:], "kind 3"), (row[:40] + b"\1" + row[41:], "40..48"),
                     (row[:264] + b"\1" + row[265:], "264..272")):
        with pytest.raises(ValueError, match=why):
            Wd.Entry.from_row(bad)
    for bad in (Wd.Entry("tile"), Wd.Entry("call", root_a=bytes(63)), Wd.Entry("call", root_h3=bytes(32)),
                Wd.Entry("call", lo=1 << 64), Wd.Entry("call", m=-1)):
        with pytest.raises(ValueError):
            bad.row()
    with pytest.raises(ValueError):
        Wd.Entry.from_words([1 << 16] * 168)
    assert Wd.record_digest({"b": 1, "a": [2]}) == hashlib.sha512(b'{"a":[2],"b":1}').digest()


def test_the_layout_bounds_each_entry():
    ok = Wd.Entry("call", 0, 6, 2, 3, _d(1), _d(2), H3, _d(3), tile0=5, cols=1, m=16)
    assert Wd.layout_problem(ok) is None and Wd.layout_problem(Wd.Entry("none")) is None
    big = replace(ok, n_a=1 << 13, n_b=1 << 17, m=1 << 13, cols=1 << 11, tile0=(1 << 40) - (1 << 18))
    assert Wd.layout_problem(big) is None and Wd.pass_tiles(big) == 1 << 18
    filler = replace(ok, kind="filler", root_h3=bytes(64), record=bytes(64), tile0=0, cols=0, m=0)
    assert Wd.layout_problem(filler) is None
    for bad, why in ((replace(ok, hi=1 << 40), "40-bit"), (Wd.Entry("none", hi=1), "zero row"),
                     (Wd.Entry("none", record=_d(1)), "zero row"), (Wd.Entry("none", root_h3=_d(1)), "zero row"),
                     (Wd.Entry("none", m=16), "zero row"),
                     (replace(ok, n_a=0), "side a holds 0"), (replace(ok, n_a=(1 << 13) + 1), "2\\^13"),
                     (replace(ok, n_b=(1 << 17) + 1), "side b"), (replace(filler, root_h3=H3), "filler's -h3 root"),
                     (replace(filler, record=_d(3)), "filler's record"), (replace(filler, tile0=1), "filler's tile0"),
                     (replace(filler, cols=1), "filler's tile0"), (replace(filler, m=16), "filler's tile0"),
                     (replace(ok, m=0), "its m 0"), (replace(ok, m=24), "multiple of 16"),
                     (replace(ok, m=(1 << 13) + 16), "its m"), (replace(ok, n_a=17, m=16), "from n_a = 17"),
                     (replace(ok, cols=0), "its cols 0"), (replace(ok, cols=(1 << 11) + 1), "its cols"),
                     (replace(ok, n_b=65), "n_b = 65"), (replace(big, tile0=big.tile0 + 1), "pass tiles")):
        assert re.search(why, Wd.layout_problem(bad)), (bad, why)


def test_a_side_row_is_its_operand_row_then_its_index_and_annotations():
    """A side row is the operand row's words, then one 128-bit word: its index as a u32 (two u16 words, low first), its ok
    and admit words and four zero words; fp8-p at k = 4096 gives 2,120 words.  `side_field` reads the field and whether
    the operand words are blank."""
    words = list(range(1, 17))
    row = Wd.side_row(words, 0x1_2345, 1, 1)
    assert row == words + [0x2345, 1, 1, 1, 0, 0, 0, 0] and Wd.side_field(row) == (0x1_2345, 1, 1, 0)
    assert Wd.side_field(Wd.side_row([0] * 16, 7, 0, 0)) == (7, 0, 0, 1)
    assert Wd.side_field(Wd.side_row(words, 3, 0, 1)) == (3, 0, 1, 0)
    assert len(Wd.side_row([0] * (4096 // 2 + 64), 8191, 0, 1)) == 2120
    for bad in ((words[:-1], 0, 1, 1), (words, -1, 1, 1), (words, 1 << 32, 1, 1), (words, True, 1, 1), (words, 0, 2, 1),
                (words, 0, 1, -1), (words, 0, True, 1)):
        with pytest.raises(ValueError):
            Wd.side_row(*bad)
    for bad in (row[:-1] + [1], row[:8], words[:8] + [1 << 16] + [0] * 7, row[:-6] + [2, 0, 0, 0, 0, 0],
                row[:-5] + [3, 0, 0, 0, 0]):
        with pytest.raises(ValueError):
            Wd.side_field(bad)


def test_tile_read_finds_the_pass_leaf_from_the_rows_a_unit_reads():
    """p and at from the entry row and the two side rows' indices alone are the pass's tile holding word (I, J), in
    `audit.tile_refs`'s order (workload order, then each call's 64 x 64 tiles row-major, the last row and column of tiles
    possibly short), and the word's (row, column, tile height) in it."""
    shapes = [(80, 128), (16, 64), (64, 192), (48, 64)]
    refs, tile0 = [], []
    for c, (m, n) in enumerate(shapes):
        tile0.append(len(refs))
        refs += [(c, t) for t in grid_tiles(Shape(m, 32, n), 64, 64)]
    for c, (m, n) in enumerate(shapes):
        entry = Wd.Entry("call", tile0=tile0[c], cols=-(-n // 64), m=m)
        for i in range(m):
            for j in range(0, n, 7):
                p, at = Wd.tile_read(entry, i, j)
                cc, tile = refs[p]
                assert cc == c and i in tile.rows and j in tile.cols, (c, i, j)
                assert at == tile.rows.index(i) | tile.cols.index(j) << 6 | (len(tile.rows) // 16 - 1) << 12
    for bad in ((80, 0), (0, 128), (-1, 0)):
        with pytest.raises(ValueError):
            Wd.tile_read(Wd.Entry("call", tile0=0, cols=2, m=80), *bad)
    with pytest.raises(ValueError):
        Wd.tile_read(Wd.Entry("filler"), 0, 0)


def test_tile_at_is_a_function_of_the_shape():
    """Every slot and weight row is in place, so index u of call c's range is (u div n_c, u mod n_c), and N − 1 is the
    filler's (C, 0, 0), whatever the window's real tokens."""
    assert Wd.shape_tiles(SHAPE) == 129
    assert Wd.tile_at(SHAPE, 0) == (0, 0, 0) and Wd.tile_at(SHAPE, 11) == (0, 2, 3) and Wd.tile_at(SHAPE, 63) == (0, 15, 3)
    assert Wd.tile_at(SHAPE, 64) == (1, 0, 0) and Wd.tile_at(SHAPE, 127) == (1, 15, 3) and Wd.tile_at(SHAPE, 128) == (2, 0, 0)
    for bad in (129, -1, True, 1.0):
        with pytest.raises(ValueError):
            Wd.tile_at(SHAPE, bad)
    for t in range(129):
        e, i, j = Wd.tile_at(SHAPE, t)
        entry = Wd.shape_entries(SHAPE, 5)[e]
        assert entry.lo <= t < entry.hi and t - entry.lo == i * entry.n_b + j and i < entry.n_a


def test_the_shape_is_a_windows_or_refused():
    assert Wd.shape_problem(SHAPE, 3) is None and Wd.shape_of(Wd.shape_json(SHAPE)) == SHAPE
    big = (Wd.Call(1 << 13, 1 << 17, 0),)
    assert Wd.shape_problem(big, 2) is None and Wd.shape_tiles(big) == (1 << 30) + 1
    cases = {"2 calls, not 1 to the bound 2": (SHAPE, 2), "call 0: its m 24": ((Wd.Call(24, 4, 0),), 3),
             "call 0: its m 0": ((Wd.Call(0, 4, 0),), 3), "call 0: its n 0": ((Wd.Call(16, 0, 0),), 3),
             "its n 131073": ((Wd.Call(16, (1 << 17) + 1, 0),), 3),
             "call 1: its pass tiles start at 0, before the last call's end 1": ((SHAPE[0], Wd.Call(16, 4, 0)), 3),
             "beyond 2^40": ((Wd.Call(16, 4, 1 << 40),), 3), "0 calls": ((), 3)}
    for want, (shape, bound) in cases.items():
        got = Wd.shape_problem(shape, bound)
        assert got and want in got, (want, got)
    for bad in ([], [{"m": 16, "n": 4}], [{"m": 16, "n": 4, "tile0": -1}], [{"m": True, "n": 4, "tile0": 0}], {"m": 16},
                [{"m": 16, "n": 4, "tile0": 0, "k": 1}]):
        with pytest.raises(ValueError):
            Wd.shape_of(bad)


def test_the_window_statement_refuses_each_departure_from_the_layout():
    w, es, fs = Wd.credit(20, 129), _entries(), _fields()
    assert Wd.window_problem(_rows(es), w, 5, H3, SHAPE, fs) is None

    def at(e, side, i, field):
        out = json.loads(json.dumps(fs))
        out[e][side][i] = field
        return out

    swapped = json.loads(json.dumps(fs))
    swapped[0]["a"][:2] = swapped[0]["a"][1::-1]
    cases = {
        "entry 1: its -h3 root is not the pass's": (es[:1] + [replace(es[1], root_h3=_d(61))] + es[2:], 5, w, fs),
        "not the layout: the filler's -h3 root": (es[:2] + [replace(es[2], root_h3=H3)] + es[3:], 5, w, fs),
        "5 entries, not the bound 6": (es, 6, w, fs),
        "not the registered level 22 of 129": (es, 5, Wd.credit(30, 129), fs),
        "not the registered level 17 of 130": (es, 5, Wd.Work.at(w.level, 130), fs),
        "entry 0: its kind, range, rows, tile0, cols and m are not the shape's":
            ([replace(es[0], hi=63)] + es[1:], 5, w, fs),
        "entry 1: its kind, range, rows, tile0, cols and m": (es[:1] + [replace(es[1], tile0=2)] + es[2:], 5, w, fs),
        "entry 2: its kind, range": (es[:2] + [replace(es[2], n_a=2)] + es[3:], 5, w, fs),
        "entry 3: its kind": (es[:3] + [es[0]] + es[4:], 5, w, fs),
        "not the layout: side b holds 0": (es[:1] + [replace(es[1], n_b=0)] + es[2:], 5, w, fs),
        "not the layout: an empty entry": (es[:4] + [Wd.Entry("none", n_a=1)], 5, w, fs),
        "entry 0: its side-a rows' indices are not their leaves 0 to 15": (es, 5, w, swapped),
        "entry 1: its side-b rows' indices": (es, 5, w, at(1, "b", 3, [4, 1, 1, 0])),
        "entry 0: a side-a row's ok, admit or blank is not 0 or 1": (es, 5, w, at(0, "a", 0, [0, 2, 1, 0])),
        "entry 1: a side-b row's ok, admit or blank": (es, 5, w, at(1, "b", 0, [0, 1, 1, 7])),
        "entry 2: the filler's side-a row is admitted": (es, 5, w, at(2, "a", 0, [0, 0, 1, 1])),
        "entry 2: side-b row 0 is not admitted and not the canonical empty row": (es, 5, w, at(2, "b", 0, [0, 1, 0, 1])),
        "entry 0: side-a row 9 is not admitted and not the canonical empty row": (es, 5, w, at(0, "a", 9, [9, 0, 0, 0])),
        "no tile is credited": (es, 5, w, _fields(real=(0, 0))),
    }
    for want, (entries, bound, work, fields) in cases.items():
        got = Wd.window_problem(_rows(entries), work, bound, H3, SHAPE, fields)
        assert got and want in got, (want, got)
    for bad, why in ((es[0].row()[:-2], "entry 0: not an entry row: an entry row is 336 bytes"),
                     (b"\7" + es[0].row()[1:], "entry 0: not an entry row: kind 7")):
        assert Wd.window_problem([bad] + _rows(es[1:]), w, 5, H3, SHAPE, fs).startswith(why)
    assert "entry 0: its -h3 root" in Wd.window_problem(_rows(es), w, 5, _d(61), SHAPE, fs)
    with pytest.raises(ValueError, match="64 bytes"):
        Wd.window_problem(_rows(es), w, 5, H3[:32], SHAPE, fs)
    with pytest.raises(ValueError, match="the shape is not a window's"):
        Wd.window_problem(_rows(es), w, 5, H3, (Wd.Call(24, 4, 0),), fs)
    for bad in (fs[:2], [*fs[:2], {"a": [[0, 0, 0]], "b": [[0, 0, 0, 1]]}], [*fs[:2], {"a": [[0, 0, 0, 1]]}]):
        with pytest.raises(ValueError, match="fields"):
            Wd.window_problem(_rows(es), w, 5, H3, SHAPE, bad)
    assert {p[0] for p in Wd.PUBLIC} == {"verdict", "work", "check", "sampling", "shape", "drawn", "root_W"}
    assert set(Wd.REGISTERED) == {p[0] for p in Wd.PUBLIC if p[2] != "outcome"}


def test_windows_of_one_shape_have_one_structure_whatever_their_real_tokens():
    """Two windows of the same public parameters, 3 and 9 real tokens a call: the same N, entries' public fields, side
    rows (count and index), tile map and draw size, so nothing but the one registered level depends on the values."""
    few, many = _fields(real=(3, 3)), _fields(real=(9, 9))
    w_few, w_many = Wd.credit(24, 129), Wd.credit(72, 129)
    es = _entries()
    assert Wd.window_problem(_rows(es), w_few, 5, H3, SHAPE, few) is None
    assert Wd.window_problem(_rows(es), w_many, 5, H3, SHAPE, many) is None
    assert w_few.n == w_many.n == Wd.shape_tiles(SHAPE) and w_few.level != w_many.level
    assert [[[p[0] for p in f[s]] for s in "ab"] for f in few] == [[[p[0] for p in f[s]] for s in "ab"] for f in many]
    assert [Wd.public_fields(e) for e in es] == [Wd.public_fields(e) for e in Wd.shape_entries(SHAPE, 5)]
    eps, delta = Fraction(1, 2), Fraction(1, 2**20)
    assert Wd.k_prime(eps, delta, w_few.credited, w_few.n, 1) > Wd.k_prime(eps, delta, w_many.credited, w_many.n, 1)


def test_a_refused_row_stays_in_place_and_is_not_credited():
    """Slot 1 of call 0 is admitted and its forming refused: its row stays leaf 1 with ok 0, every later row keeps its
    leaf, and the window credits its 2 other slots' tiles only (16, not 20)."""
    fs = _fields(refused=((1,), ()))
    assert [p[0] for p in fs[0]["a"]] == list(range(16)) and fs[0]["a"][1] == [1, 0, 1, 0]
    assert Wd.window_problem(_rows(_entries()), Wd.credit(16, 129), 5, H3, SHAPE, fs) is None
    got = Wd.window_problem(_rows(_entries()), Wd.credit(20, 129), 5, H3, SHAPE, fs)
    assert "16 credited tiles of 129 are level" in got and "not the registered level" in got


def test_a_forged_annotation_on_an_empty_slot_credits_nothing():
    """FMT-R45's attack: slot 5 of call 0 is empty (no request: the canonical empty row).  Setting its ok to 1 fails the
    emptiness check; setting its admit to 1 leaves an admitted zero row whose ok is 0, credited nothing, so the window's
    credit stays 20 and 24 (its 4 tiles) is refused."""
    honest, es = _fields(), _entries()
    assert honest[0]["a"][5] == [5, 0, 0, 1]
    ok = json.loads(json.dumps(honest))
    ok[0]["a"][5] = [5, 1, 0, 1]
    for work in (Wd.credit(20, 129), Wd.credit(24, 129)):
        got = Wd.window_problem(_rows(es), work, 5, H3, SHAPE, ok)
        assert "entry 0: side-a row 5 is not admitted and not the canonical empty row" in got
    admit = json.loads(json.dumps(honest))
    admit[0]["a"][5] = [5, 0, 1, 1]
    assert Wd.selected(admit[0]) == Wd.selected(honest[0]) == {"a": 3, "b": 4}
    assert Wd.window_problem(_rows(es), Wd.credit(20, 129), 5, H3, SHAPE, admit) is None
    assert "not the registered level" in Wd.window_problem(_rows(es), Wd.credit(24, 129), 5, H3, SHAPE, admit)


def test_a_side_tree_is_its_rows_then_frame_v3_pads_to_two_to_the_d():
    """`side_levels` and `side_path` are `tree_levels` over the n leaves and the domain's pad leaves at ranks n..2^D − 1,
    computed without the pads' own leaves; and each window's side domains are its own, one per side, at the public depth."""
    small = CommitmentDomain(hashlib.sha512(b"side").digest(), -1, RangeIndexedDomain(0, 16), hash="sha512")
    leaves = [hashlib.sha512(bytes([i])).digest() for i in range(16)]
    for n in (1, 3, 5, 8, 16):
        full = tree_levels(small, leaves[:n] + [small.pad(r) for r in range(n, 16)])
        mine = Wd.side_levels(small, leaves[:n])
        assert mine[-1] == full[-1] and all(m == f[:len(m)] for m, f in zip(mine, full))
        for i in range(n):
            sib = Wd.side_path(small, mine, i)
            assert sib == [full[d][(i >> d) ^ 1] for d in range(4)]
    window = hashlib.sha512(b"window").digest()
    a, b = Wd.side_domain(window, "a"), Wd.side_domain(window, "b")
    assert (a.count, b.count) == (1 << 13, 1 << 17) and a.domain_id != b.domain_id
    assert a == Wd.side_domain(window, "a") and a.domain_id != Wd.side_domain(hashlib.sha512(b"other").digest(), "a").domain_id
    three = leaves[:3]
    assert Wd.side_levels(a, three)[-1] == tree_levels(a, three + [a.pad(r) for r in range(3, 1 << 13)])[-1]
    for bad in (("c", window), ("a", window[:32])):
        with pytest.raises(ValueError):
            Wd.side_domain(bad[1], bad[0])
    for n in (0, 17):
        with pytest.raises(ValueError):
            Wd.side_levels(small, (leaves * 2)[:n])


def _registration(n=9, k=3, root=_d(7), dom=_d(8)):
    return {"population": n, "law": {"law": "subset", "k": k},
            "roots": [{"port": "entries", "root": root.hex(), "leaves": 4, "domain": dom.hex()},
                      {"port": "W", "root": _d(9).hex(), "leaves": 1, "domain": _d(10).hex()}]}


def test_the_public_inputs_file_is_c_flocks_and_pads_with_the_last_filler_tile():
    """F_σ is C-Flock's ``verity/flock-public-inputs/v1`` (a key-sorted JSON header line, then the instances' words, u16 LE,
    instance-major, the ports in order): t as a u128, the window root, its domain, the two side domains and the pass's -h3
    tile domain, from the registration, the draw and the pass alone.  Session σ of ⌈K′/cap⌉ holds drawn[σ·cap ..], then
    N − 1."""
    reg, drawn, h3 = _registration(), [1, 4, 6], _d(11)
    assert Wd.sessions(3, 2) == 2 and Wd.sessions(4, 2) == 2 and Wd.sessions(1, 4096) == 1
    sides = Wd.side_domain(_d(8), "a").domain_id + Wd.side_domain(_d(8), "b").domain_id
    for sigma, ts in ((0, [1, 4]), (1, [6, 8])):
        assert Wd.instance_ts(reg, drawn, sigma, 2) == ts
        head, body = Wd.public_inputs(reg, drawn, sigma, 2, h3).split(b"\n", 1)
        h = json.loads(head)
        assert head == json.dumps(h, sort_keys=True).encode() and h["format"] == Wd.PUBLIC_INPUTS_FORMAT
        assert h["instances"] == 2 and h["ports"] == list(Wd.PUBLIC_PORTS)
        assert [p["port"] for p in h["ports"]] == ["t", "root", "domain", "domain_a", "domain_b", "domain_h3"]
        assert all(p["words"] % 8 == 0 for p in h["ports"])
        per = 2 * sum(p["words"] for p in h["ports"])
        assert per == 336 and body == b"".join(t.to_bytes(16, "little") + _d(7) + _d(8) + sides + h3 for t in ts)
    assert Wd.instance_ts(reg, drawn, 0, 4) == [1, 4, 6, 8]
    bad = {"population": (_registration(n=0), drawn, 0, 2), "2^40": (_registration(n=(1 << 40) + 1), drawn, 0, 2),
           "subset": ({**reg, "law": {"law": "bernoulli", "k": 3}}, drawn, 0, 2),
           "distinct": (reg, [4, 1, 6], 0, 2), "distinct ascending": (reg, [1, 1, 6], 0, 2),
           "not 3": (reg, [1, 4], 0, 2), "of [0, 9)": (reg, [1, 4, 9], 0, 2), "session 2": (reg, drawn, 2, 2),
           "roots of port": ({**reg, "roots": reg["roots"][1:]}, drawn, 0, 2),
           "window root and its domain id are 64 bytes": (_registration(root=b"\1" * 32), drawn, 0, 2)}
    for why, args in bad.items():
        with pytest.raises(ValueError, match=re.escape(why)):
            Wd.public_inputs(*args, h3)
    with pytest.raises(ValueError, match="-h3 domain id is 64 bytes"):
        Wd.public_inputs(reg, drawn, 0, 2, h3[:32])
    with pytest.raises(ValueError):
        Wd.sessions(3, 0)


#: the served sizing points (ε, δ) and the levels a served window registers
POINTS = ((Fraction(1, 1000), Fraction(1, 2**40)), (Fraction(1, 100), Fraction(1, 2**128)))
LEVELS = range(87, 144)


def test_sessions_hold_the_largest_served_draw_at_both_points():
    """Over levels 87 to 143 (N from 9,216 to 1,179,648) K′ is N at the small levels and at most 31,178 at ε = 0.1%,
    δ = 2^-40 (27,713 unpadded, times up to g_(ℓ+1)/g_ℓ = 9/8), and at most 9,937 at ε = 1%, δ = 2^-128, both at level 143.
    A session of the tile unit at k_log 24 (`Pc8TileLeafH3`'s) holds at most 2,048 instances (m <= 35), so S is 16 and 5;
    at k_log 25, 31 and 10.  F_σ at the largest draw is every session's `cap` instances, the last padded with N − 1."""
    assert (Wd.point(87), _at(87).n, Wd.point(143), _at(143).n) == (8192, 9216, 1 << 20, 1179648)
    assert [Wd.session_cap(k) for k in (24, 25)] == [2048, 1024] and Wd.session_cap(24, 4) == 8192
    for bad in (0, 33):
        with pytest.raises(ValueError):
            Wd.session_cap(bad)
    largest = []
    for (eps, delta), (k_max, s24, s25) in zip(POINTS, ((31178, 16, 31), (9937, 5, 10))):
        ks = {ell: Wd.k_prime(eps, delta, _at(ell).credited, _at(ell).n, 1) for ell in LEVELS}
        assert max(ks.values()) == ks[143] == k_max and ks[87] == _at(87).n
        assert all(k <= _at(ell).n for ell, k in ks.items())
        assert (Wd.sessions(k_max, Wd.session_cap(24)), Wd.sessions(k_max, Wd.session_cap(25))) == (s24, s25)
        largest.append(k_max)
    n, k, cap = _at(143).n, largest[0], Wd.session_cap(24)
    drawn = sorted(random.Random(0).sample(range(n - 1), k))
    reg = _registration(n=n, k=k)
    s = Wd.sessions(k, cap)
    last = Wd.instance_ts(reg, drawn, s - 1, cap)
    assert last[:k - (s - 1) * cap] == drawn[(s - 1) * cap:] and last[k - (s - 1) * cap:] == [n - 1] * (s * cap - k)
    head, body = Wd.public_inputs(reg, drawn, s - 1, cap, _d(11)).split(b"\n", 1)
    assert json.loads(head)["instances"] == cap and len(body) == cap * 336
    assert int.from_bytes(body[-336:-320], "little") == n - 1 < 1 << Wd.T_BITS


def test_the_pinned_vectors_are_the_layout():
    """``vectors/window.json`` is `window_vectors.vectors()`: the toy window's shape, entry rows, records and side rows'
    fields, (p, at) at the pinned indices, a side tree, the root and F_σ.  Every slot is in place (call 0's 80, of which
    71 admitted, slot 2 refused), and the vectors agree with the window statement, records included."""
    pinned = json.loads(V.OUT.read_text())
    assert pinned == json.loads(json.dumps(V.vectors()))
    rows = [bytes.fromhex(e["row"]) for e in pinned["entries"]]
    shape = Wd.shape_of(pinned["shape"])
    work = Wd.Work.at(pinned["window"]["work"]["level"], pinned["window"]["work"]["n"])
    root_h3, domain_h3 = (bytes.fromhex(pinned["window"][k]) for k in ("root_h3", "domain_h3"))
    records, fields = pinned["records"], pinned["fields"]
    assert Wd.window_problem(rows, work, V.P, root_h3, shape, fields, records=records) is None
    assert (work.n, pinned["window"]["real"], work.credited) == (5345, 4562, 4096)
    call = Wd.Entry.from_row(rows[0])
    assert (call.n_a, call.n_b, call.hi, call.root_h3, call.tile0, call.cols, call.m) == (80, 66, 5280, root_h3, 3, 2, 80)
    assert Wd.Entry.from_row(rows[2]).root_h3 == bytes(64) and Wd.Entry.from_row(rows[2]).lo == 5344
    side = pinned["side"]
    assert side["rows"] == 16 and len(side["path_of_1"]) == 13
    assert Wd.side_field(side["row_1"]) == (1, 1, 1, 0) and Wd.side_field(side["row_15"]) == (15, 0, 0, 1)
    assert [(r["p"], r["at"] >> 12) for r in pinned["tile_reads"] if "p" in r][2:4] == [(6, 0), (5, 0)]
    moved = [{**records[0], "tile0": 4}] + records[1:]
    assert "not the one the entry binds" in Wd.window_problem(rows, work, V.P, root_h3, shape, fields, records=moved)
    swapped = json.loads(json.dumps(fields))
    swapped[0]["b"][3:5] = swapped[0]["b"][4:2:-1]
    assert "entry 0: its side-b rows' indices are not their leaves 0 to 65" in Wd.window_problem(
        rows, work, V.P, root_h3, shape, swapped, records=records)
    reg = pinned["public_inputs"]["registration"]
    assert reg["population"] == 5345
    for sigma, f in enumerate(pinned["public_inputs"]["sessions"]):
        assert Wd.public_inputs(reg, pinned["public_inputs"]["drawn"], sigma, V.CAP, domain_h3) == bytes.fromhex(f)


def test_a_record_is_held_to_its_entry():
    """A call's record gives its entry's tile0, cols = ⌈n/64⌉ and m, and its entry binds its SHA-512."""
    rec = {"shape": [80, 32, 100], "tile0": 3}
    entry = Wd.Entry("call", 0, 8000, 80, 100, record=Wd.record_digest(rec), tile0=3, cols=2, m=80)
    assert Wd.record_problem(entry, rec) is None
    for bad, why in (({**rec, "tile0": 4}, "tile0, cols and m"), ({**rec, "shape": [80, 32, 129]}, "cols and m"),
                     ({**rec, "shape": [96, 32, 100]}, "cols and m"), ({"tile0": 3}, "no shape")):
        assert why in Wd.record_problem(replace(entry, record=Wd.record_digest(bad)), bad), (bad, why)
    assert "binds" in Wd.record_problem(entry, {**rec, "tile0": 4})
