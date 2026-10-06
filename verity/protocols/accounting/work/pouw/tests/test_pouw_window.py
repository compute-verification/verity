"""PoUW's audit window (`pouw.window`): the grid a window's tile count is padded to and the q it gives the leakage bound,
L3's K′, the entry row and side trees of the layout for gates, the map from a drawn index to a tile, the window statement
on the entry rows, and the tile instances' public inputs."""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import replace
from fractions import Fraction

import pytest

import window_vectors as V
from verity.primitives.commitments.indexed import RangeIndexedDomain
from verity.primitives.commitments.merkle import CommitmentDomain, tree_levels
from verity.primitives.commitments.rowleaf import sha512_row_v2_digest
from verity.protocols.accounting.work.pouw import window as Wd


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


def test_bucket_credits_the_floor_point_and_pads_past_the_real_count():
    for real in (1, 7, 16, 17, 24, 25, 1000, 10**7, 2**33 + 1):
        w = Wd.bucket(real)
        assert w.credited <= real < w.n and w.n == Wd.point(w.level + 1) and Wd.Work.at(w.level) == w
        if real > 16:
            assert Fraction(w.credited, real) > Fraction(8, 9) and Fraction(w.n, w.credited) <= Fraction(9, 8)
    assert Wd.bucket(24).to_json(4096) == {"level": 19, "n": 26, "credited_tiles": 24, "credited": 24 * 4096}


def test_the_padded_count_is_the_unit_count_bucket_of_the_real_tiles_and_one_filler():
    """N is the least grid point at or above N_real + 1, the private partition's n_b on the same grid."""
    for n in range(1, 3000):
        g = Wd.pad(n)
        assert Wd.on_grid(g) and g >= n and (g == n or Wd.floor_point(n) < n < g)
        assert Wd.bucket(n).n == Wd.pad(n + 1)
    assert [Wd.pad(n) for n in (16, 17, 18, 19, 8192, 8193)] == [16, 18, 18, 20, 8192, 9216]


def test_k_prime_is_the_smallest_k_whose_miss_probability_is_at_most_delta():
    eps, delta = Fraction(1, 1000), Fraction(1, 2**40)
    full = Wd.k_prime(eps, delta, 10**7, 10**7, 1)
    assert full == 27713
    w = Wd.bucket(10**7)
    k = Wd.k_prime(eps, delta, w.credited, w.n, 1)
    p = float(eps * w.credited / w.n)
    assert (1 - p) ** k <= float(delta) < (1 - p) ** (k - 1) and full <= k <= math.ceil(full * 9 / 8)
    assert Wd.k_prime(Fraction(1, 2), Fraction(1, 10), Wd.bucket(8192).credited, Wd.bucket(8192).n, 1) == 4
    assert Wd.k_prime(Fraction(1), Fraction(1, 10), 5, 5, 1) == 1
    assert Wd.k_prime(Fraction(1, 2), Fraction(1, 2**40), 2, 3, 1) == 3
    with pytest.raises(ValueError):
        Wd.k_prime(Fraction(0), Fraction(1, 2), 1, 1, 1)
    s = Wd.sampling(Fraction(1, 2), Fraction(1, 10), Wd.bucket(8192), 4096)
    assert s == {"eps": "1/2", "delta": "1/10", "grid_bits": 3, "law": {"law": "subset", "k": 4}}


def _d(n: int) -> bytes:
    return bytes([n]) * 64


def _entries(work, real_a=(3, 2), rows_b=4, filler=(1, 2), bound=5):
    out, at = [], 0
    for c, a in enumerate(real_a):
        out.append(Wd.Entry("call", at, at + a * rows_b, a, rows_b, _d(2 * c + 1), _d(2 * c + 2), _d(100 + c)))
        at += a * rows_b
    out.append(Wd.Entry("filler", at, work.n, *filler, _d(50), _d(51)))
    return out + [Wd.Entry("none")] * (bound - len(out))


def _rows(entries):
    return [e.row() for e in entries]


def test_an_entry_is_one_fixed_width_row():
    """An entry row is 240 bytes (120 u16 words): kind, lo, hi, n_a, n_b and a zero word as u64 LE, then root_a, root_b and
    the record's SHA-512; it round-trips through its bytes, its words and its JSON, and a row of another length, an unknown
    kind or a nonzero pad is no entry."""
    e = Wd.Entry("call", 5, 17, 3, 4, _d(1), _d(2), _d(3))
    row = e.row()
    assert len(row) == Wd.ENTRY_BYTES == 240 and len(e.words()) == Wd.ENTRY_WORDS == 120 and Wd.ENTRY_BITS == 1920
    assert struct.unpack_from("<6Q", row) == (1, 5, 17, 3, 4, 0)
    assert (row[48:112], row[112:176], row[176:240]) == (_d(1), _d(2), _d(3))
    assert Wd.Entry.from_row(row) == e == Wd.Entry.from_words(e.words()) == Wd.Entry.from_json(e.to_json())
    assert Wd.Entry("none").row() == bytes(240) and Wd.Entry("filler").row()[0] == 2
    assert Wd.entry_digest(e) == sha512_row_v2_digest(row, 1920)
    for bad, why in ((row[:-1], "240 bytes"), (b"\3" + row[1:], "kind 3"), (row[:40] + b"\1" + row[41:], "40..48")):
        with pytest.raises(ValueError, match=why):
            Wd.Entry.from_row(bad)
    for bad in (Wd.Entry("tile"), Wd.Entry("call", root_a=bytes(63)), Wd.Entry("call", lo=1 << 64)):
        with pytest.raises(ValueError):
            bad.row()
    with pytest.raises(ValueError):
        Wd.Entry.from_words([1 << 16] * 120)
    assert Wd.record_digest({"b": 1, "a": [2]}) == hashlib.sha512(b'{"a":[2],"b":1}').digest()


def test_the_layout_bounds_each_entry():
    ok = Wd.Entry("call", 0, 6, 2, 3, _d(1), _d(2), _d(3))
    assert Wd.layout_problem(ok) is None and Wd.layout_problem(Wd.Entry("none")) is None
    assert Wd.layout_problem(replace(ok, n_a=1 << 13, n_b=1 << 17)) is None
    for bad, why in ((replace(ok, hi=1 << 40), "40-bit"), (Wd.Entry("none", hi=1), "zero row"),
                     (Wd.Entry("none", record=_d(1)), "zero row"), (replace(ok, n_a=0), "side a holds 0"),
                     (replace(ok, n_a=(1 << 13) + 1), "2\\^13"), (replace(ok, n_b=(1 << 17) + 1), "side b"),
                     (replace(ok, kind="filler"), "filler's record")):
        assert re.search(why, Wd.layout_problem(bad)), (bad, why)


def test_tile_at_maps_a_drawn_index_to_its_entry_and_dense_rows():
    """A side tree holds its call's kept rows only, so index u of a range is (u div n_b, u mod n_b)."""
    w = Wd.bucket(20)
    es = _entries(w)
    assert (w.credited, w.n) == (20, 22)
    assert Wd.tile_at(es, 0) == (0, 0, 0) and Wd.tile_at(es, 11) == (0, 2, 3) and Wd.tile_at(es, 12) == (1, 0, 0)
    assert Wd.tile_at(es, 20) == (2, 0, 0) and Wd.tile_at(es, 21) == (2, 0, 1)
    with pytest.raises(ValueError):
        Wd.tile_at(es, 22)
    with pytest.raises(ValueError):
        Wd.tile_in(es[4], 0)


def test_the_window_statement_refuses_each_departure_from_the_layout():
    w = Wd.bucket(20)
    es = _entries(w)
    assert Wd.window_problem(_rows(es), w, 5) is None
    cases = {
        "bound 6": (es, 6),
        "padded count": (es, 5, Wd.bucket(30)),
        "registered level": (_entries(w, real_a=(2, 2), filler=(2, 3)), 5),
        "does not start": (es[:1] + [replace(es[1], lo=13)] + es[2:], 5),
        "or is empty": ([replace(es[0], hi=0)] + es[1:], 5),
        "not its 12 tiles": ([replace(es[0], hi=11)] + es[1:], 5),
        "beyond its 1": (_entries(w, filler=(1, 1)), 5),
        "calls, then one filler": (es[:2] + [es[2], es[2]] + es[4:], 5),
        "calls, then one filler, then": (es[:3] + [es[0]] + es[4:], 5),
        "then one filler": ([replace(es[2], lo=0)] + [Wd.Entry("none")] * 4, 5),
        "not the layout: side b holds 0": (es[:1] + [replace(es[1], n_b=0)] + es[2:], 5),
        "not the layout: an empty entry": (es[:4] + [Wd.Entry("none", n_a=1)], 5),
    }
    for want, (entries, bound, *work) in cases.items():
        got = Wd.window_problem(_rows(entries), work[0] if work else w, bound)
        assert got and want in got, (want, got)
    for bad, why in ((es[0].row()[:-2], "entry 0: not an entry row: an entry row is 240 bytes"),
                     (b"\7" + es[0].row()[1:], "entry 0: not an entry row: kind 7")):
        assert Wd.window_problem([bad] + _rows(es[1:]), w, 5).startswith(why)
    assert {p[0] for p in Wd.PUBLIC} == {"verdict", "work", "check", "sampling", "drawn", "root_W"}
    assert set(Wd.REGISTERED) == {p[0] for p in Wd.PUBLIC if p[2] != "outcome"}


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
    instance-major, the ports in order): t as a u128, the window root, its domain and the two side domains, from the
    registration and the draw alone.  Session σ of ⌈K′/cap⌉ holds drawn[σ·cap ..], then N − 1."""
    reg, drawn = _registration(), [1, 4, 6]
    assert Wd.sessions(3, 2) == 2 and Wd.sessions(4, 2) == 2 and Wd.sessions(1, 4096) == 1
    sides = Wd.side_domain(_d(8), "a").domain_id + Wd.side_domain(_d(8), "b").domain_id
    for sigma, ts in ((0, [1, 4]), (1, [6, 8])):
        assert Wd.instance_ts(reg, drawn, sigma, 2) == ts
        head, body = Wd.public_inputs(reg, drawn, sigma, 2).split(b"\n", 1)
        h = json.loads(head)
        assert head == json.dumps(h, sort_keys=True).encode() and h["format"] == Wd.PUBLIC_INPUTS_FORMAT
        assert h["instances"] == 2 and h["ports"] == list(Wd.PUBLIC_PORTS)
        assert [p["port"] for p in h["ports"]] == ["t", "root", "domain", "domain_a", "domain_b"]
        assert all(p["words"] % 8 == 0 for p in h["ports"])
        per = 2 * sum(p["words"] for p in h["ports"])
        assert per == 272 and body == b"".join(t.to_bytes(16, "little") + _d(7) + _d(8) + sides for t in ts)
    assert Wd.instance_ts(reg, drawn, 0, 4) == [1, 4, 6, 8]
    bad = {"population": (_registration(n=0), drawn, 0, 2), "2^40": (_registration(n=(1 << 40) + 1), drawn, 0, 2),
           "subset": ({**reg, "law": {"law": "bernoulli", "k": 3}}, drawn, 0, 2),
           "distinct": (reg, [4, 1, 6], 0, 2), "distinct ascending": (reg, [1, 1, 6], 0, 2),
           "not 3": (reg, [1, 4], 0, 2), "of [0, 9)": (reg, [1, 4, 9], 0, 2), "session 2": (reg, drawn, 2, 2),
           "roots of port": ({**reg, "roots": reg["roots"][1:]}, drawn, 0, 2),
           "64 bytes": (_registration(root=b"\1" * 32), drawn, 0, 2)}
    for why, args in bad.items():
        with pytest.raises(ValueError, match=re.escape(why)):
            Wd.public_inputs(*args)
    with pytest.raises(ValueError):
        Wd.sessions(3, 0)


def test_the_pinned_vectors_are_the_layout():
    """``vectors/window.json`` is `window_vectors.vectors()`: the toy window's entry rows, a side tree, the root and F_σ.
    The rows are dense (call 0's refused A row 2 is no leaf), and the vectors agree with the window statement."""
    pinned = json.loads(V.OUT.read_text())
    assert pinned == json.loads(json.dumps(V.vectors()))
    rows = [bytes.fromhex(e["row"]) for e in pinned["entries"]]
    work = Wd.Work.at(pinned["window"]["work"]["level"])
    assert Wd.window_problem(rows, work, V.P) is None and work.n == 9
    call = Wd.Entry.from_row(rows[0])
    assert (call.n_a, call.n_b, call.hi) == (3, 2, 6) and pinned["side"]["rows"] == 3 and len(pinned["side"]["path_of_1"]) == 13
    reg = pinned["public_inputs"]["registration"]
    for sigma, f in enumerate(pinned["public_inputs"]["sessions"]):
        assert Wd.public_inputs(reg, pinned["public_inputs"]["drawn"], sigma, V.CAP) == bytes.fromhex(f)
