"""PoUW's audit window (`pouw.window`): the grid a window's tile count is padded to and the q it gives the leakage bound,
L3's K′, the map from a drawn index to a tile, and the window statement on the advice."""
from __future__ import annotations

import math
from fractions import Fraction

import pytest
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


def _entries(work, real_a=(3, 2), rows_b=4, filler=(1, 2), bound=5):
    out, at = [], 0
    for a in real_a:
        e = {"kind": "call", "lo": at, "a": {"rows": a + 1, "keep": list(range(1, a + 1))},
             "b": {"rows": rows_b, "keep": list(range(rows_b))}}
        at += a * rows_b
        e["hi"] = at
        out.append(e)
    out.append({"kind": "filler", "lo": at, "hi": work.n, "a": {"rows": filler[0], "keep": list(range(filler[0]))},
                "b": {"rows": filler[1], "keep": list(range(filler[1]))}})
    return out + [{"kind": "none", "index": i} for i in range(len(out), bound)]


def test_tile_at_maps_a_drawn_index_to_its_entry_and_rows():
    w = Wd.bucket(20)
    es = _entries(w)
    assert (w.credited, w.n) == (20, 22)
    assert Wd.tile_at(es, 0) == (0, 1, 0) and Wd.tile_at(es, 11) == (0, 3, 3) and Wd.tile_at(es, 12) == (1, 1, 0)
    assert Wd.tile_at(es, 20) == (2, 0, 0) and Wd.tile_at(es, 21) == (2, 0, 1)
    with pytest.raises(ValueError):
        Wd.tile_at(es, 22)


def test_the_window_statement():
    w = Wd.bucket(20)
    es = _entries(w)
    assert Wd.window_problem(es, w, 5) is None
    assert "bound" in Wd.window_problem(es, w, 6)
    assert "padded count" in Wd.window_problem(es, Wd.bucket(30), 5)
    assert "registered level" in Wd.window_problem(_entries(w, real_a=(2, 2), filler=(2, 3)), w, 5)
    gap = [dict(e) for e in es]
    gap[1] = {**gap[1], "lo": 13}
    assert "start" in Wd.window_problem(gap, w, 5)
    short = [dict(e) for e in es]
    short[0] = {**short[0], "hi": 11}
    assert "range" in Wd.window_problem(short, w, 5)
    small = _entries(w, filler=(1, 1))
    assert "beyond" in Wd.window_problem(small, w, 5)
    two = es[:2] + [es[2], es[2]] + es[4:]
    assert "one filler" in Wd.window_problem(two, w, 5)
    unsorted = [dict(e) for e in es]
    unsorted[0] = {**unsorted[0], "a": {"rows": 4, "keep": [2, 1, 3]}}
    assert "ascending" in Wd.window_problem(unsorted, w, 5)
    assert {p[0] for p in Wd.PUBLIC} == {"verdict", "work", "check", "sampling", "drawn", "root_W"}
    assert set(Wd.REGISTERED) == {p[0] for p in Wd.PUBLIC if p[2] != "outcome"}
