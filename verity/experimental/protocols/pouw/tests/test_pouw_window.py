"""PoUW's audit window (`pouw.window`): the grid a window's tile count is padded to and the q it gives the leakage bound,
L3's K′, the map from a drawn index to a tile, and the window statement on the advice."""
from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest
from verity.experimental.protocols.pouw import window as Wd


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


def _k_naive(eps, delta, credited, n, w):
    q = 1 - eps * credited / (n * w)
    k = 1
    while k < n and q ** k > delta:
        k += 1
    return k


def test_k_prime_is_exact_in_rationals():
    """(1 − p)^K = δ exactly takes the K that reaches it: at level 6 (7 tiles credited, 8 padded) with ε = 1, (1/8)^3 = 2^-9,
    where binary64 logarithms gave 4."""
    w = Wd.Work.at(6)
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
    assert "kind" in Wd.window_problem(es[:4] + [7], w, 5)
    for bad in (True, 0.0, None, "0"):
        assert "range" in Wd.window_problem([{**es[0], "lo": bad}] + es[1:], w, 5)
    assert "ascending" in Wd.window_problem([{**es[0], "a": {"rows": 4, "keep": [True, 2, 3]}}] + es[1:], w, 5)
    assert {p[0] for p in Wd.PUBLIC} == {"verdict", "work", "check", "sampling", "drawn", "root_W"}
    assert set(Wd.REGISTERED) == {p[0] for p in Wd.PUBLIC if p[2] != "outcome"}
