"""Falsification tests of H-1T's two conjectures (`Security/Models/Pouw/Assumptions/Fp8Atom/H1TAssumptions.lean`), the
tests that `shape.conjectures` in `Security/lean-audit.json` names:

* `DistinctLiveH1T`: on a legal unit `U`, `U.word` is injective and no credited word is free (a constant or a raw XOF
  word);
* `HalfApartH1T`: on a legal unit, for each row, two distinct credited words agree on at most half of the row's salts,
  and a credited word takes any one value on at most half of them.

The words are `lean/scripts/h1t_vectors.py`'s `row_words`, the function whose vectors the Lean kernel checks against
`Unit.word` (`H1TChecks`); `test_the_model_is_the_leans_word` replays those vectors. A unit is decided two ways:

* **exhaustively** (`Case.live`): a unit of one slice whose columns are zero outside a few real lanes. A word reads only
  the salt fields `extract` returns, every field value has the same number of raw preimages, and a lane whose column
  codes are all zero adds a zero product, which the atom skips, so its sign bit is read by no word. Enumerating the
  tag fields and the live lanes' signs then counts exactly what the statements count;
* **by sampling** (`Case.salts`): `N` uniform unit salts, independent per row. A word's values on them separate it from
  every other word and from the constants, and a salt that differs only in bit 0 of every raw word (a bit `extract`
  never reads) leaves it unchanged, which no raw XOF word does: where `DistinctLiveH1T` holds, the samples prove it.
  For `HalfApartH1T` a frequency over `N` salts above `1/2 + margin(N)` is a counterexample with probability at least
  `1 − 10⁻⁹` (Hoeffding).

A counterexample fails the test with the unit, the words and the salts."""
from __future__ import annotations

import functools
import importlib.util
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
LEAN = HERE.parents[3] / "Security"
SCRIPT = HERE / "lean" / "scripts" / "h1t_vectors.py"
SEED = 2026101001
N = 256


def margin(n: int) -> float:
    """How far above `1/2` a frequency over `n` uniform salts must be to be a counterexample with probability at least
    `1 − 10⁻⁹` (Hoeffding)."""
    return math.sqrt(math.log(1e9) / (2 * n))


@functools.cache
def model():
    """`h1t_vectors.py`, with its pure forming functions memoized so that a salt sweep doesn't redo their Fractions."""
    spec = importlib.util.spec_from_file_location("h1t_vectors", SCRIPT)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    h.enc = functools.cache(h.enc)
    h.form_real = functools.cache(h.form_real)
    raw_floor, raw_slice = h.floor_f, h.form_slice
    floor_f = functools.cache(raw_floor)
    form_slice = functools.lru_cache(maxsize=1 << 16)(lambda xs, r: raw_slice(xs, r))
    h.floor_f = lambda xs: floor_f(tuple(xs))
    h.form_slice = lambda xs, r: form_slice(tuple(xs), tuple(r))
    return h


@dataclass(frozen=True)
class Unit:
    """`Unit`: `m` rows of `X_q` (`xq`) against `n` columns of B′'s real codes (`b`, column-major) over `k`."""
    name: str
    k: int
    xq: tuple[tuple[int, ...], ...]
    b: tuple[tuple[int, ...], ...]

    @property
    def T(self) -> int:
        return (self.k + 28) // 29

    def legal(self) -> bool:
        finite = all(c & 0x7F != 0x7F for v in self.xq + self.b for c in v)
        return self.k <= 32768 and len(self.b) <= 88412 and finite

    def describe(self) -> str:
        hexes = lambda vs: [bytes(v).hex() for v in vs]
        return f"unit {self.name} (k={self.k}, T={self.T}, m={len(self.xq)}, n={len(self.b)}): xq={hexes(self.xq)}, " \
               f"b={hexes(self.b)}"


def keys(U: Unit) -> list[tuple]:
    """The credited words `(i, j, a)`: `a = ("I", p)` for `p < 2T`, `("R", p)` for `1 ≤ p ≤ 2T − 3` (`Pos`)."""
    pos = [("I", p) for p in range(2 * U.T)] + [("R", p) for p in range(1, 2 * U.T - 2)]
    return [(i, j, a) for i in range(len(U.xq)) for j in range(len(U.b)) for a in pos]


def words(U: Unit, g) -> list[float]:
    """`U.word c g` for every credited word `c`, in `keys` order, as values: the FP32 words of `row_words` read as
    numbers, so `−0` and `+0` are one value, as in ℚ."""
    h, T, out = model(), U.T, []
    pad = lambda v: list(v) + [0] * (29 * T - U.k)
    for i in range(len(U.xq)):
        for j, col in enumerate(U.b):
            I, R, _, _ = h.row_words(T, pad(U.xq[i]), pad(col), j, g[i])
            vi, vr = np.array(I, np.uint32).view(np.float32), np.array(R, np.uint32).view(np.float32)
            out += [float(v) + 0.0 for v in vi] + [float(v) + 0.0 for v in vr[1:2 * T - 2]]
    return out


# ---------------------------------------------------------------------------------------------------------------
# the units
# ---------------------------------------------------------------------------------------------------------------

FINITE = [c for c in range(256) if c & 0x7F != 0x7F]
SUBNORMAL = [c for c in FINITE if (c >> 3) & 0xF == 0 and c & 7]


@dataclass(frozen=True)
class Case:
    """A unit, swept over every value of the salt fields its words read (`live`: the real lanes whose columns aren't all
    zero) or over `salts` sampled unit salts."""
    unit: Unit
    live: tuple[int, ...] | None = None
    salts: int = N


def sampled_units() -> list:
    """Legal units of one to three slices (and one of eight), with columns `j` whose tuples differ in one lane (`j` and
    `j + 1`, `j + 23`), columns that differ only by their tuple, and loud, zero, subnormal and padded rows."""
    rng = random.Random(SEED)
    codes = lambda n, pool=FINITE: tuple(rng.choice(pool) for _ in range(n))
    out = []

    def unit(name, k, rows, cols, slow=False, salts=N):
        case = Case(Unit(name, k, tuple(rows), tuple(cols)), salts=salts)
        out.append(pytest.param(case, id=name, marks=[pytest.mark.slow] if slow else []))

    for T in (1, 2, 3):
        unit(f"random-T{T}", 29 * T, [codes(29 * T) for _ in range(2)], [codes(29 * T) for _ in range(24)])
    same = codes(58)
    unit("tuple-only-T2", 58, [codes(58), codes(58)], [same] * 24)
    unit("zero-cols-T3", 87, [codes(87)], [(0,) * 87] * 24)
    unit("padded-T3", 61, [codes(61)], [codes(61) for _ in range(24)])
    unit("loud-T2", 58, [codes(58, [0x7E, 0xFE])], [(0x7E,) * 58] * 12 + [codes(58, [0x7E, 0xFE]) for _ in range(12)])
    unit("zero-row-T2", 58, [(0,) * 58], [codes(58) for _ in range(24)])
    unit("subnormal-T2", 58, [codes(58, SUBNORMAL)], [codes(58) for _ in range(24)])
    unit("k1", 1, [codes(1), (0,)], [codes(1) for _ in range(24)])
    unit("random-T8", 232, [codes(232)], [codes(232) for _ in range(24)], slow=True)
    unit("tuple-only-T1-n1430", 29, [codes(29)], [codes(29)] * 1430, slow=True, salts=64)
    return out


def exact_units() -> list:
    """Legal one-slice units whose columns are zero outside the lanes `live`, small enough to enumerate every salt
    field a word reads."""
    rng = random.Random(SEED + 1)
    out = []

    def unit(name, rows, cols, live, slow=False):
        cols = [tuple(c if l in live else 0 for l, c in enumerate(col)) for col in cols]
        case = Case(Unit(name, 29, tuple(rows), tuple(cols)), live=tuple(live))
        out.append(pytest.param(case, id=name, marks=[pytest.mark.slow] if slow else []))

    rand = lambda pool=FINITE: tuple(rng.choice(pool) for _ in range(29))
    unit("tags-n2", [rand(), (0,) * 29], [(0,) * 29] * 2, [])
    unit("tags-n24", [rand()], [(0,) * 29] * 24, [], slow=True)
    unit("live2-n4", [rand(), rand([0x7E, 0xFE])], [rand() for _ in range(4)], [3, 17], slow=True)
    return out


CASES = sampled_units() + exact_units()


# ---------------------------------------------------------------------------------------------------------------
# the sweeps
# ---------------------------------------------------------------------------------------------------------------

def raw_of(pos: dict[int, bool], neg: tuple[bool, ...], man: tuple[int, ...]) -> list[int]:
    """18 raw XOF words whose fields (`extract`, `fields`) are the given signs and mantissas, every other bit zero."""
    r = [0] * 18
    for lane, bit in pos.items():
        r[lane // 2] |= int(bit) << (15 if lane % 2 == 0 else 31)
    r[17] |= int(neg[0]) << 31 | man[0] << 23
    r[16] |= int(neg[1]) << 15 | man[1] << 7 | int(neg[2]) << 31 | man[2] << 23
    return r


def flip0(g):
    return [[[w ^ 1 for w in r] for r in row] for row in g]


@dataclass
class Sweep:
    unit: Unit
    keys: list[tuple]
    values: np.ndarray        # salts × words
    salts: list               # each a unit salt: rows × slices × 18 raw words
    exact: bool               # every row read the same salt, over every field value it reads
    unread: list[tuple]       # words that a flip of bit 0 of every raw word moved


@functools.cache
def sampled(U: Unit, n: int) -> Sweep:
    rng = random.Random(f"{SEED}:{U.name}")
    salts = [[[[rng.getrandbits(32) for _ in range(18)] for _ in range(U.T)] for _ in U.xq] for _ in range(n)]
    values = np.array([words(U, g) for g in salts])
    ks = keys(U)
    unread = [ks[c] for c, (a, b) in enumerate(zip(values[0], words(U, flip0(salts[0])))) if a != b]
    return Sweep(U, ks, values, salts, False, unread)


@functools.cache
def exhaustive(U: Unit, live: tuple[int, ...]) -> Sweep:
    h = model()
    assert U.T == 1 and set(live) <= set(range(29))
    assert all(col[l] == 0 for col in U.b for l in range(29) if l not in live)
    salts = []
    for signs in range(1 << len(live)):
        pos = {l: bool(signs >> n & 1) for n, l in enumerate(live)}
        for tags in range(1 << 12):
            neg = tuple(bool(tags >> (4 * t) & 1) for t in range(3))
            man = tuple(tags >> (4 * t + 1) & 7 for t in range(3))
            r = raw_of(pos, neg, man)
            p, n_, m_ = h.fields(r)
            assert (n_, m_) == (list(neg), list(man)) and all(p[l] == b for l, b in pos.items())
            salts.append([[r]] * len(U.xq))
    values = np.array([words(U, g) for g in salts])
    ks = keys(U)
    unread = [ks[c] for c, (a, b) in enumerate(zip(values[-1], words(U, flip0(salts[-1])))) if a != b]
    return Sweep(U, ks, values, salts, True, unread)


def salt_hex(g) -> str:
    """A unit salt as hex, one little-endian string of 18 raw words per (row, slice)."""
    return str([[np.array(r, "<u4").tobytes().hex() for r in row] for row in g])


def distinct_live_failures(s: Sweep) -> list[str]:
    """`DistinctLiveH1T` on the sweep's unit: every word is non-constant, no raw XOF word, and the words' value
    vectors are pairwise distinct. On an exhaustive sweep every row read the same salt, so only one row's words are
    compared; two words of different rows read different coordinates, so they are equal only if both are constant."""
    out = [f"{c} moved when only bit 0 of every raw word did, so it may be a raw XOF word" for c in s.unread]
    for c, col in zip(s.keys, s.values.T):
        if (col == col[0]).all():
            how = "every salt field value it reads" if s.exact else f"all {len(col)} sampled salts"
            out.append(f"{c} is {col[0]} on {how}: a constant, which is free")
    seen: dict[tuple, tuple] = {}
    for c, col in zip(s.keys, s.values.T):
        key = (c[0] if s.exact else None, col.tobytes())
        if key in seen:
            how = "every salt field value they read" if s.exact else f"all {len(col)} sampled salts"
            out.append(f"{seen[key]} and {c} agree on {how}, e.g. {col[0]} at salt {salt_hex(s.salts[0])}")
        seen.setdefault(key, c)
    return out


def half_apart_failures(s: Sweep) -> list[str]:
    """`HalfApartH1T` on the sweep's unit, row by row: on an exhaustive sweep the counts are exact
    (`2·count ≤ total`); on a sampled one a frequency above `1/2 + margin` fails."""
    total = len(s.salts)
    limit = total / 2 if s.exact else total * (0.5 + margin(total))
    out = []
    for i in range(len(s.unit.xq)):
        cols = [c for c, k in enumerate(s.keys) if k[0] == i]
        vals = s.values[:, cols]
        for c in range(len(cols)):
            v, n = np.unique(vals[:, c], return_counts=True)
            if n.max() > limit:
                t = int(np.argmax(vals[:, c] == v[n.argmax()]))
                out.append(f"{s.keys[cols[c]]} is {v[n.argmax()]} on {n.max()} of {total} salts (limit {limit:g}), "
                           f"e.g. salt {salt_hex([s.salts[t][i]])}")
        agree: dict[tuple[int, int], int] = {}
        for row in vals:
            v, inv, n = np.unique(row, return_inverse=True, return_counts=True)
            for g in np.flatnonzero(n > 1):
                at = np.flatnonzero(inv == g)
                for x in range(len(at)):
                    for y in range(x + 1, len(at)):
                        agree[at[x], at[y]] = agree.get((at[x], at[y]), 0) + 1
        for (a, b), n in sorted(agree.items()):
            if n > limit:
                t = int(np.argmax(vals[:, a] == vals[:, b]))
                out.append(f"{s.keys[cols[a]]} and {s.keys[cols[b]]} agree on {n} of {total} salts (limit {limit:g}), "
                           f"e.g. salt {salt_hex([s.salts[t][i]])}")
    return out


def report(s: Sweep, fails: list[str], what: str) -> str:
    return f"{what} is falsified on {s.unit.describe()}:\n" + "\n".join(fails[:20])


# ---------------------------------------------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------------------------------------------

ROW = re.compile(r"⟨(\d+), (\d+), (0x[0-9A-F]+), (0x[0-9A-F]+), \[([^\]]*)\], \[([^\]]*)\], \[([^\]]*)\]⟩")


def test_the_model_is_the_leans_word() -> None:
    """`row_words` gives every I and R word of the Lean's `rowVectors` (`H1TVectors.lean`), which the kernel checks
    against `freshWord` and `rnAdd` (`H1TChecks`): the words these tests sweep are `Unit.word`'s."""
    text = (LEAN / "Definitions/Pouw/Fp8Atom/H1TVectors.lean").read_text()
    rows = ROW.findall(text)
    assert len(rows) >= 30
    nums = lambda s: [int(w, 16) for w in s.split(", ") if w]
    unpack = lambda n, bits, count: [(n >> (bits * i)) & ((1 << bits) - 1) for i in range(count)]
    for T, j, x, b, raws, I, R in rows:
        T, j = int(T), int(j)
        xs, bs = unpack(int(x, 16), 8, 29 * T), unpack(int(b, 16), 8, 29 * T)
        got_i, got_r, _, _ = model().row_words(T, xs, bs, j, [unpack(r, 32, 18) for r in nums(raws)])
        assert (got_i, got_r) == (nums(I), nums(R)), (T, j)


def sweep(case: Case) -> Sweep:
    assert case.unit.legal(), case.unit.describe()
    return sampled(case.unit, case.salts) if case.live is None else exhaustive(case.unit, case.live)


def test_a_counterexample_fails(monkeypatch) -> None:
    """With one tag tuple for every column, two columns with the same codes have the same words, and both checks name
    them."""
    monkeypatch.setattr(model(), "tuple_codes", lambda j: [0x68, 0x60, 0x60])
    rng = random.Random(SEED + 2)
    col = tuple(rng.choice(FINITE) for _ in range(29))
    s = sampled(Unit("one-tuple", 29, (tuple(rng.choice(FINITE) for _ in range(29)),), (col, col)), 64)
    same = "(0, 0, ('I', 0)) and (0, 1, ('I', 0)) agree on"
    assert any(f.startswith(same + " all 64 sampled salts") for f in distinct_live_failures(s))
    assert any(f.startswith(same + " 64 of 64 salts") for f in half_apart_failures(s))


@pytest.mark.parametrize("case", CASES)
def test_distinct_live_h1t(case: Case) -> None:
    """`DistinctLiveH1T`'s falsification test: on each unit, exhaustively or on sampled salts."""
    s = sweep(case)
    assert not (fails := distinct_live_failures(s)), report(s, fails, "DistinctLiveH1T")


@pytest.mark.parametrize("case", CASES)
def test_half_apart_h1t(case: Case) -> None:
    """`HalfApartH1T`'s falsification test: on each unit, counted exactly or on sampled salts."""
    s = sweep(case)
    assert not (fails := half_apart_failures(s)), report(s, fails, "HalfApartH1T")
