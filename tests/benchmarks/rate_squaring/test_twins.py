"""`sq.cu`'s FOLD 5, FOLD 3 and FOLD 7 transcribed into Python word by word, with PTX's carry flag, and held to `pow`.

Each twin follows its kernel's carry chains (`mad.lo.cc`, `madc.hi.cc`, `addc`, the funnel-shift doubling) and asserts
every carry the kernel drops is zero, which is where a form's bound fails (FOLD 3's first build carried 2^37 out of
2^512 where it assumed < 570). The device is gated on the node (`sq.sh`: `sq gate` against `sq_gate.py` before any
timing); these catch a carry-bound slip before a GPU is leased.
"""

from __future__ import annotations

import random

import pytest

NL, CF = 16, 569
P = 2**512 - CF
M = 0xFFFFFFFF


class Cc:
    """The PTX carry flag: `.cc` writes it, `c` reads it; a form without `.cc` drops the carry, asserted zero."""

    def __init__(self) -> None:
        self.cf = 0

    def _out(self, v: int, cc: bool) -> int:
        if cc:
            self.cf = v >> 32
        else:
            assert v >> 32 == 0, "a dropped carry"
        return v & M

    def mad_lo(self, a: int, b: int, c: int, *, carry_in: bool = False, cc: bool = True) -> int:
        return self._out((a * b & M) + c + (self.cf if carry_in else 0), cc)

    def mad_hi(self, a: int, b: int, c: int, *, carry_in: bool = False, cc: bool = True) -> int:
        return self._out((a * b >> 32) + c + (self.cf if carry_in else 0), cc)

    def add(self, a: int, b: int, *, carry_in: bool = False, cc: bool = True) -> int:
        return self._out(a + b + (self.cf if carry_in else 0), cc)


def words(x: int, n: int) -> list[int]:
    return [(x >> (32 * k)) & M for k in range(n)]


def value(w: list[int]) -> int:
    return sum(v << (32 * k) for k, v in enumerate(w))


def mac(r: list[int], a: int, b: int) -> None:
    cc = Cc()
    r[0] = cc.mad_lo(a, b, r[0])
    r[1] = cc.mad_hi(a, b, r[1], carry_in=True)
    r[2] = cc.add(r[2], 0, carry_in=True, cc=False)


def add3(r: list[int], a: int, b: int) -> None:
    cc = Cc()
    r[0] = cc.add(r[0], a)
    r[1] = cc.add(r[1], b, carry_in=True)
    r[2] = cc.add(r[2], 0, carry_in=True, cc=False)


def add_small(x: list[int], a: int) -> int:
    x[0] = (x[0] + a) & M
    if x[0] >= a:
        return 0
    cc = Cc()
    x[1] = cc.add(x[1], 1)
    for k in range(2, NL):
        x[k] = cc.add(x[k], 0, carry_in=True)
    return cc.add(0, 0, carry_in=True, cc=False)


def add_small2(x: list[int], lo: int, hi: int) -> int:
    cc = Cc()
    x[0] = cc.add(x[0], lo)
    x[1] = cc.add(x[1], hi, carry_in=True)
    if not cc.add(0, 0, carry_in=True, cc=False):
        return 0
    cc = Cc()
    x[2] = cc.add(x[2], 1)
    for k in range(3, NL):
        x[k] = cc.add(x[k], 0, carry_in=True)
    return cc.add(0, 0, carry_in=True, cc=False)


def sqr_dbl_n(a: list[int]) -> list[int]:
    """FOLD 5's square: the off-diagonal columns chained by their carries, one doubling, the squares in one chain."""
    n = len(a)
    s = [0] * (2 * n)
    c0 = c1 = 0
    for m in range(1, 2 * n - 2):
        r = [c0, c1, 0]
        i = 0 if m < n else m - n + 1
        while 2 * i < m:
            mac(r, a[i], a[m - i])
            i += 1
        s[m], c0, c1 = r
    s[2 * n - 2], s[2 * n - 1] = c0, c1
    assert s[2 * n - 1] >> 31 == 0, "the doubling's funnel shift drops S's top bit"
    d = [0] + [((s[k] << 1) | (s[k - 1] >> 31)) & M for k in range(1, 2 * n)]
    cc = Cc()
    y = [0] * (2 * n)
    y[0] = cc.mad_lo(a[0], a[0], d[0])
    y[1] = cc.mad_hi(a[0], a[0], d[1], carry_in=True)
    for i in range(1, n):
        y[2 * i] = cc.mad_lo(a[i], a[i], d[2 * i], carry_in=True)
        y[2 * i + 1] = cc.mad_hi(a[i], a[i], d[2 * i + 1], carry_in=True)
    assert cc.cf == 0, "the square's chain carries out of 2^(64 n)"
    return y


def fold_ptx(t: list[int]) -> list[int]:
    cc = Cc()
    x = [0] * NL
    x[0] = cc.mad_lo(t[NL], CF, t[0])
    for k in range(1, NL):
        x[k] = cc.mad_lo(t[NL + k], CF, t[k], carry_in=True)
    h = cc.add(0, 0, carry_in=True, cc=False)
    x[1] = cc.mad_hi(t[NL], CF, x[1])
    for k in range(2, NL):
        x[k] = cc.mad_hi(t[NL + k - 1], CF, x[k], carry_in=True)
    h = cc.mad_hi(t[2 * NL - 1], CF, h, carry_in=True, cc=False)
    assert h * CF >> 32 == 0
    if add_small(x, h * CF):
        assert add_small(x, CF) == 0, "the second fold carries out"
    return x


def fold_wide(t: list[int]) -> list[int]:
    x = [0] * NL
    h = 0
    for k in range(NL):
        p = t[NL + k] * CF + t[k]
        cc = Cc()
        x[k] = cc.add(p & M, h)
        h = cc.add(p >> 32, 0, carry_in=True, cc=False)
    assert h * CF >> 32 == 0
    if add_small(x, h * CF):
        assert add_small(x, CF) == 0, "the second fold carries out"
    return x


def sqr_fused(a: list[int]) -> list[int]:
    h = [0] * NL
    c0 = c1 = 0
    for m in range(NL, 2 * NL - 1):
        r = [0, 0, 0]
        i = m - NL + 1
        while 2 * i < m:
            mac(r, a[i], a[m - i])
            i += 1
        assert r[2] >> 31 == 0
        r = [(r[0] << 1) & M, ((r[1] << 1) | (r[0] >> 31)) & M, ((r[2] << 1) | (r[1] >> 31)) & M]
        if m % 2 == 0:
            mac(r, a[m // 2], a[m // 2])
        add3(r, c0, c1)
        h[m - NL], c0, c1 = r
    h[NL - 1] = c0
    assert c1 == 0, "the high columns carry out of 2^1024"
    c0 = c1 = 0
    x = [0] * NL
    for m in range(NL):
        r = [0, 0, 0]
        i = 0
        while 2 * i < m:
            mac(r, a[i], a[m - i])
            i += 1
        assert r[2] >> 31 == 0
        r = [(r[0] << 1) & M, ((r[1] << 1) | (r[0] >> 31)) & M, ((r[2] << 1) | (r[1] >> 31)) & M]
        if m % 2 == 0:
            mac(r, a[m // 2], a[m // 2])
        mac(r, h[m], CF)
        add3(r, c0, c1)
        x[m], c0, c1 = r
    lo, hi = (c0 * CF) & M, ((c0 * CF) >> 32) + c1 * CF
    assert hi >> 32 == 0
    if add_small2(x, lo, hi):
        assert add_small(x, CF) == 0, "the second fold carries out"
    return x


FORMS = {
    "FOLD 5": lambda a: fold_ptx(sqr_dbl_n(a)),
    "FOLD 7": lambda a: fold_wide(sqr_dbl_n(a)),
    "FOLD 3": sqr_fused,
}

EDGES = [0, 1, 2, P - 1, P, P + 1, 2**512 - 1, 2**511, 2**511 - 1, 2**256 - 1, value([M] * 8 + [0] * 8),
         value([0] * 8 + [M] * 8), value([M, 0] * 8), value([0, M] * 8)]


def inputs(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    out = list(EDGES)
    for _ in range(n):
        out.append(rng.getrandbits(512))
        out.append(2**512 - 1 - rng.getrandbits(rng.randrange(1, 64)))
    return out


@pytest.mark.parametrize("form", sorted(FORMS))
def test_square_matches_pow(form: str) -> None:
    for x in inputs(569, 200):
        y = value(FORMS[form](words(x, NL)))
        assert y < 2**512
        assert y % P == x * x % P, (form, hex(x))


@pytest.mark.parametrize("form", sorted(FORMS))
def test_chain_matches_pow(form: str) -> None:
    """The lazy output (< 2^512, not reduced below p) is the next squaring's input, as in `sq.cu`'s chain."""
    rng = random.Random(512)
    for _ in range(4):
        x0 = x = rng.getrandbits(512)
        for _ in range(64):
            x = value(FORMS[form](words(x, NL)))
        assert x % P == pow(x0, 2**64, P)


def test_square_at_half_width() -> None:
    """FOLD 6 squares 8-limb halves with FOLD 5's square."""
    for x in inputs(8, 200):
        h = x % 2**256
        assert value(sqr_dbl_n(words(h, 8))) == h * h
