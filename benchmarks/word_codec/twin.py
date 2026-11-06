"""`wc.cu`'s arithmetic step for step on Python integers, every intermediate checked to fit the register it lives in.

The tests run the twin against `codec.py`: what holds here for every input in a lazy form's range holds on the device,
since each step below is one of the kernel's 32- or 64-bit operations.
"""

from __future__ import annotations

from codec import P

U32 = (1 << 32) - 1
U64 = (1 << 64) - 1
M31 = (1 << 31) - 1
LAZY = (1 << 61) + 8       # `sqr`'s and `canon`'s inputs are below this


def _fits(v: int, bits: int) -> int:
    assert 0 <= v < 1 << bits, (v, bits)
    return v


def fold(z: int) -> int:
    _fits(z, 64)
    zl, zh = z & U32, z >> 32
    return _fits(((zh & 0x1FFFFFFF) << 32 | zl) + (zh >> 29), 64)


def sqr(a: int, form: int = 0) -> int:
    assert 0 <= a < LAZY
    if form == 1:
        lo, hi = a * a & U64, a * a >> 64
        s = (lo & P) + (_fits(hi << 3, 64) | lo >> 61)
        assert s < 1 << 62
        out = fold(s)
    else:
        lo, hi = a & U32, a >> 32
        a0 = lo & M31
        a1 = (hi << 32 | lo) >> 31 & U32
        s = _fits((a1 << 1 & U32) * a1, 64)
        s = _fits(s + a0 * a0, 64)
        m = _fits(a0 * a1, 64)
        mlo, mhi = m & U32, m >> 32
        f1 = (mhi << 32 | mlo) >> 29 & U32
        s = _fits(s + ((mlo & 0x1FFFFFFF) << 32 | f1), 64)
        out = fold(s)
    assert out < (1 << 61) + 4 and out % P == a * a % P
    return out


def canon(r: int) -> int:
    assert 0 <= r < LAZY
    r1 = r + 1
    out = r1 & P if r1 >> 61 else r
    assert out < P and out % P == r % P
    return out


def dec_round(y: int, nt: int, c: int, form: int = 0) -> int:
    assert 0 <= y < P
    ka, kb = 2 * P - c + 1, 6 * P - c + 1
    v = sqr(y, form)
    z = (kb - v + nt) & U64 if y & 1 else (v + ka + nt) & U64
    return canon(fold(z))


def decode_word(y: int, t: int, c: tuple[int, ...], form: int = 0) -> int:
    nt = ~t & U64
    for cj in reversed(c):
        y = dec_round(y, nt, cj, form)
    return y


def root(a: int, form: int = 0) -> int:
    s = a
    for _ in range(59):
        s = sqr(s, form)
    s = canon(s)
    square = canon(sqr(s, form)) == a
    ns = P - s if s else 0
    return s if (s & 1) != square else ns


def encode_word(x: int, t: int, c: tuple[int, ...], form: int = 0) -> int:
    for cj in c:
        x = root(canon(fold(x + cj + t)), form)
    return x


def take4(payloads: list[int], bit: int, pb: int = 56) -> int:
    """Stream bits `bit` … `bit` + 31 of `pb`-bit payloads."""
    w, sh = divmod(bit, pb)
    v = payloads[w] >> sh & U32
    if sh + 32 > pb:
        v |= payloads[w + 1] << (pb - sh) & U32
    return v


def lane_registers(payloads: list[int], pb: int = 56) -> list[int]:
    """One lane's 64 A-fragment registers (16 tiles × 4) from its decoded payloads, as the fused kernel forms them."""
    return [take4(payloads, 128 * tt + 32 * r, pb) for tt in range(16) for r in range(4)]
