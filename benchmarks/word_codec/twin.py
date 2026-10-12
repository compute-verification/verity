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


def horner31(acc: int, g: int, w: int) -> int:
    _fits(acc, 32)
    _fits(w, 32)
    assert 0 <= g < 1 << 31
    s = _fits(acc * g + w, 64)
    f = _fits((s & M31) + (s >> 31), 64)
    out = _fits((f & M31) + (f >> 31), 32)
    assert out % M31 == (acc * g + w) % M31
    return out


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


B3_CHUNK_START, B3_CHUNK_END, B3_ROOT, B3_KEYED = 1, 2, 8, 16


def b3_lane_messages(j: int, words: list[int]) -> list[tuple[list[int], int, int]]:
    """`b3_lane_tag`'s compressions of block j (its words, encoded): each one's 16 message words, length and flags."""
    _fits(j, 64)
    msg = 2 + 2 * len(words)
    blocks = (msg + 15) // 16
    bswap = lambda v: int.from_bytes(v.to_bytes(4, "little"), "big")  # noqa: E731
    out = []
    for b in range(blocks):
        m = []
        for i in range(16):
            u = 16 * b + i
            if u == 0:
                m.append(bswap(j >> 32))
            elif u == 1:
                m.append(bswap(j & U32))
            elif u < msg:
                m.append(_fits(words[(u - 2) // 2], 64) >> (32 * (u & 1)) & U32)
            else:
                m.append(0)
        length = 64 if b + 1 < blocks else 4 * (msg - 16 * (blocks - 1))
        flags = B3_KEYED | (B3_CHUNK_START if b == 0 else 0) | (B3_CHUNK_END | B3_ROOT if b + 1 == blocks else 0)
        out.append((m, length, flags))
    return out

