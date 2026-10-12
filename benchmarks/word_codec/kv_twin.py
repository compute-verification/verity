"""`kv.cu`'s arithmetic step for step on Python limbs, every intermediate checked to fit the register it lives in.

The tests run the twin against `kv.py` at every field: what holds here for every input in a lazy form's range holds on
the device, since each step below is one of the kernel's 32- or 64-bit operations.
"""

from __future__ import annotations

from kv import Field, from_limbs, to_limbs

U32 = (1 << 32) - 1


def _fits(v: int, bits: int) -> int:
    assert 0 <= v < 1 << bits, (v, bits)
    return v


def _lazy(x: list[int], f: Field) -> list[int]:
    assert len(x) == f.limbs and all(0 <= v <= U32 for v in x)
    return x


def sqr_full(a: list[int], f: Field) -> list[int]:
    nl = f.limbs
    t = [0] * (2 * nl)
    c0 = c1 = 0
    for m in range(2 * nl - 1):
        r = 0
        for i in range(0 if m < nl else m - nl + 1, (m + 1) // 2):
            r = _fits(r + a[i] * a[m - i], 96)
        r = _fits(r << 1, 96)
        if m % 2 == 0:
            r = _fits(r + a[m // 2] * a[m // 2], 96)
        r = _fits(r + c0 + (c1 << 32), 96)
        t[m], c0, c1 = r & U32, (r >> 32) & U32, r >> 64
    assert c1 == 0
    t[2 * nl - 1] = c0
    return t


def mul_full(a: list[int], b: list[int], f: Field) -> list[int]:
    nl = f.limbs
    t = [0] * (2 * nl)
    c0 = c1 = 0
    for m in range(2 * nl - 1):
        r = 0
        for i in range(0 if m < nl else m - nl + 1, (m if m < nl else nl - 1) + 1):
            r = _fits(r + a[i] * b[m - i], 96)
        r = _fits(r + c0 + (c1 << 32), 96)
        t[m], c0, c1 = r & U32, (r >> 32) & U32, r >> 64
    assert c1 == 0
    t[2 * nl - 1] = c0
    return t


def fold(t: list[int], f: Field) -> list[int]:
    nl, c = f.limbs, f.c
    x = [0] * nl
    v = 0
    for k in range(nl):
        v = _fits(v + t[nl + k] * c + t[k], 64)
        x[k], v = v & U32, v >> 32
    assert v <= c
    v = _fits(v * c, 64)
    for k in range(nl):
        v = _fits(v + x[k], 64)
        x[k], v = v & U32, v >> 32
    if v:
        assert from_limbs(x) < c * c
        z = c
        for k in range(nl):
            z = _fits(z + x[k], 64)
            x[k], z = z & U32, z >> 32
        assert z == 0
    return x


def sqr(a: list[int], f: Field) -> list[int]:
    return fold(sqr_full(_lazy(a, f), f), f)


def mul(a: list[int], b: list[int], f: Field) -> list[int]:
    return fold(mul_full(_lazy(a, f), _lazy(b, f), f), f)


def canon(x: list[int], f: Field) -> list[int]:
    z = [0] * f.limbs
    v = f.c
    for k in range(f.limbs):
        v = _fits(v + x[k], 64)
        z[k], v = v & U32, v >> 32
    return z if v else list(x)


def add_mod(x: list[int], b: list[int], f: Field) -> list[int]:
    assert from_limbs(x) < f.p and from_limbs(b) < f.p
    y = [0] * f.limbs
    v = 0
    for k in range(f.limbs):
        v = _fits(v + x[k] + b[k], 64)
        y[k], v = v & U32, v >> 32
    if v:
        z = f.c
        for k in range(f.limbs):
            z = _fits(z + y[k], 64)
            y[k], z = z & U32, z >> 32
        assert z == 0
    return canon(y, f)


def _sub(x: list[int], b: list[int], br: int) -> tuple[list[int], int]:
    y = [0] * len(x)
    for k in range(len(x)):
        d = (x[k] - b[k] - br) % (1 << 64)
        y[k], br = d & U32, d >> 63
    return y, br


def sub_mod(x: list[int], b: list[int], f: Field) -> list[int]:
    assert from_limbs(x) < f.p and from_limbs(b) < f.p
    y, br = _sub(x, b, 0)
    if br:
        y, bc = _sub(y, [f.c] + [0] * (f.limbs - 1), 0)
        assert bc == 0
    return y


def neg(s: list[int], f: Field) -> list[int]:
    assert from_limbs(s) <= f.p
    y, br = _sub([v ^ U32 for v in s], [f.c - 1] + [0] * (f.limbs - 1), 0)
    assert br == 0
    return y


def root(a: list[int], f: Field) -> tuple[list[int], int, int]:
    """a^((p+1)/4) lazily, with the squarings and multiplies it took."""
    m, low = f.chain
    x = list(a)
    cur, sq, mu = 1, 0, 0
    for b in range(m.bit_length() - 2, -1, -1):
        s = list(x)
        for _ in range(cur):
            x = sqr(x, f)
        x = mul(x, s, f)
        sq, mu, cur = sq + cur, mu + 1, cur * 2
        if (m >> b) & 1:
            x = mul(sqr(x, f), a, f)
            sq, mu, cur = sq + 1, mu + 1, cur + 1
    assert cur == m
    for b in range(f.low_bits - 1, -1, -1):
        x = sqr(x, f)
        sq += 1
        if (low >> b) & 1:
            x = mul(x, a, f)
            mu += 1
    return x, sq, mu


def rho(x: list[int], f: Field) -> list[int]:
    s = canon(root(x, f)[0], f)
    q = canon(sqr(s, f), f)
    square = q == x
    if (s[0] & 1) != (0 if square else 1):
        s = neg(s, f)
    return s


def rho_inv(y: list[int], f: Field) -> list[int]:
    q = canon(sqr(y, f), f)
    return neg(q, f) if y[0] & 1 else q


def tweak(i: int, tau: list[int], f: Field) -> list[int]:
    nl, c = f.limbs, f.c
    ilo, ihi = i & U32, i >> 32
    _fits(i, 64)
    r = [0] * (nl + 2)
    v = 0
    for k in range(nl):
        v = _fits(v + tau[k] * ilo, 64)
        r[k], v = v & U32, v >> 32
    r[nl] = v
    v = 0
    for k in range(nl):
        v = _fits(v + tau[k] * ihi + r[k + 1], 64)
        r[k + 1], v = v & U32, v >> 32
    r[nl + 1] = v
    t = [0] * nl
    v = _fits(r[nl] * c + r[0], 64)
    t[0], v = v & U32, v >> 32
    v = _fits(v + r[nl + 1] * c + r[1], 64)
    t[1], v = v & U32, v >> 32
    for k in range(2, nl):
        v = _fits(v + r[k], 64)
        t[k], v = v & U32, v >> 32
    if v:
        z = c
        for k in range(nl):
            z = _fits(z + t[k], 64)
            t[k], z = z & U32, z >> 32
        assert z == 0
    return canon(t, f)


def encode_word(m: int, i: int, c: tuple[int, ...], tau: int, f: Field) -> int:
    x = to_limbs(m, f)
    t = tweak(i, to_limbs(tau, f), f)
    for cj in c:
        x = rho(add_mod(add_mod(x, to_limbs(cj, f), f), t, f), f)
    return from_limbs(x)


def decode_word(y: int, i: int, c: tuple[int, ...], tau: int, f: Field) -> int:
    x = to_limbs(y, f)
    t = tweak(i, to_limbs(tau, f), f)
    for cj in reversed(c):
        x = sub_mod(sub_mod(rho_inv(x, f), to_limbs(cj, f), f), t, f)
    return from_limbs(x)
