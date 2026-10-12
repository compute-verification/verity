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
    if x[-1] != U32:
        assert from_limbs(x) < f.p
        return list(x)
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
    if s[0] > (1 << 32) - f.c:
        y, br = _sub([v ^ U32 for v in s], [f.c - 1] + [0] * (f.limbs - 1), 0)
        assert br == 0
        return y
    return [(1 << 32) - f.c - s[0]] + [v ^ U32 for v in s[1:]]


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


PRIMS = ("add_mod", "sub_mod", "neg_if", "canon", "sqr")  # kv.cu's kv_prim ops, in order


def rare_cases(f: Field, rng, searches: int = 300_000) -> dict[int, list[tuple[int, int]]]:
    """Inputs (a, b) for each of PRIMS that take its rare branch (a carry or borrow past limb 0, or canon's
    subtraction of p), beside edges and random words. neg_if negates a when b is odd; the square's rare fold is
    searched for, at most `searches` tries."""
    p, c, n = f.p, f.c, f.n
    top = (1 << n) - 1

    def low(lo_min: int, lo_max: int, bound: int) -> int:
        return (rng.randrange(bound >> 32) << 32) | rng.randrange(lo_min, lo_max + 1)

    add = [(p - 1, (1 << 32) + c), (p - 1, p - 1), (p - 1, 1), (p - 2, 1), (0, 0), (1, p - 1)]
    for _ in range(8):
        x = p - 1 - rng.randrange(1 << 64)
        w = low((1 << 32) - c, U32, 1 << (n - 72))
        add.append((x, w + (1 << n) - x))
    sub = [(0, (1 << n) - (1 << 32)), (0, p - 1), (5, 3), (3, 5), (0, 1), (p - 1, 0)]
    for _ in range(8):
        x = rng.randrange(1 << 64)
        r = low(0, c - 1, 1 << n) | (1 << (n - 1))
        sub.append((x, x - r + (1 << n)))
    neg_ = [(s, 1) for s in (U32, (1 << 32) - c + 1, (1 << 32) - c, 0, 1, p - 1, (1 << 32) + U32)]
    neg_ += [(low((1 << 32) - c + 1, U32, p), 1) for _ in range(8)] + [(s, 0) for s in (U32, p - 1, 0)]
    can = [(v, 0) for v in (p - 1, p, p + 1, top, top - 1, (1 << n) - (1 << 32), 0)]
    sq = [(v, 0) for v in (top, top - 1, p, p - 1, 0, 1, 1 << (n - 1))]
    for _ in range(searches):
        if len(sq) == 15:
            break
        a = rng.randrange(1 << n)
        s1 = (a * a & top) + c * (a * a >> n)
        if (s1 & U32) + (s1 >> n) * c > U32:
            sq.append((a, 0))
    for _ in range(8):
        add.append((rng.randrange(p), rng.randrange(p)))
        sub.append((rng.randrange(p), rng.randrange(p)))
        neg_.append((rng.randrange(p), rng.randrange(2)))
        can.append((rng.randrange(1 << n), 0))
        sq.append((rng.randrange(1 << n), 0))
    for a, b in add + sub:
        assert 0 <= a < p and 0 <= b < p, (a, b)
    return {0: add, 1: sub, 2: neg_, 3: can, 4: sq}


def prim_want(f: Field, op: int, a: int, b: int) -> int:
    """kv_prim's result: the field's arithmetic, neg_if taking 0 to p, and sqr the twin's lazy square."""
    if op == 0:
        return (a + b) % f.p
    if op == 1:
        return (a - b) % f.p
    if op == 2:
        return f.p - a if b & 1 else a
    if op == 3:
        return a % f.p
    return from_limbs(sqr(to_limbs(a, f), f))


def prim_twin(f: Field, op: int, a: int, b: int) -> int:
    la, lb = to_limbs(a, f), to_limbs(b, f)
    if op == 0:
        return from_limbs(add_mod(la, lb, f))
    if op == 1:
        return from_limbs(sub_mod(la, lb, f))
    if op == 2:
        return from_limbs(neg(la, f)) if b & 1 else a
    if op == 3:
        return from_limbs(canon(la, f))
    return from_limbs(sqr(la, f))
