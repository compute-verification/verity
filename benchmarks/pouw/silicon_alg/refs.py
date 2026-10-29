"""Exact reference models and test vectors for the SI-ALG MACs in rtl/.

Each design is a dict entry: its Verilog module, its MAC lanes (MACs a step), its accumulator width, a test-vector
generator and an exact reference written independently of the RTL (FP32 rounding by exact integer arithmetic, fields by
Python's `%`).
"""

from __future__ import annotations

import random

M31 = (1 << 31) - 1
BB = 15 * (1 << 27) + 1
M61 = (1 << 61) - 1
GL = (1 << 64) - (1 << 32) + 1
NAN32 = 0x7FC00000


def s8(x: int) -> int:
    return x - 256 if x & 0x80 else x


def dot4(v: dict, i: int) -> int:
    return sum(s8(v[f"a{k}"][i]) * s8(v[f"b{k}"][i]) for k in range(4))


# ---- FP8 E4M3 and FP32, exactly ----


def e4m3(x: int) -> tuple[int, int, int, bool]:
    """(sign, M, E, nan) with value (-1)^sign * M * 2^E."""
    e, m = (x >> 3) & 15, x & 7
    nan = (x & 0x7F) == 0x7F
    big_m = m | (8 if e else 0)
    return x >> 7, big_m, max(e, 1) - 10, nan


def e2m1(x: int) -> tuple[int, int, int]:
    e, m = (x >> 1) & 3, x & 1
    return (x >> 3) & 1, (m if e == 0 else 2 + m), (0 if e == 0 else e - 1) - 1


def f32_round(sign: int, num: int, exp2: int) -> int:
    """RNE of (-1)^sign * num * 2^exp2 (num > 0) to FP32 bits, subnormals, overflow to infinity."""
    lead = num.bit_length() - 1 + exp2
    e = max(lead - 23, -149)
    k = e - exp2
    if k <= 0:
        q = num << -k
    else:
        q0, rem, half = num >> k, num & ((1 << k) - 1), 1 << (k - 1)
        q = q0 + (1 if rem > half or (rem == half and q0 & 1) else 0)
    if q == 1 << 24:
        q, e = 1 << 23, e + 1
    if q >= 1 << 23:
        biased = e + 150
        if biased >= 255:
            return (sign << 31) | 0x7F800000
        return (sign << 31) | (biased << 23) | (q - (1 << 23))
    return (sign << 31) | q


def fp8_fp32(a: int, b: int, acc: int) -> int:
    sa, ma, ea, na = e4m3(a)
    sb, mb, eb, nb = e4m3(b)
    ec, fc = (acc >> 23) & 255, acc & 0x7FFFFF
    if na or nb or (ec == 255 and fc):
        return NAN32
    if ec == 255:
        return acc
    sp, mp, ep = sa ^ sb, ma * mb, ea + eb
    sc, mc, e_c = acc >> 31, fc | ((1 << 23) if ec else 0), max(ec, 1) - 150
    e0 = min(ep, e_c)
    total = (-1) ** sp * (mp << (ep - e0)) + (-1) ** sc * (mc << (e_c - e0))
    if total == 0:
        return (sc << 31) if sp == sc else 0
    return f32_round(1 if total < 0 else 0, abs(total), e0)


def e4m3_fx(a: int, b: int) -> int:
    """Exact E4M3 product as an integer at LSB 2^-18."""
    sa, ma, ea, _ = e4m3(a)
    sb, mb, eb, _ = e4m3(b)
    return (-1) ** (sa ^ sb) * ((ma * mb) << (ea + eb + 18))


def is_nan8(x: int) -> bool:
    return (x & 0x7F) == 0x7F


# ---- binary fields ----


def clmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def gf_reduce(x: int, n: int, low: int) -> int:
    f = (1 << n) | low
    for i in range(x.bit_length() - 1, n - 1, -1):
        if x >> i & 1:
            x ^= f << (i - n)
    return x


# ---- test vectors ----


def rbits(rng: random.Random, n: int, w: int) -> list[int]:
    return [rng.getrandbits(w) for _ in range(n)]


def rmod(rng: random.Random, n: int, p: int) -> list[int]:
    edge = [0, 1, p - 1, p - 2, p // 2, (p + 1) // 2]
    return [edge[i] if i < len(edge) else rng.randrange(p) for i in range(n)]


def s8_vec(rng: random.Random, n: int) -> list[int]:
    edge = [0x80, 0x7F, 0x81, 0xFF, 0x00, 0x01]
    return [rng.choice(edge) if rng.random() < 0.2 else rng.getrandbits(8) for _ in range(n)]


def f32_acc_vec(rng: random.Random, a: list[int], b: list[int]) -> list[int]:
    """FP32 accumulators that exercise cancellation, ties, specials and subnormals."""
    special = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0x00000001, 0x807FFFFF,
               0x7F7FFFFF, 0xFF7FFFFF, 0x00800000]
    out = []
    for x, y in zip(a, b):
        r = rng.random()
        if r < 0.15:
            out.append(rng.getrandbits(32))
        elif r < 0.55:
            out.append((rng.getrandbits(1) << 31) | (rng.randrange(80, 160) << 23) | rng.getrandbits(23))
        elif r < 0.90:
            p = fp8_fp32(x, y, 0)
            if (p >> 23) & 255 in (0, 255):
                out.append(rng.getrandbits(32))
                continue
            neg = p ^ 0x80000000
            shift = rng.randrange(0, 30)
            jitter = rng.randrange(-3, 4)
            # near -p (cancellation), or p's magnitude scaled so the product lands on a tie or a sticky bit
            if rng.random() < 0.5:
                out.append((neg + jitter) & 0xFFFFFFFF)
            else:
                e = min(254, ((p >> 23) & 255) + shift)
                out.append((p & 0x807FFFFF) | (e << 23) ^ (rng.getrandbits(1) << 31))
        else:
            out.append(rng.choice(special))
    return out


def gen_s8x4(rng, n, accw, accgen=None):
    v = {f"a{k}": s8_vec(rng, n) for k in range(4)}
    v.update({f"b{k}": s8_vec(rng, n) for k in range(4)})
    v["acc"] = accgen(rng, n) if accgen else rbits(rng, n, accw)
    return v


def gen_fp8(rng, n, lanes, accgen):
    v = {}
    names = ["a", "b"] if lanes == 1 else [f"{c}{k}" for c in "ab" for k in range(lanes)]
    for nm in names:
        v[nm] = rbits(rng, n, 8)
    if lanes == 1:
        # distinct (a, b) pairs drawn from all 65,536 (every pair once at n >= 65,536), then random
        pairs = rng.sample(range(65536), min(n, 65536))
        for i, ab in enumerate(pairs):
            v["a"][i], v["b"][i] = ab >> 8, ab & 255
        v["acc"] = accgen(rng, v["a"], v["b"])
    else:
        v["acc"] = accgen(rng, n)
    return v


def designs() -> dict:
    d = {}
    d["mac_s8x4_s32"] = dict(
        lanes=4, accw=32, role="baseline A",
        gen=lambda rng, n: gen_s8x4(rng, n, 32),
        ref=lambda v, i: {"y": (v["acc"][i] + dot4(v, i)) % (1 << 32)})
    d["mac_fp8_fp32"] = dict(
        lanes=1, accw=32, role="baseline B",
        gen=lambda rng, n: gen_fp8(rng, n, 1, f32_acc_vec),
        ref=lambda v, i: {"y": fp8_fp32(v["a"][i], v["b"][i], v["acc"][i])})
    for w in (48, 64):
        d[f"mac_s8x4_s{w}"] = dict(
            lanes=4, accw=w, role="wide integer",
            gen=lambda rng, n, w=w: gen_s8x4(rng, n, w),
            ref=lambda v, i, w=w: {"y": (v["acc"][i] + dot4(v, i)) % (1 << w)})
    for name, p, w in (("m31", M31, 31), ("bb", BB, 31), ("m61", M61, 61), ("gl", GL, 64)):
        d[f"mac_s8x4_{name}"] = dict(
            lanes=4, accw=w, role="s8 dot, field accumulate",
            gen=lambda rng, n, p=p, w=w: gen_s8x4(rng, n, w, lambda r, m, p=p: rmod(r, m, p)),
            ref=lambda v, i, p=p: {"y": (v["acc"][i] + dot4(v, i)) % p})
        rinv = pow(1 << 32, -1, p) if name == "bb" else 1
        d[f"mac_{name}_full"] = dict(
            lanes=1, accw=w, role="full field MAC",
            gen=lambda rng, n, p=p: {k: rmod(rng, n, p) if k != "b" else rmod(rng, n, p)[::-1]
                                     for k in ("a", "b", "acc")},
            ref=lambda v, i, p=p, rinv=rinv: {"y": (v["acc"][i] + v["a"][i] * v["b"][i] * rinv) % p})
    d["mac_fp8_k64"] = dict(
        lanes=1, accw=64, role="Kulisch FP8",
        gen=lambda rng, n: gen_fp8(rng, n, 1, lambda r, a, b: rbits(r, len(a), 64)),
        ref=lambda v, i: {"y": (v["acc"][i] + e4m3_fx(v["a"][i], v["b"][i])) % (1 << 64),
                          "nan": int(is_nan8(v["a"][i]) or is_nan8(v["b"][i]))})
    d["mac_fp8x4_k64"] = dict(
        lanes=4, accw=64, role="Kulisch FP8",
        gen=lambda rng, n: gen_fp8(rng, n, 4, lambda r, m: rbits(r, m, 64)),
        ref=lambda v, i: {"y": (v["acc"][i] + sum(e4m3_fx(v[f"a{k}"][i], v[f"b{k}"][i]) for k in range(4)))
                          % (1 << 64),
                          "nan": int(any(is_nan8(v[f"{c}{k}"][i]) for c in "ab" for k in range(4)))})

    def nvfp4_ref(v, i):
        tot = 0
        for k in range(16):
            sa, ma, ea = e2m1(v["a"][i] >> (4 * k) & 15)
            sb, mb, eb = e2m1(v["b"][i] >> (4 * k) & 15)
            tot += (-1) ** (sa ^ sb) * ((ma * mb) << (ea + eb + 2))       # product * 4
        ssa, msa, esa, na = e4m3(v["sa"][i])
        ssb, msb, esb, nb = e4m3(v["sb"][i])
        val = (-1) ** (ssa ^ ssb) * tot * ((msa * msb) << (esa + esb + 18))  # * 2^20 overall
        return {"y": (v["acc"][i] + val) % (1 << 64), "nan": int(na or nb)}

    d["mac_nvfp4_b16_k64"] = dict(
        lanes=16, accw=64, role="Kulisch NVFP4",
        gen=lambda rng, n: {"a": rbits(rng, n, 64), "b": rbits(rng, n, 64), "sa": rbits(rng, n, 8),
                            "sb": rbits(rng, n, 8), "acc": rbits(rng, n, 64)},
        ref=nvfp4_ref)
    d["mac_gf8x4"] = dict(
        lanes=4, accw=8, role="binary field",
        gen=lambda rng, n: {k: rbits(rng, n, 8) for k in
                            ["a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3", "acc"]},
        ref=lambda v, i: {"y": v["acc"][i] ^ gf_reduce(
            clmul(v["a0"][i], v["b0"][i]) ^ clmul(v["a1"][i], v["b1"][i]) ^ clmul(v["a2"][i], v["b2"][i])
            ^ clmul(v["a3"][i], v["b3"][i]), 8, 0x1B)})
    for n_, low in ((32, 0x8D), (64, 0x1B), (128, 0x87)):
        d[f"mac_gf{n_}"] = dict(
            lanes=1, accw=n_, role="binary field",
            gen=lambda rng, n, n_=n_: {k: rbits(rng, n, n_) for k in ("a", "b", "acc")},
            ref=lambda v, i, n_=n_, low=low: {"y": v["acc"][i] ^ gf_reduce(clmul(v["a"][i], v["b"][i]), n_, low)})
    primes = (251, 241, 239, 233)

    def rns_gen(rng, n):
        v = {}
        for nm in ["a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3", "acc"]:
            v[nm] = [sum(rng.randrange(m) << (8 * c) for c, m in enumerate(primes)) for _ in range(n)]
        return v

    def rns_ref(v, i):
        y = 0
        for c, m in enumerate(primes):
            g = lambda nm: v[nm][i] >> (8 * c) & 255
            s = g("acc") + sum(g(f"a{k}") * g(f"b{k}") for k in range(4))
            y |= (s % m) << (8 * c)
        return {"y": y}

    d["mac_rns4_s8x4"] = dict(lanes=4, accw=32, role="RNS", gen=rns_gen, ref=rns_ref)
    return d
