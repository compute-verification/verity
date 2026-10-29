"""C-Flock's proof cost of each SI-ALG MAC: live ANDs (one R1CS row each, XOR and NOT free) in the repository's own
circuit builder (`verity.ml.circuits.boolean.forms`, and `silicon.fp`'s FP32 adder), each circuit checked bit for bit
against refs.py on random vectors.

    python3 cflock_ands.py --out OUT [--vectors N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import refs  # noqa: E402
from verity.ml.circuits.boolean import forms as F  # noqa: E402
from verity.ml.circuits.boolean.forms import ONE, ZERO, NOT, Circuit, add, compress_columns, const_bits, lzc  # noqa: E402
from verity.ml.circuits.boolean.forms import multiply, mux, or_reduce, shl_window, sub  # noqa: E402
from verity.ml.circuits.silicon import fp  # noqa: E402


def put(cols, bits, shift, width):
    for i, b in enumerate(bits):
        if i + shift < width:
            cols[i + shift].append(b)


def put_const(cols, value, width):
    for i in range(width):
        if value >> i & 1:
            cols[i].append(ONE)


def put_neg(cols, bits, shift, width):
    """Add -(bits << shift) mod 2^width: NOT of the shifted word over the full width, plus one."""
    word = [ZERO] * width
    for i, b in enumerate(bits):
        if i + shift < width:
            word[i + shift] = b
    put(cols, [NOT(w) for w in word], 0, width)
    put_const(cols, 1, width)


def csd(c):
    """Canonical signed digits of c > 0: [(shift, +1|-1)]."""
    out, k = [], 0
    while c:
        if c & 1:
            d = 2 - (c & 3)
            out.append((k, d))
            c -= d
        c >>= 1
        k += 1
    return out


def put_cmul(cols, bits, c, width):
    for k, d in csd(c):
        (put if d > 0 else put_neg)(cols, bits, k, width)


def signed_prod_cols(C, cols, a, b, width):
    """(a * b) mod 2^width for two's-complement a, b (Baugh-Wooley: negative-weight partial products inverted)."""
    na, nb = len(a) - 1, len(b) - 1
    const = 0
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            pp = C.AND(x, y)
            if (i == na) != (j == nb):
                put(cols, [NOT(pp)], i + j, width)
                const -= 1 << (i + j)
            else:
                put(cols, [pp], i + j, width)
    return const


def reduce_cols(C, cols, const, width):
    put_const(cols, const % (1 << width), width)
    return compress_columns(C, cols)[:width]


def cond_sub(C, x, p, width):
    d, borrow = sub(C, x, const_bits(p, width), width)
    return mux(C, NOT(borrow), d, x)


def s8x4_acc(C, width):
    a = [C.input(f"a{k}", 8) for k in range(4)]
    b = [C.input(f"b{k}", 8) for k in range(4)]
    acc = C.input("acc", width if width <= 64 else 64)
    return a, b, acc


def b_s8x4_int(width):
    def build(C):
        a, b, acc = s8x4_acc(C, width)
        cols = [[] for _ in range(width)]
        put(cols, acc, 0, width)
        const = sum(signed_prod_cols(C, cols, a[k], b[k], width) for k in range(4))
        C.output("y", reduce_cols(C, cols, const, width))
    return build


def b_s8x4_modp(p, w):
    def build(C):
        a, b, acc = s8x4_acc(C, w)
        tw = w + 2
        cols = [[] for _ in range(tw)]
        put(cols, acc, 0, tw)
        const = sum(signed_prod_cols(C, cols, a[k], b[k], tw) for k in range(4))
        t = reduce_cols(C, cols, const, tw)
        neg = t[tw - 1]
        tp = add(C, t, const_bits(p, tw), width=tw)
        tm, borrow = sub(C, t, const_bits(p, tw), tw)
        y = mux(C, neg, tp, mux(C, NOT(borrow), tm, t))
        C.output("y", y[:w])
    return build


def b_mersenne(p_bits):
    p = (1 << p_bits) - 1

    def build(C):
        a, b, acc = (C.input(n, p_bits) for n in ("a", "b", "acc"))
        x = multiply(C, a, b)
        cols = [[] for _ in range(p_bits + 2)]
        put(cols, x[:p_bits], 0, p_bits + 2)
        put(cols, x[p_bits:2 * p_bits], 0, p_bits + 2)
        put(cols, acc, 0, p_bits + 2)
        s = compress_columns(C, cols)[:p_bits + 2]
        s1 = add(C, s[:p_bits], s[p_bits:p_bits + 2], width=p_bits + 1)
        C.output("y", cond_sub(C, s1, p, p_bits + 1)[:p_bits])
    return build


def b_babybear(C):
    P, NINV = refs.BB, 0x77FFFFFF
    a, b, acc = (C.input(n, 31) for n in ("a", "b", "acc"))
    cols = [[] for _ in range(64)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            cols[i + j].append(C.AND(x, y))
    put(cols, acc, 32, 64)
    x = compress_columns(C, cols)[:64]
    mcols = [[] for _ in range(32)]
    put_cmul(mcols, x[:32], NINV, 32)
    m = compress_columns(C, mcols)[:32]
    tcols = [[] for _ in range(65)]
    put(tcols, x, 0, 65)
    put_cmul(tcols, m, P, 65)
    t = compress_columns(C, tcols)[:65]
    u = t[32:65]
    C.output("y", cond_sub(C, cond_sub(C, u, P, 33), P, 33)[:31])


def b_goldilocks(C):
    P, EPS = refs.GL, (1 << 32) - 1
    a, b, acc = (C.input(n, 64) for n in ("a", "b", "acc"))
    x = multiply(C, a, b)
    W = 67
    cols = [[] for _ in range(W)]
    put(cols, x[:64], 0, W)
    put_cmul(cols, x[64:96], EPS, W)
    put_neg(cols, x[96:128], 0, W)
    put(cols, acc, 0, W)
    put_const(cols, P, W)
    r = compress_columns(C, cols)[:W]
    c2 = [[] for _ in range(W)]
    put(c2, r[:64], 0, W)
    put_cmul(c2, r[64:67], EPS, W)
    r2 = compress_columns(C, c2)[:W]
    C.output("y", cond_sub(C, cond_sub(C, r2, P, W), P, W)[:64])


def e4m3_fields(C, x):
    e = x[3:7]
    nz = or_reduce(C, e)
    m = x[0:3] + [nz]
    xe = [e[0] ^ NOT(nz)] + e[1:]                          # max(e, 1): e[0] is 0 whenever nz is
    return x[7], m, xe


def e4m3_nan(C, x):
    return F.and_reduce(C, x[0:7])


def kulisch_prod(C, a, b, cols, width):
    sa, ma, xa = e4m3_fields(C, a)
    sb, mb, xb = e4m3_fields(C, b)
    mp = multiply(C, ma, mb)
    sh = add(C, add(C, xa, xb, width=5, carry_out=False), const_bits(32 - 2, 5), width=5)   # xa + xb - 2 mod 32
    mag = shl_window(C, mp, sh, 0, 37)
    s = sa ^ sb
    put(cols, [m ^ s for m in mag] + [s] * (width - 37), 0, width)
    cols[0].append(s)


def b_fp8_k64(lanes):
    def build(C):
        names = [("a", "b")] if lanes == 1 else [(f"a{k}", f"b{k}") for k in range(lanes)]
        ins = [(C.input(x, 8), C.input(y, 8)) for x, y in names]
        acc = C.input("acc", 64)
        cols = [[] for _ in range(64)]
        put(cols, acc, 0, 64)
        for a, b in ins:
            kulisch_prod(C, a, b, cols, 64)
        C.output("y", compress_columns(C, cols)[:64])
        C.output("nan", [or_reduce(C, [e4m3_nan(C, v) for ab in ins for v in ab])])
    return build


def b_fp8_fp32(C):
    a, b, acc = C.input("a", 8), C.input("b", 8), C.input("acc", 32)
    sa, ma, xa = e4m3_fields(C, a)
    sb, mb, xb = e4m3_fields(C, b)
    mp = multiply(C, ma, mb)[:8]
    az, cnt = lzc(C, mp)
    mn = shl_window(C, mp, cnt, 0, 8)
    e9 = add(C, add(C, xa, xb, width=9), const_bits(114, 9), width=9)
    e9, _ = sub(C, e9, cnt + [ZERO] * 6, 9)
    word = [ZERO] * 16 + mn[:7] + [C.AND(NOT(az), e) for e in e9[:8]] + [sa ^ sb]
    y = fp.f32_add(C, acc, word)
    nan = F.OR(C, e4m3_nan(C, a), e4m3_nan(C, b))
    C.output("y", mux(C, nan, const_bits(refs.NAN32, 32), y))


def b_gf(n, low, karatsuba, lanes=1):
    def kara(C, a, b):
        if len(a) <= 1:
            return [C.AND(a[0], b[0])]
        h = (len(a) + 1) // 2
        a0, a1, b0, b1 = a[:h], a[h:], b[:h], b[h:]
        lo = kara(C, a0, b0)
        hi = kara(C, a1 + [ZERO] * (h - len(a1)), b1 + [ZERO] * (h - len(b1)))
        mid = kara(C, [x ^ y for x, y in zip(a0, a1 + [ZERO] * (h - len(a1)))],
                   [x ^ y for x, y in zip(b0, b1 + [ZERO] * (h - len(b1)))])
        out = [ZERO] * (2 * len(a) - 1 + 2)
        for i, v in enumerate(lo):
            out[i] ^= v
            out[i + h] ^= v
        for i, v in enumerate(hi):
            if i + 2 * h < len(out):
                out[i + 2 * h] ^= v
            if i + h < len(out):
                out[i + h] ^= v
        for i, v in enumerate(mid):
            out[i + h] ^= v
        return out[:2 * len(a) - 1]

    def school(C, a, b):
        out = [ZERO] * (2 * len(a) - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[i + j] ^= C.AND(x, y)
        return out

    def build(C):
        mul = kara if karatsuba else school
        if lanes == 1:
            pairs = [(C.input("a", n), C.input("b", n))]
        else:
            pairs = [(C.input(f"a{k}", n), C.input(f"b{k}", n)) for k in range(lanes)]
        acc = C.input("acc", n)
        q = [ZERO] * (2 * n - 1)
        for a, b in pairs:
            q = [x ^ y for x, y in zip(q, mul(C, a, b))]
        for i in range(2 * n - 2, n - 1, -1):
            v = q[i]
            q[i] = ZERO
            for k in range(n):
                if low >> k & 1:
                    q[i - n + k] ^= v
        C.output("y", [x ^ y for x, y in zip(acc, q[:n])])
    return build


def b_rns(C):
    primes = (251, 241, 239, 233)
    ins = {nm: C.input(nm, 32) for nm in ["a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3", "acc"]}
    y = []
    for c, m in enumerate(primes):
        g = lambda nm: ins[nm][8 * c:8 * c + 8]
        cols = [[] for _ in range(18)]
        put(cols, g("acc"), 0, 18)
        for k in range(4):
            for i, xa in enumerate(g(f"a{k}")):
                for j, xb in enumerate(g(f"b{k}")):
                    cols[i + j].append(C.AND(xa, xb))
        s = compress_columns(C, cols)[:18]
        cc = 256 - m
        for w_out in (16, 13, 11, 10, 10):
            fc = [[] for _ in range(w_out)]
            put(fc, s[:8], 0, w_out)
            put_cmul(fc, s[8:], cc, w_out)
            s = compress_columns(C, fc)[:w_out]
        s = cond_sub(C, cond_sub(C, s, m, 10), m, 10)
        y += s[:8]
    C.output("y", y)


def b_nvfp4(C):
    a, b = C.input("a", 64), C.input("b", 64)
    sa8, sb8, acc = C.input("sa", 8), C.input("sb", 8), C.input("acc", 64)
    bcols = [[] for _ in range(14)]
    for k in range(16):
        x, y = a[4 * k:4 * k + 4], b[4 * k:4 * k + 4]
        nzx, nzy = F.OR(C, x[1], x[2]), F.OR(C, y[1], y[2])
        mx, my = [x[0], nzx], [y[0], nzy]
        # exponent - 1 when nonzero, 0 when subnormal: (e - 1) for e in 1..3 is e ^ 1 on bit 0 with a borrow
        ex = [C.AND(nzx, NOT(x[1])), C.AND(x[2], x[1])]
        ey = [C.AND(nzy, NOT(y[1])), C.AND(y[2], y[1])]
        mp = multiply(C, mx, my)[:4]
        sh = add(C, ex, ey, width=3)
        mag = shl_window(C, mp, sh, 0, 8)
        s = x[3] ^ y[3]
        put(bcols, [m ^ s for m in mag] + [s] * 6, 0, 14)
        bcols[0].append(s)
    blk = compress_columns(C, bcols)[:14]
    ssa, ma, xa = e4m3_fields(C, sa8)
    ssb, mb, xb = e4m3_fields(C, sb8)
    ms = multiply(C, ma, mb)[:8]
    vcols = [[] for _ in range(23)]
    const = signed_prod_cols(C, vcols, blk, ms + [ZERO], 23)
    v = reduce_cols(C, vcols, const, 23)
    sh = add(C, add(C, xa, xb, width=5), const_bits(30, 5), width=5)
    vx = shl_window(C, v + [v[22]] * 41, sh, 0, 64)
    s = ssa ^ ssb
    cols = [[] for _ in range(64)]
    put(cols, acc, 0, 64)
    put(cols, [w ^ s for w in vx], 0, 64)
    cols[0].append(s)
    C.output("y", compress_columns(C, cols)[:64])
    C.output("nan", [F.OR(C, e4m3_nan(C, sa8), e4m3_nan(C, sb8))])


BUILDERS = {
    "mac_s8x4_s32": b_s8x4_int(32),
    "mac_fp8_fp32": b_fp8_fp32,
    "mac_s8x4_s48": b_s8x4_int(48),
    "mac_s8x4_s64": b_s8x4_int(64),
    "mac_s8x4_m31": b_s8x4_modp(refs.M31, 31),
    "mac_m31_full": b_mersenne(31),
    "mac_s8x4_bb": b_s8x4_modp(refs.BB, 31),
    "mac_bb_full": b_babybear,
    "mac_s8x4_m61": b_s8x4_modp(refs.M61, 61),
    "mac_m61_full": b_mersenne(61),
    "mac_s8x4_gl": b_s8x4_modp(refs.GL, 64),
    "mac_gl_full": b_goldilocks,
    "mac_fp8_k64": b_fp8_k64(1),
    "mac_fp8x4_k64": b_fp8_k64(4),
    "mac_nvfp4_b16_k64": b_nvfp4,
    "mac_gf8x4": b_gf(8, 0x1B, False, 4),
    "mac_gf8x4@karatsuba": b_gf(8, 0x1B, True, 4),
    "mac_gf32": b_gf(32, 0x8D, False),
    "mac_gf32@karatsuba": b_gf(32, 0x8D, True),
    "mac_gf64": b_gf(64, 0x1B, False),
    "mac_gf64@karatsuba": b_gf(64, 0x1B, True),
    "mac_gf128": b_gf(128, 0x87, False),
    "mac_gf128@karatsuba": b_gf(128, 0x87, True),
    "mac_rns4_s8x4": b_rns,
}


def run(name, n, seed):
    top = name.split("@")[0]
    spec = refs.designs()[top]
    C = Circuit()
    BUILDERS[name](C)
    rng = random.Random(f"cflock:{name}:{seed}")
    vec = spec["gen"](rng, n)
    n = len(next(iter(vec.values())))
    ins = {nm: [sum(((x >> j) & 1) << i for i, x in enumerate(vec[nm])) for j in range(len(bits))]
           for nm, bits in C.inputs.items()}
    outs, _ = C.evaluate(ins, n)
    bad = 0
    for i in range(n):
        want = spec["ref"](vec, i)
        for k, w in want.items():
            got = sum(((outs[k][j] >> i) & 1) << j for j in range(len(outs[k])))
            bad += got != w
    c = C.counts()
    return {"design": name, "lanes": spec["lanes"], "ands": c["ands"], "ands_per_mac": c["ands"] / spec["lanes"],
            "vectors": n, "mismatches": bad}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in [x for x in a.only.split(",") if x] or BUILDERS:
        r = run(name, a.vectors, a.seed)
        print(f"{name:24s} ANDs {r['ands']:7d}  a MAC {r['ands_per_mac']:8.1f}  check "
              f"{'ok' if not r['mismatches'] else 'FAIL'} ({r['vectors']})", flush=True)
        rows.append(r)
    (out / "cflock_ands.json").write_text(json.dumps(rows, indent=1))
    return 1 if any(r["mismatches"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
