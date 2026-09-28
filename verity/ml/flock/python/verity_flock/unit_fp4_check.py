"""unit_fp4 against verity.ml.circuits.silicon BLACKWELL_SM120_NVF4.step_scaled on random and directed units (bit-sliced)."""
from __future__ import annotations

import random
import sys

from verity.ml.circuits.silicon.models import BLACKWELL_SM120_NVF4 as M

from . import unit_fp4


def cases(rng, n):
    out = []
    for t in range(n):
        fam = t % 14
        if fam >= 8:
            out.append(directed(rng, fam))
            continue
        def scale():
            r = rng.random()
            if r < 0.1:
                return 0
            if r < 0.25:
                return rng.randint(1, 7)                      # subnormal field
            if r < 0.35:
                return rng.choice([0x78, 0x79, 0x7E, 0x70])  # top exponent field
            return rng.choice([b for b in (rng.randint(0, 0x7E),) if b != 0x7F])
        def code():
            if fam == 1 and rng.random() < 0.8:
                return rng.choice([0, 8])
            return rng.randint(0, 15)
        a = [code() for _ in range(64)]
        b = [code() for _ in range(64)]
        if fam == 2:
            b = [bb ^ 8 if i % 2 else bb for i, bb in enumerate(a)]   # cancelling pairs
            a = [aa for aa in a]
        sa = [scale() for _ in range(4)]
        sb = [scale() for _ in range(4)]
        if fam == 3:
            c = rng.choice([0, 0x8000_0000])
        elif fam == 4:
            c = (rng.getrandbits(1) << 31) | rng.getrandbits(23)          # subnormal
        elif fam == 5:
            c = (rng.getrandbits(1) << 31) | (rng.randint(200, 254) << 23) | rng.getrandbits(23)
        elif fam == 6:
            c = (rng.getrandbits(1) << 31) | (rng.randint(90, 140) << 23) | rng.getrandbits(23)
        else:
            c = (rng.getrandbits(1) << 31) | (rng.randint(0, 254) << 23) | rng.getrandbits(23)
        out.append((c, a, b, sa, sb))
    return out


def directed(rng, fam):
    """8: no products, any accumulator (passes through, +0 for +-0); 9: no products, subnormal accumulator; 10: exactly
    cancelling pairs in every group, zero accumulator (+0 from participating groups); 11: cancelling pairs with a random
    accumulator (the groups still anchor lsb); 12: zero scale mantissas (products but no participation); 13: subnormal
    scales (unit weight 2^-20) against a subnormal / tiny accumulator."""
    acc = lambda: (rng.getrandbits(1) << 31) | (rng.randint(0, 254) << 23) | rng.getrandbits(23)  # noqa: E731
    sub = lambda: (rng.getrandbits(1) << 31) | rng.getrandbits(23)  # noqa: E731
    sc = lambda: rng.choice([b for b in (rng.randint(1, 0x7E),) if b != 0x7F] or [8])  # noqa: E731
    if fam in (8, 9):
        a = [rng.choice([0, 8]) if rng.random() < 0.5 else rng.randint(0, 15) for _ in range(64)]
        b = [0 if rng.random() < 0.5 else 8 for _ in range(64)]
        sa, sb = [sc() for _ in range(4)], [sc() for _ in range(4)]
        c = sub() if fam == 9 else rng.choice([acc(), 0, 0x8000_0000, sub()])
        return c, a, b, sa, sb
    if fam in (10, 11):
        a, b = [], []
        for _ in range(32):
            x, y = rng.randint(1, 15), rng.randint(1, 15)
            a += [x, x]
            b += [y, y ^ 8]
        sa, sb = [sc() for _ in range(4)], [sc() for _ in range(4)]
        c = rng.choice([0, 0x8000_0000]) if fam == 10 else acc()
        return c, a, b, sa, sb
    if fam == 12:
        a = [rng.randint(0, 15) for _ in range(64)]
        b = [rng.randint(0, 15) for _ in range(64)]
        sa = [rng.choice([0, 0x08, 0x40]) if g % 2 else 0 for g in range(4)]
        sb = [0 if g % 2 else rng.choice([0x38, 0x50]) for g in range(4)]
        return rng.choice([acc(), sub(), 0]), a, b, sa, sb
    a = [rng.randint(0, 15) for _ in range(64)]
    b = [rng.randint(0, 15) for _ in range(64)]
    sa = [rng.randint(1, 7) for _ in range(4)]
    sb = [rng.randint(1, 7) for _ in range(4)]
    c = rng.choice([sub(), (rng.getrandbits(1) << 31) | (rng.randint(1, 110) << 23) | rng.getrandbits(23)])
    return c, a, b, sa, sb


def run(n=4000, seed=20260926):
    C = unit_fp4.unit()
    cs = cases(random.Random(seed), n)
    T = len(cs)
    vals = {nm: [0] * len(bits) for nm, bits in C.inputs.items()}
    def put(nm, i, t, v, width):
        for j in range(width):
            if v >> j & 1:
                vals[nm][i * width + j] |= 1 << t
    for t, (c, a, b, sa, sb) in enumerate(cs):
        for i in range(64):
            put("x", i, t, a[i], 4)
            put("w", i, t, b[i], 4)
        for g in range(4):
            put("sx", g, t, sa[g], 8)
            put("sw", g, t, sb[g], 8)
        put("c", 0, t, c, 32)
    outs, fail = C.evaluate(vals, T)
    bad = 0
    for t, (c, a, b, sa, sb) in enumerate(cs):
        want = M.step_scaled(c, a, b, sa, sb)
        got = sum(((outs["c_out"][j] >> t) & 1) << j for j in range(32))
        if got != want or (fail >> t) & 1:
            if bad < 5:
                print(f"case {t}: c={c:#010x} got {got:#010x} want {want:#010x} fail={(fail >> t) & 1}")
            bad += 1
    cnt = C.counts()
    print(f"units {T}, mismatches {bad}; counts {cnt}")
    print("phases", C.phase_counts())
    return bad


if __name__ == "__main__":
    sys.exit(1 if run(int(sys.argv[1]) if len(sys.argv) > 1 else 4000) else 0)
