"""A32: A21's P-HYB3 rows against D5 v2's, in D5 v2's own LP (public_best at 34866fd9f8).

1. c_t's Z Uᵀ: D5's price, then with shared memory (and, on sm_120, register-file reads) unlimited, which is the
   tensor-only count A21 and nofold's C_T make.
2. b cᵀ at 3 × 3 bits on the RTX PRO 6000: D5's hull, then the hull plus A21's FP4 route, priced the way D5 prices its
   own FP4 bit route (`d5_routes.methods`' bc_fixed: every MMA load × the FP4 cost, here 0.5 at depth 0), plus the
   shift's rank-1 corrections (row and column sums, one add an output) on the lanes.
3. b Uᵀ at 3 bits × ±80: naive alone against D5's hull, and the overflow density q of each fast scheme's U-side forms.

    cd <D5 v2>/benchmarks/pouw/theory && ACCEL_SCORE_DIR=<nofold dir> python3 phyb3_check.py OUT.txt
"""
import math
import sys
from dataclasses import replace

import d5_v2 as D
import public_best as pb

out = []


def say(s):
    print(s, flush=True)
    out.append(s)


b1, b3, u80 = D._uniform(0, 1), D._uniform(0, 7), D._uniform(-80, 80)
eU = pb.entry_pmf(pb.data_pmf("zero"), pb.V2_U)

say("1. c_t = 1 + price(Z Uᵀ), standalone at θ, L' 256, n 16512, beside=True (1-bit b; 3-bit b alike)")
for name, c in D.CARDS.items():
    free = replace(c, smem=math.inf, rf=math.inf)
    for t in ((2, 3) if name == "H200" else (3, 4)):
        eX = D._plus(b1, u80)
        st = pb._stats("A", eX, (1,) * t)
        leaf = pb.Leaf(1.0, st, pb._stats("B", eU, (1,)), 1)
        z = pb.price(c, [leaf], 256, 16512, pb.TOKENS, beside=True)
        zf = pb.price(free, [leaf], 256, 16512, pb.TOKENS, beside=True)
        zt = pb.price(free, [leaf], 256, 16512, pb.TOKENS, beside=False)
        r = lambda p: {k: round(float(v), 3) for k, v in p.routes.items() if v > 1e-6}  # noqa: E731
        say(f"  {name:12} t {t}: |Z| ≤ {st.max_abs}, q {st.q:.4f} | θ: c_t {1 + z.c:.3f} (binding {z.binding}; "
            f"routes {r(z)}) | smem, rf unlimited: {1 + zf.c:.3f} ({zf.binding}; {r(zf)}) | the same, MMA only: "
            f"{1 + zt.c:.3f} ({zt.binding})")

say("\n2. b cᵀ at 3 × 3 bits, L 1024, n 16512, standalone at θ")
menu = pb.methods(pb.orbit(), pb.catalogue())
for name, c in D.CARDS.items():
    L, n = 2 * 512, 16512
    chosen = {"naive": (None, 0)}
    for mn, (schemes, d) in menu.items():
        if mn != "naive":
            chosen[mn] = (pb.best_scheme(c, d, b3, b3, L, n, pb.TOKENS, schemes, beside=True)[1], d)
    ms = [pb.scheme_method(lb, s, d, b3, b3) for lb, (s, d) in chosen.items()]
    h = pb.price_methods(c, ms, L, n, pb.TOKENS, beside=True)
    say(f"  {name:12} D5's hull: {h.c:.4f} (binding {h.binding}; mix "
        f"{ {k: round(float(v), 3) for k, v in h.mix.items() if v > 1e-3} })")
    if c.family != "sm_120":
        continue
    ops = 1 / L + 1 / pb.TOKENS + 1 / n
    x = 0.5
    fixed = dict(tensor=x / c.s8, smem=x * (1 / c.honest_tile[0] + 1 / c.honest_tile[1]), rf=x * c.mma_rf,
                 issue=x * c.mma_issue)
    for r_, y in (("alu", ops), ("issue", ops), ("rf", 3 * ops), ("vec", ops)):
        fixed[r_] = fixed.get(r_, 0.0) + y
    fp4 = pb.Method("FP4, (b - 3)(c - 3)ᵀ in E2M1, depth 0", [], fixed=fixed)
    alone = pb.price_methods(c, [fp4], L, n, pb.TOKENS, beside=True)
    hf = pb.price_methods(c, ms + [fp4], L, n, pb.TOKENS, beside=True)
    say(f"  {name:12} FP4 route alone: {alone.c:.4f} (binding {alone.binding}) | D5's hull + FP4: {hf.c:.4f} "
        f"(binding {hf.binding}; mix { {k: round(float(v), 3) for k, v in hf.mix.items() if v > 1e-3} })")
    lo, hi = -3, 4
    e2m1 = {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    vals = set(range(0 - 3, 7 - 3 + 1))
    say(f"    b - 3 takes {sorted(vals)}; all in E2M1: {all(abs(v) in e2m1 for v in vals)}. D5's own FP4 route "
        f"(BIT_COST sm_120, Strassen depth 2 on 0/1 bits) takes forms in [-4, 4], the same set or wider; "
        f"products ≤ {max(abs(lo), hi) ** 2}, a segment's sum ≤ {max(abs(lo), hi) ** 2 * L} < 2^24")

say("\n3. b Uᵀ at 3 bits × ±80, L 1024, n 16512, standalone at θ")
for name, c in D.CARDS.items():
    L, n = 2 * 512, 16512
    nv = pb.price_methods(c, [pb.scheme_method("naive", None, 0, b3, u80)], L, n, pb.TOKENS, beside=True)
    chosen = {"naive": (None, 0)}
    for mn, (schemes, d) in menu.items():
        if mn != "naive":
            chosen[mn] = (pb.best_scheme(c, d, b3, u80, L, n, pb.TOKENS, schemes, beside=True)[1], d)
    h = pb.price_hull(c, chosen, b3, u80, L, n, pb.TOKENS, beside=True)
    say(f"  {name:12} naive alone {nv.c:.4f} (binding {nv.binding}) | D5's hull {h.c:.4f} (binding {h.binding}; mix "
        f"{ {k: round(float(v), 3) for k, v in h.mix.items() if v > 1e-3} })")
    for lb, (s, d) in chosen.items():
        if lb == "naive" or h.mix.get(lb, 0) < 1e-3:
            continue
        lv = pb.leaves_of(s, d, b3, u80)
        qs = sorted({(round(lf.B.q, 4), lf.B.max_abs) for lf in lv})
        say(f"    {lb}: U-side forms (q, max |form|) {qs}: the forms past int8 are priced as overflow corrections")

open(sys.argv[1], "w").write("\n".join(out) + "\n")
