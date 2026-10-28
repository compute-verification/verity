#!/usr/bin/env python3
"""A check of checked_strip.py's priced zero-instance routes (its check H and P7).

`checked_strip.py` prices every zero-instance route as bit-sliced binary products at x warp instructions per 32 × 128
bit products (T increments at 8x an IMMA, D_d and W at 7x, O′'s c·b at x·k/32), and treats x = 12 as conservative because
a binary word costs about one IMMA word in time. That holds for O′'s c·b, a 1-bit × 1-bit product. The T increment
Σ (c + U)·b is an 8-bit × 1-bit product and D_d = U·A_d, W = U·b are 7-bit × 1-bit, which IDP.4A takes four to a lane
instruction: 8 IDP.4A a word, 32 warp instructions an IMMA's 128 words, which is x = 4 for T and 32/7 for D and W, with the
counting included. IDP.4A takes s8 × s8 bytes, so the same 8 instructions make a T word at any instance: this is the
vector route at q = 8, which the stated thresholds close only while JI3e_0 binds honest (2.03 there, 11.29 on JI3e_1). This
script prices those routes on checked_strip's own facets and tiles (`ts_tile`, `ts_pv`, `ts_best`) and finds the fold
below which they pay.

  uv run python benchmarks/pouw/theory/zero_routes_check.py [--quick] [--mutations] [--verdict PATH]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checked_strip as cs
from exact_kit.checks import Checks

#: Warp instructions an IMMA's 128 words cost when each word is 8 IDP.4A, and when it is the measured dp4a word (10.59
#: issue units a word, `scalar_route.py`'s no-table costs).
IDP4A, DP4A_MEASURED = 128 * 8 / 32, 128 * 10.59 / 32


def dp4a(a, b, acc):
    """IDP.4A.S8.U8: acc + Σ_{j<4} a_j·b_j over the four bytes of each 32-bit lane, wrapping at 32 bits."""
    return (acc + (a.reshape(*a.shape[:-1], -1, 4).astype(np.int64) * b.reshape(*b.shape[:-1], -1, 4)).sum(-1)
            .sum(-1)) & cs.MASK


def lin_routes(per_imma, xbin=12):
    def routes(k):
        m = cs.n_digits((k // 32) * -128, (k // 32) * 127)
        return [(-(k // 32), per_imma * (k // 32)), (-(m - 1), per_imma * (m - 1)),
                cs.o_routes("co32c", k)["scalar"](xbin)]
    return routes


def pv_routes(per_imma, xbin=12):
    def routes(k):
        g = k // 32 - 1
        m = cs.n_digits(g * -128, g * 127) if g else 1
        r = [(-(k // 32), per_imma * (k // 32)), (-1, per_imma), (-(2 if g else 1), xbin * k / 32)]
        return r + ([(-(m - 1), per_imma * (m - 1))] if m > 1 else [])
    return routes


def lin_x(x):
    return lambda k: cs.word_routes("co32c", k, x) + [cs.o_routes("co32c", k)["scalar"](x)]


def pv_x(x):
    return lambda k: cs.pv_routes(k, x)


def linears(fold, model, routes=None):
    h = c = 0.0
    for _, k, n in cs.TS_SHAPES[model]:
        th, tb = cs.ts_best("co32c", k, n, fold, routes(k) if routes else [])
        h, c = h + th * n, c + tb * n
    return h / c - 1


def pv(fold, rep, ctx, routes=None):
    h = c = 0.0
    for beta in range(ctx // 32):
        if beta >= 256 and beta % 8:
            continue
        wt, k = (1 if beta < 256 else 8), 32 * (beta + 1)
        hon, cht, _, _ = cs.ts_pv(k, fold, rep)
        h += wt * cs.ts_time(*hon)
        c += wt * (cs.ts_best(None, k, 0, fold, routes(k), (hon, cht))[1] if routes else cs.ts_time(*cht))
    return h / c - 1


def main() -> int:
    checks = Checks()
    quick = "--quick" in checks.args
    rng = np.random.default_rng(20261017)

    cells, k = 64, 32
    c = rng.integers(0, 2, (cells, k))
    U = rng.choice(np.r_[-64:0, 1:65], (cells, k))
    b = rng.integers(0, 2, (cells, k))
    acc0 = rng.integers(0, 1 << 32, cells)
    Pg, Xg = rng.integers(-64, 64, (cells, k)), rng.integers(-128, 128, (cells, k))
    ok = []
    for A, B in (((c + U), b), (Pg, Xg)):
        got = acc0
        for s in range(0, k, 4):
            got = dp4a(A[:, s:s + 4], B[:, s:s + 4], got)
        ok.append(np.array_equal(got, (acc0 + (A * B).sum(-1)) & cs.MASK))
    checks.check("8 IDP.4A make a T increment exactly (s32 accumulate), at the zero instance (Σ (c + U)·b, s8 × u8) and "
                 "at a general one (P̃·X, s8 × s8), so the T route costs 32 warp instructions an IMMA's 128 words (x = 4) "
                 "with the counting included", all(ok) and k // 4 == 8, f"{cells} cells each, 8 IDP.4A a word: {ok}")

    models = list(cs.TS_SHAPES)
    folds = (3, 2.75, 2.6, 2.5, 2.25, 2, 1.5, 1)
    tab = {(m, f): (linears(f, m), linears(f, m, lin_x(12)), linears(f, m, lin_routes(IDP4A)),
                    linears(f, m, lin_routes(DP4A_MEASURED))) for m in models for f in folds}
    summary = {f"{m} f={f}": "strip-only {:.3%}, x=12 {:.3%}, IDP.4A {:.3%}, dp4a 10.59 {:.3%}".format(*v)
               for (m, f), v in tab.items()}
    checks.check("co32c's linears at the duo fold (f ≥ 2.6): IDP.4A-made T and D words gain nothing",
                 all(abs(tab[(m, f)][2] - tab[(m, f)][0]) < 1e-12 for m in models for f in (3, 2.75, 2.6)),
                 "; ".join(summary[f"{m} f={f}"] for m in models for f in (3, 2.6)))
    a0, b0 = next((a, b_) for n_, a, b_ in cs.TS_FACETS if n_ == "JI3e_0")
    knee = max((a - a0) / (4 * (b0 - b_)) for n_, a, b_ in cs.TS_FACETS if n_.startswith("JI3e") and b_ < b0)
    checks.check(f"the routes open below f = {knee:.2f}, where JI3e_0 stops binding honest's tile (a + 4f on JI3e_0 "
                 "against a′ + b′·4f on JI3e_1): gain at f = 2.5 and below, none at 2.6",
                 2.5 < knee <= 2.6 and all(tab[(m, f)][2] > tab[(m, f)][0] + 5e-4 for m in models
                                           for f in (2.5, 2.25, 2, 1.5, 1)),
                 "; ".join(summary[f"{m} f={f}"] for m in models for f in (2.5, 2)))
    checks.check("at the XOR-in fold (f = 1.5) IDP.4A routes put 8B's linears at ≈ 3.0% (0.83% at the measured "
                 "dp4a word), against checked_strip's 0.08% at x = 12",
                 0.025 < tab[(models[0], 1.5)][2] < 0.035 and 0.006 < tab[(models[0], 1.5)][3] < 0.011
                 and tab[(models[0], 1.5)][1] < 0.001, summary[f"{models[0]} f=1.5"])

    ctxs = (2048,) if quick else (2048, 8192, 32768)
    pvt = {(ctx, f): (pv(f, 4, ctx), pv(f, 4, ctx, pv_x(12)), pv(f, 4, ctx, pv_routes(IDP4A)))
           for ctx in ctxs for f in (3, 1.5)}
    pvs = {f"8B ctx={ctx} f={f}": "own ε {:.3%}, x=12 {:.3%}, IDP.4A {:.3%}".format(*v) for (ctx, f), v in pvt.items()}
    checks.check("P·V (8B, served output free): at f = 3 IDP.4A routes add < 0.01% of P·V's time; at f = 1.5 they add "
                 "> 2% (x = 12 adds < 0.2%)",
                 all(pvt[(x, 3)][2] - pvt[(x, 3)][0] < 1e-4 for x in ctxs)
                 and all(pvt[(x, 1.5)][2] - pvt[(x, 1.5)][0] > 0.02 and pvt[(x, 1.5)][1] - pvt[(x, 1.5)][0] < 0.002
                         for x in ctxs), "; ".join(pvs.values()))
    if checks.mutations:
        checks.mutation("pricing T and D at x = 12 (checked_strip's) hides the f = 2 gain",
                        all(abs(tab[(m, 2)][1] - tab[(m, 2)][0]) < 1e-12 for m in models),
                        "; ".join(summary[f"{m} f=2"] for m in models))
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
