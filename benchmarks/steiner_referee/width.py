"""Round 3 referee checks: exact schemes against the full floor in the width-aware cost model.

A product is priced by its operand format: s8 x s8 = 1 int8 slot, FP16/BF16 = 2, E4M3 = 1, FP4 (E2M1) = 1/2.
Operands are wrapping casts of linear forms of the inputs; product results are only added (mod 2^32), each into
its own cell's accumulator, so a product used by two cells is paid twice. The floor for prefix targets at
granularity L = 2 on <n, k, n2> is n * k * n2 slots (one int8 slot a MAC).

  schemes  every exact scheme, run on random and extreme s8 inputs, checked mod 2^32, with its slot count
           (junk included) at <2,2,2>, <2,2,4> and per MAC asymptotically.
  toy      exhaustive search on one cell <1,2,1>, inputs w-bit signed (the residues of Z/4 and Z/8), for the
           cheapest set of width-labelled products whose span holds the target x1 y1 + x2 y2, with every
           function of X alone and of Y alone free (the junk, amortised). Formats in the toy: narrow (cost 1/2,
           operands narrower than w bits, the FP4 analogue), slot (cost 1, w-bit operands), wide (cost 2,
           w + 4 bits, the FP16 analogue). Spans are tested over GF(P), so a hit is exact over Q and is then
           checked for an integral (odd-denominator) combination; a miss rules a scheme out over Z.
           `--ring W` instead tests single products modulo 2^W (W = w is the literal Z/4, Z/8 output ring).
  shared   Strassen over FP16 Winograd with each product's value reused by several cells: exact while the
           Winograd operands stay within FP16's integers (three levels), below the floor, and outside a count
           that pays a product once per accumulator step.
"""

import argparse
import itertools
import json
from fractions import Fraction

import numpy as np

MOD = 2 ** 32
E2M1 = {s * m for m in (0, 0.5, 1, 1.5, 2, 3, 4, 6) for s in (1, -1)}
P = 2 ** 31 - 1


def wrap8(t):
    return ((t + 128) % 256) - 128


def fits_fp16(v):
    v = np.asarray(v, dtype=np.int64)
    return bool((np.float16(v).astype(np.int64) == v).all())


def fits_bf16(v):
    f = np.asarray(v, dtype=np.float32)
    return bool(((f.view(np.uint32) & 0xFFFF) == 0).all() and (f.astype(np.int64) == v).all())


def fits_fp4(v):
    return all(float(x) in E2M1 for x in np.unique(v))


def fits_s8(v):
    return bool((np.asarray(v) >= -128).all() and (np.asarray(v) <= 127).all())


def winograd_operands(X, Y):
    a0, a1 = X[:, 0::2], X[:, 1::2]
    b0, b1 = Y[0::2, :], Y[1::2, :]
    s = a0[:, :, None] + b1[None, :, :]
    t = a1[:, :, None] + b0[None, :, :]
    jx = (a0 * a1).sum(axis=1)
    jy = (b0 * b1).sum(axis=0)
    return s, t, jx, jy


def run_schemes(X, Y):
    """Each scheme's value of X @ Y mod 2^32, with its per-pair product list (format, count) and format checks."""
    T = (X @ Y) % MOD
    s, t, jx, jy = winograd_operands(X, Y)
    junk = jx[:, None] + jy[None, :]
    out = {}

    out["plain_int8"] = ((X @ Y) % MOD, {"int8": 2}, fits_s8(X) and fits_s8(Y))

    ok = fits_fp16(s) and fits_fp16(t)
    w = (np.float32(s) * np.float32(t)).astype(np.int64)
    out["fp16_winograd"] = (((w.sum(axis=1)) - junk) % MOD, {"fp16": 1}, ok and bool((w == s * t).all()))
    ok = fits_bf16(s) and fits_bf16(t)
    out["bf16_winograd"] = (((s * t).sum(axis=1) - junk) % MOD, {"bf16": 1}, ok)

    u, v = wrap8(s), wrap8(t)
    c1, c2 = (s - u) // 256, (t - v) // 256
    p1, p2 = u + 128 * c1, v + 128 * c2
    ok = fits_s8(u) and fits_s8(v) and fits_s8(p1) and fits_s8(p2) and fits_fp4(c1) and fits_fp4(c2)
    val = 2 * (p1 * p2) - u * v + 32768 * (c1 * c2)
    out["wrapped_winograd_carry_adjusted"] = ((val.sum(axis=1) - junk) % MOD, {"int8": 2, "fp4": 1}, ok)
    out["control_without_carry_product"] = (((2 * (p1 * p2) - u * v).sum(axis=1) - junk) % MOD, {"int8": 2}, ok)

    a, b = -((-s) // 2), s // 2
    c, d = -((-t) // 2), t // 2
    s0, t0 = s - 2 * b, t - 2 * d
    ok = all(fits_s8(z) for z in (a, b, c, d)) and fits_fp4(s0) and fits_fp4(t0)
    val = 2 * a * c + 2 * b * d - s0 * t0
    out["ceil_floor_halves"] = ((val.sum(axis=1) - junk) % MOD, {"int8": 2, "fp4": 1}, ok)

    val = u * v + 256 * (u * c2 + c1 * v)
    out["wrapped_winograd_r2"] = ((val.sum(axis=1) + 65536 * (c1 * c2).sum(axis=1) - junk) % MOD,
                                  {"int8": 3, "cuda": 1}, fits_s8(u) and fits_s8(v))

    return T, out


def strassen_fp16(A, B, levels, count):
    """Strassen's 7 products, `levels` deep, over FP16 Winograd at the base. Each product is computed once and its
    value added to several cells, which Theorem B's per-step count pays once per cell (so this is outside it)."""
    if levels == 0:
        s, t, jx, jy = winograd_operands(A, B)
        ok = fits_fp16(s) and fits_fp16(t) and fits_fp16(A) and fits_fp16(B)
        w = (np.float32(s) * np.float32(t)).astype(np.int64)
        ok &= bool((w == s * t).all())
        count["fp16"] += s.size + jx.size + jy.size
        count["ok"] &= ok
        return w.sum(axis=1) - jx[:, None] - jy[None, :]
    h = A.shape[0] // 2
    a11, a12, a21, a22 = A[:h, :h], A[:h, h:], A[h:, :h], A[h:, h:]
    b11, b12, b21, b22 = B[:h, :h], B[:h, h:], B[h:, :h], B[h:, h:]
    r = lambda x, y: strassen_fp16(x, y, levels - 1, count)
    m1, m2, m3 = r(a11 + a22, b11 + b22), r(a21 + a22, b11), r(a11, b12 - b22)
    m4, m5, m6, m7 = r(a22, b21 - b11), r(a11 + a12, b22), r(a21 - a11, b11 + b12), r(a12 - a22, b21 + b22)
    return np.block([[m1 + m4 - m5 + m7, m3 + m5], [m2 + m4, m1 - m2 + m3 + m6]])


def mode_shared(args):
    rng = np.random.default_rng(args.seed)
    rows = []
    for levels in range(0, 5):
        n = args.n
        res = {"levels": levels, "exact": True, "fp16_ok": True}
        for trial in range(args.count + 2):
            if trial == 0:
                X, Y = np.full((n, n), -128), np.full((n, n), -128)
            elif trial == 1:
                X, Y = np.full((n, n), -128), np.full((n, n), 127)
            else:
                X, Y = rng.integers(-128, 128, (n, n)), rng.integers(-128, 128, (n, n))
            count = {"fp16": 0, "ok": True}
            C = strassen_fp16(X.astype(np.int64), Y.astype(np.int64), levels, count)
            res["exact"] &= bool(((C - X @ Y) % MOD == 0).all())
            res["fp16_ok"] &= count["ok"]
        res["worst_winograd_operand"] = 128 * 2 * 2 ** levels
        res["fp16_ok"] &= res["worst_winograd_operand"] <= 2048
        res["fp16_products"] = count["fp16"]
        res["slots"] = 2 * count["fp16"]
        res["ratio"] = res["slots"] / n ** 3
        res["mixed_ratio"] = 2 * (7 ** levels) * ((n // 2 ** levels) ** 3 // 2) / n ** 3
        rows.append(res)
    best = min((r for r in rows if r["exact"] and r["fp16_ok"]), key=lambda r: r["mixed_ratio"])
    return {"mode": "shared", "n": args.n, "rows": rows, "best_levels": best["levels"],
            "best_mixed_ratio": best["mixed_ratio"], "failures": [] if rows[3]["fp16_ok"] and not rows[4]["fp16_ok"] else ["range"]}


PRICE = {"int8": 1.0, "fp16": 2.0, "bf16": 2.0, "fp4": 0.5, "e4m3": 1.0, "cuda": 0.0}


def slot_cost(name, per_pair, shape):
    n, k, n2 = shape
    pairs = n * n2 * (k // 2)
    cost = pairs * sum(PRICE[f] * c for f, c in per_pair.items())
    if name != "plain_int8":
        cost += (n + n2) * (k // 2)
    return cost


def mode_schemes(args):
    rng = np.random.default_rng(args.seed)
    rows, fails = {}, []
    tests = []
    for _ in range(args.count):
        tests.append((rng.integers(-128, 128, (args.n, args.n)), rng.integers(-128, 128, (args.n, args.n))))
    for xv, yv in itertools.product((-128, 127, -1, 0), repeat=2):
        tests.append((np.full((4, 4), xv), np.full((4, 4), yv)))
    tests.append((np.full((4, 4), -128) * np.array([1, 1, 0, 0]), np.full((4, 4), -128)))
    for X, Y in tests:
        X, Y = X.astype(np.int64), Y.astype(np.int64)
        T, out = run_schemes(X, Y)
        for name, (val, per_pair, fmt_ok) in out.items():
            r = rows.setdefault(name, {"per_pair": per_pair, "exact": True, "formats_ok": True})
            r["exact"] &= bool((val % MOD == T).all())
            r["formats_ok"] &= fmt_ok
    for name, r in rows.items():
        r["slots_222"] = slot_cost(name, r["per_pair"], (2, 2, 2))
        r["slots_224"] = slot_cost(name, r["per_pair"], (2, 2, 4))
        r["ratio_222"] = r["slots_222"] / 8
        r["ratio_224"] = r["slots_224"] / 16
        r["per_mac_asymptotic"] = sum(PRICE[f] * c for f, c in r["per_pair"].items()) / 2
        if name.startswith("control"):
            if r["exact"]:
                fails.append(name)
        elif not (r["exact"] and r["formats_ok"]):
            fails.append(name)
    return {"mode": "schemes", "floor_222": 8, "floor_224": 16, "tests": len(tests), "schemes": rows,
            "failures": fails}


# ---- toy search -------------------------------------------------------------------------------------------

def toy_space(w):
    lo, hi = -(2 ** (w - 1)), 2 ** (w - 1) - 1
    vals = np.arange(lo, hi + 1)
    grid = np.array(list(itertools.product(vals, repeat=4)), dtype=np.int64)
    return grid, lo, hi


def toy_operands(w, grid, support):
    lo, hi = -(2 ** (w - 1)), 2 ** (w - 1) - 1
    nar = 1 if w == 2 else 2
    ops = {}
    for coeffs in itertools.product((-1, 0, 1), repeat=4):
        if not any(coeffs) or sum(c != 0 for c in coeffs) > support:
            continue
        f = grid @ np.array(coeffs)
        wr = ((f - lo) % 2 ** w) + lo
        forms = {"id": f, "wrap": wr, "carry": (f - wr) // 2 ** w, "half": f >> 1, "ceil": -((-f) >> 1),
                 "low": f & 1}
        for tname, g in forms.items():
            if not g.any():
                continue
            key = tuple(g)
            if key in ops:
                continue
            fmts = []
            if g.min() >= -nar and g.max() <= nar:
                fmts.append("narrow")
            if g.min() >= lo and g.max() <= hi:
                fmts.append("slot")
            if abs(g).max() <= 2 ** (w + 3):
                fmts.append("wide")
            if fmts:
                ox = any(coeffs[:2])
                oy = any(coeffs[2:])
                ops[key] = {"v": np.array(key), "fmts": fmts, "desc": f"{tname}{coeffs}", "x": ox, "y": oy}
    return list(ops.values())


TOY_PRICE = {"narrow": 0.5, "slot": 1.0, "wide": 2.0}


def interaction(h, gx):
    m = h.reshape(gx, gx)
    z = gx // 2
    return (m - m[:, z][:, None] - m[z, :][None, :] + m[z, z]).ravel()


def toy_products(w, support):
    grid, lo, hi = toy_space(w)
    gx = (hi - lo + 1) ** 2
    ops = toy_operands(w, grid, support)
    prods = {}
    for i in range(len(ops)):
        for j in range(i, len(ops)):
            a, b = ops[i], ops[j]
            common = [f for f in a["fmts"] if f in b["fmts"]]
            if not common:
                continue
            cost = min(TOY_PRICE[f] for f in common)
            q = interaction(a["v"] * b["v"], gx)
            if not q.any():
                continue
            key = tuple(q) if q[np.nonzero(q)[0][0]] > 0 else tuple(-q)
            if key not in prods or prods[key]["cost"] > cost:
                prods[key] = {"q": np.array(key), "cost": cost, "desc": f"{a['desc']}*{b['desc']}"}
    target = interaction(grid[:, 0] * grid[:, 2] + grid[:, 1] * grid[:, 3], gx)
    return grid, list(prods.values()), target, len(ops)


def rref_mod(M):
    M = M.copy() % P
    piv = []
    r = 0
    for c in range(M.shape[1]):
        if r == M.shape[0]:
            break
        nz = np.nonzero(M[r:, c])[0]
        if not len(nz):
            continue
        k = r + nz[0]
        M[[r, k]] = M[[k, r]]
        M[r] = (M[r] * pow(int(M[r, c]), P - 2, P)) % P
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] = (M[i] - M[i, c] * M[r]) % P
        piv.append(c)
        r += 1
    return M[:r], piv


def reduce_all(V, B, piv):
    V = V.copy() % P
    for row, c in zip(B, piv):
        V = (V - V[:, c:c + 1] * row[None, :]) % P
    return V


def integral_combo(vecs, target):
    """An exact rational solution of sum a_i vecs_i = target (vecs independent), or None."""
    A = [[Fraction(int(x)) for x in col] for col in zip(*vecs)]
    b = [Fraction(int(x)) for x in target]
    m = len(vecs)
    rows = [A[i] + [b[i]] for i in range(len(A))]
    sol, r = [None] * m, 0
    pivs = []
    for c in range(m):
        k = next((i for i in range(r, len(rows)) if rows[i][c] != 0), None)
        if k is None:
            return None
        rows[r], rows[k] = rows[k], rows[r]
        rows[r] = [x / rows[r][c] for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] != 0:
                f = rows[i][c]
                rows[i] = [x - f * y for x, y in zip(rows[i], rows[r])]
        pivs.append(c)
        r += 1
    if any(row[m] != 0 for row in rows[r:]):
        return None
    for i, c in enumerate(pivs):
        sol[c] = rows[i][m]
    return sol


def mode_toy(args):
    w = args.w
    grid, prods, target, nops = toy_products(w, args.support)
    rng = np.random.default_rng(args.seed)
    sketch = rng.integers(0, P, (len(target), args.sketch))
    Qs = (np.array([p["q"] for p in prods]) @ sketch) % P
    ts = (target @ sketch) % P
    cost = np.array([p["cost"] for p in prods])
    by = {c: np.nonzero(cost == c)[0] for c in (0.5, 1.0, 2.0)}
    hits = []

    def exact_hit(idx):
        vecs = [prods[i]["q"] for i in idx]
        sol = integral_combo(vecs, target)
        if sol is None:
            return None
        return {"cost": float(sum(prods[i]["cost"] for i in idx)), "products": [prods[i]["desc"] for i in idx],
                "coefficients": [str(x) for x in sol], "odd_denominators": all(x.denominator % 2 for x in sol)}

    def in_span_with(base_idx, cand):
        B, piv = rref_mod(np.vstack([ts[None, :]] + [Qs[i][None, :] for i in base_idx]))
        Bq, pq = rref_mod(np.vstack([Qs[i][None, :] for i in base_idx])) if base_idx else (np.zeros((0, ts.size), np.int64), [])
        if base_idx and not reduce_all(ts[None, :], Bq, pq).any():
            return []
        if len(B) == len(base_idx):
            return []
        R = reduce_all(Qs[cand], B, piv)
        Rb = reduce_all(Qs[cand], Bq, pq) if base_idx else Qs[cand] % P
        good = ~R.any(axis=1) & Rb.any(axis=1)
        return list(cand[good])

    for i in range(len(prods)):
        if not reduce_all(ts[None, :], *rref_mod(Qs[i][None, :])).any():
            h = exact_hit([i])
            if h:
                hits.append(h)
    levels = [("narrow+narrow", by[0.5], by[0.5]), ("slot+narrow", by[1.0], by[0.5])]
    for name, first, second in levels:
        for i in first:
            for j in in_span_with([i], second):
                if name == "narrow+narrow" and j <= i:
                    continue
                h = exact_hit([i, j])
                if h:
                    hits.append(h)
    if args.triples:
        nar = by[0.5]
        for a in range(args.shard[0], len(nar), args.shard[1]):
            for b in range(a + 1, len(nar)):
                for k in in_span_with([nar[a], nar[b]], nar[b + 1:]):
                    h = exact_hit([nar[a], nar[b], k])
                    if h:
                        hits.append(h)
    integral = [h for h in hits if h["odd_denominators"]]
    below = [h for h in integral if h["cost"] < 2.0]
    gx = int(round(len(target) ** 0.5))
    plain = integral_combo([interaction(grid[:, 0] * grid[:, 2], gx), interaction(grid[:, 1] * grid[:, 3], gx)], target)
    s_, t_ = grid[:, 0] + grid[:, 3], grid[:, 1] + grid[:, 2]
    known = [-((-s_) // 2) * -((-t_) // 2), (s_ >> 1) * (t_ >> 1), (s_ & 1) * (t_ & 1)]
    known = [interaction(k, gx) for k in known]
    keys = {tuple(p["q"]): p["cost"] for p in prods}
    norm = lambda q: tuple(q) if q[np.nonzero(q)[0][0]] > 0 else tuple(-q)
    halves = {"coefficients": [str(x) for x in integral_combo(known, target) or []],
              "costs_in_space": [keys.get(norm(k)) for k in known]}
    wide = [exact_hit([i]) for i in by[2.0]]
    wide = [h for h in wide if h and h["odd_denominators"]]
    return {"mode": "toy", "w": w, "support": args.support, "operands": nops, "mixed_products": len(prods),
            "by_cost": {str(c): int(len(v)) for c, v in by.items()}, "floor": 2.0, "triples": args.triples, "shard": args.shard,
            "rational_hits_below_floor": [h for h in hits if h["cost"] < 2.0][:20],
            "integral_hits_below_floor": below[:20], "min_cost_below_floor": min((h["cost"] for h in below), default=None),
            "plain_two_slots": [str(x) for x in plain] if plain else None,
            "ceil_floor_halves_2.5": halves, "single_wide_hits": wide[:5], "failures": [] if plain and None not in halves["costs_in_space"] else ["known scheme missing"]}


def mode_ring(args):
    """Single products modulo 2^W: the cheapest one product whose multiple equals the target's interaction mod 2^W."""
    w, W = args.w, args.ring
    grid, prods, target, nops = toy_products(w, args.support)
    m = 2 ** W
    t = target % m
    Qm = np.array([p["q"] for p in prods]) % m
    cost = np.array([p["cost"] for p in prods])
    best = {}
    for a in range(1, m):
        ok = ~(((a * Qm) - t) % m).any(axis=1)
        for i in np.nonzero(ok)[0]:
            c = float(cost[i])
            if c not in best:
                best[c] = {"product": prods[i]["desc"], "coefficient": a}
    return {"mode": "ring", "w": w, "W": W, "single_product_hits_by_cost": {str(k): v for k, v in sorted(best.items())},
            "min_single_cost": min(best) if best else None, "failures": []}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["schemes", "toy", "ring", "shared"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--w", type=int, default=2)
    ap.add_argument("--support", type=int, default=2)
    ap.add_argument("--ring", type=int, default=2)
    ap.add_argument("--sketch", type=int, default=24)
    ap.add_argument("--triples", action="store_true")
    ap.add_argument("--shard", type=lambda s: [int(x) for x in s.split("/")], default=[0, 1],
                    help="I/N: this job takes the narrow triples whose first index is I mod N")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = {"schemes": mode_schemes, "toy": mode_toy, "ring": mode_ring, "shared": mode_shared}[args.mode](args)
    print(json.dumps(out))
    return 1 if out["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
