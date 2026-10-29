"""HASH's FOLD/TENSOR lane, the lever: a v3 word form whose step-0 value carries a wide rank-ρ coin term, and the
write count it gives the tensor-core fold (`hash_tensor.py`).

    python3 hash_lever.py LEAD.json [--verdict PATH] [--json OUT]

The word form. At zero data the lead's words are W_t(i, j) = s(i, j) + r_t(i) + q_t(p(i), j): a static step-0 value s
(the lead's tag), a row walk r_t (the b bits) and a column walk q_t a row class p (the coin lanes). The lever adds
L(i, j) = Σ_{l<ρ} U_l(i)·V_l(j) to s, with U uniform in [-2^15, 2^15) and V in [-2^14, 2^14), drawn by live coins after
the operands' commitment. Honest sets the accumulator to L before the call's first k32 step and subtracts it after the
last, exactly mod 2^32: 2ρ IMADs a word a chunk.

The floor. Cut the fold's product bytes(W_t)·R into row-planes: one row i, one step t and one byte position b, 256
cells, whose product is 8 values. ByteEntry, the premise this count reads: a program that outputs the sealed state
obtains every row-plane's product from one of the routes below, each a correction Δ added to products it already
holds, and each correction cell is a product with R that it forms:
- scratch: Δ is the row-plane itself;
- prev and partner: Δ is the row-plane less one other row-plane, of row i at t - 1, or of another row at t or t - 1;
- incr: Δ is the row-plane less row i's at t - 1 less the bytes of step t's increment, which are one of four patterns a
  step (a class and a b bit), so the carries alone (`mma_pairsum`'s route for a fold linear in its words);
- decomp: Δ is the row-plane less the bytes of s, r_t and q_t, each shared (s by the chunk, q_t by a class, r_t by its
  row), so the carries between the three (COUNTER's `decomp`);
- walk: Δ is the row-plane less the bytes of s and of the walk r_t + q_t, the walk's own row-plane priced by its own
  cheapest scratch, prev or partner route.
Under H32 a register write forms at most 32 products with s8 R entries, so a nonzero Δ cell (8 products, 16 past
|Δ| = 128) costs at least 8/32 of a register. Under DistinctWrites a write carries at least 32 operand bits, so the cell
costs at least its width over 32: 1 bit for |Δ| = 1, 4 to 8, 8 to 128, 16 past it. Zero cells cost nothing, which
grants the prover free gathering. The state adds Z (16 bytes) and Z·Q (128 products) a row a step: 8/BN registers a
word. The floor is the mean over row-planes of the cheapest route, summed over b.

- D. The lever: every byte plane full rank, no row close to another, and the words bit-exact with the lead's plus L.
- F. The floor at K 2,048 and 8,192, ρ = 2 and 1, against the lead's tag form, with the winning route per byte.
- X. Attacks priced as programs: the class-sharing prover (`hash_tensor.pack_share`) on lever words, `pack`'s lane
  widths, and each route's natural-layout cost (every cell of a nonzero fragment pays, and Δ's registers are written).
- H. Honest's extra writes and the lever's coin bytes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exact_kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_kit.checks import Checks
from hash_tensor import (BN, M32, ROWS, STEPS, cell_bytes, chain, draw_coins, lead_words, pack_share, rank_mod_p,
                         seal)

U_BITS, V_BITS = 16, 15
STATE = 8 / BN
ROUTES = ("scratch", "prev", "partner", "incr", "decomp", "walk")


def lever_words(rng, m: int, n: int, S: int, rho: int) -> dict:
    """The lead's zero-data words at the rot layout with its step-0 tag, plus L = U·V of rank rho (0: none)."""
    d = lead_words(rng, "rot", m, n, S, "tag")
    U = rng.integers(-(1 << (U_BITS - 1)), 1 << (U_BITS - 1), (m, rho), dtype=np.int64)
    V = rng.integers(-(1 << (V_BITS - 1)), 1 << (V_BITS - 1), (rho, n), dtype=np.int64)
    L = U @ V
    d["lead_wint"] = d["wint"]
    d["L"] = L
    d["d0"] = d["d0"] + L
    d["wint"] = d["wint"] + L[None]
    d["w"] = d["wint"] & M32
    d["row"] = np.cumsum(d["b"], axis=0)
    d["colw"] = np.cumsum(d["col"], axis=0)
    return d


def digit_cost(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    bits = np.where(a <= 1, 1, np.where(a <= 8, 4, 8))
    return np.where(a == 0, 0, 8 + bits) / 32


def cell_cost(delta: np.ndarray) -> np.ndarray:
    """Registers a cell, at least, to add Δ·R for one cell: Δ = d0 + 256·d1 with d0 in [-128, 128), and each nonzero
    digit costs 8 products (with R, or 256·R mod 2^16) and its operand width (1, 4 or 8 bits), over 32."""
    d1 = (np.asarray(delta) + 128) // 256
    return digit_cost(delta - 256 * d1) + digit_cost(d1)


def plane_cost(delta: np.ndarray) -> np.ndarray:
    return cell_cost(delta).sum(axis=-1)


def best_other(x: np.ndarray, y: np.ndarray, exclude_self: bool) -> np.ndarray:
    """For each row of x (m, n), the cheapest Δ against any row of y (m, n)."""
    c = plane_cost(x[:, None, :] - y[None, :, :])
    if exclude_self:
        c = c + np.where(np.eye(len(x), dtype=bool), np.inf, 0)
    return c.min(axis=1)


def route_costs(d: dict, steps, fresh: bool = False) -> dict:
    """Each route's cost (registers a row-plane) for every (t, i, b) in `steps`, and the shared parts' amortized cost.
    With `fresh` R a step, no product carries from one step to the next: prev and incr are gone, partners are the
    step's own rows, and decomp and walk pay the static part's product every step."""
    wint = d["wint"]
    S, m, n = wint.shape
    Pi = cell_bytes(wint & M32)
    Bs = cell_bytes(d["d0"] & M32)
    Br = cell_bytes(d["row"] & M32)
    Bq = cell_bytes(d["colw"] & M32)
    walk = d["row"][:, :, None] + d["colw"][:, d["cls"], :]
    Bw = cell_bytes(walk & M32)
    Bc = cell_bytes(d["c"] & M32)
    per_class = m // (int(d["cls"].max()) + 1)
    out = {r: [] for r in ROUTES}
    for t in steps:
        for b in range(4):
            x = Pi[t, :, :, b]
            static = plane_cost(Bs[:, :, b]) if fresh else plane_cost(Bs[:, :, b]).mean() / S
            out["scratch"].append(plane_cost(x))
            if t > 0 and not fresh:
                xp = Pi[t - 1, :, :, b]
                out["prev"].append(plane_cost(x - xp))
                out["partner"].append(np.minimum(best_other(x, x, True), best_other(x, xp, False)))
                out["incr"].append(plane_cost(x - xp - Bc[t, :, :, b]) + plane_cost(Bc[t, :, :, b]) / per_class / 2)
            else:
                for r in ("prev", "incr"):
                    out[r].append(np.full(m, np.inf))
                out["partner"].append(best_other(x, x, True))
            shared = static + plane_cost(Bq[t, d["cls"], :, b]) / per_class + 8 / 32
            out["decomp"].append(plane_cost(x - Bs[:, :, b] - Br[t, :, None, b] - Bq[t, d["cls"], :, b]) + shared)
            wx = Bw[t, :, :, b]
            wcost = plane_cost(wx)
            if t > 0 and not fresh:
                wcost = np.minimum(wcost, plane_cost(wx - Bw[t - 1, :, :, b]))
            wcost = np.minimum(wcost, best_other(wx, wx, True))
            out["walk"].append(plane_cost(x - Bs[:, :, b] - wx) + wcost + static)
    return {r: np.array(v).reshape(len(steps), 4, m) for r, v in out.items()}


def floor(d: dict, steps, fresh: bool = False) -> dict:
    """The per-byte floor in registers a word: the mean over row-planes of the cheapest route, and each route's share."""
    rc = route_costs(d, steps, fresh)
    stack = np.stack([rc[r] for r in ROUTES])
    best = stack.min(axis=0)
    win = stack.argmin(axis=0)
    n = d["wint"].shape[2]
    per_byte = (best.mean(axis=(0, 2)) / n).tolist()
    shares = [{r: float(np.mean(win[:, b, :] == k)) for k, r in enumerate(ROUTES)} for b in range(4)]
    means = {r: [float(rc[r][:, b][np.isfinite(rc[r][:, b])].mean() / n) if np.isfinite(rc[r][:, b]).any()
                 else float("inf") for b in range(4)] for r in ROUTES}
    return {"per_byte": per_byte, "total": sum(per_byte) + STATE, "winners": shares, "route_per_byte": means}


def part_d(checks: Checks, rng) -> dict:
    S = STEPS["k2048"]
    d = lever_words(rng, ROWS, BN, S, 2)
    t = S - 1
    cb = cell_bytes(d["w"][t])
    ranks = [rank_mod_p(cb[:, :, b]) for b in range(4)]
    gap = np.abs(d["wint"][t][:, None, :] - d["wint"][t][None, :, :]).astype(float)
    gap += np.where(np.eye(ROWS, dtype=bool)[..., None], np.inf, 0)
    med = float(np.log2(np.median(gap.min(axis=1))))
    exact = bool(np.array_equal(d["wint"] - d["lead_wint"], np.broadcast_to(d["L"], d["wint"].shape)))
    span = float(np.log2(np.abs(d["wint"]).max()))
    R, Q = draw_coins(rng, BN)
    z = chain(d["w"], R, Q)
    out = {"ranks": ranks, "nearest_row_log2": med, "exact": exact, "span_log2": span, "seal": seal(z)[:16]}
    checks.check("D: the lever's words are the lead's zero-data words plus L = U·V (rank 2, 16- and 15-bit coins), "
                 "exactly; every byte plane is full rank over the tile, a cell's nearest other row is past 2^16 (so "
                 "bytes 0-2 differ), and the words stay inside s32",
                 exact and min(ranks) == ROWS and med > 16 and span < 31,
                 f"ranks {ranks}, nearest row 2^{med:.1f} (median), |W| < 2^{span:.2f}")
    return out


def part_f(checks: Checks, rng) -> dict:
    out = {}
    k2, k8 = STEPS["k2048"], STEPS["k8192"]
    cases = (("k2048/rho2", k2, 2, range(k2), False), ("k2048/rho1", k2, 1, range(k2), False),
             ("k2048/lead", k2, 0, range(k2), False), ("k8192/rho2", k8, 2, range(0, k8, 4), False),
             ("k2048/rho2/fresh", k2, 2, range(k2), True), ("k2048/rho1/fresh", k2, 1, range(k2), True),
             ("k2048/lead/fresh", k2, 0, range(k2), True), ("k8192/rho2/fresh", k8, 2, range(0, k8, 4), True))
    for name, S, rho, steps, fresh in cases:
        d = lever_words(rng, ROWS, BN, S, rho)
        out[name] = floor(d, list(steps), fresh)
    return out


def part_x(checks: Checks, rng) -> dict:
    out = {}
    S = STEPS["k2048"]
    d = lever_words(rng, ROWS, BN, S, 2)
    R, Q = draw_coins(rng, BN)
    honest = chain(d["w"], R, Q)
    z, model = pack_share(d, R, Q)
    out["share"] = {"same": bool(np.array_equal(z, honest)), "free": model["free"], "lanes_ok": model["lanes_ok"]}
    span = int((d["wint"].max(axis=0) - d["wint"].min(axis=0)).max())
    rest = d["wint"] - d["L"][None]
    split = cell_bytes(d["w"]) != cell_bytes(d["L"] & M32)[None] + cell_bytes(rest & M32)
    out["pack"] = {"span_log2": float(np.log2(span)), "abs_log2": float(np.log2(np.abs(d["wint"]).max())),
                   "bytes_split": float(split.any(axis=-1).mean())}
    Pi = cell_bytes(d["w"])
    Bc = cell_bytes(d["c"] & M32)
    Bs, Br, Bq = cell_bytes(d["d0"] & M32), cell_bytes(d["row"] & M32), cell_bytes(d["colw"] & M32)
    nat = {}
    frag = (16, 32)
    for name, delta in (("scratch", Pi[1:]),
                        ("prev", Pi[1:] - Pi[:-1]),
                        ("incr", Pi[1:] - Pi[:-1] - Bc[1:]),
                        ("decomp", Pi[1:] - Bs[None] - Br[1:, :, None] - Bq[1:][:, d["cls"]])):
        per = []
        for b in range(4):
            x = delta[..., b]
            Sx = x.shape[0]
            f = x.reshape(Sx, ROWS // frag[0], frag[0], BN // frag[1], frag[1])
            live = (f != 0).any(axis=(2, 4))
            mma = 0.25 * (2 if np.abs(x).max() > 128 else 1) * live.mean()
            regs = x.reshape(Sx, ROWS, BN // 4, 4)
            moved = np.concatenate([regs[:1] != 0, regs[1:] != regs[:-1]]).any(axis=3).mean() / 4
            per.append(float(mma + moved))
        nat[name] = per
    out["natural"] = {k: {"per_byte": v, "total": sum(v) + STATE} for k, v in nat.items()}
    best = [min(nat[k][b] for k in nat) for b in range(4)]
    out["natural_best"] = {"per_byte": best, "total": sum(best) + STATE}
    return out


def part_h(checks: Checks, lead: dict) -> dict:
    """The lever: ρ IMADs set the accumulator to L and ρ remove it, once a chunk. A fresh R a step is 4·BN·8 bytes a
    step a call, read by wgmma from shared memory (no register writes) or by ldmatrix for mma.sync (2 registers a
    thread a MMA, shared by the warp's m16 fragments)."""
    out = {"fresh_r_bytes_a_step": 4 * BN * 8,
           "fresh_r_bytes_a_call": {k: 4 * BN * 8 * s for k, s in STEPS.items()},
           "fresh_r_mma_sync_writes": {f"{f} m16 a warp": 0.5 / f for f in (1, 4)}}
    for rho in (1, 2):
        writes = {k: 2 * rho / s for k, s in STEPS.items()}
        out[f"rho{rho}"] = {"writes_a_word": writes,
                            "coin_bytes_a_row_and_column": rho * (U_BITS + V_BITS) / 8,
                            "h_add": {pt: lead["points"][pt]["r"] * writes["k2048" if pt != "t8192" else "k8192"]
                                      for pt in lead["points"]}}
    return out


def main() -> int:
    checks = Checks()
    args = checks.args
    lead = json.loads(Path(args[0]).read_text())
    rng = np.random.default_rng(31)
    stock = {pt: 1 / v["r"] for pt, v in lead["points"].items()}
    dd = part_d(checks, rng)
    f = part_f(checks, rng)
    x = part_x(checks, rng)
    h = part_h(checks, lead)
    s8192 = stock["t8192"]
    for name, v in f.items():
        v["over_stock"] = v["total"] / s8192
    lv, lf = f["k2048/rho2"], f["k2048/rho2/fresh"]
    fixed = [k for k in f if not k.endswith("fresh")]
    fresh = [k for k in f if k.endswith("fresh")]

    def show(keys):
        return "; ".join(f"{k}: {f[k]['total']:.3f} registers a word ({f[k]['over_stock']:.2f}x stock), per byte "
                         + ", ".join(f"{p:.3f}" for p in f[k]["per_byte"]) for k in keys)

    checks.check("F: with one R a call the lever doesn't lift the count to stock: a row-plane's product is last step's "
                 "plus the carries of adding the step's increment (four byte patterns a step), about a quarter "
                 "register a byte at bytes 0-1, and bytes 2-3 barely move",
                 all(f[k]["over_stock"] < 1 for k in fixed), show(fixed))
    checks.check("F: with a fresh R each step (no product carries between steps) the lever's floor clears stock at "
                 "rank 1 and 2 and both K, and the lead's tag form without the lever still doesn't",
                 all(f[k]["over_stock"] > 1 for k in fresh if "lead" not in k) and f["k2048/lead/fresh"]["over_stock"] < 1,
                 show(fresh))
    for name, v in (("one R", lv), ("fresh R", lf)):
        for b in range(4):
            checks.check(f"F: byte {b}'s winning routes (lever, rank 2, K 2,048, {name})", True,
                         ", ".join(f"{r} {s:.1%}" for r, s in v["winners"][b].items() if s > 0)
                         + "; route means " + ", ".join(f"{r} {v['route_per_byte'][r][b]:.3f}" for r in ROUTES
                                                        if np.isfinite(v['route_per_byte'][r][b])))
    checks.check("X: the class-sharing prover still matches honest on lever words, and pays more than honest",
                 x["share"]["same"] and x["share"]["lanes_ok"] and x["share"]["free"] > 2.047,
                 f"{x['share']['free']:.3f} registers a word")
    checks.check("X: pack's lanes could hold W - L (a cell moves 2^18 over the call), but the MMA reads bytes(W), and "
                 "bytes(L + x) is not bytes(L) + bytes(x) on most cells: that is the decomp and walk routes, priced in F",
                 x["pack"]["span_log2"] < 20 and x["pack"]["bytes_split"] > 0.5,
                 f"a cell's span over the call 2^{x['pack']['span_log2']:.2f}; bytes split on "
                 f"{x['pack']['bytes_split']:.1%} of (cell, step)")
    checks.check("X: each route as a program in the natural layout (a fragment with any nonzero cell pays its MMA, Δ's "
                 "registers are written when they move), registers a word",
                 True, "; ".join(f"{k}: {v['total']:.3f} (" + ", ".join(f"{p:.3f}" for p in v["per_byte"]) + ")"
                                 for k, v in x["natural"].items())
                 + f"; best a byte: {x['natural_best']['total']:.3f}")
    checks.check("H: honest's extra writes for the lever (2ρ IMADs a word a chunk, set and removed), its coins, and a "
                 "fresh R a step",
                 True, "; ".join(f"{k}: " + ", ".join(f"{s} {w:.4f}" for s, w in v["writes_a_word"].items())
                                 + " registers a word (h +" + ", ".join(f"{p} {a:.4f}" for p, a in v["h_add"].items())
                                 + f"), {v['coin_bytes_a_row_and_column']:.2f} coin bytes a row and a column"
                                 for k, v in h.items() if k.startswith("rho"))
                 + f"; fresh R: {h['fresh_r_bytes_a_step']} bytes a step, "
                 + ", ".join(f"{k} {v / 1024:.0f} KiB" for k, v in h["fresh_r_bytes_a_call"].items())
                 + " a call; mma.sync's B loads " + ", ".join(f"{k} {v:.3f}" for k, v in
                                                              h["fresh_r_mma_sync_writes"].items())
                 + " registers a word, wgmma's none")
    if "--json" in args:
        Path(args[args.index("--json") + 1]).write_text(json.dumps(
            {"lever": dd, "floor": f, "attacks": x, "honest": h, "stock": stock}, indent=1, sort_keys=True,
            default=float) + "\n")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
