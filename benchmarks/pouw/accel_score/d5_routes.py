"""PD3-4: D5's `tile_job` hull (BET-ALGO-LAB, public_best.py at 126ba960ca) plus two routes it lacks at A = B = 0:
PD2-1's O-route (the dense words, O = bc^T at `bit` from the 1-bit dither, no peels) and the zero-high-digit peel
(Y_W's high digit is 0, so the peels cost 3 digits of inner L' instead of 4).

    PUBLIC_BEST_DIR=<checkout of cursor/compute-research-accel-bet-algo-lab-ea24>/benchmarks/pouw/theory \
        python3 benchmarks/pouw/accel_score/d5_routes.py
"""
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.environ.get("PUBLIC_BEST_DIR", str(Path(__file__).resolve().parents[1] / "theory")))
import public_best as pb  # noqa: E402


def job(card, t, Lp, K, n, bit, depth, v, hash_scale, o_route=True, zhd=True):
    L, zleaf, sha, peels, dense, honest = pb._tile_honest(card, t, Lp, K, n, v, hash_scale)
    dense_m = pb.Method("dense", dense, items=(peels,))
    best = pb.price_methods(card, [dense_m], L, n, pb.TOKENS, beside=True, items=(sha,))
    mma = dict(smem=1 / card.honest_tile[0] + 1 / card.honest_tile[1], rf=card.mma_rf, issue=card.mma_issue)

    def bc_fixed(x):
        f = {r: y * x for r, y in mma.items()}
        f["tensor"] = x / card.s8
        if depth:
            ops = (pb.assembly(pb.STRASSEN, depth) + 1) / L
            for r, y in (("alu", ops), ("issue", ops), ("rf", 3 * ops), ("vec", ops)):
                f[r] = f.get(r, 0.0) + y
        return f

    tiles = pb.Method("tiles", [replace(zleaf, weight=1 / t)], fixed=bc_fixed(bit + 1 / t))
    ms = [tiles, dense_m]
    if o_route:
        ms.append(pb.Method("o_route", dense, fixed=bc_fixed(bit)))
    if zhd:
        ms.append(pb.Method("zhd", dense, items=((peels[0], peels[1] * 3 / 4, peels[2]),)))
    hull = pb.price_methods(card, ms, L, n, pb.TOKENS, beside=True, items=(sha,))
    return dict(c_opt=min(1.0, hull.c / best.c), eps_alg=1 - best.c / honest.c, mix=hull.mix, best=best,
                honest=honest)


def job_at(card, t, Lp, K, n, bit, depth, v, alu, **kw):
    lo, hi = 0.0, 4.0
    load = lambda s: pb._alu_load(pb._tile_honest(card, t, Lp, K, n, v, s)[-1])  # noqa: E731
    if load(hi) < alu:
        return None
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if load(mid) < alu else (lo, mid)
    return job(card, t, Lp, K, n, bit, depth, v, hi, **kw)


def report(ts=None, **kw):
    for card in (pb.H200, pb.SM120):
        t = (ts or pb.V2_T)[card.family]
        _, bit, depth = pb.BIT_COST[card.family][0]
        for model in ("frozen", "Llama-3.1-8B", "Llama-3.1-70B", "DeepSeek-V4-Pro"):
            num = den = 0.0
            cells = []
            for mdl, K, N, reps, Lp, alu in pb.V2_PICKS[card.family]:
                if mdl != model:
                    continue
                r = job_at(card, t, Lp, K, N, bit, depth, pb.V2_V, alu, **kw)
                w = K * N * reps
                num, den = num + w * r["best"].c * (1 - r["c_opt"]), den + w * r["best"].c
                mix = ",".join(f"{k.split(':')[-1]} {x:.2f}" for k, x in r["mix"].items() if x > 1e-3)
                cells.append(f"{K}x{N} L'{Lp}: {r['c_opt']:.3f} [{mix}]")
            print(f"{card.name:12} t {t} {model:16} leak {num / den:6.2%} | " + "; ".join(cells))


if __name__ == "__main__":
    for o, z in ((False, False), (True, False), (False, True), (True, True)):
        print(f"\n-- o_route {o} zhd {z}")
        report(o_route=o, zhd=z)
