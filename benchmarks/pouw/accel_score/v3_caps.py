"""Protocol v3 (P12, 06:20Z; PD3-4): per static shape, the largest noise period L' at which no route wins at the
period's dual prices, by D5's job LP with every route in it (`d5_routes.job`: the tile route, PD2-1's O-route and the
zero-high-digit peel beside the dense product), hashing at nofold's 16-bit word model with b streamed (P2), in model
units (θ's H200 without the 1830 MHz long-window cap, M14). H200 at t = 2; RTX PRO 6000 at t = 3 with L' ≤ 256 (PB2).

On the RTX PRO 6000 no L' ≥ 64 closes the zero-high-digit route (it binds on ALU and register file; 0.02–0.17% at L'
64, where γ is about 3× that at 256). Under credit at the public best (06:10Z) a winning public route lowers pub, so
what it leaves is uncredited honest time, not ε: there the cap is the largest L' ≤ 256 that leaves at most 1%.
PV (R-ACCEL-8's credited attention) is capped the same way at every attended length n, rounded to 64 (P12).

    PUBLIC_BEST_DIR=<checkout of cursor/compute-research-accel-bet-algo-lab-ea24>/benchmarks/pouw/theory \\
        python3 benchmarks/pouw/accel_score/v3_caps.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d5_routes as dr  # noqa: E402
import nofold as nf  # noqa: E402

pb = dr.pb
CARDS = {"H200": (pb.H200, 2, None, 1e-4), "RTX PRO 6000": (pb.SM120, 3, 256, 0.01)}
LPS = (64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048)


def leak(card: str, K: int, N: int, t: int, Lp: int, model_sha: bool = False, **kw) -> tuple[float, dict, dict]:
    r = nf.v2_round(card, K, N, t * Lp, 16, tiles=t, **kw)
    c = CARDS[card][0]
    _, bit, depth = pb.BIT_COST[c.family][0]
    j = dr.job(c, t, Lp, K, N, bit, depth, nf.V_WIDTH[card], 1.0, digits=r["digits"], comps=r["comps"],
               model_sha=model_sha)
    return 1 - j["c_opt"], j, r


def cap(card: str, K: int, N: int, **kw) -> dict:
    """The largest L' in LPS (t L' ≤ K, under the card's limit) whose leak is within the card's tolerance (zero on the
    H200); else the least-leak L'. Below t·64 inner indices, one tile (t = 1)."""
    _, t, lmax, tol = CARDS[card]
    t = t if t * LPS[0] <= K else 1
    tried = []
    for Lp in sorted((x for x in LPS if t * x <= K and (lmax is None or x <= lmax)), reverse=True) or [K]:
        lk, j, r = leak(card, K, N, t, Lp, **kw)
        tried.append(dict(Lp=Lp, t=t, leak=lk, gamma_lp=j["honest"].c, round=r))
        if lk < tol:
            return tried[-1]
    return min(tried, key=lambda x: x["leak"])


def static_caps() -> dict:
    out = {}
    print("\nv3 caps (P12): per static shape, the largest L' with zero leak under D5's LP and every route")
    for card in CARDS:
        for model, shapes in nf.SHAPES.items():
            cells = []
            for K, N, _ in shapes:
                p = cap(card, K, N)
                out[card, K, N] = p
                cells.append(f"{K}x{N} L' {p['Lp']}" + (f" leak {p['leak']:.2%}" if p["leak"] >= 1e-4 else ""))
            print(f"  {card:13} t {CARDS[card][1]} {model:16} " + "; ".join(cells))
    return out


def pv_caps() -> dict:
    out = {}
    print("\nv3 PV caps: per attended length n (rounded to 64), K = n, N = d_v, weight side streamed")
    for card in CARDS:
        for m in nf.WORLD:
            hist = nf.world_macs(m)[0]
            for nb in sorted({nf.pv_bucket(n) for n in hist}):
                if (card, nb, m["dv"]) not in out:
                    out[card, nb, m["dv"]] = cap(card, nb, m["dv"], weight_coins=False)
            ps = [out[card, nb, m["dv"]] for nb in sorted({nf.pv_bucket(n) for n in hist})]
            print(f"  {card:13} {m['name']:16} d_v {m['dv']}: L' {min(p['Lp'] for p in ps)}–{max(p['Lp'] for p in ps)}, "
                  f"worst leak {max(p['leak'] for p in ps):.2%}")
    return out


def world(static: dict, pv: dict) -> list[dict]:
    """v3 at the world model (nofold.v2_world with the caps): γ at the measured SHA-512 rates (built) and the paper
    ceiling, ε in model units (c_mm = c_sha = 1: no route wins, or what wins is uncredited), and the routes' uncredited
    share, MAC- and cost-weighted over credited static and PV work."""
    picks = {k: (p["t"], p["Lp"]) for k, p in {**static, **pv}.items()}
    rows = []
    print("\nv3 at the world model (16-bit words, b streamed, overlapped; R-ACCEL-8 attention)")
    for built in (True, False):
        for r in nf.v2_world(0.0, built=built, picks=picks):
            card = r["card"].split(" built")[0]
            m = next(x for x in nf.WORLD if x["name"] == r["model"])
            hist, st, qk, pvm, ix = nf.world_macs(m)
            num = den = 0.0
            for K, N, reps in nf.SHAPES[r["model"]]:
                p = static[card, K, N]
                w = K * N * reps * p["gamma_lp"]
                num, den = num + w * p["leak"], den + w
            sw = den / sum(K * N * reps for K, N, reps in nf.SHAPES[r["model"]])
            for n, c in hist.items():
                p = pv[card, nf.pv_bucket(n), m["dv"]]
                w = c * m["hq"] * m["dv"] * n * p["gamma_lp"]
                num, den = num + w * p["leak"], den + w
            r["route_uncredited"] = num / den
            r["rate"] = "built" if built else "paper"
            rows.append(r)
            print(f"  {r['card']:18} {r['model']:16} γ ovl {r['gamma']:5.2f} add {r['gamma_add']:5.2f} util "
                  f"{r['util']:4.0%} coverage {1 - r['uncredited']:5.1%} eps_alg {r['eps_alg']:6.2%} routes "
                  f"uncredited {r['route_uncredited']:6.3%} (static LP γ {sw:.2f})")
    return rows


if __name__ == "__main__":
    world(static_caps(), pv_caps())
