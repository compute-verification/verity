"""A33: pub at the tensor-only price μ_t (08:24Z), per round at v3.1's picks, in D6's job LPs (D5 v2's hull, c75b7dabfe).

Per credited MAC, in honest-kernel tensor MAC-times (FP8 FLOP / 2 at the s8 rate):
  honest   the deployed kernel's tensor (the dense product and its peels on the MMA)
  OPT*(μ_t), pub(μ_t)            the least tensor over the honest words' programs and over the hull, lanes free
                                 (`d5_routes.mu_price` at {tensor: 1}); with the peels' DP4A route (`peel_lanes`) and
                                 without it (D5 v2's `tile_job` menu)
  pub, bounded at s              the hull's least tensor with the round held to OPT*'s own makespan × (1 + s)
                                 (`d5_routes.row_min`), with the peels' DP4A route
and, for the pub(μ_t) rows, the route that carries the tensor that's left.

    PUBLIC_BEST_DIR=<D5 v2>/benchmarks/pouw/theory python3 mu_t_check.py <D6 c75b7dabfe accel_score dir> PICKS OUT
"""
import json
import pickle
import sys

sys.path.insert(0, sys.argv[1])
import d5_routes as dr  # noqa: E402
import d6  # noqa: E402
import nofold as nf  # noqa: E402

MU_T = {"tensor": 1.0}
SLACK = (0.0, 0.25, 1.0)


def row(card, K, N, p, wc):
    out = {}
    for lanes in (False, True):
        c, m, _ = d6.build_job(card, K, N, p["t"], p["Lp"], weight_coins=wc, peel_lanes=lanes)
        best = dr.priced(c, m, "dense")
        r = dict(honest=m["honest_tensor"], opt=dr.mu_price(c, m, "dense", MU_T),
                 pub=dr.mu_price(c, m, "hull", MU_T), best_c=best.c)
        hull_t = dr.priced(c, m, "hull", hook=lambda lp, cc, **kw: lp([0.0] + list(
            kw["A_ub"][dr.rows(c).index("tensor")][1:]), **kw))
        r["pub_mix"] = {k: round(float(v), 3) for k, v in hull_t.mix.items() if v > 1e-3}
        r["pub_routes"] = {k: round(float(v), 4) for k, v in hull_t.routes.items()
                           if v > 1e-4 and not k.startswith("sha")}
        if lanes:
            r["bounded"] = {s: dr.row_min(c, m, "hull", "tensor", best.c * (1 + s)) for s in SLACK}
        out["lanes" if lanes else "mma"] = r
    return out


def main():
    P = pickle.load(open(sys.argv[2], "rb"))
    static, pv = P["v3.1"]
    lines, res = [], {}
    for card in ("H200", "RTX PRO 6000"):
        keys = [(k, p, True) for k, p in static.items() if k[0] == card]
        keys += [(k, p, False) for k, p in pv.items() if k[0] == card and p is not None and k[1] in
                 (128, 256, 512, 1024, 2048, 4096)]
        for (c_, K, N), p, wc in sorted(keys, key=lambda x: (not x[2], x[0][1], x[0][2])):
            r = row(card, K, N, p, wc)
            res[f"{card} {'' if wc else 'PV '}{K}x{N} L' {p['Lp']}"] = r
            a, b = r["mma"], r["lanes"]
            bd = " / ".join(f"{x:.3f}" for x in b["bounded"].values())
            lines.append(f"{card:12} {'' if wc else 'PV '}{K}x{N} t {p['t']} L' {p['Lp']}: honest {a['honest']:.3f} | "
                         f"D5 menu: OPT* {a['opt']:.3f}, pub {a['pub']:.3f} ({a['pub_routes']}) | peels on DP4A: "
                         f"OPT* {b['opt']:.3f}, pub {b['pub']:.3f} ({b['pub_routes']}) | pub held to OPT*'s "
                         f"makespan ×1 / 1.25 / 2: {bd}")
            print(lines[-1], flush=True)
    open(sys.argv[3] + ".txt", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(sys.argv[3] + ".json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
