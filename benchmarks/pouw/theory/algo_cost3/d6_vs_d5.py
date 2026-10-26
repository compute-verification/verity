"""A31: D5 v2's served column at a stated beside load x against D6's, step by step inside D6's own LPs.

D5 v2 (`analyse`'s `fixed`) prices each static round alone beside X MAC-times of plain s8 tensor work a credited MAC, at
that round's own μ°, and quotes 1 − (the MAC-weighted mean of the rounds' c). D6 (`d6.period(beside=x)`) prices one
period LP over every credited round (static and PV) with the period's own uncredited work (QKᵀ's two passes, V4's
indexer, plain PV) and x as fixed load, at that period's one μ°, and quotes 1 − Σ pub(μ°) / Σ OPT*(μ°) (cost-weighted).
This walks from one to the other, one change at a time, at the same picks and the same job LPs (`d6.job_lps`):

  a  per round, static only, x only,     MAC-weighted mean of 1 − pub/OPT   (D5 v2's reading)
  b  per round, static only, x only,     cost-weighted
  c  one period LP, static only, x only, cost-weighted
  d  c plus the period's own uncredited work as fixed load
  e  d plus the credited PV rounds                                           (D6's reading; checked against d6.period)

    PUBLIC_BEST_DIR=<D5 v2 checkout>/benchmarks/pouw/theory python3 d6_vs_d5.py <D6 accel_score dir> OUT_PREFIX [PICKS]

PICKS: a pickle of d6.picks()' v3.1 and v3 picks, written if absent and read if present, so two D6 heads compare at
the same picks.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import d5_routes as dr  # noqa: E402
import d6  # noqa: E402
import nofold as nf  # noqa: E402
import v3_caps as vc  # noqa: E402

XS = (0.25, 1.0, 4.0)
MODELS = ("Llama-3.1-8B", "Llama-3.1-70B", "DeepSeek-V4-Pro")


def items(card, model, static, pv):
    """d6.period's credited rounds: (K, N, MAC weight, pick, static?), and the period's own uncredited MACs."""
    hist, st, qk, pvm, ix = nf.world_macs(model)
    out = [(K, N, K * N * reps * st / model["static"], static[card, K, N], True)
           for K, N, reps in nf.SHAPES[model["name"]]]
    bins = {}
    for n, cnt in hist.items():
        bins[nf.pv_bucket(n)] = bins.get(nf.pv_bucket(n), 0) + cnt * model["hq"] * model["dv"] * n
    plain = 0.0
    for nb, x in sorted(bins.items()):
        p = pv[card, nb, model["dv"]]
        if p is None:
            plain += x
        else:
            out.append((nb, model["dv"], x, p, False))
    return out, qk * (1 + nf.SOFTMAX_PASS) + ix + plain


def lps(card, K, N, p, wc):
    m, l = d6.job_lps(card, K, N, p["t"], p["Lp"], wc)
    return m, l


def fixed_alone(c, lp, F):
    """d6.fixed_alone (6b3202bd2a): the fixed load's makespan alone, honest-kernel units."""
    f = []
    for r in dr.rows(c):
        if r.startswith("joint"):
            a, b = c.facets[int(r[5:])]
            f.append(a * F.get("tensor", 0.0) + b * F.get(c.facet_other, 0.0) / 32)
        else:
            f.append(F.get(r, 0.0))
    return float(max(np.array(f) / -np.array([row[0] for row in lp["A_ub"]]))) * c.s8


def mu_str(mu, c):
    return " + ".join(f"{r} {v:.3g}" for r, v in sorted(mu.items(), key=lambda kv: -kv[1]) if v > 1e-9)


def per_round(card, its, x):
    c = vc.CARDS[card][0]
    rows = []
    for K, N, w, p, wc in its:
        if not wc:
            continue
        m, l = lps(card, K, N, p, wc)
        F = {r: v * x for r, v in d6.serve_loads(c).items()}
        per = dr.period(c, [(1.0, l["dense"][0])], F)
        pub, opt = dr.mu_price(c, m, "hull", per["mu"]), dr.mu_price(c, m, "dense", per["mu"])
        rows.append(dict(K=K, N=N, Lp=p["Lp"], w=w, pub=pub, opt=opt, leak=1 - pub / opt, mu=mu_str(per["mu"], c)))
    W = sum(r["w"] for r in rows)
    return dict(mac_mean=sum(r["w"] * r["leak"] for r in rows) / W,
                cost=1 - sum(r["w"] * r["pub"] for r in rows) / sum(r["w"] * r["opt"] for r in rows), rows=rows)


def pooled(card, its, own, x, with_pv, with_own):
    c = vc.CARDS[card][0]
    sel = [it for it in its if it[4] or with_pv]
    cred = sum(it[2] for it in sel)
    jobs, ms = [], []
    for K, N, w, p, wc in sel:
        m, l = lps(card, K, N, p, wc)
        jobs.append((w / cred, l["dense"][0]))
        ms.append((w / cred, m, K, N, wc))
    F_macs = (own / cred if with_own else 0.0) + x
    F = {r: v * F_macs for r, v in d6.serve_loads(c).items()}
    served = dr.period(c, jobs, F)
    hull = dr.period(c, [(w, lps(card, K, N, p, wc)[1]["hull"][0]) for K, N, w0, p, wc in sel
                         for w in [w0 / cred]], F)
    mu = served["mu"]
    pub = sum(w * dr.mu_price(c, m, "hull", mu) for w, m, *_ in ms)
    opt = sum(w * dr.mu_price(c, m, "dense", mu) for w, m, *_ in ms)
    T_F = fixed_alone(c, jobs[0][1], F)
    by = {}
    for w, m, K, N, wc in ms:
        a, b = dr.mu_price(c, m, "hull", mu), dr.mu_price(c, m, "dense", mu)
        by[f"{'' if wc else 'PV '}{K}x{N}"] = dict(w=w, leak=1 - a / b, opt=b)
    return dict(leak=1 - pub / opt, S=1 - (hull["c"] - T_F) / (served["c"] - T_F), mu=mu_str(mu, c),
                binding=served["binding"], by=by)


def main():
    out_prefix = sys.argv[2]
    import pickle
    picks_in = sys.argv[3] if len(sys.argv) > 3 else None
    if picks_in and os.path.exists(picks_in):
        P = pickle.load(open(picks_in, "rb"))
    else:
        P, _ = d6.picks()
        if picks_in:
            pickle.dump({k: P[k] for k in ("v3.1", "v3")}, open(picks_in, "wb"))
    lines, res = [], {}
    for design in ("v3.1", "v3"):
        static, pv = P[design]
        for card in ("RTX PRO 6000", "H200"):
            for mname in MODELS:
                model = d6.model_of(mname)
                its, own = items(card, model, static, pv)
                lps_static = sorted({(K, N, p["Lp"]) for K, N, _, p, wc in its if wc})
                for x in XS:
                    a = per_round(card, its, x)
                    cc = pooled(card, its, own, x, False, False)
                    dd = pooled(card, its, own, x, False, True)
                    ee = pooled(card, its, own, x, True, True)
                    d6v = d6.period(card, model, static, pv, beside=x)
                    res[design, card, mname, x] = dict(a=a["mac_mean"], b=a["cost"], c=cc["leak"], d=dd["leak"],
                                                       e=ee["leak"], d6=d6v["leak_served"], S_c=cc["S"], S_e=ee["S"],
                                                       c_inc_d6=d6v.get("c_inc"), mu_c=cc["mu"], mu_e=ee["mu"],
                                                       rounds=a["rows"], pooled_by=ee["by"], static_by=cc["by"])
                    lines.append(f"{design:5} {card:12} {mname:16} x {x:<4} a {a['mac_mean']:6.2%}  b {a['cost']:6.2%}"
                                 f"  c {cc['leak']:6.2%}  d {dd['leak']:6.2%}  e {ee['leak']:6.2%}"
                                 f"  (d6.period {d6v['leak_served']:6.2%}; makespan S: c {cc['S']:6.2%}, e {ee['S']:6.2%})"
                                 f" | mu° c: {cc['mu']} | mu° e: {ee['mu']}")
                    if x == 1.0:
                        for r in a["rows"]:
                            lines.append(f"      per round {r['K']}x{r['N']} L' {r['Lp']}: w {r['w']:.3g} leak "
                                         f"{r['leak']:6.2%} at {r['mu']}; pooled (static) leak "
                                         f"{cc['by'][str(r['K']) + 'x' + str(r['N'])]['leak']:6.2%}")
                lines.append(f"      static picks {lps_static}; credited PV classes "
                             f"{sum(1 for it in its if not it[4])}, plain PV + own {own:.4g} MACs")
                print("\n".join(lines[-1:]), flush=True)
    open(out_prefix + ".txt", "w").write("\n".join(lines) + "\n")
    json.dump({" | ".join(map(str, k)): v for k, v in res.items()}, open(out_prefix + ".json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
