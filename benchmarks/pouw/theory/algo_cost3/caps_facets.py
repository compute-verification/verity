"""Do v3.1's RTX PRO 6000 picks move under the machine model's joint issue rows?

D6's 07:04Z rule takes each round class's L′ from `d6.grid` (job-only leak and γ at every L′) by `d6.rule_pick` at a
τ it chooses per card, and the job-only leak is priced on the card's facets (JI3e_0..2, closed at VF3n-16). This swaps
them for a vector's rows as `theta_facets.py` does, and recomputes every RTX PRO 6000 round class of v3.1 (static and
PV): its grid, and its candidate at every τ in `d6.TAUS`, beside v3.1's pick. γ doesn't read the facets. τ is chosen
by the period's credited share (`d6.rule`), which this doesn't rerun; so a class whose candidate moves at the τ D6
chose moves under the same τ, and the τ could move as well. `--rule` also reruns `d6.rule` on the swapped rows (every
card; the H200's are untouched): the RTX PRO 6000's τ, its sweep, and which of its picks move. `--by-load D6PKL`
(with --rule) is D6's "the rule's τ by load" on the swapped rows: each τ's picks priced by `d6.period` at every x > 0
of `d6.BESIDE`, with share0, leak0 and γ from that D6 build's `--pickle` (x = 0 doesn't read the rows the rule's
sweep shows), and D6's own leak beside each.

    PUBLIC_BEST_DIR=<D5 v2>/benchmarks/pouw/theory python3 caps_facets.py <D6 accel_score dir> PICKS OUT_PREFIX \
        [--vector NAME|theta] [--rule [--by-load D6PKL]]

`--vector theta` keeps JI3e: the control, whose candidate at D6's τ is v3.1's pick wherever v3.1 credits the class.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import theta_facets as tf  # noqa: E402

d6 = tf.b.d6
CARD = "RTX PRO 6000"


def row(x: dict) -> dict:
    return {k: (v if isinstance(v, bool) else float(v)) for k, v in x.items()}


def main():
    card = d6.vc.CARDS[CARD][0]
    name, facets = tf.vector_facets(card)
    if tf.VECTOR != "theta":
        object.__setattr__(card, "facets", tuple((a * card.s8 / 4096, v) for _, a, v in facets))
    st, pv = pickle.load(open(sys.argv[2], "rb"))["v3.1"]
    keys = [(k, True, p) for k, p in st.items() if k[0] == CARD]
    keys += [(k, False, p) for k, p in pv.items() if k[0] == CARD]
    out = []
    for k, wc, p in keys:
        g = d6.grid(*k, weight_coins=wc)
        cand = {tau: d6.rule_pick(*k, tau, weight_coins=wc) for tau in d6.TAUS}
        out.append(dict(key=list(k), kind="static" if wc else "pv",
                        v31=None if p is None else dict(t=p["t"], Lp=p["Lp"], leak=float(p["leak"])),
                        grid=[row(x) for x in g], picks={f"{t:g}": row(x) for t, x in cand.items()}))
        print(k, "static" if wc else "pv", "v3.1:", None if p is None else p["Lp"],
              " ".join(f"{x['Lp']}:{x['leak']:.4f}" for x in g),
              "| τ " + " ".join(f"{t:g}→{x['Lp']}" for t, x in cand.items()), flush=True)
    res = dict(vector=name, rows=out)
    if "--rule" in sys.argv:
        s31, p31, sw = d6.quiet(d6.rule, list(st), list(pv))
        lp = lambda p: p and p["Lp"]  # noqa: E731
        res["rule"] = dict(tau=sw["tau"][CARD], sweep={f"{t:g}": sw["sweep"][CARD, t] for t in d6.TAUS},
                           moved=[list(k) for d, e in ((st, s31), (pv, p31)) for k in d if k[0] == CARD
                                  and lp(d[k]) != lp(e[k])])
        print("rule: τ", res["rule"]["tau"], "moved", res["rule"]["moved"])
        for t, x in res["rule"]["sweep"].items():
            print(f"  τ {t}: share", " / ".join(f"{v:.4f}" for v in x["share"]), "γ",
                  " / ".join(f"{v:.3f}" for v in x["gamma"]), "plain", x["plain"])
        if "--by-load" in sys.argv:
            res["by_load"] = by_load(sw["tau_picks"], sys.argv[sys.argv.index("--by-load") + 1])
    Path(sys.argv[3] + ".json").write_text(json.dumps(res, indent=1, default=float))


def by_load(tau_picks: dict, d6pkl: str) -> dict:
    """`d6.report`'s τ by load, from each τ's picks priced at the swapped rows."""
    B = pickle.load(open(d6pkl, "rb"))["beside"]
    models = list(dict.fromkeys(k[2] for k in B))
    out, best = {}, {}
    for x in d6.BESIDE[1:]:
        for t, (s, p) in tau_picks.items():
            n = f"τ {t:g}"
            for m in models:
                per = d6.quiet(d6.period, CARD, d6.model_of(m), s, p, beside=x)
                b = B[n, CARD, m, x]
                out[f"{n} | {m} | {x:g}"] = dict(
                    leak=per["leak_served"], d6_leak=b["leak"], binding=list(per["dense"]["served_binding"]),
                    d6_binding=list(b["binding"]), gamma=b["gamma"],
                    share=b["share0"] * (1 - per["leak_served"]) / (1 - b["leak0"]))
                print(f"  {n} {m} x {x:g}: leak {per['leak_served']:.4f} (D6 {b['leak']:.4f})",
                      ", ".join(per["dense"]["served_binding"]), flush=True)
        mean = {t: np.mean([out[f"τ {t:g} | {m} | {x:g}"]["share"] for m in models]) for t in tau_picks}
        near = [t for t in tau_picks if mean[t] >= max(mean.values()) - 1e-3]
        best[f"{x:g}"] = min(near, key=lambda t: np.mean([out[f"τ {t:g} | {m} | {x:g}"]["gamma"] for m in models]))
        print(f"x {x:g}: τ {best[f'{x:g}']:g}, mean share " + ", ".join(f"{t:g} {v:.4f}" for t, v in mean.items()))
    return dict(rows=out, tau=best)


if __name__ == "__main__":
    main()
