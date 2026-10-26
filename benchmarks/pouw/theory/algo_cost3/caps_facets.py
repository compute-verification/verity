"""Do v3.1's RTX PRO 6000 picks move under the machine model's joint issue rows?

D6's 07:04Z rule takes each round class's L′ from `d6.grid` (job-only leak and γ at every L′) by `d6.rule_pick` at a
τ it chooses per card, and the job-only leak is priced on the card's facets (JI3e_0..2, closed at VF3n-16). This swaps
them for a vector's rows as `theta_facets.py` does, and recomputes every RTX PRO 6000 round class of v3.1 (static and
PV): its grid, and its candidate at every τ in `d6.TAUS`, beside v3.1's pick. γ doesn't read the facets. τ is chosen
by the period's credited share (`d6.rule`), which this doesn't rerun; so a class whose candidate moves at the τ D6
chose moves under the same τ, and the τ could move as well. `--rule` also reruns `d6.rule` on the swapped rows (every
card; the H200's are untouched): the RTX PRO 6000's τ, its sweep, and which of its picks move.

    PUBLIC_BEST_DIR=<D5 v2>/benchmarks/pouw/theory python3 caps_facets.py <D6 accel_score dir> PICKS OUT_PREFIX \
        [--vector NAME|theta] [--rule]

`--vector theta` keeps JI3e: the control, whose candidate at D6's τ is v3.1's pick wherever v3.1 credits the class.
"""
import json
import pickle
import sys
from pathlib import Path

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
    Path(sys.argv[3] + ".json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
