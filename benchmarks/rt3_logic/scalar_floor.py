"""RT3-LOGIC-D: sm_120's measured headline in the Lean form at (I3) ScalarRoute's floor q, ½ (A6's, what
`SheetScalarRouteHalf` proves from `SheetWords`) beside the statement's 9/8 (ClassPSlots at a cap nothing proves).

Runs #2791's own exporter (`benchmarks/pouw/lean_headline.py`: `instance()`, ♭'s counts, the writer kinds, ε_skip) with
only q changed, and without `measured()`'s refusals, which hold the statement's q: at each q it takes the machine
model's certificate there (`E.eps(...).cheat.pi`, mixed or not) and evaluates the Lean bound's two branches at it the
way `Model.wStar` and `Model.phiStar` read it (the vector route `q/32 · Σ_r λ_r min_k a r k`). Light: one LP a vector
and q. Usage, from the repository root of a tree carrying #2791:

    PYTHONPATH=benchmarks/pouw/machine_model python benchmarks/rt3_logic/scalar_floor.py [ROOT] [VECTOR ...]
"""

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
vectors = sys.argv[2:] or ["VF3s-16", "VF3r-16"]
sys.path.insert(0, str(root / "benchmarks" / "pouw"))
import lean_headline as L  # noqa: E402
from verity_machine_model import eps as E  # noqa: E402

i = L.instance()
B_ = L._eps_budget()
d = E.design(L.DESIGN)
n0 = {k: i.nw.get(k, 0) + i.nf.get(k, 0) for k in {*i.nw, *i.nf}}
out = {"stated": {"q": "9/8", "eps": float(L.EPS_M)}, "readings": []}
for vec in vectors:
    pi = E.machine(vec)
    c = {k: {r: L.exact(v) for r, v in pi.c[k].items() if v} for k in i.classes}
    writers = []
    for f in sorted(E.forms(), key=lambda f: f.kind):
        if f.kind in pi.c and f.kind not in L.OFF_MENU:
            writers.append((f.kind, math.ceil(Fraction(f.words) * Fraction(round(f.depth * f.split), f.split) / d.L)))
    writers += [(k, w) for k, w in L.UNIFORM.items() if k in pi.c and k not in L.OFF_MENU]
    eps_skip = L.exact(B_.design_skip(d, vector=vec).eps_time)

    def load(r, n):
        return sum((c[k].get(r, Fraction(0)) * v for k, v in n.items()), Fraction(0))

    for q in (Fraction(1, 2), Fraction(9, 8)):
        e = E.eps(d, pi, float(q), operands="free", credit="floor")
        lam = {r: L.exact(w) for r, w in e.cheat.pi.items() if w > 1e-12}
        tot = sum(lam.values())
        lam = {r: w / tot for r, w in lam.items()}

        def price(n):
            return sum((w * load(r, n) for r, w in lam.items()), Fraction(0))

        B0 = max(load(r, n0) for r in pi.rows)
        skip_at = sum((w * c[L.SKIP].get(r, Fraction(0)) for r, w in lam.items()), Fraction(0))
        ns = {L.SKIP: eps_skip / (1 - eps_skip) * price(n0) / skip_at} if skip_at else {}
        n = {k: n0.get(k, 0) + ns.get(k, 0) for k in {*n0, *ns}}
        B = max(load(r, n) for r in pi.rows)
        demand = {k: sum((w * c[k].get(r, Fraction(0)) for r, w in lam.items()), Fraction(0)) / om
                  for k, om in writers}
        wk = min(demand, key=demand.get)
        vmin = sum((w * min(c[k].get(r, Fraction(0)) for k in i.vec) for r, w in lam.items()), Fraction(0))
        vroute = q / E.LANES * vmin
        w = min(demand[wk], vroute)
        phi = min(price(f) for f in i.folds)
        folded, unfolded = L.branches(i.Nw, w, phi, B, i.checks, L.LN_P_M)
        out["readings"].append({
            "vector": vec, "q": f"{q.numerator}/{q.denominator}",
            "pistar": {r: float(x) for r, x in lam.items()},
            "w_head": "vector route" if vroute < demand[wk] else wk,
            "w_writer": float(demand[wk]), "w_vector": float(vroute),
            "B_flat": float(B), "B_flat_at_pistar": float(price(n)),
            "folded": float(folded), "unfolded": float(unfolded), "lean_bound": float(max(folded, unfolded)),
            "machine_eps": e.machine_eps, "machine_total": e.machine_eps * (1 - float(eps_skip)) + float(eps_skip),
            "eps_skip": float(eps_skip),
        })
print(json.dumps(out, indent=1))
