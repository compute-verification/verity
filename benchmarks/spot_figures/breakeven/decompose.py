"""Per-stratum decomposition of a line's planned two-stage design at one scope and beta."""
import sys
from common import *
from math import log2, e

def decompose(name, top=12, **kw):
    m, g, row, span = design(name, **kw)
    strata = strata_of(name, kw.get("inputs_"))
    split = [replace(cl, stratum=0 if opt._size_class(cl.bits) <= row["merged"] else opt._size_class(cl.bits)) for cl in g.cells]
    st = opt.two_stage_statements(split, row["k"], row["q"], row["registration"])
    n_t = {}
    for cl in split:
        n_t[cl.stratum] = n_t.get(cl.stratum, 0) + cl.n
    tot_prove = sum(r["prove"] for r in st)
    tot_marg = sum(r["marginal"] for r in st)
    rows = []
    for x, cl, r in zip(strata, split, st):
        rows.append((r["prove"], x["block"][:70], cl, r))
    rows.sort(key=lambda t: -t[0])
    print(f"== {name}: bound {row['bound']/8e12:.4g} TB, q {row['q']:.4g}, kappa {row['kappa']}, merged {row['merged']}, "
          f"prove/sampled {tot_prove:.4g} s (marginal {tot_marg:.4g}, fixed {tot_prove-tot_marg:.4g}), replay {g.replay:.4g} s")
    bits_req = sum(cl.n * cl.bits for cl in g.cells)
    print(f"   request: blocks {sum(cl.n for cl in g.cells):.4g}, committed bits {bits_req:.4g}, S {m.served}")
    print(f"   {'block':70s} {'stratum':>7s} {'n/req':>9s} {'bits':>7s} {'k_t':>9s} {'taken':>9s} {'prove s':>9s} {'share':>6s} {'fixed s':>9s}")
    for p, blk, cl, r in rows[:top]:
        k = row["k"][cl.stratum]
        print(f"   {blk:70s} {cl.stratum:7d} {cl.n:9.3g} {cl.bits:7.0f} {k:9.3g} {r['taken']:9.3g} {p:9.3g} {p/tot_prove:6.1%} {p-r['marginal']:9.3g}")
    # bit-weighted proving per request, sum n_t (min(b,S)+l) p_t
    R = g.replay_blocks
    pb = sum(cl.n * (min(cl.bits, m.served) + log2(e * R * n_t[cl.stratum])) * (cl.prove + cl.fixed / cl.cap) for cl in split)
    p0 = sum(cl.n * (cl.prove + cl.fixed / cl.cap) for cl in split)
    print(f"   per request: full-statement proving {p0:.4g} s, bit-weighted proving sum n(b+l)p {pb:.4g} bit-s, ratio {pb/p0:.4g} bits")
    return m, g, row, split, st

if __name__ == "__main__":
    for n in sys.argv[1:] or ("tokens", "per_expert"):
        decompose(n)
