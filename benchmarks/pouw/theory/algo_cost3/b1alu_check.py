"""A34 (menu coverage, gap iii): A7's b c^T with b1 emulated on the lanes, priced as a route inside D6's period LPs, and
D6's PV picks priced as a whole.

D5 v2 lists "A7 with b1 emulated on the ALU beside it" (0.26) in `BIT_COST` with depth None, so it enters only the
standalone table: no job LP prices it, because part of it runs on lanes. Here it is a route with lane loads. A7
(c_bit.py at 26df6a5df1) prices emulated b1 (LOP3 + POPC) at about 0.77 of an INT8 product's time on the ALU, so a
bit MAC costs e = 0.77 · alu / s8 lane-ops; it is charged e on alu (or, more generously, on the FMA pipe), issue and
vec, and 3e on the register file, like D5's assembly ops. It enters two ways:
  - every method that gets O by route (`dense, O by route`, the fast schemes) gains O = b c^T by emulated b1;
  - the tile route gains a twin whose b c^T runs emulated (V (Σc)^T stays on s8), so the LP can split b c^T between
    FP4 Strassen and the lanes in any proportion (A7's 0.26 is the split on idle lanes).
Then `d6.period` at each beside load, with and without the route, on the RTX PRO 6000 (on the H200 b1 is native at
1/8 and emulation costs about 3 product-times). The unpatched pass also records D6's `leak_kind` (static and PV
priced as a whole) on both cards.

    PUBLIC_BEST_DIR=<D5 v2>/benchmarks/pouw/theory python3 b1alu_check.py <D6 accel_score dir> PICKS OUT_PREFIX
"""
import json
import pickle
import sys
from dataclasses import replace

sys.path.insert(0, sys.argv[1])
import d6  # noqa: E402

pb = d6.pb
A7_ALU = 0.77
XS = (0.0, 0.25, 1.0)
MODELS = ("Llama-3.1-8B", "Llama-3.1-70B", "DeepSeek-V4-Pro")
_orig = pb.tile_methods


def emu(card, pipe):
    e = A7_ALU * card.alu / card.s8
    return {pipe: e, "issue": e, "vec": e, "rf": 3 * e}


def patched(card, t, Lp, K, bit, depth, L, zleaf, peels, dense, digits=(2, 2), o_route=True, zhd=True,
            schemes=None, v=pb.V2_V):
    ms = _orig(card, t, Lp, K, bit, depth, L, zleaf, peels, dense, digits, o_route, zhd, schemes, v)
    if card.family != "sm_120":
        return ms
    extra = {f"o_route b1 on {p}": emu(card, p) for p in ("alu", "fma")}
    out = []
    for m in ms:
        if m.items and m.items[0][0] == "O":
            name, w, o = m.items[0]
            m = replace(m, items=((name, w, {**o, **extra}),) + tuple(m.items[1:]))
        out.append(m)
    tiles = next(m for m in ms if m.label == "tiles")
    mma = dict(smem=1 / card.honest_tile[0] + 1 / card.honest_tile[1], rf=card.mma_rf, issue=card.mma_issue)
    for p in ("alu", "fma"):
        f = {r: y / t for r, y in mma.items()}
        f["tensor"] = 1 / (t * card.s8)
        for r, y in emu(card, p).items():
            f[r] = f.get(r, 0.0) + y
        out.append(replace(tiles, label=f"tiles, b c^T by emulated b1 on {p}", fixed=f))
    return out


def run(cards, label):
    d6._JM.clear()
    d6._JOB.clear()
    d6._FREED.clear()
    res = {}
    for card in cards:
        for mname in MODELS:
            model = d6.model_of(mname)
            for x in XS:
                o = d6.period(card, model, static, pv, beside=x)
                res[card, mname, x] = dict(leak_job=o["leak_job"], leak=o["leak"], leak_served=o["leak_served"],
                                           leak_kind=o["leak_kind"], c_inc=o["c_inc"],
                                           at_tensor=(o["hull"]["at_tensor"], o["dense"]["at_tensor"]),
                                           mu=o["dense"]["served_mu"])
                k = o["leak_kind"]
                print(f"{label:22} {card:12} {mname:16} x {x:<4} job {o['leak_job']:6.2%}  served {o['leak_served']:6.2%}"
                      f"  static {k['static']:6.2%}  PV " + (f"{k['pv']:6.2%}" if k["pv"] is not None else "  -   ")
                      + f"  1-c_inc {1 - o['c_inc']:6.2%}  μ_t pub/OPT {o['hull']['at_tensor']:.4f}/"
                      f"{o['dense']['at_tensor']:.4f}", flush=True)
    return res


if __name__ == "__main__":
    P = pickle.load(open(sys.argv[2], "rb"))
    static, pv = P["v3.1"]
    out = {"menu": run(("RTX PRO 6000", "H200"), "D5 v2 menu (D6)")}
    pb.tile_methods = patched
    out["b1"] = run(("RTX PRO 6000",), "+ emulated b1 (A7)")
    json.dump({lab: {" | ".join(map(str, k)): v for k, v in r.items()} for lab, r in out.items()},
              open(sys.argv[3] + ".json", "w"), indent=1, default=str)
