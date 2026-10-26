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
        [--cost 0.77,1.25] [--pipes alu,fma] [--skip-menu | --menu-only]

--cost: emulated b1's ALU time in INT8 product-times (A7's 0.77; AND + POPC + an add on D5's 64 ALU lanes is about
1.25, and 1.6 with Harley-Seal's adds), one pass each; --pipes: the pipes it may run on (fma: LOP3 on all 128 INT32
lanes of sm_120, which D5's card doesn't model); --skip-menu: skip the unpatched pass; --menu-only: run only it (D5
v2.1's menu carries the census routes itself, so its unpatched pass is the check against `--census` at D5 v2).

--census: instead of one route at a fixed cost, every strategy of `b1_census.py` (the carry-save trees with either
accumulator, and dp4a2 on the IMAD pipe) is a route of its own, each with its instruction counts on D5's resources:
LOP3 and IADD3 on alu, IMAD and IDP.4A two slots on fma, POPC on an XU row added to θ (16 lane-ops a SM-clock, the
census's 0.5 warp instructions), every one of them on issue and vec, the registers each reads (AND 2, a carry-save
LOP3 3, POPC 1, IADD3 3, IMAD 2, IDP.4A 3, the chain's mask 1), and its operands from shared memory on D5's DP4A tile
(`BESIDE_TILE`: 8 bytes and half an LDS.128 a word pair, over t). The LP mixes them per job. --no-dp4a2: the trees
alone.
"""
import json
import pickle
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(Path(__file__).parent))
import b1_census as b1c  # noqa: E402
import d6  # noqa: E402

pb = d6.pb


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


COSTS = tuple(float(x) for x in _opt("--cost", "0.77").split(","))
PIPES = tuple(_opt("--pipes", "alu,fma").split(","))
CENSUS = "--census" in sys.argv
A7_ALU = COSTS[0]
XS = d6.BESIDE
MODELS = ("Llama-3.1-8B", "Llama-3.1-70B", "DeepSeek-V4-Pro")
_orig = pb.tile_methods


XU_LANES = 16.0


def emu(card, pipe):
    e = A7_ALU * card.alu / card.s8
    return {pipe: e, "issue": e, "vec": e, "rf": 3 * e}


def census_routes(card):
    t = pb.BESIDE_TILE
    out = {}
    for name, (g, fas, pops) in b1c.STRATEGIES.items():
        for acc in ("iadd3", "imad"):
            alu = g + 2 * fas + (pops / 2 if acc == "iadd3" else 0)
            imad = pops if acc == "imad" else 0
            vec = alu + pops + imad
            rf = 2 * g + 6 * fas + pops + (1.5 * pops if acc == "iadd3" else 2 * pops)
            ld = dict(alu=alu, xu=pops, fma=2 * imad, vec=vec, issue=vec + g / (2 * t), rf=rf, smem=8 * g / t)
            out[f"{name}/{acc}"] = {r: x / (32 * g) for r, x in ld.items() if x}
    ext = 1.5 / b1c.DP4A_CHAIN
    ld = dict(fma=2.0, alu=ext, vec=1 + ext, issue=1 + ext + 1 / (2 * t), rf=3 + 2.5 / b1c.DP4A_CHAIN, smem=8 / t)
    if "--no-dp4a2" not in sys.argv:
        out["dp4a2"] = {r: x / b1c.DP4A_BITS for r, x in ld.items()}
    return out


def routes(card):
    if CENSUS:
        return census_routes(card)
    return {f"b1 on {p}": emu(card, p) for p in PIPES}


def patched(card, t, Lp, K, bit, depth, L, zleaf, peels, dense, digits=(2, 2), o_route=True, zhd=True,
            schemes=None, v=pb.V2_V):
    ms = _orig(card, t, Lp, K, bit, depth, L, zleaf, peels, dense, digits, o_route, zhd, schemes, v)
    if card.family != "sm_120":
        return ms
    extra = {f"o_route {k}": ld for k, ld in routes(card).items()}
    out = []
    for m in ms:
        if m.items and m.items[0][0] == "O":
            name, w, o = m.items[0]
            m = replace(m, items=((name, w, {**o, **extra}),) + tuple(m.items[1:]))
        out.append(m)
    tiles = next(m for m in ms if m.label == "tiles")
    mma = dict(smem=1 / card.honest_tile[0] + 1 / card.honest_tile[1], rf=card.mma_rf, issue=card.mma_issue)
    for k, ld in routes(card).items():
        f = {r: y / t for r, y in mma.items()}
        f["tensor"] = 1 / (t * card.s8)
        for r, y in ld.items():
            f[r] = f.get(r, 0.0) + y
        out.append(replace(tiles, label=f"tiles, b c^T by emulated {k}", fixed=f))
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


def dump(out, prefix):
    """PREFIX.json holds every label's rows; PREFIX.LABEL.json each label's alone, keyed "card | model | x" as d6.py's
    --census reads them."""
    flat = {lab: {" | ".join(map(str, k)): v for k, v in r.items()} for lab, r in out.items()}
    Path(prefix + ".json").write_text(json.dumps(flat, indent=1, default=str))
    for lab, r in flat.items():
        Path(f"{prefix}.{lab.replace(' ', '_')}.json").write_text(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    P = pickle.load(open(sys.argv[2], "rb"))
    static, pv = P["v3.1"]
    out = {} if "--skip-menu" in sys.argv else {"menu": run(("RTX PRO 6000", "H200"), "D5 v2 menu (D6)")}
    if "--menu-only" in sys.argv:
        dump(out, sys.argv[3])
        sys.exit()
    pb.tile_methods = patched
    if CENSUS:
        cap0 = pb.cap
        pb.RES = pb.RES + ("xu",)
        pb.cap = lambda card: {**cap0(card), "xu": XU_LANES if card.family == "sm_120" else math.inf}
        label = "b1 census trees" if "--no-dp4a2" in sys.argv else "b1 census routes"
        out[label] = run(("RTX PRO 6000",), f"+ {label}")
    for A7_ALU in () if CENSUS else COSTS:
        out[f"b1 {A7_ALU} on {'+'.join(PIPES)}"] = run(("RTX PRO 6000",), f"+ b1 {A7_ALU} {'+'.join(PIPES)}")
    dump(out, sys.argv[3])
