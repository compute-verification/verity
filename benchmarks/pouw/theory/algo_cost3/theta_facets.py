"""D6's RTX PRO 6000 period with θ's joint issue rows taken from the machine model's vector of record.

D5's sm_120 card (`public_best.SM120`) prices MMA beside the lanes on JI3e_0..2, which the machine model closed at
VF3n-16: measured programs (IMMA m16n8k16 and TF32 m16n8k4 beside IADD3) load JI3e_1 past 1
(`machine_model/README.md`). The vector of record (`data/model.toml`'s `current`) carries its own joint issue rows
(VF3r-16: JI3s_0..20, each MMA form its own kind, every vector-pipe instruction at the row's `@V`). This swaps the
card's facets for the vector's rows at IMMA16832's coefficient, in the card's form (a · s8/4096 on tensor time, b on
vec/32, at most 4 a SM-clock), keeping D5's assumption that every MMA weighs on them in proportion to its tensor time,
and reruns `d6.period` at D6's v3.1 picks (held fixed: they were chosen at JI3e), unpatched and with A35's census
routes for emulated b1.

    PUBLIC_BEST_DIR=<D5 v2>/benchmarks/pouw/theory python3 theta_facets.py <D6 accel_score dir> PICKS OUT_PREFIX \
        [--vector NAME]

NAME defaults to the vector of record; VF3s-16 (RT-FACET's refit, which measured programs beating JI3s_0-9) is the
tighter one.
"""
import json
import math
import pickle
import sys
from pathlib import Path

import verity_machine_model.model as mm

sys.path.insert(0, str(Path(__file__).parent))
VECTOR = sys.argv.pop(sys.argv.index("--vector") + 1) if "--vector" in sys.argv else None
if VECTOR:
    sys.argv.remove("--vector")
sys.argv.append("--census")
import b1alu_check as b  # noqa: E402

FORM = "IMMA16832"


def vector_facets():
    m = mm.model()
    v = mm.vector(VECTOR or m.current)
    c = mm.census(m, v.epoch)
    joint = [r for r in v.rows if r.startswith("JI3")]
    return v.name, [(r, c[FORM][r] / m.cycle_u, c["ALU"][r] / m.cycle_u) for r in joint]


def main():
    current, facets = vector_facets()
    card = b.d6.vc.CARDS["RTX PRO 6000"][0]
    print(f"{current}: {len(facets)} joint issue rows at {FORM}; θ's JI3e_0..2 were "
          + ", ".join(f"({a * 4096 / card.s8:.3f}, {v:.3f})" for a, v in card.facets))
    for r, a, v in facets:
        print(f"  {r}: MMA {a:.3f} + vector {v:.4f}")
    full = min(max(0.0, 1 - a / 16) / v for _, a, v in facets)
    jie = min(max(0.0, 1 - a * 4096 / card.s8 / 16) / v for a, v in card.facets)
    print(f"  at a full tensor pipe: {full:.3f} vector instructions a SMSP-cycle, θ's JI3e {jie:.3f}")
    P = pickle.load(open(sys.argv[2], "rb"))
    b.static, b.pv = P["v3.1"]
    object.__setattr__(card, "facets", tuple((a * card.s8 / 4096, v) for _, a, v in facets))
    out = {"facets": [list(f) for f in facets], "vector": current,
           "menu": b.run(("RTX PRO 6000",), f"{current} facets, menu")}
    b.pb.tile_methods = b.patched
    cap0 = b.pb.cap
    b.pb.RES = b.pb.RES + ("xu",)
    b.pb.cap = lambda c: {**cap0(c), "xu": b.XU_LANES if c.family == "sm_120" else math.inf}
    out["census routes"] = b.run(("RTX PRO 6000",), f"{current} facets, + b1 census")
    json.dump({lab: ({" | ".join(map(str, k)): v for k, v in r.items()} if isinstance(r, dict) else r)
               for lab, r in out.items()}, open(sys.argv[3] + ".json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
