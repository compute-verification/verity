"""Run DESIGN's landing simulator (`tools/research/landing_sim/sim.py` on cursor/top-landing-design-8b2a, its data in
``LANDING_STUDY_DIR``) with the verdict carry: a PR whose landed head carries an older head's land verdict (`carry_study.py`'s
results, strict) gets its verdict at its mark.

    LANDING_STUDY_DIR=/tmp/ld python3 study/landing-human/sim_carry.py SIM_DIR CARRY.json [CARRY.json ...]"""

import json
import sys

sys.path.insert(0, sys.argv[1])
import sim

carried = set()
for f in sys.argv[2:]:
    carried |= {(r["pr"], r["new"]) for r in json.load(open(f)) if r["carries"]}
heads = {int(r["pr"]): r["head"] for r in json.load(open(sim.DATA + "/trace.json")) if r["pr"]}
load = sim.load


def load_carried(p_break, seed_salt="b"):
    prs, delays = load(p_break, seed_salt)
    for p in prs:
        if (p.n, heads.get(p.n)) in carried:
            p.v = 0.0
    return prs, delays


hit = sum(1 for n, h in heads.items() if (n, h) in carried)
print(f"landed heads that carry: {hit} of {len(heads)}")
for name, over, with_carry in (("current (calibrated)", {}, False), ("current + carry", {}, True),
                               ("current + review cap 120 min", {"review_cap": 120}, False),
                               ("current + carry + review cap 120 min", {"review_cap": 120}, True),
                               ("tree queue", None, False), ("tree queue + carry", None, True)):
    sim.load = load_carried if with_carry else load
    pol = "tree" if over is None else "current"
    params = dict(sim.TREE if pol == "tree" else sim.CURRENT)
    params.update(over or {})
    prs, moves, cores = sim.run(pol, params, 0.04)
    sim.report(name, prs, moves)
