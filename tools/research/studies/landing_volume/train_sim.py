"""Lane trains through DESIGN's replay simulator (`tools/research/landing_sim/sim.py` on cursor/top-landing-design-8b2a, its
data art in LANDING_STUDY_DIR): a rider is a marked PR that its carrier, also in the replay, brought with it. Under a train
it takes no admission, is no member of `next` and costs no pass time; its carrier takes on its Lean flag, its break draw
and (for `set`, a carrier of another lane's PR) its review delay; it reaches `main` when its carrier does.

Riders: landed PRs from trains.py's CSV (carrier = the first later landing that carried it), waiting ones from
carried_csv.py's CSV (marked, carried by a marked PR).

    LANDING_STUDY_DIR=/tmp/ld python3 train_sim.py SIM_PY TRAINS_CSV CARRIED_CSV
"""
from __future__ import annotations

import csv
import importlib.util
import sys


def riders(trains_csv: str, carried_csv: str, cross: bool) -> dict[int, int]:
    out: dict[int, int] = {}
    for r in csv.DictReader(open(trains_csv)):
        if r["first_carrier"] and (cross or r["same_lane"] == "1"):
            out[int(r["pr"])] = int(r["first_carrier"])
    for r in csv.DictReader(open(carried_csv)):
        if r["state"] not in ("ready", "on_next") or r["carrier_state"] not in ("ready", "on_next"):
            continue
        if r["category"] == "carried-marked" or cross and r["category"] == "carried-cross-lane":
            out.setdefault(int(r["pr"]), int(r["carrier"]))
    return out


def top(n: int, rides: dict[int, int]) -> int:
    seen = set()
    while n in rides and n not in seen:
        seen.add(n)
        n = rides[n]
    return n


def run(sim, params, p_break, rides: dict[int, int], cross: bool, marked_first: bool = False):
    """``marked_first``: a PR rides only when its carrier was marked before it (the queue refuses a mark on a PR a marked PR
    of its owner already contains), so no rider waits for a carrier nobody has marked yet."""
    prs, _ = sim.load(p_break)
    by = {p.n: p for p in prs}
    for p in prs:
        p.r0 = p.r
        if params.get("review_cap") is not None:
            p.v = min(p.v, params["review_cap"])
    attached: dict[int, list] = {}
    for p in prs:
        if p.n not in rides:
            continue
        c = by.get(top(p.n, rides))
        if c is None or c is p or marked_first and c.r > p.r:
            continue
        attached.setdefault(c.n, []).append(p)
        c.lean |= p.lean
        c.breaks |= p.breaks
        c.lowrisk &= p.lowrisk
        if cross:
            c.v = max(c.v, p.r + p.v - c.r)
    ride = {p.n for ps in attached.values() for p in ps}
    rng = sim.random.Random(1)
    moves, diag = sim.sim_current([p for p in prs if p.n not in ride], params, rng)
    for c, ps in attached.items():
        for p in ps:
            p.main = by[c].main
    return prs, moves, len(ride), diag


def main(sim_py, trains_csv, carried_csv):
    spec = importlib.util.spec_from_file_location("sim", sim_py)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    base = dict(sim.CURRENT)
    s3 = dict(base, revert=True, pin=False, set_landing=True, review_cap=120)
    s4 = dict(s3, admit=None, pass0=10, pass1=0.0)
    s7 = dict(s4, run_min=60, flake=0.0, lean_min=45)
    for pb in (0.02, 0.05):
        print(f"== p_break {pb}")
        for label, params in (("current", base), ("steps 1-3", s3), ("steps 1-4", s4), ("steps 1-7", s7)):
            for name, cross, first in (("", None, False), (" + lane trains", False, False),
                                       (" + lane trains, carrier marked first", False, True),
                                       (" + carrier sets (any lane)", True, False)):
                rides = {} if cross is None else riders(trains_csv, carried_csv, cross)
                prs, moves, n, diag = run(sim, dict(params), pb, rides, bool(cross), first)
                pend = sum(x[0] for x in diag) / max(1, len(diag))
                sim.report(label + name, prs, moves, extra=f"riders={n} pending mean={pend:.0f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
