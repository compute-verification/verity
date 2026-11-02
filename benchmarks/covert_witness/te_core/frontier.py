"""The search's frontier, read from the ledger's index.

    python -m te_core.frontier [--root DIR] [--json OUT] [--pin COMMIT ...]

It reads only the submissions measured by the pinned measurement commit or an earlier commit its `pinned.json` names as
measuring the same version (`same_version`; covert_witness's `pin.py`; `--pin`, repeated, names others), and leaves
out, each marked with its reason (`status`): an invalid record, an unscored one, one measured by any other commit
(`other-version`), one whose E digest (`output.e_sha256`) equals an earlier record's (`duplicate-e`: the
first to submit an E is the one its compute is counted for), and one with covered steps ingested by a te_core before
3.0 (`re-ingest`: 3.0 prices ρ by thresholds; `python -m te_core.ledger ingest RUN_DIR` prices it again). Per (k, r)
cell of the grid (score.K_BITS x score.R):

- the frontier: the valid submissions no other beats on both |C| ρ and val bits per byte;
- the elites: per niche (mechanism tag x β range x φ range x start), the submission with the least val bits per byte,
  the least |C| ρ breaking ties;
- the coverage map: every niche of the observed mechanism tags, β and φ ranges and starts, the empty ones listed;
- W*(l*), honest(l*) and the effective slowdown W*/honest at l* = the start's bits per byte less 0.5%, 1%, 2% and 5%,
  honest(l*) being the least |C| over the reference runs (`reference` records) on that start.

`candidates(rows, T)` are the frontier members and elites at tier T - 1 not yet promoted to T; which of them are promoted
is decided on the sealed promotion set, one bit each (covert_witness's `promote.py --tier T`, run as the scorer).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .ledger import Ledger, coverage_of
from .score import K_BITS, R, cell, effective_slowdown, honest, w_star

BETA_EDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
PHI_EDGES = (0.1, 0.25, 0.5, 0.75)
TARGETS = (0.005, 0.01, 0.02, 0.05)
KEYS = [cell(k, r) for k in K_BITS for r in R]


def bin_of(x: float, edges: tuple[float, ...]) -> int:
    return sum(x >= e for e in edges)


#: the first te_core whose ingest prices ρ by thresholds (a row's `te_core`): an earlier ingest could price a run with
#: covered steps at its cheapest step's β, so such a row counts once its run is ingested again
PRICED_FROM = (3, 0)


def version(v) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return ()


def pinned() -> tuple[str, ...]:
    """The pin's commit, then the earlier commits that measure the same version (`same_version`)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pin import pinned as read

    pin = read()
    return (pin["commit"], *pin.get("same_version", []))


def commits(pin: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return (pin,) if isinstance(pin, str) else tuple(pin)


def status(rows: list[dict], pin: str | tuple[str, ...]) -> dict[str, str]:
    """Each left-out record's reason: invalid, unscored, other-version, duplicate-e, re-ingest. `pin` is a commit or the
    commits of one version. An E repeats an earlier record's when that record was measured at the pin or is another
    agent's: an agent's own relaunch at a new pin of an E it had at an old one counts."""
    pins = set(commits(pin))
    out, first = {}, {}
    for r in sorted(rows, key=lambda r: (r.get("written_ts", 0), r["id"])):
        e = (r.get("output") or {}).get("e_sha256")
        if e and any(at_pin or agent != r.get("agent") for at_pin, agent in first.get(e, ())):
            out[r["id"]] = "duplicate-e"
        elif r.get("invalid"):
            out[r["id"]] = "invalid"
        elif r.get("val_bpb") is None:
            out[r["id"]] = "unscored"
        elif r.get("measurement_commit") not in pins:
            out[r["id"]] = "other-version"
        elif version(r.get("te_core")) < PRICED_FROM and coverage_of(r.get("families", [])):
            out[r["id"]] = "re-ingest"
        if e and out.get(r["id"]) != "duplicate-e":
            first.setdefault(e, set()).add((r.get("measurement_commit") in pins, r.get("agent")))
    return out


def valid(rows: list[dict], pin: str | tuple[str, ...] | None = None) -> list[dict]:
    left_out = status(rows, pin or pinned())
    return [r for r in rows if r["id"] not in left_out]


def cost(r: dict, key: str) -> float:
    return r["macs"] * r["rho"][key]


def pareto(rows: list[dict], key: str) -> list[dict]:
    pts = sorted(rows, key=lambda r: (cost(r, key), r["val_bpb"]))
    out, best = [], math.inf
    for r in pts:
        if r["val_bpb"] < best:
            out.append(r)
            best = r["val_bpb"]
    return out


def niche(r: dict, key: str) -> tuple:
    return (r.get("mechanism", ""), bin_of(r["beta"][key], BETA_EDGES), bin_of(r["phi"][key], PHI_EDGES), r["start"])


def elites(rows: list[dict], key: str) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for r in rows:
        n = niche(r, key)
        if n not in out or (r["val_bpb"], cost(r, key)) < (out[n]["val_bpb"], cost(out[n], key)):
            out[n] = r
    return out


def coverage_map(rows: list[dict], key: str) -> dict:
    mechs = sorted({r.get("mechanism", "") for r in rows})
    starts = sorted({r["start"] for r in rows})
    filled = set(elites(rows, key))
    every = [(m, b, f, s) for m in mechs for b in range(len(BETA_EDGES) + 1) for f in range(len(PHI_EDGES) + 1)
             for s in starts]
    return {"filled": len(filled), "cells": len(every), "empty": [list(c) for c in every if c not in filled]}


def targets(rows: list[dict], key: str) -> list[dict]:
    out = []
    for start in sorted({r["start"] for r in rows}):
        on = [r for r in rows if r["start"] == start]
        l0 = next((r["start_bpb"] for r in on if r.get("start_bpb") is not None), None)
        if l0 is None:
            continue
        subs, refs = [r for r in on if not r.get("reference")], [r for r in on if r.get("reference")]
        for f in TARGETS:
            l_star = l0 * (1 - f)
            out.append({"start": start, "drop": f, "l_star": l_star, "w_star": w_star(subs, l_star, key),
                        "honest": honest(refs, l_star), "effective_slowdown": effective_slowdown(subs, refs, l_star, key)})
    return out


def report(rows: list[dict], pin: str | tuple[str, ...] | None = None) -> dict:
    rows = valid(rows, pin)
    out = {}
    for key in KEYS:
        el = elites(rows, key)
        out[key] = {
            "frontier": [{"id": r["id"], "cost": cost(r, key), "val_bpb": r["val_bpb"], "rho": r["rho"][key]}
                         for r in pareto(rows, key)],
            "elites": [{"niche": list(n), "id": r["id"], "val_bpb": r["val_bpb"], "cost": cost(r, key)}
                       for n, r in sorted(el.items())],
            "coverage": coverage_map(rows, key),
            "targets": targets(rows, key),
        }
    return out


def candidates(rows: list[dict], tier: int, pin: str | tuple[str, ...] | None = None) -> list[str]:
    """The frontier members and elites, in any cell, among submissions run at tier - 1 and not yet promoted to `tier`."""
    pool = [r for r in valid(rows, pin) if r["tier"] == tier - 1 and r.get("promoted", 0) < tier]
    ids = set()
    for key in KEYS:
        ids |= {r["id"] for r in pareto(pool, key)} | {r["id"] for r in elites(pool, key).values()}
    return sorted(ids)


def markdown(rep: dict) -> str:
    lines = ["| cell | frontier | elites | niches filled | W*/honest at -0.5%, -1%, -2%, -5% |", "|---|---|---|---|---|"]
    for key, v in rep.items():
        eff = ", ".join("—" if t["effective_slowdown"] is None else f"{t['effective_slowdown']:.3g}" for t in v["targets"])
        lines.append(f"| {key} | {len(v['frontier'])} | {len(v['elites'])} | {v['coverage']['filled']}/"
                     f"{v['coverage']['cells']} | {eff} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--pin", action="append", default=None,
                    help="a measurement commit, repeated for one version's (default covert_witness's pin.py)")
    ap.add_argument("--promote", type=int, choices=(1, 2), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--as", dest="agent", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.promote:
        print(f"promotions are decided on the sealed promotion set, one bit each: from benchmarks/covert_witness/, "
              f"`python promote.py --tier {a.promote}` (the scorer's)", file=sys.stderr)
        return 2
    rows = Ledger(a.root).index()
    pin = tuple(a.pin) if a.pin else pinned()
    rep = report(rows, pin)
    left_out = status(rows, pin)
    if a.json:
        a.json.write_text(json.dumps({"pin": list(pin), "left_out": left_out, "cells": rep}, indent=1) + "\n")
    print(markdown(rep))
    reasons = sorted(set(left_out.values()))
    print(f"pin {' + '.join(c[:12] for c in pin)}: {len(rows) - len(left_out)} of {len(rows)} submissions; left out: "
          + (", ".join(f"{sum(v == x for v in left_out.values())} {x}" for x in reasons) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
