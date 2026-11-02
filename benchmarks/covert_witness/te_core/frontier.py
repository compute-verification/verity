"""The search's frontier, read from the ledger's index.

    python -m te_core.frontier [--root DIR] [--json OUT] [--pin COMMIT ...] [--facts FACTS.json] [--formalism]

It reads only the submissions measured by the pinned measurement commit or an earlier commit its `pinned.json` names as
measuring the same version (`same_version`; covert_witness's `pin.py`; `--pin`, repeated, names others), and leaves
out, each marked with its reason (`status`): an invalid record, one whose read-outs `cw.readout_leaks` lists
(`read-out-leak`, invalid: the coordinator's 14:22Z rule), one that updates the tied embedding with no block below the
top (`tied-lookup`: the scorer's rule, coordinator 13:32Z), an unscored one, one measured by any other commit
(`other-version`), one whose E digest (`output.e_sha256`) equals an earlier record's (`duplicate-e`: the
first to submit an E is the one its compute is counted for), one with covered steps ingested by a te_core before
3.0 (`re-ingest`: 3.0 prices ρ by thresholds; `python -m te_core.ledger --agent AGENT ingest RUN_DIR ... --run-id RUN`
prices it again), and one a later ingest of its run supersedes (`re-ingested`). A row ingested before 3.13 carries
neither the read-outs nor the changed tensors; `--facts` gives them by run id (`with_facts`). Per (k, r) cell of the
grid (score.K_BITS x score.R):

- the frontier: the valid submissions no other beats on both |C| ρ and val bits per byte;
- the elites: per niche (mechanism tag x β range x φ range x start), the submission with the least val bits per byte,
  the least |C| ρ breaking ties;
- the coverage map: every niche of the observed mechanism tags, β and φ ranges and starts, the empty ones listed;
- W*(l*), honest(l*) and the effective slowdown W*/honest at l* = the start's bits per byte less 0.5%, 1%, 2% and 5%,
  by the coordinator's 14:57Z ruling (`target`): honest(l*) is the least |C| over every valid submission of any class
  or direction that reaches l* by a complete schedule (`schedule_ends`), W*(l*) the least |C| ρ over the same ones, so
  W*/honest ≥ 1; where nothing complete reaches l*, any checkpoint that does, labelled `UPPER_BOUND`. The reference
  directions' honest(l*) alone, the rule before, is printed beside it as context. The table prints training FLOPs,
  2|C|.

With `--formalism` it scores the valid submissions by the training formalism v1 instead (te_core.formalism): |C| in
Boolean gates, ρ at every (c, k) of the start's policy grid (te_core.policy), and per start and l* honest H, W* in each
cell and S = W*/H by the same 14:57Z rule (`formalism_report`), with every quantity's source.

`candidates(rows, T)` are the frontier members and elites at tier T - 1 not yet promoted to T; which of them are promoted
is decided on the sealed promotion set, one bit each (covert_witness's `promote.py --tier T`, run as the scorer).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from . import formalism, policy
from .ledger import Ledger, coverage_of, version
from .score import K_BITS, R, cell

BETA_EDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
PHI_EDGES = (0.1, 0.25, 0.5, 0.75)
TARGETS = (0.005, 0.01, 0.02, 0.05)
KEYS = [cell(k, r) for k in K_BITS for r in R]
REF_DIRECTIONS = ("ref-optim", "ref-data")
UPPER_BOUND = "upper bound (checkpoint)"


def bin_of(x: float, edges: tuple[float, ...]) -> int:
    return sum(x >= e for e in edges)


#: the first te_core whose ingest prices ρ by thresholds (a row's `te_core`): an earlier ingest could price a run with
#: covered steps at its cheapest step's β, so such a row counts once its run is ingested again
PRICED_FROM = (3, 0)


def _run_key(r: dict):
    return (r["run_id"], r.get("measurement_commit"), r.get("checkpoint_step")) if r.get("run_id") else None


def _run_of(r: dict):
    return (r["run_id"], r.get("measurement_commit")) if r.get("run_id") else r["id"]


def arg_value(args: str | list | None, flag: str) -> str | None:
    """`flag`'s value in a program's arguments (a string, or `script_args`' list), "" when it is given none, None when
    absent."""
    a = list(args) if isinstance(args, list) else str(args or "").split()
    for i, t in enumerate(a):
        if t == flag:
            return "" if i + 1 == len(a) or a[i + 1].startswith("--") else a[i + 1]
        if t.startswith(flag + "="):
            return t[len(flag) + 1:]
    return None


def fork_ladder(r: dict) -> bool:
    """The scorer's rule: a reference direction's train.py run with --forks and an empty --snap-at checkpoints only at
    its cooldown branches' ends (a branch decays the rate to 0, is snapshotted, and the run returns to where it forked)
    and its own end, so each of its checkpoints ends a complete schedule of that length (its |C| includes the earlier
    branches', so honest is overstated there, never understated)."""
    return (r.get("direction") in REF_DIRECTIONS and bool(arg_value(r.get("args"), "--forks"))
            and arg_value(r.get("args"), "--snap-at") == "")


def schedule_ends(rows: list[dict]) -> set[str]:
    """The ids of the rows that end a complete schedule: each run's last checkpoint (a run is its run id and
    measurement commit; a row without a run id is its own run) and every checkpoint of a fork ladder (`fork_ladder`).
    A checkpoint inside a longer schedule is not what an honest run of that length reaches (coordinator 12:05Z)."""
    top: dict = {}
    for r in rows:
        top[_run_of(r)] = max(top.get(_run_of(r), -1), r.get("checkpoint_step", -1))
    return {r["id"] for r in rows if r.get("checkpoint_step", -1) == top[_run_of(r)] or fork_ladder(r)}


def tied_lookup(changed: list[str], layers: int | None) -> bool:
    """The scorer's rule (coordinator 13:32Z): a record that updates the tied embedding with no block below the top is
    priced wrong, v2.3's counter pricing the updated table's lookup at 0. `changed` is the witness's
    `changed.actual`, `layers` its `structure.layers`."""
    below = [n for n in changed if (m := re.match(r"blocks\.(\d+)\.", n)) and layers and int(m[1]) < layers - 1]
    return "embed" in changed and not below


def with_facts(rows: list[dict], facts: dict[str, dict]) -> list[dict]:
    """The rows with what the rules read beyond the price, by run id (`facts`: run id -> any of `args`, `changed`,
    `layers` and `readout_leaks`, as `ledger.facts` gives them), for rows ingested before te_core 3.13."""
    return [r | facts.get(r.get("run_id"), {}) for r in rows]


def pinned() -> tuple[str, ...]:
    """The pin's commit, then the earlier commits that measure the same version (`same_version`)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pin import pinned as read

    pin = read()
    return (pin["commit"], *pin.get("same_version", []))


def commits(pin: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return (pin,) if isinstance(pin, str) else tuple(pin)


def status(rows: list[dict], pin: str | tuple[str, ...]) -> dict[str, str]:
    """Each left-out record's reason: re-ingested, invalid, read-out-leak, tied-lookup, unscored, other-version,
    duplicate-e, re-ingest. `pin` is a
    commit or the commits of one version. A later ingest of the same run's checkpoint (its run id; a row without one is
    never superseded) supersedes an earlier one (`re-ingested`), which then seeds no duplicate. An E repeats an earlier
    record's when that record was measured at the pin or is another agent's: an agent's own relaunch at a new pin of an
    E it had at an old one counts."""
    pins = set(commits(pin))
    out, first = {}, {}
    ordered = sorted(rows, key=lambda r: (r.get("written_ts", 0), r["id"]))
    latest = {_run_key(r): r["id"] for r in ordered}
    for r in ordered:
        e = (r.get("output") or {}).get("e_sha256")
        key = _run_key(r)
        if key is not None and latest[key] != r["id"]:
            out[r["id"]] = "re-ingested"
            continue
        if e and any(at_pin or agent != r.get("agent") for at_pin, agent in first.get(e, ())):
            out[r["id"]] = "duplicate-e"
        elif r.get("invalid"):
            out[r["id"]] = "invalid"
        elif r.get("readout_leaks"):
            out[r["id"]] = "read-out-leak"
        elif r.get("changed") is not None and tied_lookup(r["changed"], r.get("layers")):
            out[r["id"]] = "tied-lookup"
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


def cost(r: dict, key: str, size: str = "macs") -> float:
    return r[size] * r["rho"][key]


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


def _of(r: dict | None) -> dict | None:
    return r and {"id": r["id"], "checkpoint": r.get("checkpoint_step", r.get("cp")), "direction": r.get("direction")}


def target(on: list[dict], l_star: float, keys=KEYS, reference=None, size: str = "macs") -> dict:
    """The coordinator's 14:57Z ruling at one l*, over one start's valid submissions `on` (each with `id`, `macs`,
    `val_bpb`, `rho` and `schedule_end`). honest(l*) is the least |C| over every submission of any class or direction
    that reaches l* by a complete schedule, and W*(l*) in each cell of `keys` the least |C| ρ over the same
    submissions, so W*/honest ≥ 1 (ρ ≥ 1) and is the effective slowdown. Where no complete schedule reaches l*, both
    read every checkpoint that does and the target is labelled `UPPER_BOUND`: the checkpoint's |C| bounds honest(l*)
    from above. `honest_reference` is the least |C| over the reference directions' complete schedules alone (the rule
    before, as context; `reference` picks them, by default a direction in `REF_DIRECTIONS`). |C| is each row's `size`:
    MACs, or the formalism's gates (`formalism_targets`)."""
    is_ref = reference or (lambda r: r.get("direction") in REF_DIRECTIONS)
    reach = [r for r in on if r.get("val_bpb") is not None and r["val_bpb"] <= l_star]
    ends = [r for r in reach if r.get("schedule_end")]
    pool = ends or reach
    h = min(pool, key=lambda r: r[size], default=None)
    ref = min((r for r in ends if is_ref(r)), key=lambda r: r[size], default=None)
    out = {"l_star": l_star, "bound": UPPER_BOUND if reach and not ends else "", "honest": h and h[size],
           "honest_of": _of(h), "honest_reference": ref and ref[size], "honest_reference_of": _of(ref), "cells": {}}
    for key in keys:
        w = min(pool, key=lambda r: cost(r, key, size), default=None)
        out["cells"][key] = {"w_star": w and cost(w, key, size), "of": _of(w), "rho": w and w["rho"][key],
                             "effective_slowdown": cost(w, key, size) / h[size] if w and h[size] else None}
    return out


def start_bpb(on: list[dict]) -> float | None:
    """A start's l0: the whole-pool bits per byte its rows carry (the median, should they differ in the last digits)."""
    v = sorted(r["start_bpb"] for r in on if r.get("start_bpb") is not None)
    return v[len(v) // 2] if v else None


def targets(rows: list[dict], key: str, ends: set[str] | None = None) -> list[dict]:
    """`target` per start and l* = its l0 less each of `TARGETS`, in cell `key`. `ends` are the rows that end a complete
    schedule (`schedule_ends`, by default over `rows`)."""
    ends = schedule_ends(rows) if ends is None else ends
    out = []
    for start in sorted({r["start"] for r in rows}):
        on = [r | {"schedule_end": r["id"] in ends} for r in rows if r["start"] == start]
        l0 = start_bpb(on)
        if l0 is None:
            continue
        for f in TARGETS:
            t = target(on, l0 * (1 - f), [key])
            c = t["cells"][key]
            out.append({"start": start, "drop": f, "l_star": t["l_star"], "w_star": c["w_star"], "w_star_of": c["of"],
                        "rho": c["rho"], "honest": t["honest"], "honest_of": t["honest_of"], "bound": t["bound"],
                        "honest_reference": t["honest_reference"], "honest_reference_of": t["honest_reference_of"],
                        "effective_slowdown": c["effective_slowdown"]})
    return out


def report(rows: list[dict], pin: str | tuple[str, ...] | None = None) -> dict:
    ends = schedule_ends(rows)
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
            "targets": targets(rows, key, ends),
        }
    return out


def formalism_rows(rows: list[dict], shapes: dict[str, dict]) -> tuple[list[dict], dict[str, str]]:
    """Each row whose start has a shape, with its formalism quantities (`measured`), |C| in gates and ρ in every cell of
    its start's policy grid (te_core.policy); and each row left out, with why (`no-shape`, or `unmeasured`)."""
    out, skipped, cells = [], {}, {}
    for r in rows:
        shape = shapes.get(r["start"])
        if shape is None:
            skipped[r["id"]] = "no-shape"
            continue
        if r["start"] not in cells:
            cells[r["start"]] = {policy.key(c, k): (c, k) for c, k in policy.grid(shape)}
        try:
            m = formalism.measure(r, shape)
        except ValueError:
            skipped[r["id"]] = "unmeasured"
            continue
        out.append(r | {"gates": m.gates, "rho": formalism.rhos(m, cells[r["start"]]), "measured": m})
    return out, skipped


def formalism_targets(rows: list[dict], shapes: dict[str, dict], ends: set[str]) -> dict[str, dict]:
    """Per start (its rows `formalism_rows` gives): its reference shapes, and at each l* = l0 less each of `TARGETS`
    honest H(l*) (least gates over every valid complete schedule reaching l*, `target`), and per policy cell W*(l*) (least
    gates·ρ over the same) and S = W*/H."""
    out = {}
    for start in sorted({r["start"] for r in rows}):
        on = [r | {"schedule_end": r["id"] in ends} for r in rows if r["start"] == start]
        l0 = start_bpb(on)
        if l0 is None:
            continue
        keys = [policy.key(c, k) for c, k in policy.grid(shapes[start])]
        marks = {m: s | {"key": policy.key(10 ** round(math.log10(s["c"])), 10 ** round(math.log10(s["k"])))}
                 for m, s in policy.shapes(shapes[start]).items()}
        out[start] = {"l0": l0, "shape": shapes[start], "marks": marks, "targets": []}
        for f in TARGETS:
            t = target(on, l0 * (1 - f), keys, size="gates")
            out[start]["targets"].append({"drop": f, "l_star": t["l_star"], "honest": t["honest"],
                                          "honest_of": t["honest_of"], "bound": t["bound"],
                                          "cells": {k: {"w_star": v["w_star"], "of": v["of"], "rho": v["rho"],
                                                        "S": v["effective_slowdown"]} for k, v in t["cells"].items()}})
    return out


def formalism_report(rows: list[dict], pin: str | tuple[str, ...] | None = None,
                     shapes: dict[str, dict] | None = None) -> dict:
    """The formalism's frontier over the valid rows (`valid`): `formalism_targets`, with every quantity's source (the
    labels of the rows' measurements) and the rows it couldn't score."""
    shapes = policy.starts() if shapes is None else shapes
    ends = schedule_ends(rows)
    scored, skipped = formalism_rows(valid(rows, pin), shapes)
    labels: dict[str, dict[str, int]] = {}
    for r in scored:
        for k, v in r["measured"].labels.items():
            labels.setdefault(k, {}).setdefault(v, 0)
            labels[k][v] += 1
    return {"starts": formalism_targets(scored, shapes, ends), "sources": labels, "skipped": skipped,
            "scored": len(scored)}


def formalism_markdown(rep: dict) -> str:
    lines = ["S = W*/H per start and target, at each reference shape (te_core.policy; ρ is Cover's target where Fit "
             "fails, |C| in Boolean gates):", "", "| start | target | l* | H (gates) | " + " | ".join(
                 f"S at {m}" for m in policy.MARKS) + " |", "|---|---|---|---|" + "---|" * len(policy.MARKS)]
    for start, v in rep["starts"].items():
        for t in v["targets"]:
            h = "—" if t["honest"] is None else f"{t['honest']:.3g}" + (f", {t['bound']}" if t["bound"] else "")
            ss = []
            for m in policy.MARKS:
                cell = t["cells"].get(v["marks"][m]["key"], {})
                ss.append("—" if cell.get("S") is None else f"{cell['S']:.3g} (ρ {cell['rho']:.3g})")
            lines.append(f"| {start} | -{100 * t['drop']:g}% | {t['l_star']:.4f} | {h} | " + " | ".join(ss) + " |")
    lines += ["", "Sources:"] + [f"- {k}: " + "; ".join(f"{n} rows {s}" for s, n in sorted(v.items()))
                                  for k, v in sorted(rep["sources"].items())]
    return "\n".join(lines)


def candidates(rows: list[dict], tier: int, pin: str | tuple[str, ...] | None = None) -> list[str]:
    """The frontier members and elites, in any cell, among submissions run at tier - 1 and not yet promoted to `tier`."""
    pool = [r for r in valid(rows, pin) if r["tier"] == tier - 1 and r.get("promoted", 0) < tier]
    ids = set()
    for key in KEYS:
        ids |= {r["id"] for r in pareto(pool, key)} | {r["id"] for r in elites(pool, key).values()}
    return sorted(ids)


def _flops(macs: float | None, of: dict | None) -> str:
    return "—" if macs is None else f"{2 * macs:.3g} ({of['id']}@{of['checkpoint']})"


def markdown(rep: dict) -> str:
    lines = ["| cell | frontier | elites | niches filled |", "|---|---|---|---|"]
    for key, v in rep.items():
        lines.append(f"| {key} | {len(v['frontier'])} | {len(v['elites'])} | {v['coverage']['filled']}/"
                     f"{v['coverage']['cells']} |")
    keys = list(rep)
    lines += ["", "W*/honest per cell (training FLOPs, 2|C|; honest over every complete schedule, the reference "
              "directions' alone as context):", "",
              "| start | target | l* | honest | reference directions only | " + " | ".join(keys) + " |",
              "|---|---|---|---|---|" + "---|" * len(keys)]
    for i, t in enumerate(rep[keys[0]]["targets"] if keys else []):
        honest = _flops(t["honest"], t["honest_of"]) + (f", {t['bound']}" if t["bound"] else "")
        eff = ["—" if rep[k]["targets"][i]["effective_slowdown"] is None
               else f"{rep[k]['targets'][i]['effective_slowdown']:.3g}" for k in keys]
        lines.append(f"| {t['start']} | -{100 * t['drop']:g}% | {t['l_star']:.4f} | {honest} | "
                     f"{_flops(t['honest_reference'], t['honest_reference_of'])} | " + " | ".join(eff) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--pin", action="append", default=None,
                    help="a measurement commit, repeated for one version's (default covert_witness's pin.py)")
    ap.add_argument("--facts", type=Path, default=None,
                    help="JSON, run id -> args, changed, layers, readout_leaks, for rows ingested before 3.13")
    ap.add_argument("--formalism", action="store_true",
                    help="score by the training formalism v1 over the policy grid (te_core.formalism, te_core.policy)")
    ap.add_argument("--promote", type=int, choices=(1, 2), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--as", dest="agent", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.promote:
        print(f"promotions are decided on the sealed promotion set, one bit each: from benchmarks/covert_witness/, "
              f"`python promote.py --tier {a.promote}` (the scorer's)", file=sys.stderr)
        return 2
    rows = Ledger(a.root).index()
    if a.facts:
        rows = with_facts(rows, json.loads(a.facts.read_text()))
    pin = tuple(a.pin) if a.pin else pinned()
    if a.formalism:
        rep = formalism_report(rows, pin)
        if a.json:
            a.json.write_text(json.dumps({"pin": list(pin), "left_out": status(rows, pin)} | rep, indent=1) + "\n")
        print(formalism_markdown(rep))
        print(f"pin {' + '.join(c[:12] for c in pin)}: {rep['scored']} scored, {len(rep['skipped'])} without a start shape "
              "or measurement")
        return 0
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
