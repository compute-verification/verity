"""The block's sample sized to a CPU proving budget: `h200.py`'s windows, one a template, each sampling a uniform
`subset:k` of its n units, priced by C-Flock `prove --zk` on the CPU nodes instead of the GPU fit, and the share of
each template's units the budget certifies.

    python benchmarks/sampling_plan/cpu_budget.py --plan PLAN.json --slots 17 --hours 2.5 [--out DIR]

* **A statement** holds up to `UNITS` sampled proof blocks of one template, at no more cost than one, since C-Flock pads a
  statement to 8 instance slots. Its m is k_log + 3, with k_log = ceil(log2(proved ANDs)), the unit's ANDs with its
  rows' openings (`h200.py`'s ``proved_ands``). C-Flock proves no instance past k_log `K_LOG_MAX`, so a unit with more
  ANDs is sampled whole and proved as its least number of parts under it, each part a template of its own (a row cut
  where a partial sum is committed).
* **Its cost** comes from `CALIBRATION`: firewalled `--zk` cold sessions on Genoa proofs slots, every one accepted by
  the live verifier and by Lean. Prove wall time is scaled to a 42-core slot by thread count, with the peak summed
  over the prover, the live verifier and the ZK server. Lean `verify --zk` (one core) and staging are counted apart.
  Past m = 30 time is extrapolated at 2.2x and memory at 2x per m, and a statement takes as many 120 GB slots as its
  memory needs. No statement is smaller than k_log `K_LOG_MIN` (m = 25), so a smaller unit costs the m = 25 point.
* **The plan**: every template draws k = ceil(ln(S/1%)/share), rounded up to fill its last statement and at most n.
  The share is the least whose statements fit `slots` x `hours` of slot time. Each window then has fewer than
  K0 = ln(S/delta) n/k units wrong outside delta (`IntegrityPeriodLog`, delta split over the S windows), and its
  windows over any period fewer than that in all (`IntegrityHorizon`), as in `h200.py`.
"""
from __future__ import annotations

import argparse
import json
from math import ceil, log, log2
from pathlib import Path

DELTAS = (1e-2, 1e-6)
PERIODS = {"week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}
SERVERS = 10_000
UNITS = 8
SLOT_CORES, SLOT_GB = 42, 120
#: m: (prove wall s, threads, peak GiB, Lean verify s, stage s, run): `zk_cost_cpu.sh`'s points
CALIBRATION = {25: (21.0, 34, 5.0, 9.5, 0.8, "r20261026-005203-c613"),
               26: (32.6, 34, 8.2, 13.7, 1.3, "r20261026-005203-c613"),
               27: (42.4, 34, 14.0, 20.8, 2.3, "r20261025-233133-6b59"),
               28: (119.7, 30, 28.1, 70.4, 9.1, "r20261025-233158-1a1c"),
               29: (194.3, 30, 55.9, 138.5, 18.8, "r20261025-234257-5cf3"),
               30: (384.1, 42, 107.4, 276.8, 39.4, "r20261025-234304-229c")}
#: the least k_log a statement takes: a unit of 21,963 ANDs (ScaledMm K=128) and one of 139,283 (K=1024) both prove at 22
K_LOG_MIN = 22
#: `Stmt.InRange`'s k_log bound (`live/src/circuit.rs`'s IN_RANGE_K_LOG, `verity_flock.circuit.K_MAX`): C-Flock refuses an
#: instance past it, so a unit with more proved ANDs is cut into parts, each its own template.
K_LOG_MAX = 27


def k_log(proved_ands: int) -> int:
    return max(K_LOG_MIN, ceil(log2(max(proved_ands, 2))))


def statement(m: int) -> dict:
    """One statement at m on 42-core slots: its wall time, slots, memory, Lean verify and staging (single-core s), and
    whether it is measured."""
    lo, hi = min(CALIBRATION), max(CALIBRATION)
    at = min(max(m, lo), hi)
    wall, threads, gib, verify, stage, _ = CALIBRATION[at]
    wall *= threads / SLOT_CORES
    if m > hi:
        wall, gib, verify, stage = wall * 2.2 ** (m - hi), gib * 2 ** (m - hi), verify * 2 ** (m - hi), stage * 2 ** (m - hi)
    slots = max(1, ceil(gib * 1.074 / SLOT_GB))
    return {"m": m, "wall_s": wall, "slots": slots, "peak_gib": gib, "verify_s": verify, "stage_s": stage,
            "measured": lo <= m <= hi}


def window(t: dict, k: int, windows: int) -> dict:
    n = t["units"]
    k = min(n, ceil(k / UNITS) * UNITS)
    parts = ceil(t["proved_ands"] / 2 ** K_LOG_MAX)
    st = statement(k_log(ceil(t["proved_ands"] / parts)) + 3)
    count = parts * ceil(k / UNITS)
    K0 = {str(d): min(ceil(log(windows / d) * n / k), n - k + 1) for d in DELTAS}
    pooled = {str(d): min(log(windows / d) * n / k, n - k + 1) for d in DELTAS}
    return {"template": t["template"], "units": n, "k": k, "parts": parts, "statements": count, "m": st["m"],
            "measured": st["measured"],
            "slot_s": count * st["wall_s"] * st["slots"], "slots_each": st["slots"], "wall_each_s": st["wall_s"],
            "peak_gib": st["peak_gib"], "core_s": count * (st["verify_s"] + st["stage_s"]), "K0": K0, "pooled": pooled,
            "share": {d: K0[d] / n for d in K0}}


def plan(ts: list[dict], share: float) -> list[dict]:
    k = ceil(log(len(ts) / DELTAS[0]) / share)
    return [window(t, k, len(ts)) for t in ts]


def cost(ws: list[dict]) -> float:
    return sum(w["slot_s"] for w in ws)


def budget_share(ts: list[dict], budget: float) -> float | None:
    """The least share whose plan's statements fit `budget` slot-seconds, or None when share 1 doesn't."""
    if cost(plan(ts, 1.0)) > budget:
        return None
    lo, hi = 1e-9, 1.0
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        lo, hi = (lo, mid) if cost(plan(ts, mid)) <= budget else (mid, hi)
    return hi


def report(o: dict, slots: int, hours: float) -> dict:
    ts = [{"template": w["template"], "units": w["units"], "proved_ands": w["proved_ands"]} for w in o["windows"]]
    budget = slots * hours * 3600
    s = budget_share(ts, budget)
    ws = plan(ts, s if s is not None else 1.0)
    units = sum(t["units"] for t in ts)
    periods = []
    for fleet, count in (("1 H200", 1), (f"{SERVERS:,} H200s", SERVERS)):
        for name, seconds in PERIODS.items():
            blocks = count * seconds / o["block_s"]
            each = {str(d): sum(min(ceil(log(len(ws) * blocks / d) * w["units"] / w["k"]), w["units"] - w["k"] + 1)
                                for w in ws) for d in DELTAS}
            pooled = {str(d): sum(w["pooled"][str(d)] for w in ws) for d in DELTAS}
            periods.append({"period": name, "fleet": fleet, "blocks": blocks, "pooled": pooled,
                            "pooled_share": {d: v / (units * blocks) for d, v in pooled.items()}, "each_block": each,
                            "each_share": {d: v / units for d, v in each.items()}})
    return {"plan_of": {"block_s": o["block_s"], "units": units, "windows": len(ws), "gpu_share": o["share"],
                        "gpu_proving_s": o["cost"]["proving"]},
            "budget": {"slots": slots, "slot_cores": SLOT_CORES, "slot_gb": SLOT_GB, "hours": hours, "slot_s": budget},
            "calibration": {str(m): dict(zip(("wall_s", "threads", "peak_gib", "verify_s", "stage_s", "run"), v))
                            for m, v in CALIBRATION.items()},
            "share": s, "floor_slot_s": cost(plan(ts, 1.0)), "slot_s": cost(ws),
            "core_s": sum(w["core_s"] for w in ws), "statements": sum(w["statements"] for w in ws),
            "drawn": sum(w["k"] for w in ws), "windows": ws, "periods": periods}


def render(r: dict) -> str:
    b = r["budget"]
    s = r["share"]
    lines = [f"# The H200 block's draw on {b['slots']} CPU slots ({b['slot_cores']} cores, {b['slot_gb']} GB) for "
             f"{b['hours']:g} h", "",
             (f"Share at 1%: **{s:.3g}** of each template's units ({r['drawn']:,} drawn in {r['statements']:,} statements, "
              f"{r['slot_s']:.4g} of {b['slot_s']:.4g} slot-s; Lean verify and staging {r['core_s']:.4g} core-s)."
              if s is not None else f"No plan fits: the floor (k = 9 a template) is {r['floor_slot_s']:.4g} slot-s."), "",
             "| template | units a block | k | parts | statements | m | slots each | wall each s | slot-s | K0 at 1% "
             "| share at 1% | K0 at 1e-6 | share at 1e-6 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in r["windows"]:
        lines.append(f"| {w['template']} | {w['units']:,} | {w['k']:,} | {w['parts']} | {w['statements']} | {w['m']}"
                     f"{'' if w['measured'] else '*'} | {w['slots_each']} | {w['wall_each_s']:.0f} | {w['slot_s']:.4g} | "
                     f"{w['K0']['0.01']:,} | {w['share']['0.01']:.3g} | {w['K0']['1e-06']:,} | {w['share']['1e-06']:.3g} |")
    lines += ["", f"\\* m outside the calibration ({min(CALIBRATION)} to {max(CALIBRATION)}): extrapolated.", "",
              "Faulty proof blocks over a period, summed over the templates: pooled over all its blocks (`IntegrityHorizon`, "
              "with its share of the period's units), and in every block at once (delta split over the period's windows, "
              "with its share of a block's units).", "",
              "| period | fleet | blocks | pooled at 1% | share | pooled at 1e-6 | share | each block at 1% | at 1e-6 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for p in r["periods"]:
        lines.append(f"| {p['period']} | {p['fleet']} | {p['blocks']:,.0f} | {p['pooled']['0.01']:.4g} | "
                     f"{p['pooled_share']['0.01']:.3g} | {p['pooled']['1e-06']:.4g} | {p['pooled_share']['1e-06']:.3g} | "
                     f"{p['each_block']['0.01']:.4g} ({p['each_share']['0.01']:.3g}) | {p['each_block']['1e-06']:.4g} "
                     f"({p['each_share']['1e-06']:.3g}) |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", type=Path, required=True, help="h200.py's plan.json")
    ap.add_argument("--slots", type=int, default=17)
    ap.add_argument("--hours", type=float, default=2.5)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    r = report(json.loads(a.plan.read_text()), a.slots, a.hours)
    text = render(r)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "cpu-budget.json").write_text(json.dumps(r, indent=1))
        (a.out / "cpu-budget.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
