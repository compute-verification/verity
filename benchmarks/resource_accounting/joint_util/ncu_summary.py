"""ncu's per-kernel counters (a `--csv --page raw --print-units base` export of `loop.py --count UNITS` under
`--profile-from-start off`) as one protocol unit's (a call's or a pass's) consumption of each resource.

Each `<counter>.avg.pct_of_peak_sustained_elapsed` times its unit's elapsed cycles (`sm__cycles_elapsed.avg` for the
SMs and L1, `lts__cycles_elapsed.avg` for L2, `dram__cycles_elapsed.avg` for DRAM) is the kernel's busy time at peak,
in that unit's cycles; summed over the window's kernels and divided by UNITS, a unit's. Busy cycles are a property of
the instruction stream (and, for L2 and DRAM, of what misses), not of what runs beside it, so a run's utilization of a
resource is the sum over its protocols of busy cycles per unit times units per second, over that unit's clock.
Stdlib only.

    python3 ncu_summary.py RAW.csv UNITS > SUMMARY.json
"""
from __future__ import annotations

import csv
import json
import sys

PCT = ".avg.pct_of_peak_sustained_elapsed"
CLOCK = {"sm": "sm__cycles_elapsed.avg", "l1tex": "sm__cycles_elapsed.avg", "lts": "lts__cycles_elapsed.avg",
         "dram": "dram__cycles_elapsed.avg"}


def num(s: str) -> float | None:
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def main(path: str, units: int) -> dict:
    rows = list(csv.reader(open(path)))
    head = next(i for i, r in enumerate(rows) if "Kernel Name" in r)
    cols, data = rows[head], [r for r in rows[head + 2:] if len(r) == len(rows[head])]
    at = {c: i for i, c in enumerate(cols)}
    metrics = [c for c in cols if "__" in c]
    busy: dict[str, float] = {}
    sums: dict[str, float] = {}
    kernels: dict[str, dict] = {}
    for r in data:
        v = {c: num(r[at[c]]) for c in metrics}
        name = r[at["Kernel Name"]].split("(")[0][:80]
        k = kernels.setdefault(name, {"count": 0, "ns": 0.0, "grid": r[at["Grid Size"]] if "Grid Size" in at else None,
                                      "block": r[at["Block Size"]] if "Block Size" in at else None, "busy": {}})
        k["count"] += 1
        k["ns"] += v.get("gpu__time_duration.sum") or 0.0
        for c in metrics:
            if v[c] is None:
                continue
            if c.endswith(PCT):
                base = c[: -len(PCT)]
                cyc = v.get(CLOCK.get(base.split("__")[0], "sm__cycles_elapsed.avg")) or 0.0
                b = v[c] / 100.0 * cyc
                busy[base] = busy.get(base, 0.0) + b
                k["busy"][base] = k["busy"].get(base, 0.0) + b
            elif c.endswith(".sum"):
                sums[c] = sums.get(c, 0.0) + v[c]
    clocks = {}
    for c in ("sm__cycles_elapsed.avg.per_second", "lts__cycles_elapsed.avg.per_second", "dram__cycles_elapsed.avg.per_second"):
        xs = sorted(num(r[at[c]]) for r in data if c in at and num(r[at[c]]))
        if xs:
            clocks[c] = xs[len(xs) // 2]
    peak = {}
    for c in cols:
        if c.endswith(".peak_sustained"):
            xs = [num(r[at[c]]) for r in data if num(r[at[c]]) is not None]
            if xs:
                peak[c] = max(xs)
    per_unit = {b: x / units for b, x in busy.items()}
    kt = {nm: {"count": k["count"] / units, "ns": k["ns"] / units, "grid": k["grid"], "block": k["block"],
               "busy": {b: x / units for b, x in sorted(k["busy"].items())}} for nm, k in kernels.items()}
    return {"units": units, "kernels_profiled": len(data), "kernel_ns_per_unit": sum(k["ns"] for k in kernels.values()) / units,
            "busy_cycles_per_unit": dict(sorted(per_unit.items())), "sums_per_unit": {c: x / units for c, x in sorted(sums.items())},
            "clocks_median": clocks, "peak_sustained": peak,
            "kernels": dict(sorted(kt.items(), key=lambda kv: -kv[1]["ns"]))}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], int(sys.argv[2])), indent=1))
