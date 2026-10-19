"""RT3-DC's timings as costs: a unit's SMSP-cycles in each checkpoint form, the tensor beside it, and the atomic units' rates.

Reads a `run.sh time` output directory (probe.py's `probe/rows.jsonl` and `probe/times.jsonl`) and prints, a probe a line:

- fold probes (lr_*, wgcp_*, wgli_redux, tcU_*): a unit's SMSP-cycles (4096 words a warp; 64 IMAD a lane), 256 over the
  loop's IMAD a SM-cycle, so an issue-bound loop of N slots a unit reads N at one issue a cycle; and its excess over the
  same loop with no checkpoint (lr_none, wgcp_none, tcU_none): what the checkpoint form costs a unit;
- beside-tensor probes: the tensor's rate over its rate alone (wg_s8_n256_a1, tc_i8_m128_n256), the B200's from the MMA
  warp's own stop (4 tcgen05.mma an iteration over the stop's cycles, block by block);
- atomic probes (at_*): warp atomics and lane atomics a SM-cycle.

Run: `python3 benchmarks/pouw/rt3_dc/analyze.py OUT [--json FILE]`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

BASE = {"lr": "lr_none", "lr8": "lr_none", "wgcp": "wgcp_none", "tcU": "tcU_none"}
ALONE = {"wgcp": "wg_s8_n256_a1", "wgli": "wg_s8_n256_a1", "tcU": "tc_i8_m128_n256"}


def imad_rate(row: dict) -> float | None:
    """The fold's IMAD a SM-cycle: the loop holding IMAD (the lane loop; an until probe's until loop)."""
    hits = [v for k, v in row["rates"].items() if k.split(":", 1)[1] == "IMAD"]
    return hits[0] if hits else None


def tensor_rate(row: dict, times: list[dict]) -> float | None:
    """Tensor instructions a SM-cycle: the fixed MMA loop's, by its stop when the probe has an until loop."""
    keys = [k for k in row["rates"] if "GMMA" in k or "UTCIMMA" in k]
    if not keys:
        return None
    if not any(t.get("stop") for t in times):
        return row["rates"][keys[0]]
    # the until probes' MMA warp: 4 tcgen05.mma an iteration (UTCIMMA*=4), done at its stop
    return statistics.median(statistics.median([4 * t["iters"] / s for s in t["stop"] if s > 0]) for t in times)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    rows = {r["probe"]: r for r in map(json.loads, (a.out / "probe" / "rows.jsonl").read_text().splitlines())}
    times: dict[str, list[dict]] = {}
    for ln in (a.out / "probe" / "times.jsonl").read_text().splitlines():
        if ln.startswith("{"):
            x = json.loads(ln)
            if x.get("op") == "time":
                times.setdefault(x["probe"], []).append(x)
    out = {}
    for name, r in rows.items():
        rec = {"probe": name, "spread": r["spread"], "sm_mhz": r["sm_mhz"], "stop_gap": r.get("stop_gap"),
               "tags": r["tags"]}
        im = imad_rate(r)
        if im:
            rec["smsp_cycles_a_unit"] = 256.0 / im
        tr = tensor_rate(r, times.get(name, []))
        if tr is not None:
            rec["tensor_a_sm_cycle"] = tr
        at = [v for k, v in r["rates"].items() if "ATOMS" in k or "REDG" in k]
        if name.startswith("at"):
            rec["warp_atomics_a_sm_cycle"] = at[0]
            rec["lane_atomics_a_sm_cycle"] = 32 * at[0]
            rec["sm_cycles_a_warp_atomic"] = 1 / at[0]
        out[name] = rec
    for name, rec in out.items():
        fam = name.split("_", 1)[0]
        b = out.get(BASE.get(fam, ""))
        if b and "smsp_cycles_a_unit" in rec and b is not rec:
            rec["over_none"] = rec["smsp_cycles_a_unit"] - b["smsp_cycles_a_unit"]
        al = out.get(ALONE.get(fam, ""))
        if al and "tensor_a_sm_cycle" in rec and "tensor_a_sm_cycle" in al:
            rec["tensor_fraction"] = rec["tensor_a_sm_cycle"] / al["tensor_a_sm_cycle"]
    for rec in out.values():
        cells = [f"{rec['probe']:<16}"]
        for k, f in (("smsp_cycles_a_unit", "{:8.2f} cyc/unit"), ("over_none", "{:+7.2f} over none"),
                     ("tensor_fraction", "tensor {:.4f}"), ("lane_atomics_a_sm_cycle", "{:.3f} lane-atomics/SM-cyc"),
                     ("sm_cycles_a_warp_atomic", "{:.2f} SM-cyc/warp-atomic"), ("spread", "spread {:.4f}"),
                     ("sm_mhz", "{} MHz"), ("stop_gap", "stop_gap {:.4f}")):
            if rec.get(k) is not None:
                cells.append(f.format(rec[k]))
        if rec["tags"]:
            cells.append("[" + ",".join(rec["tags"]) + "]")
        print("  ".join(cells))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
