"""Each pipe's duty over a profiled range: run.sh MODE=duty's Nsight Compute raw pages (duty-TAG-SHAPE-{ovl,wgl}.csv,
duty.metrics' metrics over OVL_CALLS overlapped calls or as many wgmma.linear calls, in base units) as one Markdown
table, each pipe's active share of the range's elapsed cycles (idle is its complement)."""

import csv
import re
import sys
from pathlib import Path

NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

COLS = (
    ("tensor", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("ALU", "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("FMA", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("FMA heavy (IMAD)", "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("shared", "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("LSU inst", "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed"),
    ("uniform inst", "sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_elapsed"),
    ("issue", "smsp__issue_active.avg.pct_of_peak_sustained_elapsed"),
    ("DRAM", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
)
TIME = "gpu__time_duration.sum"


def ranges(path):
    """The raw page's data rows (after its header and units rows) as dicts of floats where they parse."""
    rows = [r for r in csv.reader(Path(path).read_text().splitlines()) if r]
    head = next((i for i, r in enumerate(rows) if "ID" in r), None)
    if head is None:
        return []
    out = []
    for r in rows[head + 2:]:
        d = {}
        for k, v in zip(rows[head], r):
            s = v.replace(",", "")
            d[k] = float(s) if NUM.fullmatch(s) else v
        out.append(d)
    return out


def main(paths):
    print("| range | time (ms) | " + " | ".join(f"{c} %" for c, _ in COLS) + " |")
    print("|---|---:|" + "---:|" * len(COLS))
    for p in sorted(paths):
        name = Path(p).stem.removeprefix("duty-")
        for i, d in enumerate(ranges(p)):
            t = d.get(TIME)
            ms = f"{t / 1e6:.3f}" if isinstance(t, float) else "-"
            vals = [d.get(m) for _, m in COLS]
            cells = [f"{v:.1f}" if isinstance(v, float) else "-" for v in vals]
            print(f"| {name}{'' if i == 0 else f' #{i}'} | {ms} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
