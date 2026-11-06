"""run.sh's summary: the gates (the device's and hgref.py's), the per-call ratio to plain cuBLASLt of each arm at each
linear (with the flush, per 8 calls, and steady, calls 1..7), against the earlier 2.40x at o_proj, and ncu's counters at
o_proj.
python3 summary.py OUT_DIR"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

E2E = {"o_proj": 2.40}   # the e2e slice: edited:w_l3qm_s16 and its commitments, pipelined


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.startswith("{")] if p.exists() else []


def ncu(p: Path) -> dict[str, str]:
    if not p.exists():
        return {}
    lines = p.read_text().splitlines()
    start = next((i for i, x in enumerate(lines) if x.startswith('"ID"')), None)
    if start is None:
        return {}
    return {r["Metric Name"]: r["Metric Value"] for r in csv.DictReader(io.StringIO("\n".join(lines[start:])))}


def main(o: Path) -> None:
    print(f"# edited GEMM with in-CTA commitment hashing ({o})\n\n## gates\n")
    for g in jl(o / "gate.jsonl"):
        bad = {k: v for k, v in g["device"].items() if any(v.values())}
        print(f"- device {g['m']} x {g['n']} x {g['k']} (grid {g['grid']}, D {g['D']}): device_ok {g['device_ok']}, "
              f"{len(g['device'])} kernel-calls, failing {bad or 'none'}")
    for r in jl(o / "ref.jsonl"):
        t = r["trees"]
        rec = r["receipts"]
        print(f"- hgref {r['m']} x {r['n']} x {r['k']}: ok {r['ok']}; params {r['params_ok']}; {sum(x['ok'] for x in t)}/{len(t)} "
              f"trees bit-exact at every level ({sum(x['openings'] for x in t)} openings, {sum(x['controls'] for x in t)} "
              f"controls rejected); roots distinct {r['roots_distinct']}; receipts {sum(x['bad'] == 0 for x in rec)}/{len(rec)} "
              f"tile-calls bit-exact ({sum(x['words'] for x in rec)} words); {r['ref_s']} s")
    rows = jl(o / "pipe.jsonl")
    if not rows:
        return
    layers = list(dict.fromkeys(r["layer"] for r in rows))
    arms = list(dict.fromkeys(r["pipe"] for r in rows))
    by = {(r["layer"], r["pipe"]): r for r in rows}
    print("\n## per call, ratio to plain (with the flush / steady)\n")
    print("| arm | " + " | ".join(layers) + " |")
    print("|---|" + "---|" * len(layers))
    for a in arms:
        cells = []
        for la in layers:
            r = by.get((la, a))
            cells.append("" if r is None else f"{r['ratio']:.2f}x / {r['steady_ratio']:.2f}x ({r['ms']:.3f} ms)")
        print(f"| {a} | " + " | ".join(cells) + " |")
    print("| E2E r4 (edited + commitments, pipelined) | " + " | ".join(f"{E2E[la]:.2f}x" if la in E2E else "" for la in layers) + " |")
    print("\nclocks (MHz, median / min): " + ", ".join(f"{la} {by[(la, 'plain')]['mhz']:.0f} / {by[(la, 'plain')]['mhz_min']:.0f}"
                                                      for la in layers if (la, "plain") in by))
    print("\n## ncu at o_proj\n")
    for p in sorted(o.glob("ncu-*.csv")):
        m = ncu(p)
        if m:
            print(f"- {p.stem[4:]}: " + ", ".join(f"{k.split('__', 1)[-1]} {v}" for k, v in m.items()))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
