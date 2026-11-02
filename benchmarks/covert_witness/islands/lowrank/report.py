"""The lowrank direction's results: every collected candidate in drive.py's state, by start, class and bpb.

    python report.py [--state ~/.research/lowrank/state.jsonl] [--start s100]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state", type=Path, default=Path.home() / ".research" / "lowrank" / "state.jsonl")
    ap.add_argument("--start", default=None)
    a = ap.parse_args()
    rows, flying = [], {}
    for x in a.state.read_text().splitlines():
        s = json.loads(x)
        if "row" not in s:
            flying[s["key"]] = s.get("run")
            continue
        flying.pop(s["key"], None)
        r = s.get("row") or {}
        head, note, args = (y.strip() for y in s["key"].split(";", 2))
        rows.append((head.split()[1], head.split()[0], r.get("bpb"), r.get("E_bytes"), r.get("train_flops"), r.get("phi"),
                     r.get("valid"), s.get("run"), (r.get("art") or "")[4:16], note, args))
    for start, klass, bpb, e, f, phi, valid, run, art, note, args in sorted(
            rows, key=lambda t: (t[0], t[1], t[2] if isinstance(t[2], float) else 9)):
        if a.start and start != a.start:
            continue
        bpb_s = f"{bpb:.4f}" if isinstance(bpb, float) else str(bpb)
        print(f"{start:5} {klass:7} {bpb_s:7} E={e!s:>7} F={f and f'{f:.2e}'} phi={phi and round(phi, 3)} "
              f"{'ok' if valid else 'INVALID'} {run} art:{art} | {note} | {args}")
    for k, run in sorted(flying.items()):
        print(f"in flight {run}: {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
