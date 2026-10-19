"""The dedup's upper-bound table from `step.py`'s dedup.json: per phase, today's -h3 step against each arm, median over the
timed reps (decode: a whole-graph step's wall, prefill: an eager pass's), with the saving against -h3 and its share of -h3's
hashing (h3 − nohash).

  python benchmarks/hash/dedup/table.py OUT/dedup.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

SHOWN = ("h3", "pre", "inline", "tree", "pre-od", "inline-od", "nohash")


def rows(doc: dict) -> list[dict]:
    out = []
    for phase, arms in doc["timing"].items():
        key, rep_key = ("step_ms", "wall_ms") if phase.startswith("decode") else ("pass_ms", "ms")
        ms = {a: v[key] for a, v in arms.items()}
        spread = {a: (min(r[rep_key] for r in v["reps"]), max(r[rep_key] for r in v["reps"])) for a, v in arms.items()}
        sm = [r["clock"]["sm_mhz"] for v in arms.values() for r in v["reps"] if "sm_mhz" in r.get("clock", {})]
        hashing = ms["h3"] - ms["nohash"] if "h3" in ms and "nohash" in ms else None
        saved = {a: ms["h3"] - ms[a] for a in ms if a not in ("h3", "nohash") and "h3" in ms}
        out.append({"phase": phase, "ms": ms, "spread": spread, "hashing": hashing, "saved": saved,
                    "share": {a: s / hashing for a, s in saved.items()} if hashing else {},
                    "reps": len(next(iter(arms.values()))["reps"]),
                    "sm_mhz": (min(sm), statistics.median(sm), max(sm)) if sm else None})
    return out


def saving(r: dict, arm: str) -> str:
    if arm not in r["saved"] or not r["share"]:
        return ""
    s = r["saved"][arm]
    return f"{s:.2f} ({100 * s / r['ms']['h3']:.0f}% of the step, {100 * r['share'][arm]:.0f}% of hashing)"


def markdown(doc: dict) -> str:
    rs = rows(doc)
    arms = [a for a in SHOWN if any(a in r["ms"] for r in rs)]
    lines = [f"| phase | {' | '.join(arms)} | h3's hashing | pre saves | inline saves | pre-od saves | SM MHz (min/med/max) |",
             "|" + "---|" * (len(arms) + 6)]
    for r in rs:
        cells = [f"{r['ms'][a]:.2f}" if a in r["ms"] else "" for a in arms]
        sm = "/".join(f"{x:.0f}" for x in r["sm_mhz"]) if r["sm_mhz"] else ""
        lines.append(f"| {r['phase']} | {' | '.join(cells)} | {r['hashing']:.2f} | {saving(r, 'pre')} | {saving(r, 'inline')} | "
                     f"{saving(r, 'pre-od')} | {sm} |")
    return "\n".join(lines)


def main() -> int:
    for p in sys.argv[1:]:
        doc = json.loads(Path(p).read_text())
        print(f"run {doc.get('run')} at {doc.get('source')}, {doc['layers']} layers, checks_ok {doc.get('checks_ok')}\n")
        print(markdown(doc))
        print()
        for r in rows(doc):
            print(r["phase"], "reps", r["reps"], "min-max ms:", {a: f"{lo:.2f}-{hi:.2f}" for a, (lo, hi) in r["spread"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
