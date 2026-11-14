"""The Metrics dashboard's goal, as a number: of the top goals the dashboard lists, how many have a number that runs,
measured as `console/top-goals-running` against the count of top goals as its target. A goal's number is the first metric
the dashboard lists under it, and it runs when its newest value has a target, is younger than its `every` and carries a
mark. Higher is better; the dashboard holds to its goal when every top goal's number runs.

    research run --on auto --cwd source -- uv run python benchmarks/dashboard/run.py

It reads the dashboard's own data (`research metrics show --json`, in the checkout it runs in) and writes under `--out`
(`out/` outside a research run). The validation's detail names each goal whose number doesn't run, with the first of
"no number", "never measured", "no target", "stale" and "no mark" it fails.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from verity_bench import kit

ROOT = Path(__file__).resolve().parents[2]
SHOW = ["metrics", "show", "--json"]
MARKS = ("measured", "modeled", "partial")


def _miss(goal: dict, rows: dict[str, dict]) -> str | None:
    """The first point at which `goal`'s number doesn't run, or None when it runs."""
    row = next((rows[i] for i in goal.get("metrics") or [] if i in rows), None)
    if row is None:
        return "no number"
    latest = row.get("latest")
    if not latest:
        return "never measured"
    if (latest.get("target") if latest.get("target") is not None else row.get("target")) is None:
        return "no target"
    if row.get("stale"):
        return "stale"
    if (latest.get("mark") or row.get("mark")) not in MARKS:
        return "no mark"
    return None


def running(report: dict) -> tuple[int, int, list[tuple[str, str]]]:
    """How many of `report`'s top goals have a number that runs, how many top goals it lists, and each other goal's id
    with the first point it fails."""
    rows = {m["id"]: m for m in report.get("metrics") or []}
    top = report["top"]
    misses = [(g["id"], why) for g in top if (why := _miss(g, rows))]
    return len(top) - len(misses), len(top), misses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="out", help="where the result goes outside a research run")
    out = ap.parse_args().out
    p = subprocess.run([sys.executable, "-m", "research", *SHOW], cwd=ROOT, capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"research {' '.join(SHOW)} failed: {p.stdout.strip() or p.stderr.strip()}")
    report = json.loads(p.stdout)
    if report.get("top") is None:
        sys.exit(f"the dashboard can't read the goals: {report.get('topWhy') or 'its top goals are null'}")
    ok, total, misses = running(report)
    w = kit.writer(out)
    w.measurement("console/top-goals-running", ok, "goals", target=total, mark="measured")
    w.validation("passed", "benchmarks/dashboard/run.py",
                 f"{ok} of {total} top goals' numbers run" + "".join(f"; {g}: {why}" for g, why in misses))


if __name__ == "__main__":
    main()
