"""Label hash-loop runs and their result artifacts in the evidence store, with the existing vocabulary only (`research data vocab`):
campaign=hash-loop, sweep=iteration, row=N, label=<hypothesis>, source=<branch>@<commit>, hardware=<GPU>; the arms are in the
result's measurements. `--outcome kept|dropped --reason TEXT` adds the iteration's verdict (kernel_outcome, kernel_reason) to each
result. Re-running it is harmless: a label already asserted with the same value is skipped.

    python3 benchmarks/hash/loop/label.py RUN... --row N --hypothesis TEXT [--branch B] [--outcome kept|dropped --reason TEXT] [--by hash]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

HARDWARE = {"vy-dc-h200": "H200", "vy-dc-b200": "B200", "vy-nebius-2": "RTX PRO 6000"}
BRANCH = "cursor/hash-commit-loop-15ad"


def research(*args: str) -> str:
    return subprocess.run(["research", *args], check=True, capture_output=True, text=True).stdout


def attempt(run: str) -> dict:
    subprocess.run(["research", "fetch", run], capture_output=True, text=True)
    return json.loads(research("data", "show", run, "--json"))


def asserted(target: str, by: str) -> set[tuple[str, str]]:
    try:
        rows = json.loads(research("data", "labels", target, "--json"))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return set()
    return {(r["key"], str(r["value"])) for r in rows if r.get("by") == by}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--row", required=True)
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--outcome", choices=("kept", "dropped"))
    ap.add_argument("--reason")
    ap.add_argument("--by", default="hash")
    a = ap.parse_args()
    if bool(a.outcome) != bool(a.reason):
        ap.error("--outcome and --reason go together")
    rc = 0
    for run in a.runs:
        att = attempt(run)
        machine = (att.get("execution") or {}).get("machine")
        commit = (att.get("source") or {}).get("commit")
        result = (att.get("outputs") or {}).get("result")
        if machine not in HARDWARE or not commit:
            print(f"{run}: machine {machine!r} / commit {commit!r}: not a hash-loop run", file=sys.stderr)
            rc = 1
            continue
        labels = [("campaign", "hash-loop"), ("sweep", "iteration"), ("row", str(a.row)), ("label", a.hypothesis),
                  ("source", f"{a.branch}@{commit}"), ("hardware", HARDWARE[machine])]
        targets = [run] + ([result] if result else [])
        if not result:
            print(f"{run}: no result artifact (state {(att.get('status') or {}).get('state')}): labeling the run only", file=sys.stderr)
            rc = 1
        for t in targets:
            have = asserted(t, a.by)
            todo = labels + ([("kernel_outcome", a.outcome), ("kernel_reason", a.reason)] if a.outcome and t == result else [])
            for k, v in todo:
                if (k, v) in have:
                    continue
                research("data", "label", t, k, v, "--by", a.by, "--ref", run)
            print(f"{run} {HARDWARE[machine]:<13} {t}: {len(todo)} labels")
    return rc


if __name__ == "__main__":
    sys.exit(main())
