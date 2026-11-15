"""The proofs' status, as numbers: what `verity lean deployed` reads from the Lean locks at the checkout's HEAD, measured as
`lean/deployed-guarantees` (the deployed guarantees), `lean/deployed-gap-zero` (those at a trust gap of zero, its target
all of them), `lean/trust-gap` (each deployed guarantee's gap in premises, its `member` the guarantee's Lean name) and,
for each lead the locks name, `lean/open-premises` and `lean/no-honest-witness` (its `member` the lead's handle).

    research run --on auto --cwd source -- uv run python benchmarks/proof_status/run.py

It runs the command in the checkout it is in, writes under `--out` (`out/` outside a research run), and fails with the
command's message where the command fails.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from verity_bench import kit

ROOT = Path(__file__).resolve().parents[2]
COMMAND = ("verity", "lean", "deployed")


def deployed() -> dict:
    r = subprocess.run(COMMAND, cwd=ROOT, capture_output=True, text=True, check=False)
    if r.returncode:
        sys.exit(f"{' '.join(COMMAND)} exited {r.returncode}: {(r.stderr or r.stdout).strip()}")
    return json.loads(r.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="out", help="where the result goes outside a research run")
    out = ap.parse_args().out
    doc = deployed()
    totals, guarantees = doc["totals"], doc["guarantees"]
    w = kit.writer(out)
    w.measurement("lean/deployed-guarantees", totals["deployed"], "guarantees", mark="measured")
    w.measurement("lean/deployed-gap-zero", totals["gap_zero"], "guarantees", target=totals["deployed"], mark="measured")
    for g in guarantees:
        w.measurement("lean/trust-gap", len(g["gap"]), "premises", target=0, mark="measured", member=g["name"])
    for lead, counts in totals["by_owner"].items():
        w.measurement("lean/open-premises", counts["open"], "premises", mark="measured", member=lead)
        w.measurement("lean/no-honest-witness", counts["no_honest_witness"], "guarantees", mark="measured", member=lead)
    w.validation("passed", "benchmarks/proof_status/run.py",
                 f"read {len(doc['locks'])} Lean locks at {doc['commit']}: {totals['deployed']} deployed guarantees, "
                 f"{len(totals['by_owner'])} leads")


if __name__ == "__main__":
    main()
