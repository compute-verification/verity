"""Promotions, decided on the sealed promotion set one bit each (fixed; the scorer's, from an agent VM's committed checkout).

    python benchmarks/covert_witness/promote.py --in CANDIDATES.json [--out BITS.json]   # any candidates
    python benchmarks/covert_witness/promote.py --tier T [--root LEDGER]                 # te_core's frontier, promoted

CANDIDATES is a list of `{"id", "output_sha256"[, "node"]}`: a te_core ledger submission's id, its
`output.output_sha256`, and the node its run ran on (default the node list's first, `nodes.txt`). For each node it runs
`sealed.py decide` there, as a `research run` in the node's CPU pool from the pinned measurement commit (`pin.py`), with
the candidates sent along, and prints and writes the bits it returns: for each id, `promote` true, false, or null (that
node holds no value for the output), and the value record's salted sha256. The rule and the values never leave the node
(`sealed.py`).
`--tier T` takes te_core's promotion candidates at tier T - 1 (`te_core.frontier.candidates`, on the local ledger) and
writes a promotion record, as the scorer, for each promoted one.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = Path.home() / ".research" / "runs"
NODES = Path(os.environ.get("CW_NODES", HERE / "nodes.txt"))
sys.path.insert(0, str(HERE))

from pin import source  # noqa: E402
from te_core import nodes as te_nodes  # noqa: E402


def research(*args: str) -> str:
    r = subprocess.run(["research", *args], capture_output=True, text=True, cwd=REPO)
    if r.returncode:
        raise SystemExit(f"promote: research {' '.join(args[:2])} failed ({r.returncode}): {r.stderr[-2000:]}")
    return r.stdout + r.stderr


def decide_on(node: te_nodes.Node, candidates: list[dict]) -> tuple[str, dict]:
    if not node.cpu:
        raise SystemExit(f"promote: {node.name} names no CPU pool in {NODES} (`{node.name} pool=gpu cpu=research`)")
    _, src = source()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidates.json"
        path.write_text(json.dumps([{"id": c["id"], "output_sha256": c["output_sha256"]} for c in candidates]))
        out = research("run", "--on", node.name, "--project", "verity", "--source", str(src), "--pool", node.cpu,
                       "--as", "training", "--pool-wait", "1800", "--timeout", "10m", "--declared-output", "out/*",
                       "--send", str(path), "--wait", "--", "bash", "-c",
                       'python3 benchmarks/covert_witness/sealed.py decide --in "$RESEARCH_RUN_DIR/inputs/candidates.json" '
                       '--out "$OUT/bits.json"')
    m = re.search(r"run (r\d{8}-\d{6}-[0-9a-f]{4}) launched", out)
    if not m:
        raise SystemExit(f"promote: no run id in research run's output:\n{out[-2000:]}")
    run = m.group(1)
    research("fetch", run, "--all")
    bits = RUNS / run / "out" / "bits.json"
    if not bits.is_file():
        raise SystemExit(f"promote: {run} wrote no bits.json (research inspect {run} --on {node.name})")
    return run, json.loads(bits.read_text())


def decide(candidates: list[dict]) -> dict:
    """{id: {promote, value_sha256, output_sha256, node, run}} over every candidate, one decision run per node."""
    known = {n.name: n for n in te_nodes.nodes(NODES)}
    default = next(iter(known))
    by_node: dict[str, list[dict]] = {}
    for c in candidates:
        by_node.setdefault(c.get("node") or default, []).append(c)
    out = {}
    for name, cs in by_node.items():
        if name not in known:
            raise SystemExit(f"promote: {name} is not in {NODES}")
        run, bits = decide_on(known[name], cs)
        out |= {k: v | {"node": name, "run": run} for k, v in bits.items()}
    return out


def promote_tier(tier: int, root: Path | None) -> dict:
    from te_core.frontier import candidates
    from te_core.ledger import SCORER, Ledger

    led = Ledger(root, SCORER, publish=True)
    rows = {r["id"]: r for r in led.index()}
    ids = candidates(list(rows.values()), tier)
    cands = [{"id": i, "output_sha256": rows[i]["output"]["output_sha256"], "node": rows[i].get("node")}
             for i in ids if rows[i].get("output", {}).get("output_sha256")]
    bits = decide(cands) if cands else {}
    for rid, b in sorted(bits.items()):
        if b["promote"]:
            led.promote(rid, tier, f"the sealed promotion set ({b['run']} on {b['node']}, value {b['value_sha256']})")
    return bits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--in", dest="inp", type=Path)
    which.add_argument("--tier", type=int, choices=(1, 2))
    ap.add_argument("--root", type=Path, default=None, help="--tier: the te_core ledger (default $TE_LEDGER)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    bits = decide(json.loads(a.inp.read_text())) if a.inp else promote_tier(a.tier, a.root)
    if a.out:
        a.out.write_text(json.dumps(bits, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v["promote"] for k, v in sorted(bits.items())}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
