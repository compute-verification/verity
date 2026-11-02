"""Launch a submission at a tier, through `research run --on NODE --gpus 1` (gpu-lease hands out the GPU).

    python -m te_core.launch --program train.py --tier 0 --agent A --direction D --mechanism M [--parent ID] \\
        [--entry ID] [--nodes FILE] [--starts PATH_ON_NODE] [--evaluator MODULE:FUNCTION] [--dry-run]

- Tier 0 (screen): about 3 GPU-minutes on the small start, one seed; anyone may launch.
- Tier 1: about 10 GPU-minutes on the medium start, one seed; only an entry the scorer promoted to tier 1 or above
  (`--entry`, its program the one launched), at most `TIER1_PER_HOUR` launches per agent in any hour.
- Tier 2: about 30 GPU-minutes, 3 seeds, every start, checkpoints kept; only an entry promoted to tier 2.

The node comes from the node list the coordinator maintains (`--nodes`, default `$TE_NODES`, one node per line, `#`
comments), the agent's launches spread over it in turn; `vy-dc-h200*` is refused whatever the list says. The program is
sent with the run (`--send`) and screened first (run.py's `screen`). Each launch is a `launch` record in the agent's
ledger.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .ledger import Ledger
from .run import file_sha256, screen


@dataclass(frozen=True)
class Tier:
    minutes: float
    starts: tuple[str, ...]
    seeds: tuple[int, ...]
    promoted: bool
    keep: bool = False


TIERS = {
    0: Tier(3, ("small",), (0,), False),
    1: Tier(10, ("medium",), (0,), True),
    2: Tier(30, ("small", "medium", "large"), (0, 1, 2), True, keep=True),
}
TIER1_PER_HOUR = 4
REFUSED_NODES = ("vy-dc-h200*",)
PROJECT = "te-search"
RUN_LINE = re.compile(r"research: run (\S+) launched on")


class Refused(RuntimeError):
    pass


def nodes(path: Path) -> list[str]:
    out = []
    for line in Path(path).read_text().splitlines():
        name = line.split("#", 1)[0].strip()
        if name and not any(fnmatch.fnmatchcase(name, pat) for pat in REFUSED_NODES):
            out.append(name)
    if not out:
        raise Refused(f"{path}: no usable node")
    return out


def gate(ledger: Ledger, tier: int, program: Path, entry: str, clock=time.time) -> None:
    """Refuse a launch the tier doesn't allow."""
    t = TIERS[tier]
    bad = screen(Path(program).read_text())
    if bad:
        raise Refused(f"{program}: names {bad}")
    if not t.promoted:
        return
    if not entry:
        raise Refused(f"tier {tier} runs only a promoted entry: pass --entry")
    if not ledger.promoted(entry, tier):
        raise Refused(f"{entry} is not promoted to tier {tier}")
    sub = next((r for r in ledger.records() if r.get("id") == entry), None)
    if sub is None or sub.get("program_sha256") != file_sha256(program):
        raise Refused(f"{program} is not {entry}'s program")
    if tier == 1 and ledger.launches(ledger.agent, 1, clock() - 3600) >= TIER1_PER_HOUR:
        raise Refused(f"{ledger.agent} has launched {TIER1_PER_HOUR} tier-1 runs in the last hour")


def commands(program: Path, tier: int, node: str, *, starts: str, evaluator: str, python: str, meta: dict) -> list[dict]:
    t = TIERS[tier]
    out = []
    for start in t.starts:
        for seed in t.seeds:
            m = meta | {"tier": tier, "start": start, "seed": seed}
            inner = [python, "-m", "te_core.run", "--program", f"$RESEARCH_RUN_DIR/inputs/{Path(program).name}",
                     "--starts", starts, "--start", start, "--tier", str(tier), "--seed", str(seed),
                     "--budget-minutes", str(t.minutes), "--evaluator", evaluator, "--out", "$OUT", "--meta",
                     json.dumps(m)] + (["--keep-checkpoints"] if t.keep else [])
            shell = " ".join(a if a.startswith("$") else shlex.quote(a) for a in inner)
            argv = ["research", "run", "--on", node, "--gpus", "1", "--project", PROJECT, "--cwd",
                    "source/benchmarks/learning_frontier", "--send", str(program), "--timeout",
                    f"{int(2 * t.minutes + 10)}m", "--", "bash", "-c", shell]
            out.append({"start": start, "seed": seed, "argv": argv})
    return out


def launch(ledger: Ledger, program: Path, tier: int, *, node_file: Path, direction: str, mechanism: str,
           parent: str = "", entry: str = "", starts: str = "starts.json", evaluator: str = "none",
           python: str = "python3", dry_run: bool = False, clock=time.time, runner=subprocess.run) -> list[dict]:
    gate(ledger, tier, program, entry, clock)
    pool = nodes(node_file)
    node = pool[sum(1 for r in ledger.records() if r["kind"] == "launch" and r["agent"] == ledger.agent) % len(pool)]
    meta = {"agent": ledger.agent, "direction": direction, "mechanism": mechanism, "parent": parent, "entry": entry}
    out = []
    for cmd in commands(program, tier, node, starts=starts, evaluator=evaluator, python=python, meta=meta):
        run_id, rc = "", None
        if not dry_run:
            res = runner(cmd["argv"], capture_output=True, text=True)
            rc = res.returncode
            m = RUN_LINE.search(res.stdout + res.stderr)
            run_id = m.group(1) if m else ""
        rec = ledger.append("launch", meta | {"tier": tier, "node": node, "start": cmd["start"], "seed": cmd["seed"],
                                              "program_sha256": file_sha256(program), "argv": cmd["argv"],
                                              "run_id": run_id, "rc": rc, "dry_run": dry_run})
        out.append(rec)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--program", type=Path, required=True)
    ap.add_argument("--tier", type=int, choices=sorted(TIERS), required=True)
    ap.add_argument("--agent", default=os.environ.get("TE_AGENT", ""), required=not os.environ.get("TE_AGENT"))
    ap.add_argument("--direction", required=True)
    ap.add_argument("--mechanism", required=True)
    ap.add_argument("--parent", default="")
    ap.add_argument("--entry", default="", help="tiers 1 and 2: the promoted submission this program is")
    ap.add_argument("--nodes", type=Path, default=Path(os.environ.get("TE_NODES", "nodes.txt")))
    ap.add_argument("--starts", default=os.environ.get("TE_STARTS", "starts.json"), help="the registry's path on the node")
    ap.add_argument("--evaluator", default=os.environ.get("TE_EVALUATOR", "none"))
    ap.add_argument("--python", default=os.environ.get("TE_PYTHON", "python3"))
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        recs = launch(Ledger(a.root, a.agent), a.program, a.tier, node_file=a.nodes, direction=a.direction,
                      mechanism=a.mechanism, parent=a.parent, entry=a.entry, starts=a.starts, evaluator=a.evaluator,
                      python=a.python, dry_run=a.dry_run)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    for r in recs:
        print(json.dumps({k: r[k] for k in ("id", "node", "start", "seed", "run_id", "rc")}))
        if a.dry_run:
            print(shlex.join(r["argv"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
