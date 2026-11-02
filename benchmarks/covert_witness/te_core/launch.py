"""Launch a submission at a tier, one `research run` a start and seed, from the committed checkout (`--source`).

    python -m te_core.launch (--script islands/CLASS/train.py | --program train.py) --tier 0 --agent A \\
        --direction D --mechanism M [--parent ID] [--entry ID] [--nodes FILE] [--dry-run] [-- TRAIN_ARGS...]

- Tier 0 (screen): about 3 GPU-minutes on the small start, one seed; anyone may launch.
- Tier 1: about 10 GPU-minutes on the medium start, one seed; only an entry the scorer promoted to tier 1 or above
  (`--entry`, its program the one launched), at most `TIER1_PER_HOUR` launches per agent in any hour.
- Tier 2: about 30 GPU-minutes, 3 seeds, the small and medium starts, checkpoints kept; only an entry promoted to tier 2.

A `--script` is a covert_witness train.py (cw.py): it gets `--start NAME --out DIR --budget-fraction F` with F the
tier's minutes over the 10 GPU-minutes `prepare.TRAIN_FLOP_BUDGET` stands for, and the harness's 5% checkpoints divide
the same F of that budget. Small and medium are `STARTS`'s covert_witness starts.

The node comes from the node list the coordinator maintains (`--nodes`, default `$TE_NODES`, one node per line, `#`
comments, `NODE pool=NAME` for a node that takes runs only through a slot pool, `cpu=NAME` its CPU pool), the agent's
launches spread over it in
turn; `vy-dc-h200*` is refused whatever the list says. On a pool node the run is `research run --on NODE --pool NAME`
with `gpu-lease 1` inside (vy-nebius-2: `--gpus` is refused there); elsewhere `research run --on NODE --gpus 1`. The
program is sent with the run (`--send`) and screened first (run.py's `screen`). Each launch is a `launch` record in the
agent's ledger.
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


def file_sha256(path: Path) -> str:
    from .run import file_sha256 as f

    return f(path)


def screen(text: str) -> list[str]:
    # run.py imports torch; the node list (`nodes`) is read where torch isn't installed
    from .run import screen as s

    return s(text)


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
    2: Tier(30, ("small", "medium"), (0, 1, 2), True, keep=True),
}
TIER1_PER_HOUR = 4
REFUSED_NODES = ("vy-dc-h200*",)
#: the project the node list's machines belong to (`research run --on` refuses any other)
PROJECT = "verity"
RUN_LINE = re.compile(r"run (\S+) launched")
#: the covert_witness starts a script runs at each size
STARTS = {"small": "s100", "medium": "m100"}
#: the GPU-minutes `prepare.TRAIN_FLOP_BUDGET` stands for
BUDGET_MINUTES = 10.0
#: the island scripts' own margin under the budget (their default --budget-fraction)
BUDGET_MARGIN = 0.98
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


class Refused(RuntimeError):
    pass


@dataclass(frozen=True)
class Node:
    name: str
    pool: str = ""
    cpu: str = ""


def nodes(path: Path) -> list[Node]:
    out = []
    for line in Path(path).read_text().splitlines():
        words = line.split("#", 1)[0].split()
        if not words or any(fnmatch.fnmatchcase(words[0], pat) for pat in REFUSED_NODES):
            continue
        opts = dict(w.split("=", 1) for w in words[1:] if "=" in w)
        out.append(Node(words[0], opts.get("pool", ""), opts.get("cpu", "")))
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


def _shell(argv: list[str]) -> str:
    return " ".join(a if a.startswith("$") else shlex.quote(a) for a in argv)


def commands(program: Path, tier: int, node: Node | str, *, starts: str, evaluator: str, python: str, meta: dict,
             script: bool = False, train_args: list[str] = (), lane: str = "training", source: Path = REPO) -> list[dict]:
    t = TIERS[tier]
    node = node if isinstance(node, Node) else Node(node)
    sent = f"$RESEARCH_RUN_DIR/inputs/{Path(program).name}"
    fraction = BUDGET_MARGIN * t.minutes / BUDGET_MINUTES
    timeout = int(3 * t.minutes + 15)
    out = []
    for size in t.starts:
        start = STARTS[size] if script else size
        for seed in t.seeds:
            m = meta | {"tier": tier, "start": start, "seed": seed, "node": node.name}
            common = ["--tier", str(tier), "--seed", str(seed), "--budget-minutes", str(3 * t.minutes),
                      "--evaluator", evaluator, "--out", "$OUT/run", "--meta", json.dumps(m)]
            # a script's checkpoints are E's, scored on the final set in the run at tier 2, never kept in full
            common += ["--keep-checkpoints"] if t.keep and not script else []
            if script:
                inner = [python, "-m", "te_core.run", "--script", sent, "--budget-fraction", f"{fraction:g}", *common,
                         "--", "--start", start, "--out", "$S/sub", "--budget-fraction", f"{fraction:g}",
                         *train_args]
            else:
                inner = [python, "-m", "te_core.run", "--program", sent, "--starts", starts, "--start", start, *common,
                         "--", *train_args]
            # the submission (its model in full) lives in scratch, removed at exit; the record keeps E's digest
            shell = ('cd benchmarks/covert_witness && R="${CW_ROOT:-/workspace/research/scratch/covert-witness}" && '
                     'mkdir -p "$R/candidates" && S=$(mktemp -d -p "$R/candidates" te.XXXXXX) && '
                     "trap 'rm -rf \"$S\"' EXIT && " +
                     (f"gpu-lease 1 --wait --max-min {timeout} -- {_shell(inner)}" if node.pool else _shell(inner)))
            where = (["--pool", node.pool, "--as", lane, "--pool-wait", "7200"] if node.pool else ["--gpus", "1"])
            argv = ["research", "run", "--on", node.name, *where, "--project", PROJECT, "--source", str(source),
                    "--declared-output", "out/*", "--send", str(program), "--timeout", f"{timeout}m", "--",
                    "bash", "-c", shell]
            out.append({"start": start, "seed": seed, "argv": argv})
    return out


def launch(ledger: Ledger, program: Path, tier: int, *, node_file: Path, direction: str, mechanism: str,
           parent: str = "", entry: str = "", starts: str = "starts.json", evaluator: str = "none",
           python: str = "python3", dry_run: bool = False, clock=time.time, runner=subprocess.run,
           script: bool = False, train_args: list[str] = (), lane: str = "training") -> list[dict]:
    gate(ledger, tier, program, entry, clock)
    pool = nodes(node_file)
    node = pool[sum(1 for r in ledger.records() if r["kind"] == "launch" and r["agent"] == ledger.agent) % len(pool)]
    meta = {"agent": ledger.agent, "direction": direction, "mechanism": mechanism, "parent": parent, "entry": entry}
    out = []
    for cmd in commands(program, tier, node, starts=starts, evaluator=evaluator, python=python, meta=meta,
                        script=script, train_args=train_args, lane=lane):
        run_id, rc = "", None
        if not dry_run:
            res = runner(cmd["argv"], capture_output=True, text=True)
            rc = res.returncode
            m = RUN_LINE.search(res.stdout + res.stderr)
            run_id = m.group(1) if m else ""
        rec = ledger.append("launch", meta | {"tier": tier, "node": node.name, "pool": node.pool, "start": cmd["start"], "seed": cmd["seed"],
                                              "program_sha256": file_sha256(program), "argv": cmd["argv"],
                                              "run_id": run_id, "rc": rc, "dry_run": dry_run})
        out.append(rec)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    train_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--program", type=Path, help="a file defining train(te)")
    which.add_argument("--script", type=Path, help="a covert_witness train.py (cw.py)")
    ap.add_argument("--tier", type=int, choices=sorted(TIERS), required=True)
    ap.add_argument("--agent", default=os.environ.get("TE_AGENT", ""), required=not os.environ.get("TE_AGENT"))
    ap.add_argument("--direction", required=True)
    ap.add_argument("--mechanism", required=True)
    ap.add_argument("--parent", default="")
    ap.add_argument("--entry", default="", help="tiers 1 and 2: the promoted submission this program is")
    ap.add_argument("--nodes", type=Path, default=Path(os.environ.get("TE_NODES", "nodes.txt")))
    ap.add_argument("--starts", default=os.environ.get("TE_STARTS", "starts.json"), help="--program: the registry on the node")
    ap.add_argument("--evaluator", default=os.environ.get("TE_EVALUATOR", "none"))
    ap.add_argument("--python", default=os.environ.get("TE_PYTHON", "/workspace/jobs/venv312/bin/python"))
    ap.add_argument("--lane", default=os.environ.get("RESEARCH_LANE", "training"))
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    program = a.script or a.program
    try:
        recs = launch(Ledger(a.root, a.agent), program, a.tier, node_file=a.nodes, direction=a.direction,
                      mechanism=a.mechanism, parent=a.parent, entry=a.entry, starts=a.starts, evaluator=a.evaluator,
                      python=a.python, dry_run=a.dry_run, script=a.script is not None, train_args=train_args,
                      lane=a.lane)
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
