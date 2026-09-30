"""`cluster`: check a cluster description, plan a node from a snapshot, and read the allocation ledger.

  cluster validate DESCRIPTION.toml
  cluster plan DESCRIPTION.toml SNAPSHOT.json         # one JSON action per line; the snapshot names the node and `now`
  cluster ledger verify LEDGER.jsonl
  cluster ledger quiet LEDGER.jsonl ALLOC [--until T]
  cluster ledger failures LEDGER.jsonl [--since T]
  cluster ledger usage LEDGER.jsonl [--since T]         # held against busy GPU time per holder, lowest share first

A snapshot is `{"node", "now", "allocations": [...], "queue": [...], "done": [...], "has": [...], "held_locks": [...],
"last_busy": {alloc: t}, "idle_s": {kind: s}}`, in the shapes `model.allocation_to_dict` and `model.job_to_dict` write.
Nothing here reads a clock or touches a node: `plan` is how a shadow run compares the planner's decisions with a live node's.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from . import ledger, model, plan


def _action(a: plan.Action) -> dict:
    d = {"action": type(a).__name__.lower()}
    for f in dataclasses.fields(a):
        v = getattr(a, f.name)
        d[f.name] = model.allocation_to_dict(v) if isinstance(v, model.Allocation) else v
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cluster", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("description")
    p = sub.add_parser("plan")
    p.add_argument("description")
    p.add_argument("snapshot")
    lg = sub.add_parser("ledger")
    lsub = lg.add_subparsers(dest="lcmd", required=True)
    lv = lsub.add_parser("verify")
    lv.add_argument("path")
    lq = lsub.add_parser("quiet")
    lq.add_argument("path")
    lq.add_argument("alloc")
    lq.add_argument("--until", type=int)
    lf = lsub.add_parser("failures")
    lf.add_argument("path")
    lf.add_argument("--since", type=int, default=0)
    lu = lsub.add_parser("usage")
    lu.add_argument("path")
    lu.add_argument("--since", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "validate":
        try:
            c = model.load(a.description)
        except model.ModelError as e:
            print(f"cluster: {a.description}: {e}", file=sys.stderr)
            return 1
        print(f"{c.name}: {len(c.nodes)} nodes, {sum(len(n.gpus) for n in c.nodes.values())} GPUs, {len(c.shares)} shares")
        return 0
    if a.cmd == "plan":
        c = model.load(a.description)
        s = json.loads(open(a.snapshot).read())
        allocs = [model.allocation_from_dict(x) for x in s.get("allocations", [])]
        queue = [model.job_from_dict(x) for x in s.get("queue", [])]
        now = model.instant(s["now"])
        acts = plan.plan(c, s["node"], allocs, queue, now, done=frozenset(s.get("done", [])),
                         has=frozenset(s.get("has", [])), held_locks=frozenset(s.get("held_locks", [])))
        if "idle_s" in s:
            acts += plan.idle(c, s["node"], allocs, s.get("last_busy", {}), now, idle_s=s["idle_s"])
        for act in acts:
            print(json.dumps(_action(act), sort_keys=True))
        return 0
    records = ledger.read(a.path)
    if a.lcmd == "verify":
        problems = ledger.verify(records)
        for pr in problems:
            print(f"cluster: {a.path}: {pr}", file=sys.stderr)
        if not problems:
            print(f"{a.path}: {len(records)} records, chain intact")
        return 1 if problems else 0
    if a.lcmd == "quiet":
        rep = ledger.quiet_report(records, a.alloc, until=a.until)
        print(json.dumps(rep, indent=1, sort_keys=True))
        return 0 if rep["quiet"] else 2
    for row in (ledger.failures if a.lcmd == "failures" else ledger.usage)(records, a.since):
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
