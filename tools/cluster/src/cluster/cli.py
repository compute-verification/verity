"""`cluster`: check a cluster description, plan a node from a snapshot, and read the allocation ledger.

  cluster validate DESCRIPTION.toml
  cluster plan DESCRIPTION.toml SNAPSHOT.json         # one JSON action per line; the snapshot names the node and `now`
  cluster ledger verify LEDGER.jsonl
  cluster ledger quiet LEDGER.jsonl ALLOC [--until T]
  cluster ledger failures LEDGER.jsonl [--since T]
  cluster ledger usage LEDGER.jsonl [--since T]         # held against busy GPU time per holder, lowest share first
  cluster agent DESCRIPTION.toml --out DIR [--mode shadow|live] [--hours H] [--stop-file P] [--copy-to DIR]
  cluster shadow replay DESCRIPTION.toml --sampler UTIL.jsonl[.gz] --preempted LOG[.gz] --out DIR
  cluster shadow evaluate DIR [--quiet-log QUIET.jsonl[.gz]] [--sampler UTIL.jsonl[.gz]...]  # the pass bar: exit 0 or 2
  cluster submit DESCRIPTION.toml --run-id R --now T --project P --submitter W --max-s S [--gpus N ...]
                                                        # the node and executor wrapper as JSON; exit 3 with {"refused"}

A snapshot is `{"node", "now", "allocations": [...], "queue": [...], "done": [...], "has": [...], "held_locks": [...],
"last_busy": {alloc: t}, "idle_s": {kind: s}}`, in the shapes `model.allocation_to_dict` and `model.job_to_dict` write.
Only `agent` reads a clock or touches a node (`agent.py`); `shadow replay` runs the same planner over a day's copied logs.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import shutil
import sys
from pathlib import Path

from . import ledger, model, plan


def _lines(path: str) -> list[str]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return f.read().splitlines()


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
    ag = sub.add_parser("agent")
    ag.add_argument("description")
    ag.add_argument("--out", required=True)
    ag.add_argument("--mode", choices=("shadow", "live"), default="shadow")
    ag.add_argument("--hours", type=float)
    ag.add_argument("--stop-file")
    ag.add_argument("--copy-to", help="copy the output directory here when the agent stops (a run's declared outputs)")
    ag.add_argument("--lease-dir", default="/run/gpu-lease")
    ag.add_argument("--status", default="/workspace/pouw/fill/status.txt")
    ag.add_argument("--tick-s", type=float, default=1.0)
    sh = sub.add_parser("shadow")
    ssub = sh.add_subparsers(dest="scmd", required=True)
    sr = ssub.add_parser("replay")
    sr.add_argument("description")
    sr.add_argument("--sampler", required=True, nargs="+")
    sr.add_argument("--preempted", required=True)
    sr.add_argument("--out", required=True)
    se = ssub.add_parser("evaluate")
    se.add_argument("dir")
    se.add_argument("--quiet-log")
    se.add_argument("--sampler", nargs="*", default=[], help="the node's util/*.jsonl[.gz] over the same span")
    se.add_argument("--description", default=str(Path(__file__).parents[2] / "descriptions" / "nebius.toml"))
    sb = sub.add_parser("submit", help="place a job and print its node and executor wrapper as JSON (research run --queue)")
    sb.add_argument("description")
    sb.add_argument("--run-id", required=True)
    sb.add_argument("--now", type=int, required=True, help="epoch seconds (the caller's clock)")
    sb.add_argument("--project", required=True)
    sb.add_argument("--submitter", required=True)
    sb.add_argument("--class", dest="priority", default="work")
    sb.add_argument("--gpus", type=int, default=0)
    sb.add_argument("--gpu-model")
    sb.add_argument("--cpus", type=int, default=1)
    sb.add_argument("--mem-gb", type=int, default=8)
    sb.add_argument("--max-s", type=int, required=True)
    sb.add_argument("--custody-s", type=int, default=0)
    sb.add_argument("--preempt", choices=("requeue", "never"), default="requeue")
    sb.add_argument("--quiet", action="store_true")
    sb.add_argument("--node")
    sb.add_argument("--gpu-ids", default="", help="comma-separated GPU indices the job must get")
    a = ap.parse_args(argv)

    if a.cmd == "submit":
        from . import submit
        c = model.load(a.description)
        j = submit.job(run_id=a.run_id, project=a.project, submitter=a.submitter, priority=a.priority, gpus=a.gpus,
                       gpu_model=a.gpu_model, cpus=a.cpus, memory_gib=a.mem_gb, max_s=a.max_s, custody_s=a.custody_s,
                       preempt=a.preempt, quiet=a.quiet, node=a.node,
                       gpu_ids=tuple(int(x) for x in a.gpu_ids.split(",") if x), submitted=a.now)
        try:
            out = submit.resolve(c, j, a.run_id, a.now)
        except submit.Refused as e:
            print(json.dumps({"refused": str(e)}))
            return 3
        print(json.dumps(out, sort_keys=True))
        return 0

    if a.cmd == "agent":
        import time

        from . import agent
        fp = agent.Footprint()
        agent.main_hook(fp)
        c = model.load(a.description)
        start = time.time()
        summary = agent.run(c, Path(a.out), mode=a.mode, clock=time.time, sleep=time.sleep, lease_dir=Path(a.lease_dir),
                            status=Path(a.status), stop_file=Path(a.stop_file) if a.stop_file else None,
                            until=start + 3600 * a.hours if a.hours else None, tick_s=a.tick_s, footprint=fp)
        if a.copy_to:
            shutil.copytree(a.out, a.copy_to, dirs_exist_ok=True)
        print(json.dumps(summary, sort_keys=True))
        return 1 if summary.get("exception") or str(summary.get("stopped", "")).startswith("invariant") else 0
    if a.cmd == "shadow":
        from . import shadow
        if a.scmd == "replay":
            c = model.load(a.description)
            rows = [json.loads(x) for f in a.sampler for x in _lines(f) if x.strip()]
            shadow.replay(c, sorted(rows, key=lambda r: r["t"]), _lines(a.preempted), Path(a.out))
            print(json.dumps(json.loads((Path(a.out) / "summary.json").read_text()), sort_keys=True))
            return 0
        q = [json.loads(x) for x in _lines(a.quiet_log) if x.strip()] if a.quiet_log else None
        sampled = None
        if a.sampler:
            c = model.load(a.description)
            sampled = shadow.sampler_windows(c, (json.loads(x) for f in a.sampler for x in _lines(f) if x.strip()))
        rep = shadow.evaluate(Path(a.dir), q, sampled=sampled)
        print(json.dumps(rep, indent=1, sort_keys=True))
        return 0 if rep["pass"] else 2

    if a.cmd == "validate":
        try:
            c = model.load(a.description)
        except model.ModelError as e:
            print(f"cluster: {a.description}: {e}", file=sys.stderr)
            return 1
        print(f"{c.name}: {len(c.nodes)} nodes, {sum(len(n.gpus) for n in c.nodes.values())} GPUs, {len(c.shares)} shares, "
              f"{len(c.workstreams)} workstreams")
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
