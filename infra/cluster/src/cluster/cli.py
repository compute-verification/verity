"""`cluster`: check a cluster description, plan a node from a snapshot, and read the allocation ledger.

  cluster validate DESCRIPTION.toml
  cluster plan DESCRIPTION.toml SNAPSHOT.json         # one JSON action per line; the snapshot names the node and `now`
  cluster ledger verify LEDGER.jsonl|DIR               # DIR: its ledger.jsonl, or every segment under it as one chain
  cluster ledger quiet LEDGER.jsonl|DIR ALLOC|RUN [--until T]   # what ran beside a window: owner, GPUs, pid (in the id), run
  cluster ledger failures LEDGER.jsonl|DIR [--since T]
  cluster ledger usage LEDGER.jsonl|DIR [--since T]     # held against busy GPU time per holder, lowest share first
  cluster agent DESCRIPTION.toml --out DIR [--mode shadow|live] [--hours H] [--stop-file P] [--copy-to DIR]
                [--roll [--roll-mb M]]                  # DIR is a root of daily segments (agent.py); exit 0 when stopped,
                                                        # 1 on an exception, 2 when a plan would break a rule
  cluster shadow replay DESCRIPTION.toml --sampler UTIL.jsonl[.gz] --preempted LOG[.gz] --out DIR
  cluster shadow evaluate DIR [--quiet-log QUIET.jsonl[.gz]] [--sampler UTIL.jsonl[.gz]...]  # the pass bar: exit 0 or 2
  cluster submit DESCRIPTION.toml --run-id R --now T --project P --submitter W [--kind K] [--max-s S] [--gpus N ...]
                                                        # the node, wrappers and kind as JSON; exit 3 with {"refused"}
  cluster status [--on nebius-1|nebius-2|all] [--owner OWNER] [--json]   # each node's jobs, GPU-h ready, holds, disk,
                                                        # bundles, read over SSH (status.py); exit 1 if a node can't be read
  cluster usage hourly [--on nebius-1|nebius-2|all] [--hours H] [--json] [--file PATH]   # per node and UTC hour: busy
                                                        # and useful against 95% / 90%, held idle, unmeasured (usage.py)
  cluster usage wasters --day YYYY-MM-DD [--top N] [--json] [--file PATH]   # held-but-idle GPU-hours by holding, most first;
                                                        # both read the console's record on vy-nebius-1, exit 1 if stale or unread
  cluster kueue-shadow tick|loop --out RECORD.jsonl [--every S --hours H] | down   # on vy-nebius-1: what Kueue alone
                                                        # would admit beside what the node did (kueue_shadow.py)
  cluster policy show [--description D] [--on N|all] [--key K] [--log L] [--json]   # the policy in force: each key, its
                                                        # owner and source, with the overrides in force from the local log
  cluster policy diff [--description D] [--on N|all] [--json]   # every key whose live value differs, read over SSH
                                                        # (policy.py); exit 1 drift, 3 a value unread, 2 a bad description
  cluster policy override KEY VALUE --until T --by WHO --reason R [--log L] [--apply] [--json]   # on KEY's node: KEY is
                                                        # VALUE until T (UTC); a dry run without --apply; exit 2 refused

A snapshot is `{"node", "now", "allocations": [...], "queue": [...], "done": [...], "has": [...], "held_locks": [...],
"last_busy": {alloc: t}, "idle_s": {kind: s}}`, in the shapes `model.allocation_to_dict` and `model.job_to_dict` write.
Only `agent` and `kueue-shadow` read a clock or touch a node (`agent.py`, `kueue_shadow.py`; the shadow writes only its own
Kueue objects), and `status`, `usage` and `policy diff`, which only read them; `policy override --apply` writes nothing but
its log. `shadow replay` runs the same planner over a day's copied logs.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import shutil
import sys
from pathlib import Path

from . import kueue_shadow, ledger, model, plan, usage


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


def renamed(argv: list[str] | None, names: dict[str, str]) -> list[str]:
    """Old flag spellings as hidden aliases (tests/test_cli_conventions.py): `--node X` and `--node=X` read as `--on`, said once
    on stderr. A copy of research.flags.renamed: this package is stdlib only."""
    out, said = [], set()
    for a in sys.argv[1:] if argv is None else argv:
        old, eq, value = a.partition("=")
        if old in names:
            if old not in said:
                print(f"cluster: {old} is now {names[old]}", file=sys.stderr)
                said.add(old)
            a = names[old] + eq + value
        out.append(a)
    return out


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
    lq.add_argument("alloc", help="an allocation id, or a run id: every quiet allocation of that run")
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
    ag.add_argument("--roll", action="store_true", help="--out is a root of segments, a new one each run and UTC day")
    ag.add_argument("--roll-mb", type=int, default=256, help="with --roll, also a new segment once the ledger passes M MiB")
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
    sb.add_argument("--kind", default="adhoc", help="the job's kind, from kinds/<owner>.toml beside the descriptions")
    sb.add_argument("--question", default="", help="the research question it answers (default: its kind's)")
    sb.add_argument("--gpus", type=int)
    sb.add_argument("--gpu-model")
    sb.add_argument("--cpus", type=int)
    sb.add_argument("--mem-gb", type=int)
    sb.add_argument("--max-s", type=int, help="its longest run (default: its kind's max_wall_min)")
    sb.add_argument("--disk-gb", type=float, help="the disk it writes (default: its kind's disk_gb, else 20)")
    sb.add_argument("--custody-s", type=int, default=0)
    sb.add_argument("--preempt", choices=("requeue", "never"), default="requeue")
    sb.add_argument("--quiet", action="store_true", default=None)
    sb.add_argument("--on", dest="node", metavar="NODE")
    sb.add_argument("--gpu-ids", default="", help="comma-separated GPU indices the job must get")
    sb.add_argument("--gpu-host", action="store_true", help="a job without --gpus that takes its GPUs with gpu-lease itself: its scope runs "
                                                               "at a GPU job's CPU weight (node 1: not idle)")
    st = sub.add_parser("status", help="each node's running, queued and held jobs, GPU-h ready, holds, disk and bundles")
    st.add_argument("--on", dest="node", choices=("nebius-1", "nebius-2", "all"), default="all")
    st.add_argument("--owner", help="only this owner's jobs (verity.dev/owner, else the workstream; a submitter on node 2)")
    st.add_argument("--json", action="store_true")
    usage.add_parser(sub)
    kueue_shadow.add_parser(sub)
    po = sub.add_parser("policy", help="the declared policy, and every key where a node's live value differs")
    posub = po.add_subparsers(dest="pcmd", required=True)
    from .policy import OVERRIDES
    for name in ("show", "diff", "override"):
        pp = posub.add_parser(name)
        pp.add_argument("--description", default=str(Path(__file__).parents[2] / "descriptions" / "nebius.toml"))
        pp.add_argument("--json", action="store_true")
        if name != "override":
            pp.add_argument("--on", dest="node", choices=("nebius-1", "nebius-2", "all"), default="all")
        if name != "diff":
            pp.add_argument("--log", default=OVERRIDES, help="the override log (this node's)")
        if name == "show":
            pp.add_argument("--key", help="only this key and the keys under it (exit 2 when there are none)")
        if name == "override":
            pp.add_argument("key")
            pp.add_argument("value")
            pp.add_argument("--until", required=True, help="when it ends: an absolute UTC time, 2026-10-04T10:30:00Z")
            pp.add_argument("--by", required=True, help="the handle who decided it")
            pp.add_argument("--reason", required=True)
            pp.add_argument("--apply", action="store_true", help="record it (else print what would be recorded)")
    a = ap.parse_args(renamed(argv, {"--node": "--on"}))

    if a.cmd == "usage":
        return usage.main(a)

    if a.cmd == "kueue-shadow":
        return kueue_shadow.main(a)

    if a.cmd == "policy":
        from . import policy
        if a.pcmd == "show":
            return policy.main_show(a.description, a.node, a.json, a.key, a.log)
        if a.pcmd == "override":
            return policy.main_override(a.description, a.key, a.value, a.until, a.by, a.reason, a.log, a.apply, a.json)
        return policy.main_diff(a.description, a.node, a.json)

    if a.cmd == "status":
        from . import status
        return status.main(list(status.NODES.values()) if a.node == "all" else [status.NODES[a.node]], a.owner, a.json)

    if a.cmd == "submit":
        from . import kinds, submit
        c = model.load(a.description)
        try:
            out = submit.submit(c, kinds.load(kinds.kinds_dir(a.description)), run_id=a.run_id, now=a.now, kind=a.kind,
                                question=a.question, gpus=a.gpus, cpus=a.cpus, memory_gib=a.mem_gb, max_s=a.max_s,
                                quiet=a.quiet, preempt=a.preempt, disk_gb=a.disk_gb, project=a.project, submitter=a.submitter,
                                priority=a.priority, gpu_model=a.gpu_model, custody_s=a.custody_s, node=a.node,
                                gpu_ids=tuple(int(x) for x in a.gpu_ids.split(",") if x), gpu_host=a.gpu_host)
        except (submit.Refused, ValueError) as e:
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
                            until=start + 3600 * a.hours if a.hours else None, tick_s=a.tick_s, footprint=fp,
                            roll=a.roll, roll_bytes=a.roll_mb << 20)
        if a.copy_to:
            shutil.copytree(a.out, a.copy_to, dirs_exist_ok=True)
        print(json.dumps(summary, sort_keys=True))
        if summary.get("exception"):
            return 1
        return 2 if str(summary.get("stopped", "")).startswith("invariant") else 0
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
        allocs, _ = ledger.history(records)
        ids = [a.alloc] if a.alloc in allocs else [h.alloc for h in allocs.values()
                                                   if h.job.get("run") == a.alloc and h.job.get("quiet")]
        if not ids:
            print(f"{a.alloc}: no such allocation, and no quiet allocation of that run", file=sys.stderr)
            return 1
        reps = [ledger.quiet_report(records, i, until=a.until) for i in ids]
        for rep in reps:
            print(json.dumps(rep, indent=1, sort_keys=True))
        return 0 if all(rep["quiet"] for rep in reps) else 2
    for row in (ledger.failures if a.lcmd == "failures" else ledger.usage)(records, a.since):
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
