#!/usr/bin/env python3
"""Who writes to /workspace on a Nebius node, and whether an allocation covers them (the steward's write probe, minimal).

    sudo -n nice -n 19 ionice -c3 python3 write_probe.py probe --node vy-nebius-N [--every 60] [--mbps 10] [--out DIR]
    uv run --no-sync python -m research.pods.nebius.write_probe relay [--every 120]                  where Slack works

`probe` reads every thread's `write_bytes` (/proc/<pid>/task/<tid>/io, as root, so pods' processes count too) once a tick
and sums each process's deltas; not /proc/<pid>/io, which also holds what every child the process reaped wrote. A process
that wrote at least --mbps MB/s over the tick and has a file open for writing under /workspace, or works there, is a line
of <out>/writes.jsonl, with the allocation that covers it:
  kueue:<cluster queue>/<pod>   a pod Kueue admitted (node 1)
  system:<namespace or unit>    a cluster namespace or a system service
  fill:<scope or job>, lease:<scope>   a job node 2's fill runner or GPU lease started (its fill-* or gpu-lease-* scope, or
                                the fill runner's running/<job>.sh it descends from; `fill:runner` is the runner itself)
  run:<id>                      a recorded `research run` (its cluster-r<id> scope, or RESEARCH_RUN_ID in its or a parent's env)
A writer none of them covers is also a line of <out>/unmatched.jsonl. <out>/status.json is the last tick. While node 2's fill
status shows a timed window running or waiting, the probe reads nothing. Bytes written by a process or thread that exits
before the next tick are missed. Stdlib only; it reads /proc and writes only under <out>.

`relay` posts each new unmatched writer from both nodes to #agent-alerts, once per process and again hourly while it writes.
The nodes hold no Slack credentials, so it runs where `research msg` works.
"""
from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

WORKSPACE = "/workspace"
OUT = "/workspace/verity-guest/write-probe"
FILL_STATUS = "/workspace/pouw/fill/status.txt"
KUBECONFIG = "/home/research/.kube/config"
SYSTEM_NAMESPACES = {"kube-system", "gpu-operator", "kueue-system", "monitoring", "skypilot"}
POD_UID = re.compile(r"pod([0-9a-f]{8}[-_][0-9a-f]{4}[-_][0-9a-f]{4}[-_][0-9a-f]{4}[-_][0-9a-f]{12})")

Pods = Callable[[], dict[str, tuple[str, str, str | None]]]


def _read(path: str, binary: bool = False) -> str | bytes | None:
    try:
        with open(path, "rb" if binary else "r") as f:
            return f.read()
    except OSError:
        return None


def _stat_fields(proc: str, pid: int) -> list[str] | None:
    """/proc/<pid>/stat from field 3 (state) on: the command name may hold spaces and parentheses."""
    s = _read(f"{proc}/{pid}/stat")
    return s.rsplit(")", 1)[1].split() if s else None


def snapshot(proc: str = "/proc") -> dict[tuple[int, str], dict[int, int]]:
    """(pid, start time) -> {thread id: write_bytes}, for every process this user may read."""
    out = {}
    for p in os.listdir(proc):
        if not p.isdigit():
            continue
        f = _stat_fields(proc, int(p))
        try:
            tids = os.listdir(f"{proc}/{p}/task")
        except OSError:
            continue
        if not f or len(f) < 20:
            continue
        threads = {}
        for t in tids:
            m = re.search(r"^write_bytes:\s*(\d+)", _read(f"{proc}/{p}/task/{t}/io") or "", re.M)
            if m:
                threads[int(t)] = int(m.group(1))
        if threads:
            out[(int(p), f[19])] = threads
    return out


def write_paths(pid: int, proc: str = "/proc", limit: int = 3) -> list[str]:
    """Up to `limit` files under /workspace the process holds open for writing."""
    try:
        fds = os.listdir(f"{proc}/{pid}/fd")
    except OSError:
        return []
    paths = []
    for fd in fds:
        try:
            target = os.readlink(f"{proc}/{pid}/fd/{fd}")
        except OSError:
            continue
        m = re.search(r"^flags:\s*([0-7]+)", _read(f"{proc}/{pid}/fdinfo/{fd}") or "", re.M)
        if target.startswith(WORKSPACE + "/") and m and int(m.group(1), 8) & 3:
            paths.append(target)
            if len(paths) == limit:
                break
    return paths


def _lineage(pid: int, proc: str, depth: int = 16):
    """The process and its ancestors, nearest first, each with its command line."""
    for _ in range(depth):
        yield pid, (_read(f"{proc}/{pid}/cmdline", binary=True) or b"").replace(b"\0", b" ").decode(errors="replace")
        f = _stat_fields(proc, pid)
        if not f or int(f[1]) <= 1:
            return
        pid = int(f[1])


def run_id(pid: int, proc: str = "/proc", depth: int = 16) -> str | None:
    """The run of the process or its nearest ancestor: RESEARCH_RUN_ID in its environment, or the harness itself
    (`research run ... --id <run>`, which sets that variable only for its children)."""
    for p, cmd in _lineage(pid, proc, depth):
        for kv in (_read(f"{proc}/{p}/environ", binary=True) or b"").split(b"\0"):
            if kv.startswith(b"RESEARCH_RUN_ID="):
                return kv.split(b"=", 1)[1].decode(errors="replace")
        m = re.search(r"\bresearch run\b.* --id (r\d{8}-\d{6}-[0-9a-f]{4})\b", cmd)
        if m:
            return m.group(1)
    return None


def fill_job(pid: int, proc: str = "/proc") -> str | None:
    """The fill job of the process or its nearest ancestor (the runner runs each from running/<job>.sh, and a CPU job
    without a named unit in an anonymous run-*.scope), or `runner` for node 2's fill runner itself."""
    for _, cmd in _lineage(pid, proc):
        m = re.search(r"/pouw/fill/running/([^/\s]+)\.sh\b", cmd)
        if m:
            return m.group(1)
        if "/pouw/infra/bin/fill_runner.py" in cmd:
            return "runner"
    return None


def kube_pods(kubeconfig: str = KUBECONFIG) -> dict[str, tuple[str, str, str | None]]:
    """pod uid -> (namespace, name, Kueue cluster queue or None); {} when kubectl can't be read."""
    try:
        r = subprocess.run(["kubectl", "get", "pods", "-A", "-o", "json"], capture_output=True, text=True, timeout=60,
                           env={**os.environ, "KUBECONFIG": kubeconfig})
        items = json.loads(r.stdout)["items"]
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return {}
    return {p["metadata"]["uid"]: (p["metadata"]["namespace"], p["metadata"]["name"],
                                   (p["metadata"].get("labels") or {}).get("kueue.x-k8s.io/cluster-queue-name")) for p in items}


def allocation(pid: int, cgroup: str, pods: Pods, proc: str = "/proc") -> tuple[str, bool]:
    """The allocation that covers the process, and whether there is one."""
    m = POD_UID.search(cgroup)
    if m:
        uid = m.group(1).replace("_", "-")
        if uid not in pods():
            return f"pod:{uid} (not in kubectl's list)", False
        ns, name, cq = pods()[uid]
        if cq:
            return f"kueue:{cq}/{name}", True
        return (f"system:{ns}/{name}", True) if ns in SYSTEM_NAMESPACES else (f"pod:{ns}/{name} (no Kueue admission)", False)
    m = re.search(r"/((fill|gpu-lease)-[^/]+)\.scope", cgroup)
    if m:
        return f"{'fill' if m.group(2) == 'fill' else 'lease'}:{m.group(1)}", True
    m = re.search(r"/cluster-(r\d{8}-\d{6}-[0-9a-f]{4})\.scope", cgroup)
    rid = m.group(1) if m else run_id(pid, proc)
    if rid:
        return f"run:{rid}", True
    job = fill_job(pid, proc)
    if job:
        return f"fill:{job}", True
    m = re.match(r"/system\.slice/([^/]+)", cgroup)
    if m:
        return f"system:{m.group(1)}", True
    return f"none:{cgroup}", False


def writers(prev: dict, now: dict, seconds: float, mbps: float, pods: Pods, node: str, proc: str = "/proc") -> list[dict]:
    """The processes that wrote at least `mbps` MB/s to /workspace between two snapshots, each with its allocation."""
    cache: dict[str, dict] = {}

    def cached_pods() -> dict:
        if "pods" not in cache:
            cache["pods"] = pods()
        return cache["pods"]

    rows = []
    for (pid, start), threads in now.items():
        if (pid, start) not in prev:
            continue
        wrote = sum(max(0, wb - prev[(pid, start)].get(tid, 0)) for tid, wb in threads.items())
        if wrote < mbps * 1e6 * seconds:
            continue
        paths = write_paths(pid, proc)
        try:
            cwd = os.readlink(f"{proc}/{pid}/cwd")
        except OSError:
            cwd = ""
        if not paths and not (cwd + "/").startswith(WORKSPACE + "/"):
            continue
        cgroup = ((_read(f"{proc}/{pid}/cgroup") or "").strip().splitlines() or [""])[-1].split(":", 2)[-1]
        alloc, matched = allocation(pid, cgroup, cached_pods, proc)
        uid = re.search(r"^Uid:\s*(\d+)", _read(f"{proc}/{pid}/status") or "", re.M)
        try:
            user = pwd.getpwuid(int(uid.group(1))).pw_name if uid else "?"
        except KeyError:
            user = uid.group(1) if uid else "?"
        cmd = (_read(f"{proc}/{pid}/cmdline", binary=True) or b"").replace(b"\0", b" ").decode(errors="replace").strip()
        rows.append({"node": node, "pid": pid, "key": f"{node}:{pid}:{start}", "user": user, "cmd": cmd[:300], "cgroup": cgroup,
                     "allocation": alloc, "matched": matched, "mb_per_s": round(wrote / seconds / 1e6, 1),
                     "paths": paths or [cwd]})
    return sorted(rows, key=lambda r: -r["mb_per_s"])


def window_running(status: str = FILL_STATUS) -> bool:
    """Node 2's fill status shows a timed window running or waiting (its first line: `timed True` / `window waiting True`)."""
    first = ((_read(status) or "").splitlines() or [""])[0]
    return "timed True" in first or "window waiting True" in first


def _append(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("a") as f:
            f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)


def probe(ns: argparse.Namespace) -> int:
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    node = ns.node or os.uname().nodename
    prev, t0 = None, 0.0
    while True:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        paused = window_running()
        rows: list[dict] = []
        if paused:
            prev = None
        else:
            now, t1 = snapshot(), time.monotonic()
            if prev is not None:
                rows = [dict(r, ts=stamp) for r in writers(prev, now, t1 - t0, ns.mbps, kube_pods, node)]
            prev, t0 = now, t1
        _append(out / "writes.jsonl", rows)
        _append(out / "unmatched.jsonl", [r for r in rows if not r["matched"]])
        tmp = out / ".status.json.tmp"
        tmp.write_text(json.dumps({"ts": stamp, "node": node, "every_s": ns.every, "mb_per_s": ns.mbps, "paused": paused,
                                   "writers": len(rows), "unmatched": sum(not r["matched"] for r in rows)}) + "\n")
        tmp.replace(out / "status.json")
        if ns.once:
            return 0
        time.sleep(ns.every)


def alert_text(row: dict) -> str:
    return (f"*Unmatched /workspace writer on {row['node']}* ({row['ts']}): {row['mb_per_s']} MB/s, PID {row['pid']} "
            f"({row['user']}), no Kueue admission, fill job, GPU lease or research run covers it.\n"
            f"• cmd: `{row['cmd'][:200]}`\n• writing: `{', '.join(row['paths'])[:300]}`\n• cgroup: `{row['cgroup'][:160]}` "
            f"({row['allocation'][:80]})")


def relay(ns: argparse.Namespace) -> int:
    from research import slack

    reg = slack.load_registry(slack.registry_path())
    channel = slack.channel_id_for(reg, "agent-alerts")
    state_path = Path(ns.state)
    state = json.loads(state_path.read_text()) if state_path.exists() else {"lines": {}, "posted": {}}
    while True:
        for host in ns.nodes:
            seen = state["lines"].get(host, 0)
            f = f"{ns.out}/unmatched.jsonl"
            r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host,
                                f"[ -f {f} ] || {{ echo missing; exit 0; }}; wc -l < {f}; tail -n +{seen + 1} {f}"],
                               capture_output=True, text=True)
            lines = r.stdout.splitlines()
            if r.returncode or not lines:
                print(f"write_probe relay: {host} unreadable ({r.stderr.strip()[:120]})", file=sys.stderr)
                continue
            if lines[0].strip() == "missing":   # /workspace unmounted (a cutover): resetting here replayed every old row
                continue
            total = int(lines[0].strip() or 0)
            if total < seen:
                state["lines"][host] = 0
                continue
            for n, ln in enumerate(lines[1:], seen + 1):
                try:
                    row = json.loads(ln)
                except ValueError:
                    continue
                last = state["posted"].get(row["key"], 0)
                if time.time() - last >= 3600:
                    slack._post(slack.Slack(os.environ.get("SLACK_BOT_TOKEN"), dry_run=ns.dry_run), channel, ns.handle,
                                alert_text(row))
                    state["posted"][row["key"]] = time.time()
                    state["lines"][host] = min(n, total)
                    state_path.write_text(json.dumps(state))   # a later post that raises must not repeat this one
            state["lines"][host] = total
        state_path.write_text(json.dumps(state))
        if ns.once:
            return 0
        time.sleep(ns.every)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="write_probe", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="on a node, as root: record /workspace writers each tick")
    p.add_argument("--every", type=float, default=60.0)
    p.add_argument("--mbps", type=float, default=10.0)
    p.add_argument("--out", default=OUT)
    p.add_argument("--node", help="the name alerts use (default: the hostname)")
    p.add_argument("--once", action="store_true", help="one tick (no delta on the first), for a smoke test")
    p.set_defaults(fn=probe)
    r = sub.add_parser("relay", help="post the nodes' new unmatched writers to #agent-alerts")
    r.add_argument("--nodes", nargs="+", default=["n1", "n2"])
    r.add_argument("--out", default=OUT)
    r.add_argument("--every", type=float, default=120.0)
    r.add_argument("--state", default=str(Path.home() / ".research" / "write-probe-relay.json"))
    r.add_argument("--handle", default="infra")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--once", action="store_true")
    r.set_defaults(fn=relay)
    ns = ap.parse_args(argv)
    return ns.fn(ns)


if __name__ == "__main__":
    sys.exit(main())
