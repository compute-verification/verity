"""`cluster status`: the node's running and queued work by owner and key, its GPU-hours ready and its disk. Read-only: every
command it sends the node (`COMMANDS`) only reads, through `df`, `ls` and `cat`.

`read` takes the node's snapshot over SSH (`reach`: the node's address is the environment's, no file's); `node2` is the pure
half, snapshot and `now` in, report out, and `render` prints reports as text.

**Node 2** is its agent's ledger state (`state.json`, `ledger.state`): leases are running jobs and waiters queued, owned by
their submitter and keyed by their run (else the lease's job id). A job's length is the model's (`Job.length_s`: its
`max_s`, else a session's `expected_s`). `gpu-lease` doesn't manage CPUs, so a lease's CPUs are 0. The agent writes no state
during a timed window (`agent.py`), so the state is stale only when it is older than `STALE_S` outside one, or than the
window's end plus `STALE_S`.
"""
from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Callable

from . import ledger, model, nebius2, settings

NODES = {"nebius-2": nebius2.NODE}
#: a node's address is the environment's (`reach`): no file holds it
SSH_KEY = Path.home() / ".ssh" / "research_key"
LIVE = "/workspace/pouw/infra/cluster/live"
STALE_S = 60

COMMANDS = {
    "df": "df -P /workspace",
    # the agent's state: <live>/state.json with --roll, else its run's <live>/<stamp>/state.json; the newest
    "state": f'cd {LIVE} && f=$(ls -t state.json */state.json 2>/dev/null | head -1) && test -n "$f" && echo "{LIVE}/$f"'
             ' && cat "$f"',
}
READS = {nebius2.NODE: ("state", "df")}


class ReadError(RuntimeError):
    pass


def ssh(dest: str, key: Path | None = SSH_KEY) -> Callable[[str], str]:
    """A runner of one shell command on `dest`, returning its stdout."""
    def run(cmd: str) -> str:
        argv = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                *(["-i", str(key)] if key and key.exists() else []), dest, cmd]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=300)
        if r.returncode:
            raise ReadError(f"{dest}: `{cmd[:60]}` exited {r.returncode}: {r.stderr.strip()[-300:]}")
        return r.stdout
    return run


def reach(node: str, key: Path | None = SSH_KEY) -> Callable[[str], str]:
    """A runner of one shell command on `node`, as `ssh`'s. The node's address is the environment's, in the variable the
    node's name gives (vy-nebius-2's is VY_NEBIUS_2_HOST), as `machines.d/<node>.toml` registers it (``host``): no file
    here holds it, and with it unset each command fails as a read does (`ReadError`), naming the variable."""
    def run(cmd: str) -> str:
        if not (host := settings.host(node)):
            raise ReadError(f"{settings.host_var(node)} is unset: {node}'s address, which machines.d/{node}.toml "
                            "registers as host")
        return ssh(f"research@{host}", key)(cmd)
    return run


def read(node: str, run: Callable[[str], str], t: int) -> dict[str, Any]:
    """The node's snapshot at `t`: each of its `READS`' raw output, the commands run at once."""
    with ThreadPoolExecutor(len(READS[node])) as ex:
        out = {k: ex.submit(run, COMMANDS[k]) for k in READS[node]}
        raw = {k: f.result() for k, f in out.items()}
    snap: dict[str, Any] = {"node": node, "t": t, "df": raw["df"]}
    path, _, body = raw["state"].partition("\n")
    snap["state_path"], snap["state"] = path, json.loads(body)
    return snap


def disk_pct(df: str) -> int:
    """`df -P`'s Capacity of its one filesystem: used / (used + available), rounded up, as `df` reports it."""
    return int(df.strip().splitlines()[-1].split()[4].rstrip("%"))


def _row(owner: str, key: str, **f: Any) -> dict[str, Any]:
    return {"owner": owner, "key": key, "state": None, "cls": None, "gpus": 0, "cpus": 0, "memory_gib": 0, "since": None,
            "for_s": None, "length_s": None, "length_is": None, "name": None, **f}


def node2(snap: dict[str, Any], now: int, owner: str | None = None) -> dict[str, Any]:
    """Node 2's report from its agent's `state.json` and `df` (the module's mapping)."""
    st = ledger.state_from_dict(snap["state"])
    allocs, _ = st.on(nebius2.NODE)

    def length(j: model.Job) -> dict[str, Any]:
        return {"length_s": j.length_s, "length_is": "max" if j.max_s is not None else "expected" if j.length_s else None}

    rows = []
    for a in allocs:
        j = a.job
        rows.append(_row(j.submitter, j.run or j.id, state="frozen" if a.state == "frozen" else "running", cls=j.priority,
                         gpus=len(a.gpus), cpus=len(a.cpus) or j.cpus, memory_gib=a.memory_gib, since=a.started,
                         name=a.id, **length(j)))
    for q in st.queue.values():
        if q.node != nebius2.NODE:
            continue
        j = q.job
        rows.append(_row(j.submitter, j.run or j.id, state="queued", cls=j.priority, gpus=j.gpus, cpus=j.cpus,
                         memory_gib=j.memory_gib, since=q.since, name=q.handle or j.id, **length(j)))
    rep = _report(snap, now, owner, rows)
    window = [a for a in allocs if a.job.quiet]
    ends = [a.ends_by for a in window]
    age = now - st.t if st.t is not None else None
    held_back = bool(window) and (None in ends or now <= max(ends) + STALE_S)
    rep["ledger"] = {"path": snap.get("state_path"), "seq": st.seq, "t": st.t, "age_s": age, "window": bool(window),
                     "stale": age is None or (age > STALE_S and not held_back)}
    return rep


ORDER = {"running": 0, "frozen": 1, "queued": 2}


def _report(snap, now, owner, rows) -> dict[str, Any]:
    if owner is not None:
        rows = [r for r in rows if r["owner"] == owner]
    for r in rows:
        r["for_s"] = now - r["since"] if r["since"] is not None else None
    rows.sort(key=lambda r: (r["owner"], r["key"], ORDER[r["state"]], r["name"] or ""))
    queued = [r for r in rows if r["state"] == "queued"]
    known = [r for r in queued if r["length_s"]]
    return {"node": snap["node"], "t": now, "snapshot_t": snap["t"], "owner": owner, "jobs": rows,
            "ready": {"jobs": len(queued), "gpus": sum(r["gpus"] for r in queued),
                      "gpu_jobs": sum(1 for r in queued if r["gpus"]), "known": sum(1 for r in known if r["gpus"]),
                      "gpu_h": round(sum(r["gpus"] * r["length_s"] for r in known) / 3600, 2)},
            "disk": {"path": "/workspace", "use_pct": disk_pct(snap["df"])}}


def span(s: int | None) -> str:
    if s is None:
        return "-"
    s = max(0, s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m" if s >= 60 else f"{s}s"


def _table(rows: list[dict[str, Any]], indent: str = "  ") -> list[str]:
    cols = ("STATE", "OWNER", "KEY", "GPU", "CPU", "FOR")
    cells = [cols] + [(r["state"], r["owner"], r["key"].removeprefix(f"{r['owner']}/"), str(r["gpus"]), str(r["cpus"] or "-"),
                       span(r["for_s"])) for r in rows]
    keep = [i for i in range(len(cols)) if cols[i] in ("GPU", "CPU") or any(c[i] != "-" for c in cells[1:])]
    w = {i: max(len(c[i]) for c in cells) for i in keep}
    return [indent + "  ".join(c[i].ljust(w[i]) for i in keep).rstrip() for c in cells]


def render(reports: list[dict[str, Any]]) -> str:
    """The reports as text: per node a header (disk, ledger state, ready), then its jobs by owner and key."""
    out = []
    for rep in reports:
        if "error" in rep:
            out += [f"{rep['node']}: could not read: {rep['error']}", ""]
            continue
        when = datetime.fromtimestamp(rep["t"], UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append(f"{rep['node']} at {when}" + (f", owner {rep['owner']}" if rep["owner"] else ""))
        out.append(f"  disk: {rep['disk']['path']} {rep['disk']['use_pct']}% used (df)")
        if "ledger" in rep:
            lg = rep["ledger"]
            note = (" (STALE: is the agent running?)" if lg["stale"]
                    else " (a timed window: the agent writes no state until it ends)" if lg["window"] else "")
            out.append(f"  ledger state: seq {lg['seq']}, {span(lg['age_s'])} old{note} ({lg['path']})")
        rd = rep["ready"]
        gh = ("" if not rd["gpu_jobs"] else f"; {rd['gpu_h']:.2f} GPU-h" if rd["known"] == rd["gpu_jobs"]
              else f"; GPU-h unknown (none of {rd['gpu_jobs']} GPU jobs declares a length)" if not rd["known"]
              else f"; at least {rd['gpu_h']:.2f} GPU-h ({rd['known']} of {rd['gpu_jobs']} GPU jobs declare a length)")
        out.append(f"  ready: {rd['jobs']} queued, {rd['gpus']} GPUs asked{gh}")
        rows = rep["jobs"]
        out += _table(rows) if rows else ["  no jobs"]
        out.append("")
    return "\n".join(out).rstrip() + "\n"


clock = time.time


def main(nodes: list[str], owner: str | None, as_json: bool, key: Path | None = SSH_KEY,
         runner: Callable[[str], Callable[[str], str]] | None = None) -> int:
    """Read `nodes` at once, print their reports; 1 when a node couldn't be read."""
    now = int(clock())
    runner = runner or (lambda node: reach(node, key))

    def one(node: str) -> dict[str, Any]:
        try:
            return node2(read(node, runner(node), now), now, owner)
        except (ReadError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError) as e:
            return {"node": node, "t": now, "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(len(nodes)) as ex:
        reps = list(ex.map(one, nodes))
    if as_json:
        print(json.dumps({"schema": "cluster/status/v1", "t": now, "owner": owner, "nodes": reps}, indent=1, sort_keys=True))
    else:
        print(render(reps), end="")
    return 1 if any("error" in r for r in reps) else 0


__all__ = ["NODES", "COMMANDS", "READS", "ReadError", "ssh", "reach", "read", "disk_pct", "node2", "render", "span", "main"]
