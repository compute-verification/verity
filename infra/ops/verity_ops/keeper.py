"""`research keeper`: standing campaign targets on a Nebius node, set and stopped from any VM over ssh. A lane that wants a stream
of chunks ("keep N of this chunk leased on node X, seed = counter, until T") sets a target there instead of keeping an agent VM
awake to relaunch them. The node's side is pods/nebius/keeper.py (`vy-keeper`, ticked every minute by vy-keeper.timer), whose
docstring says what a tick does.

  research keeper set NAME --on MACHINE --project P --n N --until UTC --max-min M [--gpus G] [--seed-from S] [--owner WHO]
                      [--source GITDIR] [--env K=V ...] [--tool T] [--campaign C] [research run flags ...] -- CMD... {seed}
  research keeper stop NAME --on MACHINE [--cancel] [--reason TEXT] [--by WHO]
  research keeper status [NAME] --on MACHINE [--json]

  research keeper set sweep-a --on vy-nebius-2 --project verity --n 4 --until 2026-10-05T12:00Z --max-min 30 \\
      --source . --cwd source --tool T --campaign C -- python -m pkg.chunk --seed {seed}

`set` ships the research package and the --source tree as `research run --on` does, and writes the target on the node: keep N
chunks of CMD live (running, or waiting for their lease) until UTC, at most 48 h ahead, each with `{seed}` replaced by the next
seed of the target's counter. Every chunk is a recorded `research run` on that machine, its workload under `gpu-lease G --wait
--preemptible --max-min M`, so a waiter can always take its GPUs back (a preempted chunk's place goes to a new seed). M counts
from the lease's grant: the node cancels a chunk whose lease would meet a window booked on node 2 or node 1's daily hold
(12:10-13:30Z), and one still waiting at UTC. --env is set inside the lease, on the chunk's command line, so never a secret
(PYTHONPATH stays the runner's, as with --on); it may not set GPU_LEASE_*, CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER or
RESEARCH_KEEPER_*.
Setting a NAME again replaces its values and re-arms it after a stop or a pause (3 failed chunks in a row); its seed counter
carries on, and another owner's NAME is refused. `stop` starts no more chunks; `--cancel` also sends SIGTERM to the live chunks'
workloads, recorded as manual cancels. Stop chunks that way, never by stopping their unit: systemd then SIGKILLs the runner,
which records no final status. job.json's run.remote.keeper names each chunk's target, owner and seed, and `research status RUN
--on MACHINE` reads its records; node 2's run custody (n2_custody.sh) preserves them, and elsewhere they stay on the node until
`research fetch --all RUN --on MACHINE`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from research import remote

MAX_AHEAD_H = 48
NODE_KEEPER = "/usr/local/bin/vy-keeper"

clock = time.time


def _connect(machine: str, project: str | None = None) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """The machine's entry, its target (host, port, rate) and the ssh argv that reaches it."""
    if project:
        m = remote.resolve_machine(machine, project)
    else:
        m = remote.load_machines().get(machine)
        if m is None:
            raise remote.Refused(f"machine {machine!r} is not registered (research pods registry)")
        m = {"root": "/workspace/research", "python": "python3", **m, "name": machine}
    tgt = remote.pod_target(m)
    return m, tgt, remote.ssh_argv(tgt["host"], tgt["port"], user=tgt.get("user", "root"))


def _node(m: dict[str, Any], ssh: list[str], argv: list[str], stdin: bytes | None = None) -> Any:
    """`vy-keeper ARGV` on the node: its JSON answer; Refused with its refusal, or its stderr when it gave none."""
    try:
        # an ssh session doesn't carry the KEEPER_NODE that vy-keeper.service sets, and set refuses without it
        r = remote.ssh_run(ssh, ["env", f"KEEPER_NODE={m['name']}", m["python"], NODE_KEEPER, *argv], stdin=stdin, timeout=120)
    except subprocess.TimeoutExpired:
        raise remote.Refused(f"`vy-keeper {argv[0]}` on {m['name']} gave no answer in 120 s; it may still apply once the node's lock "
                             f"frees (`research keeper status --on {m['name']}` says)") from None
    try:
        out = json.loads(r.stdout.decode(errors="replace").strip().splitlines()[-1])
    except (IndexError, ValueError):
        err = r.stderr.decode(errors="replace").strip()[-400:]
        raise remote.Refused(f"`vy-keeper {argv[0]}` on {m['name']} failed (exit {r.returncode}): {err or 'no output'}") from None
    if isinstance(out, dict) and out.get("refused"):
        raise remote.Refused(f"{m['name']} refuses it: {out['refused']}")
    return out


def _until(text: str, now: float) -> str | None:
    """Why `text` can't be a target's until (UTC, after now, at most MAX_AHEAD_H ahead), else None."""
    try:
        t = datetime.fromisoformat(text)
    except ValueError:
        return f"--until {text!r} is not a UTC time like 2026-10-05T12:00Z"
    if t.utcoffset() != timedelta(0):
        return f"--until {text!r} is not in UTC (end it with Z)"
    if not now < t.timestamp() <= now + MAX_AHEAD_H * 3600:
        return f"--until {text} is not in the next {MAX_AHEAD_H} h"
    return None


def cmd_set(argv: list[str]) -> int:
    from research import cli

    if "--" not in argv:
        print("research keeper set: the chunk's command goes after `--`, with {seed} where its seed goes", file=sys.stderr)
        return 2
    i = argv.index("--")
    opts, command = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="research keeper set", allow_abbrev=False,
                                 description="keep N chunks of CMD leased on MACHINE until UTC, a fresh {seed} each")
    ap.add_argument("name", help="the target's name: lower case, digits and -, at most 40")
    ap.add_argument("--on", required=True, metavar="MACHINE", help="the Nebius node (vy-nebius-2)")
    ap.add_argument("--project", required=True)
    ap.add_argument("--n", type=int, required=True, help="the chunks to keep live, running or waiting for their lease")
    ap.add_argument("--until", required=True, metavar="UTC", help=f"start none after this (e.g. 2026-10-05T12:00Z; at most {MAX_AHEAD_H} h ahead)")
    ap.add_argument("--gpus", type=int, default=1, help="each chunk's GPUs (default 1)")
    ap.add_argument("--max-min", type=int, required=True, metavar="M",
                    help="each chunk's lease ends it M minutes after its grant (gpu-lease --max-min)")
    ap.add_argument("--seed-from", type=int, default=None, metavar="S", help="raise the seed counter to S (it never goes down)")
    ap.add_argument("--owner", default=os.environ.get("RESEARCH_LANE"), help="the lane the target is (default $RESEARCH_LANE)")
    ap.add_argument("--source", default=None, metavar="GITDIR", help="the tree each chunk runs from (shipped as `research run --on` ships it)")
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--stage", default="run")
    ap.add_argument("--tool", default=None, metavar="NAME|module:NAME")
    ap.add_argument("--campaign", default=os.environ.get("RESEARCH_CAMPAIGN"))
    ns, harness_opts = ap.parse_known_args(opts)
    if not command or not any("{seed}" in a for a in command):
        print("research keeper set: the command after `--` has no {seed}; each chunk gets the next seed there", file=sys.stderr)
        return 2
    if not ns.owner:
        print("research keeper set: --owner (or $RESEARCH_LANE) names the lane the target is", file=sys.stderr)
        return 2
    if why := _until(ns.until, clock()):
        print(f"research keeper set: {why}", file=sys.stderr)
        return 2
    env = cli._parse_env(ns.env)
    if env is None:
        return 2
    from research.telemetry.job import SECRET_KEY

    for bad, why in (([k for k in env if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)], "not a variable name"),
                     ([k for k in env if re.fullmatch(r"GPU_LEASE_\w*|CUDA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER|RESEARCH_KEEPER_\w*", k)],
                      "the chunks' lease, the GPUs it grants and their seeds are the keeper's"),
                     ([k for k in env if SECRET_KEY.search(k)], "a secret's name, and a chunk's env is set on its command line, inside "
                      "its lease (launch.json records it as it is)")):
        if bad:
            print(f"research keeper set: --env {', '.join(sorted(bad))}: {why}", file=sys.stderr)
            return 2
    try:
        harness_opts = cli._normalize_timeout(harness_opts)
        cli._harness_parser("research keeper set").parse_args(["--attempt-dir", "-", "--stage", ns.stage, *harness_opts])
    except (ValueError, SystemExit) as e:
        print(f"research keeper set: unknown or malformed `research run` flags {shlex.join(harness_opts)} ({e})", file=sys.stderr)
        return 2
    if bad := sorted({o.split("=", 1)[0] for o in harness_opts} & {"--attempt-dir", "--job"}):
        print(f"research keeper set: {', '.join(bad)}: the runner sets each chunk's run dir and id (argparse would keep yours)", file=sys.stderr)
        return 2
    src = Path(ns.source).expanduser() if ns.source else None
    asked_cwd = remote.flag_value(harness_opts, "--cwd")
    if asked_cwd is not None and not os.path.isabs(asked_cwd) and src is None:
        print(f"research keeper set: --cwd {asked_cwd} is relative to the shipped source tree, and there is none; pass --source GITDIR",
              file=sys.stderr)
        return 2
    pkg = remote.source_tool_package(src)
    if ns.tool and pkg is None and ":" not in ns.tool:
        from research.store import tools_registry

        if ns.tool not in tools_registry.REGISTRY:
            print(f"research keeper set: this research package does not know tool {ns.tool!r}; pass --source with tools/research",
                  file=sys.stderr)
            return 3
    harness_args = ["--stage", ns.stage, "--project", ns.project, *harness_opts, *(["--tool", ns.tool] if ns.tool else []),
                    *(["--campaign", ns.campaign] if ns.campaign else [])]
    try:
        m, tgt, ssh = _connect(ns.on, ns.project)
        tool_dir = remote.ship_tool(ssh, m["root"], pkg)
        tree_sha, src_dir, ready = (None, None, {})
        if src is not None:
            tree_sha, src_dir, ready = remote.ship_source(ssh, m["root"], src)
            harness_args += ["--tree-sha", tree_sha]
            harness_args = remote.set_flag(harness_args, "--source-identity", json.dumps(remote.shipped_identity(tree_sha, ready), sort_keys=True))
        request = {"schema": remote.REQUEST_SCHEMA, "runs_root": f"{m['root']}/runs", "tool": tool_dir, "python": m["python"], "cwd": src_dir,
                   "env": env, "harness_args": harness_args, "exclusive": False,
                   "cost": {"provider": m.get("provider", "runpod"), "pod_id": m.get("pod_id"), "machine": ns.on, "currency": "USD",
                            "rate_usd_per_hour": tgt["cost_per_hr"], "rate_observed_utc": tgt["rate_observed_utc"], "reason": None},
                   "remote": {"machine": ns.on, "pod_id": m.get("pod_id"), "pod_name": tgt["pod_name"], "host": tgt["host"], "port": tgt["port"],
                              "root": m["root"], "tool": tool_dir, "tool_identity": remote.tool_identity(pkg) if remote.knows_version_check(pkg) else None,
                              "source_sha": tree_sha, "source_dir": src_dir, "source_archive_sha256": ready.get("archive_sha256"),
                              "source_files": ready.get("files"), "source_verified": ready.get("verified"),
                              "source_transport": ready.get("transport"), "launched_from": os.uname().nodename}}
        doc = {"owner": ns.owner, "node": ns.on, "n": ns.n, "until": ns.until, "gpus": ns.gpus, "max_min": ns.max_min, "command": command,
               "request": request, **({"seed_from": ns.seed_from} if ns.seed_from is not None else {})}
        out = _node(m, ssh, ["set", ns.name], stdin=json.dumps(doc).encode())
    except remote.Refused as e:
        print(f"research keeper set: {e}", file=sys.stderr)
        return 1
    print(f"research keeper: {ns.name} on {ns.on}: {ns.n} chunk(s) of `{shlex.join(command)}` until {ns.until}, next seed "
          f"{out['next_seed']} ({out['live']} live now); the node's next tick (within a minute) starts them. "
          f"`research keeper status {ns.name} --on {ns.on}`")
    return 0


def cmd_stop(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research keeper stop", description="start no more chunks of the target")
    ap.add_argument("name")
    ap.add_argument("--on", required=True, metavar="MACHINE")
    ap.add_argument("--cancel", action="store_true", help="also SIGTERM each live chunk's workload, recorded as a manual cancel")
    ap.add_argument("--reason", default="stopped")
    ap.add_argument("--by", default=os.environ.get("RESEARCH_LANE") or os.environ.get("USER") or "-")
    ns = ap.parse_args(argv)
    try:
        m, _, ssh = _connect(ns.on)
        out = _node(m, ssh, ["stop", ns.name, "--by", ns.by, "--reason", ns.reason, *(["--cancel"] if ns.cancel else [])])
    except remote.Refused as e:
        print(f"research keeper stop: {e}", file=sys.stderr)
        return 1
    print(f"research keeper: {ns.name} on {ns.on} stopped: no more chunks start ({len(out['live'])} live"
          + ("" if ns.cancel or not out["live"] else "; they run to their end, or `--cancel` stops them") + ")")
    for rid, result in sorted(out["cancels"].items()):
        print(f"  {rid}: {result}")
    return 0


def render(rows: list[dict[str, Any]]) -> str:
    lines = []
    for r in rows:
        state = "stopped" if r.get("stopped") else "paused" if r.get("paused") else "on"
        lines.append(f"{r['name']}  {state}  owner={r['owner']}  live={len(r['live'])}/{r['n']}  until={r['until']}  gpus={r['gpus']}  "
                     f"max_min={r['max_min'] or '-'}  next_seed={r['next_seed']}  failures={r['failures']}")
        if r.get("note"):
            lines.append(f"  {r['note']}")
        lines += [f"  {c['run']}  seed {c['seed']}  since {c['started']}" for c in r["live"]]
    return "\n".join(lines) or "no targets"


def cmd_status(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research keeper status", description="the node's targets, their live chunks and seed counters")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--on", required=True, metavar="MACHINE")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    try:
        m, _, ssh = _connect(ns.on)
        rows = _node(m, ssh, ["status", *([ns.name] if ns.name else []), "--json"])
    except remote.Refused as e:
        print(f"research keeper status: {e}", file=sys.stderr)
        return 1
    print(json.dumps(rows, indent=1) if ns.json else render(rows))
    if rows and not ns.json:
        print(f"(a chunk's records: `research status RUN --on {ns.on}`)")
    return 0


def main(argv: list[str]) -> int:
    cmds = {"set": cmd_set, "stop": cmd_stop, "status": cmd_status}
    if not argv or argv[0] not in cmds:
        print(__doc__.strip(), file=sys.stderr if argv and argv[0] not in ("-h", "--help") else sys.stdout)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return cmds[argv[0]](argv[1:])
