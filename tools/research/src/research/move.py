"""`research move RUN [--to auto|MACHINE]`: a run still waiting in a pool's line on one machine is resent where it starts sooner, and
the waiting copy is cancelled. `research move --waiting-over MIN` does that for each of this VM's runs that has waited longer.

A pool run's request on its machine (`<root>/requests/<id>/request.json`) holds what it was sent with: its command, wrapped in the
pool's `slot.py` line (`env CHECK_SLOTS=… python3 tools/verity/check/slot.py --by LANE … --run-max S -- nice -n 19 CMD`), its
environment and its harness options. `argv_for` rebuilds the `research run` that sends CMD again under the same lane, timeout,
memory and harness options, from a detached worktree of the run's source commit; the new run is launched first and the waiting one
cancelled only once it is placed, so a refused launch leaves the original in its line. A run that holds its slot, or never waited
in a pool line, is refused: moving a started run would throw away its work.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: harness options the launch writes itself, so a resend gets fresh ones
REGENERATED = {"--stage", "--project", "--tree-sha", "--source-identity", "--timeout"}
NICE = ["nice", "-n", "19"]
NEW_RUN = re.compile(r"\br(\d{8}-\d{6}-[0-9a-f]{4})\b")


class Unmovable(Exception):
    """Why a run can't be moved; nothing was launched or cancelled."""


def unwrap(command: list[str]) -> dict[str, Any]:
    """The pool line's lane, run time and pool slots file, and the command inside it; Unmovable for a command no pool line wraps."""
    if not command or command[0] != "env" or "tools/verity/check/slot.py" not in command or "--" not in command:
        raise Unmovable("its command isn't a pool run (no `slot.py` line), so it never waited in a pool's line")
    cut = command.index("--")
    head, inner = command[:cut], command[cut + 1:]
    if inner[:3] == NICE:
        inner = inner[3:]
    if not inner:
        raise Unmovable("its pool line wraps no command")

    def opt(name: str) -> str | None:
        return head[head.index(name) + 1] if name in head and head.index(name) + 1 < len(head) else None

    slots = next((w.split("=", 1)[1] for w in head if w.startswith("CHECK_SLOTS=")), None)
    run_max = opt("--run-max")
    return {"lane": opt("--by"), "run_max": float(run_max) if run_max else None, "slots": slots, "command": inner}


def harness_options(args: list[str]) -> list[str]:
    """`args` without the options the launch regenerates; each `--opt value` pair or bare flag kept in order."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        value = i + 1 < len(args) and not args[i + 1].startswith("--")
        if a.split("=", 1)[0] not in REGENERATED:
            out += [a, args[i + 1]] if value and "=" not in a else [a]
        i += 2 if value and "=" not in a else 1
    return out


def argv_for(request: dict[str, Any], *, source: Path, to: str = "auto", cpus: int | None = None) -> list[str]:
    """The `research run` argv that resends `request`'s pool run from `source` (a checkout of its commit) onto `to`."""
    line = unwrap(list(request.get("command") or []))
    env = dict(request.get("env") or {})
    mem = env.pop("CHECK_RUN_MEM_GB", None) or request.get("mem_gb")
    harness = list(request.get("harness_args") or [])
    project = harness[harness.index("--project") + 1] if "--project" in harness else "verity"
    argv = ["research", "run", "--on", to, "--project", project, "--source", str(source)]
    if line["lane"]:
        argv += ["--as", line["lane"]]
    if mem:
        argv += ["--mem-gb", f"{float(mem):g}"]
    if cpus:
        argv += ["--cpus", str(cpus)]
    for k, v in sorted(env.items()):
        argv += ["--env", f"{k}={v}"]
    if line["run_max"]:
        argv += ["--timeout", f"{line['run_max']:g}"]
    return argv + harness_options(harness) + ["--", *line["command"]]


def waited_s(slot: dict[str, Any], now: datetime) -> float | None:
    """How long a run has waited in its pool's line, or None when it isn't waiting (it holds its slot, or left the line)."""
    if slot.get("state") != "waiting" or not slot.get("since"):
        return None
    since = datetime.fromisoformat(str(slot["since"]).replace("Z", "+00:00"))
    return (now - since).total_seconds()


def _request(run_dir: Path) -> dict[str, Any]:
    from . import remote

    rj = remote._read_json(run_dir / "remote.json") or {}
    if not rj.get("machine") or not rj.get("request"):
        raise Unmovable(f"{run_dir.name} has no remote.json naming its machine and request: only a run sent with --on can move")
    t = remote.pod_target(remote.resolve_machine(rj["machine"], rj.get("project") or "verity"))
    r = remote.ssh_run(remote.ssh_argv(t["host"], t["port"], user=t["user"]), ["cat", rj["request"]], timeout=60)
    if r.returncode:
        raise Unmovable(f"{rj['machine']} has no request {rj['request']} ({r.stderr.decode(errors='replace').strip()[-200:]})")
    return {**json.loads(r.stdout), "_machine": rj["machine"], "_source_sha": rj.get("source_sha")}


def _worktree(sha: str) -> Path:
    top = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip())
    have = subprocess.run(["git", "-C", str(top), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True)
    if have.returncode:
        subprocess.run(["git", "-C", str(top), "fetch", "-q", "origin", sha], check=True)
    dest = Path(f"/tmp/research-move-{sha[:12]}")
    if not dest.exists():
        subprocess.run(["git", "-C", str(top), "worktree", "add", "-q", "--detach", str(dest), sha], check=True)
    return dest


def move(run_dir: Path, *, to: str, reason: str, by: str | None, cpus: int | None, dry_run: bool, now: datetime,
         min_wait_s: float = 0) -> str | None:
    """Resend `run_dir`'s waiting pool run onto `to` and cancel the waiting copy; the new run's id, the argv with `dry_run`, or None
    when it has waited less than `min_wait_s`."""
    from . import remote

    remote.fetch(run_dir)
    slot = remote._read_json(run_dir / "slot.json") or {}
    if (w := waited_s(slot, now)) is None:
        raise Unmovable(f"{run_dir.name} isn't waiting in a pool's line (slot.json: {slot.get('state') or 'none'}), so moving it "
                        f"would throw away what it has run")
    if w < min_wait_s:
        return None
    req = _request(run_dir)
    if not req.get("_source_sha"):
        raise Unmovable(f"{run_dir.name}'s remote.json records no source commit to resend")
    argv = argv_for(req, source=_worktree(req["_source_sha"]), to=to, cpus=cpus)
    if dry_run:
        return " ".join(argv)
    out = subprocess.run([sys.executable, "-m", *argv[1:]], capture_output=True, text=True)
    sys.stderr.write(out.stderr)
    new = next((m.group(0) for m in NEW_RUN.finditer(out.stdout + out.stderr) if m.group(0) != run_dir.name), None)
    if out.returncode or not new:
        raise Unmovable(f"the resend was refused (rc {out.returncode}); {run_dir.name} stays in its line: "
                        f"{(out.stderr or out.stdout).strip()[-300:]}")
    by = by or unwrap(list(req.get("command") or []))["lane"] or "-"
    remote.cancel(run_dir, reason=f"moved to {new} after {w / 60:.0f} min in {req['_machine']}'s line ({reason})", by=by)
    return new


def main(argv: list[str]) -> int:
    from . import remote
    from . import runs as R

    ap = argparse.ArgumentParser(prog="research move", description=__doc__.split("\n\n")[0])
    ap.add_argument("ref", nargs="?", help="a run id or prefix under the runs root")
    ap.add_argument("--to", default="auto", metavar="MACHINE", help="where to resend it (default auto: research run --on auto)")
    ap.add_argument("--waiting-over", type=float, default=None, metavar="MIN",
                    help="instead of RUN: move each run under the runs root that has waited in a pool's line more than MIN minutes")
    ap.add_argument("--reason", default="it starts sooner elsewhere")
    ap.add_argument("--by", default=None, help="who moves it (default: the run's lane)")
    ap.add_argument("--cpus", type=int, default=None, help="the resend's --cpus (default: --on auto's)")
    ap.add_argument("--dry-run", action="store_true", help="print the `research run` it would send, and send nothing")
    ap.add_argument("--runs-dir", default=None)
    ns = ap.parse_args(argv)
    if (ns.ref is None) == (ns.waiting_over is None):
        ap.error("name a RUN or --waiting-over MIN")
    root, now = R.runs_root(ns.runs_dir), datetime.now(UTC)
    if ns.ref:
        dirs = [R.resolve(root, ns.ref)]
    else:
        dirs = [d for d in sorted(root.glob("r2026*")) if (d / "remote.json").exists() and (d / "slot.json").exists()
                and (remote._read_json(d / "status.json") or {}).get("state") == "running"]
    rc = 0
    for d in dirs:
        try:
            got = move(d, to=ns.to, reason=ns.reason, by=ns.by, cpus=ns.cpus, dry_run=ns.dry_run, now=now,
                       min_wait_s=(ns.waiting_over or 0) * 60)
        except (Unmovable, remote.Refused, subprocess.CalledProcessError, ValueError) as e:
            print(f"research move: {d.name}: {e}", file=sys.stderr)
            rc = 1
            continue
        if got is not None:
            print(f"research move: {d.name}: " + (f"would send: {got}" if ns.dry_run else f"moved to {got}"))
    return rc
