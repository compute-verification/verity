"""`research agents quiet`: which worker agents have gone quiet: no commit, no file change and no live process for hours (INF-55 h).

    research agents quiet [--hours 2] [--repo DIR] [--branch NAME ...] [--json]

A worker that stops (its VM's kernel OOM-killed its processes, or it wedged) still shows RUNNING in Cursor, and its transcript and
resume are closed to its coordinator while it does: circuits' IR-typing worker made no commit for four hours on 17-18 Oct before
anyone noticed. Each git worktree of `--repo` (default: this checkout's), where a coordinator's local workers each work, is read
for its HEAD commit's time, the newest mtime among the files `git status` lists as changed, and the processes whose cwd is in it.
A worktree with no such process whose newest commit or change is older than `--hours` is QUIET. `--branch NAME` (a worker on a VM of
its own) reads the time of origin/NAME's newest commit (a fetch), and is QUIET past `--hours` the same way. A coordinator runs it
on its tick and looks at the QUIET lines; the exit status is 1 when any is.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAX_CHANGED = 2000


def git(repo: Path, *args: str, timeout: float = 60) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    return p.stdout if p.returncode == 0 else ""


def worktrees(repo: Path) -> list[dict]:
    out, cur = [], {}
    for line in git(repo, "worktree", "list", "--porcelain").splitlines() + [""]:
        if not line:
            if cur.get("path"):
                out.append(cur)
            cur = {}
        elif line.startswith("worktree "):
            cur["path"] = line.split(" ", 1)[1]
        elif line.startswith("branch "):
            cur["branch"] = line.split(" ", 1)[1].removeprefix("refs/heads/")
    return out


def cwds() -> dict[int, tuple[str, str]]:
    """pid -> (cwd, comm) of every process this user may read."""
    out = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                out[int(name)] = (os.readlink(f"/proc/{name}/cwd"), Path(f"/proc/{name}/comm").read_text().strip())
            except OSError:
                continue
    return out


def newest_change(wt: Path) -> float | None:
    raw = git(wt, "status", "--porcelain", "-z", "--untracked-files=normal")
    newest = None
    for entry in [e for e in raw.split("\0") if len(e) > 3][:MAX_CHANGED]:
        try:
            m = (wt / entry[3:]).stat().st_mtime
        except OSError:
            continue
        newest = m if newest is None else max(newest, m)
    return newest


def worktree_row(wt: dict, procs: dict[int, tuple[str, str]], now: float, hours: float) -> dict:
    path = Path(wt["path"])
    head = git(path, "log", "-1", "--format=%ct").strip()
    commit = float(head) if head.isdigit() else None
    change = newest_change(path)
    live = sorted({comm for cwd, comm in procs.values() if cwd == str(path) or cwd.startswith(str(path) + "/")})
    last = max([t for t in (commit, change) if t is not None], default=None)
    quiet = not live and (last is None or now - last > hours * 3600)
    return {"worktree": str(path), "branch": wt.get("branch"), "commit_age_h": _age(now, commit), "change_age_h": _age(now, change),
            "live": live, "quiet": quiet}


def branch_row(repo: Path, branch: str, now: float, hours: float) -> dict:
    git(repo, "fetch", "-q", "origin", branch, timeout=120)
    t = git(repo, "log", "-1", "--format=%ct", "FETCH_HEAD").strip()
    commit = float(t) if t.isdigit() else None
    return {"branch": branch, "commit_age_h": _age(now, commit), "quiet": commit is None or now - commit > hours * 3600,
            **({} if commit is not None else {"error": f"origin/{branch} not found"})}


def _age(now: float, t: float | None) -> float | None:
    return None if t is None else round((now - t) / 3600, 2)


def quiet_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research agents quiet", description=__doc__.split("\n\n")[0])
    ap.add_argument("--hours", type=float, default=2.0, help="quiet past this many hours (default 2)")
    ap.add_argument("--repo", default=".", help="a checkout whose worktrees to read (default .)")
    ap.add_argument("--branch", action="append", default=[], metavar="NAME", help="a worker's branch on origin (a worker with its own VM)")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    repo, now = Path(ns.repo).resolve(), time.time()
    procs = cwds()
    rows = [worktree_row(wt, procs, now, ns.hours) for wt in worktrees(repo)]
    rows += [branch_row(repo, b, now, ns.hours) for b in ns.branch]
    if ns.json:
        print(json.dumps(rows, indent=1))
    else:
        for r in sorted(rows, key=lambda r: not r["quiet"]):
            ages = ", ".join(f"{k.split('_')[0]} {v} h ago" for k, v in r.items() if k.endswith("_age_h") and v is not None)
            what = r.get("worktree") or f"origin/{r['branch']}"
            live = f"; live: {', '.join(r['live'])}" if r.get("live") else ("; no live process" if "live" in r else "")
            print(f"{'QUIET ' if r['quiet'] else 'active'} {what}" + (f" ({r['branch']})" if r.get("worktree") and r.get("branch") else "")
                  + (f": {ages}" if ages else ": no commit or change") + live + (f"; {r['error']}" if r.get("error") else ""))
    return 1 if any(r["quiet"] for r in rows) else 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["quiet"]:
        return quiet_main(argv[1:])
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    return 2
