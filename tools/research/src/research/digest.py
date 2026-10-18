"""`research digest --as LANE`: what changed for a lane since its last digest, on one screen, for a coordinator's
regular sync. It shows:

- the lane's doorbells, marked read;
- other handles' new replies in the threads the lane watches;
- what landed on `main` and what drive reverted there, and how many landed in the last 6 h;
- the queue's length by state, over every lane;
- the lane's PRs whose queue state or head changed;
- new or changed notes under lanes/LANE, each with the first 3 non-empty lines under its first 'Result so far' heading
  (level 2 to 4), up to the next heading;
- a TRIGGER line for each watched PR that landed or changed its queue state (not merely its head), with the step the
  lane wrote for that moment;
- Health: this VM's memory, load, disk (/, the notes, the workspace) and OOM kills; /workspace's branch (a WARNING
  when it's on one: the shell hook and the inbox run from that checkout), dirt and lag behind origin/main; each
  --roster worker's branch age and pending inbox deliveries (a WARNING past 30 min); the lane's open PRs that aren't in
  the queue at all; the largest .lake directories of finished worktrees (tip pushed, over 6 h old; listed, never
  deleted); and the live runs launched from this VM. A failing check prints its error in its place.

  research digest --as LANE [--roster FILE]          the digest; FILE has TSV lines agent_id, label, branch ('#' comments)
  research digest --as LANE --watch N 'TODO'         print TODO as a TRIGGER when #N lands or its queue state changes
  research digest --as LANE --unwatch N
  research digest --as LANE --thread NAME TS [--channel KEY]   watch a thread (default channel agent-coordination)
  research digest --as LANE --drop-thread NAME
  research digest --as LANE --show                   the watch list and the threads

State, cursors included, is one JSON file: --state, else $RESEARCH_DIGEST_STATE, else ~/.research/digest/LANE.json. A
cloud VM boots from a snapshot that resets ~/.research, so point it at the agent's store
(`RESEARCH_DIGEST_STATE=/cursor/stores/self/digest-LANE.json`). Exit 1 when a source failed; the others are still
shown."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

#: How far back a first digest reads doorbells and threads.
FIRST_LOOKBACK_S = 3600
#: The landing rate's window.
RATE_WINDOW_H = 6
CHANNEL = "agent-coordination"
QUEUE_SUBJECT = re.compile(r"^Queue: #(\d+)")
#: Drive's `git revert -m 1` of a breaking landing.
REVERT_SUBJECT = re.compile(r'^Revert "Queue: #(\d+)')
RESULT_HEADING = re.compile(r"#{2,4}\s+result so far", re.IGNORECASE)
HEADING = re.compile(r"#{1,6}\s")
RESULT_LINES = 3
#: The checkout the shell hook and the inbox run from.
WORKSPACE = Path("/workspace")
#: An inbox delivery pending longer than this is flagged: its agent has run no shell command since.
INBOX_STALE_S = 30 * 60
#: A worktree whose branch tip is on origin and older than this is a finished worker's.
FINISHED_H = 6
LAKES_SHOWN = 5
LIVE = ("running", "waiting")
INBOX_LINE = re.compile(r"^pending\s+(\S+)\s+(\d+) bytes(?:\s+since (\S+))?\s*$")


@dataclass
class Sources:
    doorbells: Callable[[str], list[dict]]            # since ts -> the lane's doorbells after it, marked read
    thread: Callable[[str, str, str], list[dict]]     # channel, thread ts, since ts -> its replies after it
    landings: Callable[[str | None], tuple[str, list[str]]]   # last main tip -> (tip, first-parent subjects, oldest first)
    landed_since: Callable[[float], int]              # unix time -> drive's landings on main since then
    queue: Callable[[], list[dict]]                   # `research queue status --json`
    pr_state: Callable[[int], str]                    # OPEN, MERGED, CLOSED, or "" when unknown
    notes: Callable[[str | None], tuple[str, list[str]]]      # last notes tip -> (tip, changed paths under lanes/LANE)
    note_text: Callable[[str, str], str | None]       # notes tip, path -> the file at that tip, None when it has none
    vm: Callable[[], dict]                            # {mem_total, mem_available, load, cpus, disks, oom, oom_error}
    workspace: Callable[[], dict]                     # /workspace: {branch (None detached), head, dirty, behind}
    branch_age: Callable[[str], str | None]           # branch -> "%cr %h" of origin's tip after a fetch, None when absent
    inbox: Callable[[], list[dict] | None]            # pending deliveries [{agent, bytes, since}], None without inbox.py
    lane_prs: Callable[[], list[dict]]                # the lane's open PRs [{number, title, baseRefName, isDraft}]
    lakes: Callable[[], list[dict]]                   # .lake dirs of finished worktrees [{path, bytes, branch, age_h}]
    runs: Callable[[], list[dict]]                    # `research status --all --json`'s runs
    now: Callable[[], float]


def state_path(lane: str, explicit: str | None = None) -> Path:
    return Path(explicit or os.environ.get("RESEARCH_DIGEST_STATE") or Path.home() / ".research" / "digest" / f"{lane}.json")


def load(path: Path) -> dict:
    s = json.loads(path.read_text()) if path.is_file() else {}
    for k in ("threads", "seen", "watch", "watch_seen", "queue"):
        s.setdefault(k, {})
    return s


def save(path: Path, s: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


def hhmm(ts: str | float) -> str:
    return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%d %H:%MZ")


def plain(text: str | None, n: int = 420) -> str:
    text = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"\2", text or "")
    text = re.sub(r"<(https?://[^>]+)>", r"\1", text)
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def why(row: dict) -> str:
    w = row.get("why") or ""
    return "unmarked" if w.startswith("not marked ready at") else w


def gib(n: float) -> str:
    return f"{n / 2**30:.1f} GiB" if n >= 2**30 else f"{n / 2**20:.0f} MiB"


def ago(seconds: float) -> str:
    return f"{int(seconds // 60)} min" if seconds < 7200 else f"{seconds / 3600:.1f} h"


def result_so_far(text: str | None, n: int = RESULT_LINES) -> list[str]:
    """The first `n` non-empty lines under a note's first 'Result so far' heading of level 2 to 4, up to the next heading."""
    lines = (text or "").splitlines()
    start = next((i + 1 for i, x in enumerate(lines) if RESULT_HEADING.match(x.strip())), None)
    out: list[str] = []
    for x in lines[start:] if start is not None else []:
        if HEADING.match(x) or len(out) == n:
            break
        if x.strip():
            out.append(x.strip())
    return out


def parse_roster(text: str, source: str = "roster") -> list[tuple[str, str, str]]:
    """A roster's workers, (agent_id, label, branch) from TSV lines; blank lines and '#' lines are skipped."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [x.strip() for x in line.strip().split("\t")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"{source}:{i}: want agent_id<TAB>label<TAB>branch, got {line!r}")
        out.append((parts[0], parts[1], parts[2]))
    return out


def parse_inbox(text: str) -> list[dict]:
    """The pending deliveries `inbox.py list` prints; `since` is None from an inbox.py older than its stamp."""
    out = []
    for line in text.splitlines():
        if m := INBOX_LINE.match(line):
            since = datetime.strptime(m[3], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc).timestamp() if m[3] else None
            out.append({"agent": m[1], "bytes": int(m[2]), "since": since})
    return out


def _doorbells(s: dict, src: Sources, out: list[str]) -> set[str]:
    since = s.get("doorbell") or f"{src.now() - FIRST_LOOKBACK_S:.6f}"
    msgs = sorted((m for m in src.doorbells(since) if float(m["ts"]) > float(since)), key=lambda m: float(m["ts"]))
    if msgs:
        s["doorbell"] = msgs[-1]["ts"]
        out.append(f"## Doorbells ({len(msgs)})")
    shown: set[str] = set()
    for m in msgs:
        t = m.get("text") or ""
        shown.update(re.findall(r"reply (\d+\.\d+)", t))
        shown.update(re.findall(r"thread (\d+\.\d+):", t))
        out.append(f"- {hhmm(m['ts'])} {plain(t)}")
    return shown


def _threads(lane: str, s: dict, src: Sources, shown: set[str], out: list[str]) -> None:
    lines = []
    floor = f"{src.now() - FIRST_LOOKBACK_S:.6f}"
    for name, (channel, ts) in sorted(s["threads"].items()):
        key = f"{channel}:{ts}"
        since = s["seen"].get(key, floor)
        new = sorted((m for m in src.thread(channel, ts, since) if float(m["ts"]) > float(since)), key=lambda m: float(m["ts"]))
        if new:
            s["seen"][key] = new[-1]["ts"]
        lines += [f"- {name} {hhmm(m['ts'])} {m.get('username') or m.get('user')}: {plain(m.get('text'))}"
                  for m in new if m.get("username") != lane and m["ts"] not in shown]
    if lines:
        out += ["## Threads", *lines]


def _landings(s: dict, src: Sources, out: list[str]) -> set[int]:
    """The PRs whose landing is on `main` at the new tip: in subject order, so a PR reverted and then relanded counts."""
    tip, subjects = src.landings(s.get("main"))
    queued = [x for x in subjects if QUEUE_SUBJECT.match(x)]
    reverted = [x for x in subjects if REVERT_SUBJECT.match(x)]
    if queued:
        out.append(f"## Landed on main ({len(queued)})")
        out += [f"- {plain(x, 140)}" for x in queued]
    if reverted:
        out.append(f"## Reverted on main ({len(reverted)})")
        out += [f"- {plain(x, 140)}" for x in reverted]
    s["main"] = tip
    landed: set[int] = set()
    for x in subjects:
        if m := QUEUE_SUBJECT.match(x):
            landed.add(int(m[1]))
        elif m := REVERT_SUBJECT.match(x):
            landed.discard(int(m[1]))
    return landed


def _summary(src: Sources, rows: list[dict], out: list[str]) -> None:
    """Every lane's queue by state and the landing rate: what a sweep of the whole queue reads."""
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    n = src.landed_since(src.now() - RATE_WINDOW_H * 3600)
    out.append(f"## Queue: {', '.join(f'{c} {st}' for st, c in sorted(counts.items())) or 'empty'}; "
               f"{n} landed in the last {RATE_WINDOW_H} h")


def _queue(lane: str, s: dict, src: Sources, landed: set[int], out: list[str]) -> list[dict]:
    listed = src.queue()
    _summary(src, listed, out)
    rows = {str(r["pr"]): r for r in listed}
    ours = {k: r for k, r in rows.items() if r.get("owner") == lane}
    old = s["queue"]
    lines = []
    for k, r in sorted(ours.items(), key=lambda kv: int(kv[0])):
        cur = [r["head"][:8], r["state"], why(r)]
        if old.get(k) != cur:
            pos = f" at {r['position']}" if r.get("position") else ""
            was = (" (new)" if k not in old else f" (was {old[k][1]})" if old[k][1] != cur[1] else
                   " (new head)" if old[k][0] != cur[0] else "")
            lines.append(f"- #{k} {cur[0]} {cur[1]}{pos}{was}{': ' + plain(cur[2], 200) if cur[2] else ''}")
    for k in sorted(set(old) - set(ours), key=int):
        lines.append(f"- #{k} landed" if int(k) in landed else f"- #{k} left the queue ({src.pr_state(int(k)) or 'state unknown'})")
    counts: dict[str, int] = {}
    for r in ours.values():
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    out.append(f"## Our PRs: {', '.join(f'{n} {st}' for st, n in sorted(counts.items())) or 'none in the queue'}")
    out += lines
    s["queue"] = {k: [r["head"][:8], r["state"], why(r)] for k, r in ours.items()}

    # A watched PR triggers on its state, not its head: a waiting PR's new head is its owner's news, not the watcher's.
    triggers = []
    for k, todo in s["watch"].items():
        r = rows.get(k)
        cur = r["state"] if r else None
        prev = s["watch_seen"].get(k)
        if int(k) in landed:
            triggers.append(f"- TRIGGER #{k} landed: {todo}")
        elif cur and prev and cur != prev:
            triggers.append(f"- TRIGGER #{k} is now {r['state']} at {r['head'][:8]}"
                            f"{': ' + plain(r.get('why'), 160) if r.get('why') else ''} ({todo})")
        if cur:
            s["watch_seen"][k] = cur
    if triggers:
        out += ["## Watch", *triggers]
    return listed


def _notes(s: dict, src: Sources, out: list[str]) -> None:
    tip, files = src.notes(s.get("notes"))
    lines = []
    for x in files:
        lines.append(f"- {Path(x).name}")
        if x.endswith(".md"):
            lines += [f"    {plain(y, 200)}" for y in result_so_far(src.note_text(tip, x))]
    if files:
        out += [f"## Notes changed ({len(files)})", *lines]
    s["notes"] = tip


def _vm(v: dict) -> list[str]:
    groups: dict[object, list[dict]] = {}
    absent = []
    for d in v["disks"]:
        if "dev" in d:
            groups.setdefault(d["dev"], []).append(d)
        else:
            absent.append(f"{d['label']} {d['path']} absent")
    disks = [f"{', '.join(d['label'] for d in g)} {g[0]['pct']}% used, {gib(g[0]['free'])} free" for g in groups.values()]
    oom = f"OOM kills {v['oom']}" if v.get("oom") is not None else f"OOM kills unreadable ({v.get('oom_error') or 'no dmesg'})"
    return [f"- VM: memory {gib(v['mem_total'] - v['mem_available'])} used, {gib(v['mem_available'])} available; load "
            f"{' '.join(f'{x:.2f}' for x in v['load'])} on {v['cpus']} CPUs; disk {'; '.join(disks + absent)}; {oom}"]


def _workspace(w: dict) -> list[str]:
    where = f"on branch {w['branch']}" if w["branch"] else f"detached at {w['head']}"
    dirt = f"dirty ({w['dirty']} paths)" if w["dirty"] else "clean"
    line = f"{WORKSPACE} {where}, {dirt}, {w['behind']} behind origin/main"
    return [f"- WARNING {line}: the shell hook and the inbox run from this checkout" if w["branch"] else f"- {line}"]


def _health(lane: str, src: Sources, roster: list[tuple[str, str, str]], rows: list[dict] | None,
            out: list[str]) -> list[str]:
    """The Health section, appended to `out`; returns the failed checks' errors. Each check fails alone, in its place."""
    lines: list[str] = []
    errors: list[str] = []
    now = src.now()

    def failed(name: str, e: Exception) -> str:
        errors.append(f"health {name}: {e}")
        return f"{name}: FAILED: {plain(str(e), 300)}"

    def check(name: str, f: Callable[[], list[str]]) -> None:
        try:
            lines.extend(f())
        except Exception as e:  # one check down must not hide the others
            lines.append(f"- {failed(name, e)}")

    check("VM", lambda: _vm(src.vm()))
    check(str(WORKSPACE), lambda: _workspace(src.workspace()))

    pending: dict[str, dict] | None = None
    try:
        got = src.inbox()
        if got is None:
            lines.append(f"- inbox: no tools/agent-guard/inbox.py in {WORKSPACE}")
        else:
            pending = {d["agent"]: d for d in got}
    except Exception as e:
        lines.append(f"- {failed('inbox', e)}")

    def inbox(agent: str) -> tuple[str, bool]:
        if pending is None:
            return "inbox unread", False
        d = pending.get(agent)
        if d is None:
            return "nothing pending in the inbox", False
        if d["since"] is None:
            return f"inbox: {d['bytes']} bytes pending since an unknown time", False
        stale = now - d["since"] > INBOX_STALE_S
        return (f"inbox: {d['bytes']} bytes pending since {hhmm(d['since'])} ({ago(now - d['since'])})"
                f"{f', over {INBOX_STALE_S // 60} min' if stale else ''}"), stale

    for agent, label, branch in roster:
        try:
            age = src.branch_age(branch)
            b = f"{branch} {age}" if age else f"no branch {branch} on origin"
        except Exception as e:
            b = failed(f"branch {branch}", e)
        text, stale = inbox(agent)
        lines.append(f"- {'WARNING ' if stale else ''}worker {label} {agent}: {b}; {text}")
    for agent in sorted(set(pending or {}) - {a for a, _, _ in roster}):
        text, stale = inbox(agent)
        lines.append(f"- {'WARNING ' if stale else ''}{agent} (not in the roster): {text}")

    def prs() -> list[str]:
        if rows is None:
            raise RuntimeError("the queue is unread (see Errors)")
        listed = {int(r["pr"]) for r in rows}
        mine = src.lane_prs()
        missing = sorted((p for p in mine if int(p["number"]) not in listed), key=lambda p: int(p["number"]))
        if not missing:
            return [f"- PRs: all {len(mine)} of the lane's open PRs are in the queue"]
        return [f"- WARNING #{p['number']} is not in the queue (base {p.get('baseRefName') or '?'}"
                f"{', draft' if p.get('isDraft') else ''}): {plain(p.get('title'), 120)}" for p in missing]

    def lakes() -> list[str]:
        got = sorted(src.lakes(), key=lambda d: -d["bytes"])[:LAKES_SHOWN]
        if not got:
            return [f"- .lake: none in finished worktrees (tip on origin, over {FINISHED_H} h old)"]
        return [f"- .lake of finished worktrees, the largest {len(got)} (listed, not deleted):",
                *[f"    {gib(d['bytes'])} {d['path']} ({d['branch']}, tip {d['age_h']:.0f} h old)" for d in got]]

    def runs() -> list[str]:
        live = [r for r in src.runs() if r.get("state") in LIVE]
        if not live:
            return ["- runs: none live from this VM"]
        out = []
        for r in live:
            cmd = r.get("command") or "?"
            out.append(f"- run {r['id']} {r['state']} on {(r.get('remote') or {}).get('machine') or 'local'}: "
                       f"{plain(cmd if isinstance(cmd, str) else ' '.join(map(str, cmd)), 80)}")
        return out

    check("PRs", prs)
    check(".lake", lakes)
    check("runs", runs)
    out += ["## Health", *lines]
    return errors


def digest(lane: str, s: dict, src: Sources, roster: list[tuple[str, str, str]] | None = None) -> tuple[list[str], list[str]]:
    """The digest's lines and the sources' errors; `s` advances past everything shown. A failing source is reported and
    leaves its cursor where it was, so its news shows next time; a failing health check is reported in its place."""
    out = [f"# research digest @{lane} {datetime.fromtimestamp(src.now(), timezone.utc):%d %H:%MZ}"]
    errors: list[str] = []

    def guard(name: str, f: Callable[[], object], default: object = None) -> object:
        try:
            return f()
        except Exception as e:  # one source down must not hide the others
            errors.append(f"{name}: {e}")
            return default

    shown = guard("doorbells", lambda: _doorbells(s, src, out), set())
    guard("threads", lambda: _threads(lane, s, src, shown, out))
    landed = guard("landings", lambda: _landings(s, src, out), set())
    rows = guard("queue", lambda: _queue(lane, s, src, landed, out))
    guard("notes", lambda: _notes(s, src, out))
    health = _health(lane, src, roster or [], rows, out)
    if errors:
        out += ["## Errors", *[f"- {plain(e, 300)}" for e in errors]]
    return out, errors + health


def _research(*args: str, cwd: Path) -> str:
    r = subprocess.run([sys.executable, "-m", "research", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"`research {' '.join(args[:2])}` exited {r.returncode}: {r.stderr.strip()[-300:]}")
    return r.stdout


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"`git {args[0]}` in {root} exited {r.returncode}: {r.stderr.strip()[-300:]}")
    return r.stdout


def real_sources(lane: str, repo: Path, notes_root: Path) -> Sources:
    from .notes_sync import remote_target

    def landings(last: str | None) -> tuple[str, list[str]]:
        _git(repo, "fetch", "-q", "origin", "main")
        tip = _git(repo, "rev-parse", "origin/main").strip()
        if not last or last == tip:
            return tip, []
        return tip, _git(repo, "log", "--first-parent", "--reverse", "--format=%s", f"{last}..{tip}").splitlines()

    def landed_since(t: float) -> int:
        since = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        log = _git(repo, "log", "--first-parent", f"--since={since}", "--format=%s", "origin/main").splitlines()
        return sum(1 for x in log if QUEUE_SUBJECT.match(x))

    def pr_state(n: int) -> str:
        try:
            r = subprocess.run(["gh", "pr", "view", str(n), "--json", "state", "-q", ".state"], cwd=repo, capture_output=True,
                               text=True)
        except FileNotFoundError:
            return ""
        return r.stdout.strip() if r.returncode == 0 else ""

    def notes(last: str | None) -> tuple[str, list[str]]:
        opts, target = remote_target(notes_root)
        _git(notes_root, *opts, "fetch", "-q", target, "main:refs/remotes/origin/main")
        tip = _git(notes_root, "rev-parse", "origin/main").strip()
        files: list[str] = []
        if last and last != tip:
            log = _git(notes_root, "log", "--format=", "--name-only", f"{last}..{tip}", "--", f"lanes/{lane}")
            files = sorted({x for x in log.splitlines() if x.strip()})
        subprocess.run(["git", "-C", str(notes_root), "merge", "-q", "--ff-only", "origin/main"], capture_output=True)
        return tip, files

    def note_text(tip: str, path: str) -> str | None:
        r = subprocess.run(["git", "-C", str(notes_root), "show", f"{tip}:{path}"], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
        if "does not exist" in r.stderr or "exists on disk, but not in" in r.stderr:
            return None
        raise RuntimeError(f"`git show {tip[:8]}:{path}` exited {r.returncode}: {r.stderr.strip()[-300:]}")

    def vm() -> dict:
        mem = {k: int(v.split()[0]) * 1024 for k, _, v in
               (line.partition(":") for line in Path("/proc/meminfo").read_text().splitlines()) if v.strip()}
        disks = []
        for label, p in (("/", Path("/")), ("notes", notes_root), ("workspace", WORKSPACE)):
            if p.exists():
                u = shutil.disk_usage(p)
                disks.append({"label": label, "path": str(p), "dev": p.stat().st_dev, "pct": round(100 * u.used / (u.used + u.free)),
                              "free": u.free})
            else:
                disks.append({"label": label, "path": str(p)})
        try:
            r = subprocess.run(["dmesg", "-T"], capture_output=True, text=True)
            oom = r.stdout.count("Out of memory: Killed process") if r.returncode == 0 else None
            oom_error = "" if r.returncode == 0 else f"dmesg: {r.stderr.strip()[-200:] or f'exit {r.returncode}'}; it may need root"
        except FileNotFoundError:
            oom, oom_error = None, "no dmesg on this machine"
        return {"mem_total": mem["MemTotal"], "mem_available": mem["MemAvailable"], "load": os.getloadavg(),
                "cpus": os.cpu_count(), "disks": disks, "oom": oom, "oom_error": oom_error}

    def workspace() -> dict:
        if not (WORKSPACE / ".git").exists():
            raise RuntimeError(f"{WORKSPACE} is not a git checkout")
        r = subprocess.run(["git", "-C", str(WORKSPACE), "symbolic-ref", "-q", "--short", "HEAD"], capture_output=True, text=True)
        _git(WORKSPACE, "fetch", "-q", "origin", "main")
        return {"branch": r.stdout.strip() or None, "head": _git(WORKSPACE, "rev-parse", "--short", "HEAD").strip(),
                "dirty": len(_git(WORKSPACE, "status", "--porcelain").splitlines()),
                "behind": int(_git(WORKSPACE, "rev-list", "--count", "HEAD..origin/main"))}

    def branch_age(branch: str) -> str | None:
        r = subprocess.run(["git", "-C", str(repo), "fetch", "-q", "origin", branch], capture_output=True, text=True)
        if r.returncode:
            if "couldn't find remote ref" in r.stderr:
                return None
            raise RuntimeError(f"`git fetch origin {branch}` exited {r.returncode}: {r.stderr.strip()[-300:]}")
        return _git(repo, "log", "-1", "--format=%cr %h", f"origin/{branch}").strip()

    def inbox() -> list[dict] | None:
        tool = WORKSPACE / "tools/agent-guard/inbox.py"
        if not tool.is_file():
            return None
        r = subprocess.run([sys.executable, str(tool), "list"], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"`inbox.py list` exited {r.returncode}: {r.stderr.strip()[-300:]}")
        return parse_inbox(r.stdout)

    def lane_prs() -> list[dict]:
        from .queue import LANE_RE

        r = subprocess.run(["gh", "pr", "list", "--state", "open", "--search", f'"Lane {lane}" in:body', "--limit", "200",
                            "--json", "number,title,body,baseRefName,isDraft"], cwd=repo, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"`gh pr list` exited {r.returncode}: {r.stderr.strip()[-300:]}")
        return [p for p in json.loads(r.stdout) if any(m[1] == lane for m in LANE_RE.finditer(p.get("body") or ""))]

    def lakes() -> list[dict]:
        now, found = time.time(), []
        for block in _git(repo, "worktree", "list", "--porcelain").strip().split("\n\n")[1:]:   # [0]: the main checkout
            kv = dict(x.split(" ", 1) for x in block.splitlines() if " " in x)
            wt, head, ref = Path(kv.get("worktree", "/nonexistent")), kv.get("HEAD", ""), kv.get("branch", "")
            if not (head and ref.startswith("refs/heads/") and wt.is_dir()):
                continue
            branch = ref.removeprefix("refs/heads/")
            pushed = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", head, f"refs/remotes/origin/{branch}"],
                                    capture_output=True).returncode == 0
            age_h = (now - int(_git(repo, "log", "-1", "--format=%ct", head))) / 3600
            if not pushed or age_h < FINISHED_H:
                continue
            for lakefile in _git(wt, "ls-files", "--", "*lakefile.lean", "*lakefile.toml").splitlines():
                if (lake := wt / Path(lakefile).parent / ".lake").is_dir():
                    found.append({"path": str(lake), "branch": branch, "age_h": age_h})
        if found:
            du = subprocess.run(["du", "-sk", *[d["path"] for d in found]], capture_output=True, text=True).stdout
            sizes = {p: int(k) * 1024 for k, p in (x.split("\t", 1) for x in du.splitlines() if "\t" in x)}
            for d in found:
                d["bytes"] = sizes.get(d["path"], 0)
        return found

    return Sources(
        doorbells=lambda since: json.loads(_research("msg", "read", "--as", lane, "--since", since, "--json", cwd=repo) or "[]"),
        thread=lambda ch, ts, since: json.loads(
            _research("msg", "read", "--channel", ch, "--thread", ts, "--since", since, "--json", cwd=repo) or "[]"),
        landings=landings,
        landed_since=landed_since,
        queue=lambda: json.loads(_research("queue", "status", "--json", cwd=repo)),
        pr_state=pr_state,
        notes=notes,
        note_text=note_text,
        vm=vm,
        workspace=workspace,
        branch_age=branch_age,
        inbox=inbox,
        lane_prs=lane_prs,
        lakes=lakes,
        runs=lambda: json.loads(_research("status", "--all", "--json", cwd=repo))["runs"],
        now=time.time,
    )


def main(argv: list[str] | None = None) -> int:
    from .notes import default_root

    ap = argparse.ArgumentParser(prog="research digest", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--as", dest="lane", required=True, help="the lane's handle (the queue's owner, the notes' lanes/LANE)")
    ap.add_argument("--state", help="the state file (default $RESEARCH_DIGEST_STATE, else ~/.research/digest/LANE.json)")
    ap.add_argument("--repo", default=".", help="a clone of the repository whose `main` and queue to read (default .)")
    ap.add_argument("--roster", metavar="FILE", help="the workers Health shows: TSV lines agent_id, label, branch ('#' comments)")
    ap.add_argument("--watch", nargs=2, metavar=("PR", "TODO"))
    ap.add_argument("--unwatch", metavar="PR")
    ap.add_argument("--thread", nargs=2, metavar=("NAME", "TS"))
    ap.add_argument("--channel", default=CHANNEL, help="--thread: the thread's registry channel (default agent-coordination)")
    ap.add_argument("--drop-thread", metavar="NAME")
    ap.add_argument("--show", action="store_true")
    ns = ap.parse_args(argv)
    lane = ns.lane.lstrip("@")
    path = state_path(lane, ns.state)
    s = load(path)
    if ns.watch or ns.unwatch or ns.thread or ns.drop_thread or ns.show:
        if ns.watch:
            s["watch"][ns.watch[0].lstrip("#")] = ns.watch[1]
        if ns.unwatch:
            s["watch"].pop(ns.unwatch.lstrip("#"), None)
            s["watch_seen"].pop(ns.unwatch.lstrip("#"), None)
        if ns.thread:
            s["threads"][ns.thread[0]] = [ns.channel, ns.thread[1]]
        if ns.drop_thread:
            s["threads"].pop(ns.drop_thread, None)
        save(path, s)
        print(json.dumps({"watch": s["watch"], "threads": s["threads"]}, indent=1, sort_keys=True))
        return 0
    try:
        roster = parse_roster(Path(ns.roster).read_text(), ns.roster) if ns.roster else []
    except (OSError, ValueError) as e:
        ap.error(f"--roster: {e}")
    out, errors = digest(lane, s, real_sources(lane, Path(ns.repo).resolve(), default_root()), roster)
    save(path, s)
    print("\n".join(out))
    return 1 if errors else 0
