"""`research digest --as LANE`: what changed for a lane since its last digest, on one screen, for a coordinator's
regular sync. It shows:

- the lane's doorbells, marked read;
- other handles' new replies in the threads the lane watches;
- what landed on `main`;
- the lane's PRs whose queue state changed;
- new or changed notes under lanes/LANE;
- a TRIGGER line for each watched PR that landed or changed its queue state, with the step the lane wrote for that
  moment.

  research digest --as LANE                          the digest
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
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

#: How far back a first digest reads doorbells and threads.
FIRST_LOOKBACK_S = 3600
CHANNEL = "agent-coordination"
QUEUE_SUBJECT = re.compile(r"^Queue: #(\d+)")


@dataclass
class Sources:
    doorbells: Callable[[str], list[dict]]            # since ts -> the lane's doorbells after it, marked read
    thread: Callable[[str, str, str], list[dict]]     # channel, thread ts, since ts -> its replies after it
    landings: Callable[[str | None], tuple[str, list[str]]]   # last main tip -> (tip, first-parent subjects, oldest first)
    queue: Callable[[], list[dict]]                   # `research queue status --json`
    pr_state: Callable[[int], str]                    # OPEN, MERGED, CLOSED, or "" when unknown
    notes: Callable[[str | None], tuple[str, list[str]]]      # last notes tip -> (tip, changed paths under lanes/LANE)
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
    tip, subjects = src.landings(s.get("main"))
    queued = [x for x in subjects if QUEUE_SUBJECT.match(x)]
    if queued:
        out.append(f"## Landed on main ({len(queued)})")
        out += [f"- {plain(x, 140)}" for x in queued]
    s["main"] = tip
    return {int(QUEUE_SUBJECT.match(x)[1]) for x in queued}


def _queue(lane: str, s: dict, src: Sources, landed: set[int], out: list[str]) -> None:
    rows = {str(r["pr"]): r for r in src.queue()}
    ours = {k: r for k, r in rows.items() if r.get("owner") == lane}
    old = s["queue"]
    lines = []
    for k, r in sorted(ours.items(), key=lambda kv: int(kv[0])):
        cur = [r["head"][:8], r["state"], why(r)]
        if old.get(k) != cur:
            pos = f" at {r['position']}" if r.get("position") else ""
            was = f" (was {old[k][1]})" if k in old else " (new)"
            lines.append(f"- #{k} {cur[0]} {cur[1]}{pos}{was}{': ' + plain(cur[2], 200) if cur[2] else ''}")
    for k in sorted(set(old) - set(ours), key=int):
        lines.append(f"- #{k} landed" if int(k) in landed else f"- #{k} left the queue ({src.pr_state(int(k)) or 'state unknown'})")
    counts: dict[str, int] = {}
    for r in ours.values():
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    out.append(f"## Our PRs: {', '.join(f'{n} {st}' for st, n in sorted(counts.items())) or 'none in the queue'}")
    out += lines
    s["queue"] = {k: [r["head"][:8], r["state"], why(r)] for k, r in ours.items()}

    triggers = []
    for k, todo in s["watch"].items():
        r = rows.get(k)
        cur = f"{r['state']}:{r['head'][:8]}" if r else None
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


def _notes(s: dict, src: Sources, out: list[str]) -> None:
    tip, files = src.notes(s.get("notes"))
    if files:
        out.append(f"## Notes changed ({len(files)})")
        out += [f"- {Path(x).name}" for x in files]
    s["notes"] = tip


def digest(lane: str, s: dict, src: Sources) -> tuple[list[str], list[str]]:
    """The digest's lines and the sources' errors; `s` advances past everything shown. A failing source is reported and
    leaves its cursor where it was, so its news shows next time."""
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
    guard("queue", lambda: _queue(lane, s, src, landed, out))
    guard("notes", lambda: _notes(s, src, out))
    if errors:
        out += ["## Errors", *[f"- {plain(e, 300)}" for e in errors]]
    return out, errors


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

    return Sources(
        doorbells=lambda since: json.loads(_research("msg", "read", "--as", lane, "--since", since, "--json", cwd=repo) or "[]"),
        thread=lambda ch, ts, since: json.loads(
            _research("msg", "read", "--channel", ch, "--thread", ts, "--since", since, "--json", cwd=repo) or "[]"),
        landings=landings,
        queue=lambda: json.loads(_research("queue", "status", "--json", cwd=repo)),
        pr_state=pr_state,
        notes=notes,
        now=time.time,
    )


def main(argv: list[str] | None = None) -> int:
    from .notes import default_root

    ap = argparse.ArgumentParser(prog="research digest", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--as", dest="lane", required=True, help="the lane's handle (the queue's owner, the notes' lanes/LANE)")
    ap.add_argument("--state", help="the state file (default $RESEARCH_DIGEST_STATE, else ~/.research/digest/LANE.json)")
    ap.add_argument("--repo", default=".", help="a clone of the repository whose `main` and queue to read (default .)")
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
    out, errors = digest(lane, s, real_sources(lane, Path(ns.repo).resolve(), default_root()))
    save(path, s)
    print("\n".join(out))
    return 1 if errors else 0
