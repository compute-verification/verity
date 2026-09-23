"""`research notes`: lane checkpoints and the coordinator's liveness table.

  research notes checkpoint LANE STATE TEXT... [--commit SHA] [--root DIR]   # STATE = open | blocked | final
  research notes status [--root DIR] [--stale-min 45] [--since-hours 24] [--all] [--lane GLOB] [--exclude GLOB] [--json]

A lane's report is `<root>/lanes/<lane>/<stamp>-report-<lane>.md` (the newest one if several).  `checkpoint` writes
`CHECKPOINT <sha> (<HH:MM>Z) [<state>] <text>` directly under the front matter (newest first, where lanes already put them),
sets the front matter's `status:` to the state, and creates the report if the lane has none.  `status` reads every report
touched in the window: the newest checkpoint (legacy free-form lines too: `CHECKPOINT <sha> (18:50Z) ...`,
`CHECKPOINT <sha> 18:51Z: ...`, `(CHECKPOINT <sha>): ...`), whether a `## FINAL` section has content, and how long ago the file
last changed.  A lane that is not final and whose report has not changed for `--stale-min` minutes is STALE: in practice a
lane that died silently (its pod idles on the bill) or one that stopped reporting, and either way the coordinator acts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

STATES = ("open", "blocked", "final")
CHECKPOINT_RE = re.compile(r"CHECKPOINT\s+(?P<sha>[0-9a-fA-F]{4,40}|none)?\)?[:\s]*(?P<fin>\*{0,2}FINAL\*{0,2})?\s*\(?"
                           r"(?:(?P<hh>\d{1,2}):(?P<mm>\d{2})Z)?[);]?\s*(?:\[(?P<state>open|blocked|final)\])?\s*[-—:]*\s*(?P<text>.*)")
FINAL_RE = re.compile(r"^##\s+FINAL\b.*$", re.M)
DONE_STATUSES = ("done", "final", "closed", "complete", "completed")


def default_root() -> Path:
    return Path(os.environ.get("RESEARCH_NOTES", Path.home() / ".research" / "notes"))


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def report_for(root: Path, lane: str) -> Path | None:
    reports = sorted((root / "lanes" / lane).glob("*report*.md"), key=lambda p: p.stat().st_mtime)
    return reports[-1] if reports else None


def _split_front_matter(text: str) -> tuple[str, str]:
    """(front matter incl. both `---` lines and the newline after, body); ("", text) when there is none."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end >= 0:
            return text[: end + 5], text[end + 5:]
    return "", text


def _head_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "none"
    except (OSError, subprocess.SubprocessError):
        return "none"


def checkpoint(root: Path, lane: str, state: str, text: str, commit: str | None = None, now: dt.datetime | None = None) -> Path:
    if state not in STATES:
        raise ValueError(f"state must be one of {', '.join(STATES)}, not {state!r}")
    now = now or _utcnow()
    path = report_for(root, lane)
    if path is None:
        path = root / "lanes" / lane / f"{now:%Y%m%dT%H%MZ}-report-{lane}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nlane: {lane}\nkind: report\ncreated: {now:%Y-%m-%dT%H:%MZ}\nstatus: {state}\n---\n\n")
    front, body = _split_front_matter(path.read_text())
    if front:
        front = re.sub(r"^status:.*$", f"status: {state}", front, count=1, flags=re.M) if re.search(r"^status:", front, re.M) \
            else front[:-4] + f"status: {state}\n---\n"
    line = f"CHECKPOINT {commit or _head_commit()} ({now:%H:%M}Z) [{state}] {' '.join(text.split())}\n"
    path.write_text(front + "\n" + line + body.lstrip("\n") if front else line + body)
    return path


def _checkpoint_time(hh: str | None, mm: str | None, mtime: dt.datetime) -> dt.datetime | None:
    """A checkpoint carries only HH:MM; it was written on or before the file's last change, so it is the latest such
    instant not after mtime.  Lanes round, or stamp the time a step will finish, hence an hour of slack."""
    if hh is None:
        return None
    t = mtime.replace(hour=int(hh) % 24, minute=int(mm), second=0, microsecond=0)
    return t - dt.timedelta(days=1) if t > mtime + dt.timedelta(hours=1) else t


def lane_status(path: Path, now: dt.datetime, stale_min: float) -> dict:
    text = path.read_text(errors="replace")
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
    best = None
    for line in text.splitlines():
        i = line.find("CHECKPOINT")
        if i < 0:
            continue
        m = CHECKPOINT_RE.match(line[i:])
        if not m:
            continue
        rest = m["text"].strip()
        final = bool(m["fin"]) or re.match(r"\*{0,2}FINAL\b", rest) is not None
        hh, mm = m["hh"], m["mm"]
        if hh is None and (late := re.search(r"\b(\d{1,2}):(\d{2})Z\b", rest[:60])):   # `CHECKPOINT <sha> D0 LANDED (17:52Z) ...`
            hh, mm = late.groups()
        when = _checkpoint_time(hh, mm, mtime)
        cand = {"commit": m["sha"], "at": when, "state": "final" if final else m["state"], "text": rest}
        if best is None or (when is not None and (best["at"] is None or when > best["at"])):
            best = cand
    finals = [m.end() for m in FINAL_RE.finditer(text)]
    final_body = text[finals[-1]:].strip() if finals else ""
    has_final = bool(final_body) and not re.fullmatch(r"\(?(pending|tbd|todo)\)?\.?", final_body.split("\n", 1)[0].strip(), re.I)
    front, _ = _split_front_matter(text)
    fm = re.search(r"^status:\s*(\S+)", front, re.M)
    fm_state = fm.group(1).lower() if fm else "open"
    state = "final" if has_final or fm_state in DONE_STATUSES else (best or {}).get("state") or fm_state
    age = (now - mtime).total_seconds() / 60
    return {"lane": path.parent.name, "report": str(path), "state": state, "age_min": round(age, 1),
            "stale": state != "final" and age > stale_min,
            "checkpoint_at": best["at"].strftime("%H:%MZ") if best and best["at"] else None,
            "commit": best["commit"] if best else None, "text": best["text"] if best else None}


def status(root: Path, stale_min: float = 45, since_hours: float | None = 24, now: dt.datetime | None = None,
           lanes: list[str] | None = None, exclude: list[str] | None = None) -> list[dict]:
    from fnmatch import fnmatch  # noqa: PLC0415

    now = now or _utcnow()
    rows = []
    for d in sorted((root / "lanes").glob("*/")):
        if (lanes and not any(fnmatch(d.name, g) for g in lanes)) or any(fnmatch(d.name, g) for g in exclude or ()):
            continue
        path = report_for(root, d.name)
        if path is None:
            continue
        if since_hours is not None and (now.timestamp() - path.stat().st_mtime) > since_hours * 3600:
            continue
        rows.append(lane_status(path, now, stale_min))
    order = {"stale": 0, "blocked": 1, "open": 2, "final": 3}
    return sorted(rows, key=lambda r: (order["stale"] if r["stale"] else order.get(r["state"], 2), -r["age_min"]))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research notes", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("checkpoint", help="write a structured CHECKPOINT line at the top of the lane's report")
    cp.add_argument("lane")
    cp.add_argument("state", choices=STATES)
    cp.add_argument("text", nargs="+")
    cp.add_argument("--commit", help="default: git rev-parse --short HEAD in the current directory")
    cp.add_argument("--root", type=Path, default=None)
    st = sub.add_parser("status", help="one row per lane: state, last checkpoint, minutes since the report changed, STALE")
    st.add_argument("--root", type=Path, default=None)
    st.add_argument("--stale-min", type=float, default=45)
    st.add_argument("--since-hours", type=float, default=24)
    st.add_argument("--all", action="store_true", help="every lane, however old its report")
    st.add_argument("--lane", action="append", default=[], metavar="GLOB", help="only lanes matching (repeatable)")
    st.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="skip lanes matching (repeatable)")
    st.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    root = ns.root or default_root()
    if ns.cmd == "checkpoint":
        print(checkpoint(root, ns.lane, ns.state, " ".join(ns.text), commit=ns.commit))
        return 0
    rows = status(root, ns.stale_min, None if ns.all else ns.since_hours, lanes=ns.lane, exclude=ns.exclude)
    if ns.json:
        print(json.dumps(rows, indent=1))
        return 0
    if not rows:
        print("no lane reports in the window")
        return 0
    w = max(len(r["lane"]) for r in rows)
    for r in rows:
        flag = "STALE" if r["stale"] else r["state"]
        cp_ = f"{r['checkpoint_at'] or '--:--Z'} {(r['commit'] or '-')[:9]:<9}"
        print(f"{r['lane']:<{w}}  {flag:<7} {r['age_min']:>6.0f}m  {cp_}  {(r['text'] or '')[:90]}")
    return 1 if any(r["stale"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
