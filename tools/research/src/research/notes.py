"""`research notes`: lane checkpoints and the coordinator's liveness table.

  research notes checkpoint LANE STATE TEXT... [--commit SHA] [--keep-pod WHY] [--root DIR]   # STATE = open | blocked | final | superseded
  research notes inbox LANE [--peek] [--urgent]   # --urgent: exit 1 unless an unread handoff's file name says `urgent`
  research notes bind LANE [--branch B] [--worktree DIR] [--pod NAME] [--succeeds LANE] [--brief PATH] [--final HH:MMZ] [--budget TEXT]
  research notes relaunch LANE [--as SUCCESSOR] [--why TEXT] [--brief PATH] [--final HH:MMZ] [--budget TEXT]   # contract §C in one command
  research notes status [--root DIR] [--repo DIR] [--pods] [--stale-min 30] [--since-hours 24] [--all] [--lane GLOB] [--exclude GLOB] [--json]
  research notes watch [--every 10] [--idle-min 5] [--runway-h 3] [--reap] [--reap-stale-min 30] [--guardian-log PATH] [status options]   # alert lines, below
  research notes gc-worktrees [--apply] [--wt-root DIR]     # clean worktrees of finished lanes whose branch tip is on origin
  research notes push LANE [--repo DIR]                     # after every commit: the lane's branch -> origin (never main, never --force);
                                                            #   in the notes clone: lanes/LANE/ -> the notes' main (`sync`)
  research notes branches [--fetch] [--all] [--lane GLOB] [--json]   # lane branches vs origin; exit 1 if an open lane's work is not there
  research notes sync [--path P]... [-m MSG] | scan          # notes_sync.py: commit, rebase, push with retries | history audit
  research notes archive LANE [--older-days 3] [--bucket month|day] [--apply]   # notes_archive.py: older notes into month folders
  research notes status|watch --liveness git [--fetch] ...   # steward.py: notes commit times and origin's lane/* tips, not mtimes
  research notes watch --lease [store|PATH] [--standby] ...   # steward.py: only the steward lease's holder acts
  research notes service systemd|launchd [...] [-- WATCH ARGS] # steward.py: the steward's service definition
  research notes index [--root DIR] [--catalog PATH] [--search TEXT | --evidence ID]   # notes_index.py: notes, full text, citations
  research notes mcp [--root DIR] [--catalog PATH]            # notes_mcp.py: that index joined to the evidence catalog, as MCP on stdio
  research notes claim | approach | approaches ...            # approaches.py: the approach registry (check, then claim)

A lane's report is its newest `<root>/lanes/<lane>/<stamp>-report-<lane>.md`, the one `checkpoint` creates, so a side report the
lane writes later (`<stamp>-report-<slug>.md`) never takes its checkpoints; with none of those, its newest `<stamp>-report-<slug>.md`;
never a report another agent sent the lane (`<stamp>-report-from-<x>-<slug>.md`) or a note of another kind whose slug says report
(`<stamp>-handoff-from-<x>-report-for.md`).  `checkpoint` writes
`CHECKPOINT <sha> (<HH:MM>Z) [<state>] <text>` directly under the front matter (newest first, where lanes already put them),
sets the front matter's `status:` to the state, and creates the report if the lane has none.  `superseded` marks a lane whose
work a successor took over: it is done, never STALE.

Mail is files named `*handoff*` / `*asks*` / `*reply*` in the recipient's lane dir: the machines' notices (the merge queue's
refusals, the steward's kills and claim refusals) and older agents' handoffs.  Since 2026-10-01 agents message each other on
Slack, never through notes, and the inbox flags an agent's handoff as sent the wrong way.  `checkpoint` (and `inbox`) print the
ones changed since the lane was last shown its inbox, by mtime, and mark them read; a successor (`bind --succeeds`, or the
predecessor's `[superseded] by <lane>` checkpoint) also gets its predecessors' unread ones.  `checkpoint LANE final` then runs
the finish checks: every `art:` id the report cites is preserved on the remote (recorded, else verified there now), no pod of
the lane is still running (`--keep-pod WHY` when one should), the worktree is clean at the branch tip, and every handoff is
named in the report.
The FINAL line is written regardless; any failed check exits 3.

`bind` records where a lane works in `<root>/lanes/<lane>/binding.json` (a successor that takes over its predecessor's
worktree names it there; lanes rewrite their reports, so the binding lives beside them).  Unbound lanes default to branch
`lane/<lane>` and the worktree `git worktree list` shows for it (`--pod none`: a lane without a pod); front matter `branch:` /
`worktree:` / `pod:` lines are read too.

`status` reads every lane touched in the window, one row per topic: the latest attempt (`#2` = the second), predecessors hidden
unless `--all`.  Liveness is the freshest of: the report's last change, its newest checkpoint, the lane branch's last commit,
the newest uncommitted edit in its worktree, and with `--pods` its pods (a work process running counts as now; otherwise the
newest file under /workspace).  Reports alone are not enough: lanes that die mid-step leave a fresh-looking checkpoint and a
worktree full of uncommitted edits.  A lane that is not done and shows no sign of life for `--stale-min` minutes is STALE;
`dirty` is how many files it would lose.  A busy pod with every other signal old can be a detached job that outlived its lane:
the `via` column says which signal is keeping the lane alive.  `inbox` is the open lane's unread handoff count (and the oldest
one's age); `MAIL n` on a finished lane with no successor is handoffs nobody will read.

Pods go by ownership (`pod_owner`): a lane owns every running pod named `vy-<topic>[<letter>][-<role>]` whose topic is the
lane or a predecessor in its succession (longest lane name wins; `vy-wave-h100b` and `vy-wave-h100-verifier` are wave-h100's,
hence wave-h100-2's once it took over); a binding's `--pod` only names a pod whose name carries no lane.  With `--pods`, status
shows each lane's pod count and $/h, and per pod the probe: work processes (`busy`), a live verifier (`serve`), GPU util, and
how long an idle pod has been idle (newest /workspace file, capped at its uptime).

`watch` prints a line only for something to act on: `STALE` / `ALIVE` (a lane goes silent / comes back), `MAIL` (handoffs to
a finished lane nobody will read; not on the first pass, never for a superseded lane), and with `--pods`: `IDLE-POD` (an open
lane's pod idle >= `--idle-min`: the earliest sign the lane died), one `ACCOUNT` line per pass (balance, $/h, runway, lane pods
vs other pods), `RUNWAY` (under `--runway-h` hours), `FINAL-POD` (a finished lane with no open successor still has a pod after
the grace) or, with `--reap`, `REAPED` (that pod terminated once the same custody as a stale lane's passes, else `REAP-BLOCKED`;
logged in `<root>/reaper.log`; `checkpoint ... --keep-pod WHY` keeps it), and `RESUMED-POD` once for such a pod started after the
FINAL (a reopened lane whose checkpoint has not arrived yet): never reaped.

The steward rules act without an agent; their memory is `<root>/steward-state.json` (the snapshot leaves it out).  With `--pods
--reap`, a STALE lane's pod idle (the probe) for `--reap-stale-min` is terminated once custody passes -- every `art:` id its newest
report cites is preserved on the remote (`preserved.check`; R2 credentials from ~/.config/verity/r2.env when unset) and every local
run launched on the pod has preserved.json (`research fetch --all`) or is in the store: `REAPED`, else `REAP-BLOCKED <lane> <pod>:
<first reason>` at most every 30 min, both in reaper.log.  `OVERDUE <lane>`, once: not done 30 min after its binding's `final` (the
first such HH:MMZ after the binding was written).  With `--pods`, `OVER-BUDGET <lane>`, once: its pods' $/h x time, accumulated per
pass, passed the binding's budget (its first `$<number>`).  `LAUNCHED <name> <run id> on <machine>` / `RUN-FAILED <name>: <why>`: the
`[[run]]` entries of `<root>/steward.toml`, each a recorded `research run --on MACHINE` launched daily at `at`, or weekly with `weekday` (`run_lines`).
`ROUTED <lane> <n>`: new KILLED lines of the laptop guardian's log
(`--guardian-log`) went to the lane that owns them (`--on <machine>`, or the killed process's cwd under a bound worktree) as one
handoff per lane per pass (a final lane's to `lanes/coordinator/`); `KILLED-UNOWNED <line>` when no lane does.  A rule that fails
prints `STEWARD-ERROR <rule>: <error>` and the watch goes on.

`relaunch` is contract §C in one command: save the lane's uncommitted work to `~/.research/relaunch/<lane>/uncommitted-<HHMMZ>*`
(`$RESEARCH_RELAUNCH`; on the worktree's machine, never in the notes), checkpoint it superseded, bind the successor (`<topic>-<N+1>` unless `--as`) to its branch, worktree and pods, print the
successor's launch message.  `gc-worktrees` lists (`--apply`: removes) clean worktrees under the coordinator's worktree root
whose lanes are all done and whose branch tip is on origin; branches are never deleted, dirty worktrees never removed.

The push rule: a lane pushes its branch after every commit (`push`), so its work survives its machine and the research coordinator
merges from origin.  `push` resolves the branch as `finish` does, refuses `main` / `master`, and never forces; a rejected push says
to integrate origin's commits first.  `branches` lists every local `lane/*` and bound branch against `origin/<branch>` as last
fetched (`--fetch` refreshes only origin's `lane/*` refs): `unpushed`, `ahead`, `behind`, `diverged`, `in sync`; BLOCKING marks
an open lane's branch whose work origin lacks, and makes the exit 1.  `checkpoint LANE final` shows a `pushed` line, advisory
unless `--require-pushed`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import remote
from .pythonpath import tree_pythonpath
from .pods import POD_SUFFIXES, strip_pod_suffix
from .runs import new_run_id
from .timefmt import both
from .pods.workproc import FIND_PRUNE, NOT_WORK_RE, WORK_RE

STATES = ("open", "blocked", "final", "superseded")
DONE = ("final", "superseded")
CHECKPOINT_RE = re.compile(r"CHECKPOINT\s+(?P<sha>[0-9a-fA-F]{4,40}|none)?\)?[:\s]*(?P<fin>\*{0,2}FINAL\*{0,2})?\s*\(?"
                           r"(?:(?P<hh>\d{1,2}):(?P<mm>\d{2})Z)?[);]?\s*(?:\[(?P<state>open|blocked|final|superseded)\])?\s*[-—:]*\s*(?P<text>.*)")
FINAL_RE = re.compile(r"^##\s+FINAL\b.*$", re.M)
DONE_STATUSES = ("done", "final", "closed", "complete", "completed")
POD_PREFIX = "vy-"
NO_POD = ("none", "-")
BINDING = "binding.json"
MAIL_GLOBS = ("*handoff*.md", "*asks*.md", "*reply*.md")
AGENT_HANDOFF = re.compile(r"-handoff-from-(?!merge-queue-)")      # the merge queue's and the steward's notices are machines'
SEEN = ".inbox-seen"
SUPERSEDED_BY_RE = re.compile(r"\[superseded\]\s+by\s+([A-Za-z0-9][\w.-]*)")
ART_RE = re.compile(r"\bart:([0-9a-f]{8,64})\b")
PROBE_MARK = "research-notes-probe"
POD_PROBE = rf"""# {PROBE_MARK}
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', *' 'NF >= 2 {{ if ($1 + 0 > u) u = $1 + 0; m += $2; n++ }} END {{ if (n) print "GPU", u + 0, m + 0; else print "GPU none" }}'
echo "GPUAPPS $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"
ps -eo etimes=,args= | grep -E '{WORK_RE}' \
  | grep -vE '{NOT_WORK_RE}' | awk '{{print "PROC", $1}}'
echo "SERVE $(ps -eo args= | grep -v {PROBE_MARK} | grep -c '[l]ive serve')"
f=$(timeout 10 find /workspace -xdev {FIND_PRUNE} -o -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1); echo "FILE ${{f:-none}} $(date +%s)"
"""


def default_root() -> Path:
    return Path(os.environ.get("RESEARCH_NOTES", Path.home() / ".research" / "notes"))


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _ts(t: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc)


_TIMES: dict[str, object] = {}


def file_time(root: Path, p: Path) -> float:
    """A notes file's time: its mtime, or with RESEARCH_LIVENESS=git its newest notes commit (steward.GitTimes), which every
    clone agrees on."""
    from . import steward  # noqa: PLC0415

    if not steward.git_liveness():
        return p.stat().st_mtime
    t = _TIMES.get(str(root))
    if t is None:
        t = _TIMES[str(root)] = steward.GitTimes(root)
    return t.time(p)


def refresh_times() -> None:
    _TIMES.clear()


def report_for(root: Path, lane: str) -> Path | None:
    """The lane's report (the module docstring's rule)."""
    reports = sorted((p for p in (root / "lanes" / lane).glob("????????T????Z-report-*.md") if "-report-from-" not in p.name),
                     key=lambda p: file_time(root, p))
    main = [p for p in reports if re.fullmatch(rf"\d{{8}}T\d{{4}}Z-report-{re.escape(lane)}\.md", p.name)]
    return (main or reports)[-1] if reports else None


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


EAGAIN_TRIES = 6


def _io(fn: Callable[[], object]):
    """fn(), retried with backoff on BlockingIOError: the Cursor Project store's mount answers EAGAIN on reads, stats and rewrites of a
    recently written file, and a write may land and still raise.  Callers pass writes of content computed once, so a retry is idempotent."""
    for i in range(EAGAIN_TRIES):
        try:
            return fn()
        except BlockingIOError:
            if i == EAGAIN_TRIES - 1:
                raise
            time.sleep(0.2 * 2**i)


_CHECK_BACK = re.compile(r"\bcheck[- ](?:back|after)\s+(\d{1,2}):(\d{2})Z", re.I)


def wake_hint(text: str, now: dt.datetime | None = None) -> str | None:
    """For a WAIT checkpoint: the one-shot timer that wakes the lane at its earliest check-back.  Only the lane's own timer is sure to fire
    (the coordinator's sweep is a backstop), and cron timers are not read in UTC, so the delay is computed here from UTC."""
    if not re.search(r"\bWAIT(?:ING)?\b", text):
        return None
    now = now or _utcnow()
    delays = []
    for m in _CHECK_BACK.finditer(text):
        try:
            at = now.replace(hour=int(m[1]), minute=int(m[2]), second=0, microsecond=0)
        except ValueError:
            continue
        delta = (at - now).total_seconds()
        delays.append((delta + 86400, at + dt.timedelta(days=1)) if delta < -12 * 3600 else (delta, at))
    if not delays:
        return None
    delta, at = min(delays)
    return (f"wake: before ending the turn, arm a one-shot timer (cursor-subscriptions subscribe_timer once=true "
            f"delaySeconds={max(60, int(delta))}) for check-back {both(at)}" + (" (already past: in 1 min)" if delta < 60 else "")
            + "; not a cron timer")


def checkpoint(root: Path, lane: str, state: str, text: str, commit: str | None = None, now: dt.datetime | None = None) -> Path:
    if state not in STATES:
        raise ValueError(f"state must be one of {', '.join(STATES)}, not {state!r}")
    now = now or _utcnow()
    path = _io(lambda: report_for(root, lane))
    if path is None:
        path = root / "lanes" / lane / f"{now:%Y%m%dT%H%MZ}-report-{lane}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        head = f"---\nlane: {lane}\nkind: report\ncreated: {now:%Y-%m-%dT%H:%MZ}\nstatus: {state}\n---\n\n"
        _io(lambda: path.write_text(head))
    front, body = _split_front_matter(_io(path.read_text))
    if front:
        front = re.sub(r"^status:.*$", f"status: {state}", front, count=1, flags=re.M) if re.search(r"^status:", front, re.M) \
            else front[:-4] + f"status: {state}\n---\n"
    line = f"CHECKPOINT {commit or _head_commit()} ({now:%H:%M}Z) [{state}] {' '.join(text.split())}\n"
    new = front + "\n" + line + body.lstrip("\n") if front else line + body
    _io(lambda: path.write_text(new))
    return path


def bind(root: Path, lane: str, branch: str | None = None, worktree: str | None = None, pod: str | None = None,
         succeeds: str | None = None, brief: str | None = None, final: str | None = None, budget: str | None = None,
         keep_pod: str | None = None) -> Path:
    """Record (merge into) the lane's binding; an empty string clears a field.  `pod` is a hint (pods go by name, `pod_owner`);
    brief / final (HH:MMZ) / budget go into a successor's launch message; keep_pod (checkpoint --keep-pod) stops the reaper."""
    if final and not re.fullmatch(r"\d{1,2}:\d{2}Z", final):
        raise ValueError(f"--final wants HH:MMZ (UTC), not {final!r}")
    path = root / "lanes" / lane / BINDING
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(_io(path.read_text)) if _io(path.exists) else {}
    for k, v in (("branch", branch), ("worktree", worktree), ("pod", pod), ("succeeds", succeeds), ("brief", brief),
                 ("final", final), ("budget", budget), ("keep_pod", keep_pod)):
        if v is not None:
            if v:
                cur[k] = str(Path(v).expanduser()) if k == "worktree" else v
            else:
                cur.pop(k, None)
    doc = json.dumps(cur, indent=1, sort_keys=True) + "\n"
    _io(lambda: path.write_text(doc))
    return path


def _binding_json(root: Path, lane: str) -> dict:
    try:
        return json.loads((root / "lanes" / lane / BINDING).read_text())
    except (OSError, ValueError):
        return {}


def binding(root: Path, lane: str, front: str = "") -> dict:
    """branch / worktree / pod for a lane: binding.json, then front matter; unset keys fall back at the call site."""
    out = {}
    for k in ("branch", "worktree", "pod"):
        if m := re.search(rf"^{k}:\s*`?([^\s`]+)", front, re.M):
            out[k] = str(Path(m.group(1)).expanduser()) if k == "worktree" else m.group(1)
    out.update({k: v for k, v in _binding_json(root, lane).items() if v})
    return out


# ---- inbox: handoffs a lane has not been shown yet ---------------------------------------------------------------------------------------

def successors(root: Path) -> dict[str, str]:
    """predecessor -> successor, from `[superseded] by <lane>` checkpoints and bindings' `succeeds`."""
    out: dict[str, str] = {}
    for d in (root / "lanes").glob("*/"):
        if pred := _binding_json(root, d.name).get("succeeds"):
            out[pred] = d.name
        rep = report_for(root, d.name)
        if rep is not None and d.name not in out and (m := SUPERSEDED_BY_RE.search(rep.read_text(errors="replace"))):
            out[d.name] = m.group(1)
    return out


def chain(root: Path, lane: str, succ: dict[str, str] | None = None) -> list[str]:
    """The lane and every lane it (transitively) took over, nearest first."""
    succ = successors(root) if succ is None else succ
    preds: dict[str, list[str]] = {}
    for p, s in succ.items():
        preds.setdefault(s, []).append(p)
    out, todo = [lane], list(preds.get(lane, []))
    while todo:
        p = todo.pop(0)
        if p not in out:
            out.append(p)
            todo.extend(preds.get(p, []))
    return out


def latest(lane: str, succ: dict[str, str], lanes: set[str] | None = None) -> str:
    """The latest attempt of the lane's succession (follow successors that are in `lanes`, when given)."""
    seen = {lane}
    while (nxt := succ.get(lane)) and (lanes is None or nxt in lanes) and nxt not in seen:
        seen.add(nxt)
        lane = nxt
    return lane


def attempt(root: Path, lane: str, succ: dict[str, str] | None = None) -> tuple[str, int]:
    """(topic, attempt number): the first attempt's name and how many attempts so far.  A first attempt the notes never linked
    but named `<topic>-<n>` (n < 100) is attempt n: verify-rs-4 is attempt 4 of verify-rs, wave-4090 attempt 1 of wave-4090."""
    ch = chain(root, lane, succ)
    m = re.fullmatch(r"(.+)-(\d{1,2})", ch[-1])
    return (m[1], int(m[2]) - 1 + len(ch)) if m else (ch[-1], len(ch))


def _seen(root: Path, lane: str) -> float | None:
    p = root / "lanes" / lane / SEEN
    try:
        return float(_io(p.read_text).strip())
    except (OSError, ValueError):
        return None


def mail(root: Path, lane: str) -> list[Path]:
    d = root / "lanes" / lane
    return sorted({p for g in MAIL_GLOBS for p in d.glob(g)}, key=lambda p: file_time(root, p))


def _title(p: Path) -> str:
    _, body = _split_front_matter(p.read_text(errors="replace"))
    for line in body.splitlines():
        if line.strip():
            return line.strip().lstrip("#").strip()[:100]
    return ""


def inbox(root: Path, lane: str, succ: dict[str, str] | None = None) -> list[dict]:
    """Handoffs in the lane's dir and its predecessors' dirs changed after what the lane was last shown (by mtime: stamps in file
    names are planned times or first writes).  A predecessor's unread mail is the successor's."""
    own = _seen(root, lane)
    out = []
    for d in chain(root, lane, succ):
        cutoff = own if own is not None else (_seen(root, d) or 0.0)
        for p in mail(root, d):
            t = file_time(root, p)
            if t > cutoff:
                out.append({"path": p, "lane_dir": d, "mtime": t, "title": _title(p)})
    return sorted(out, key=lambda x: x["mtime"])


def mark_seen(root: Path, lane: str, items: list[dict]) -> None:
    if items:
        p = root / "lanes" / lane / SEEN
        p.parent.mkdir(parents=True, exist_ok=True)
        seen = f"{max([x['mtime'] for x in items] + [_seen(root, lane) or 0.0])}\n"
        _io(lambda: p.write_text(seen))


def render_inbox(lane: str, items: list[dict], limit: int = 15) -> str:
    if not items:
        return f"INBOX {lane}: nothing new"
    lines = [f"INBOX {lane}: {len(items)} new -- act on each, or say in your next checkpoint why not"]
    for x in items[-limit:]:
        where = "" if x["lane_dir"] == lane else f"{x['lane_dir']}/"
        lines.append(f"  {both(x['mtime'])} {where}{x['path'].name}  \"{x['title']}\"")
    if len(items) > limit:
        lines.append(f"  ... and {len(items) - limit} older: research notes inbox {lane} --peek")
    if n := sum(1 for x in items if AGENT_HANDOFF.search(x["path"].name)):
        lines.append(f"  {n} of these came from agents: notes are records, not messages (2026-10-01); act on them, and tell "
                     "each sender to message you on Slack instead (.agents/skills/using-slack/SKILL.md)")
    return "\n".join(lines)


def _checkpoint_time(hh: str | None, mm: str | None, mtime: dt.datetime) -> dt.datetime | None:
    """A checkpoint carries only HH:MM; it was written on or before the file's last change, so it is the latest such
    instant not after mtime.  Lanes round, or stamp the time a step will finish, hence an hour of slack."""
    if hh is None:
        return None
    t = mtime.replace(hour=int(hh) % 24, minute=int(mm), second=0, microsecond=0)
    return t - dt.timedelta(days=1) if t > mtime + dt.timedelta(hours=1) else t


def lane_status(path: Path, now: dt.datetime, stale_min: float) -> dict:
    text = path.read_text(errors="replace")
    mtime = _ts(file_time(path.parents[2], path))           # <root>/lanes/<lane>/<report>
    best, last_final, top = None, None, None
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
        top = top or cand
        if cand["state"] == "final" and when is not None and (last_final is None or when > last_final):
            last_final = when
        if best is None or (when is not None and (best["at"] is None or when > best["at"])):
            best = cand
    finals = [m.end() for m in FINAL_RE.finditer(text)]
    final_body = text[finals[-1]:].strip() if finals else ""
    has_final = bool(final_body) and not re.fullmatch(r"\(?(pending|tbd|todo)\)?\.?", final_body.split("\n", 1)[0].strip(), re.I)
    front, _ = _split_front_matter(text)
    fm = re.search(r"^status:\s*(\S+)", front, re.M)
    fm_state = fm.group(1).lower() if fm else "open"
    cp_state = (best or {}).get("state")
    reopened = bool(best is top and best and best["at"] and cp_state not in DONE and last_final and best["at"] > last_final)
    if "superseded" in (fm_state, cp_state):
        state = "superseded"
    elif reopened:                       # a newer open checkpoint after a [final] one: its FINAL section is history
        state = cp_state
    else:
        state = "final" if has_final or fm_state in DONE_STATUSES else cp_state or fm_state
    age = (now - mtime).total_seconds() / 60
    done_at = (best["at"] if best and best["at"] and best["state"] in DONE else mtime) if state in DONE else None
    return {"lane": path.parent.name, "report": str(path), "state": state, "age_min": round(age, 1),
            "done_ts": done_at.timestamp() if done_at else None,
            "stale": state not in DONE and age > stale_min,
            "checkpoint_at": best["at"].strftime("%H:%MZ") if best and best["at"] else None,
            "checkpoint_min": round((now - best["at"]).total_seconds() / 60, 1) if best and best["at"] else None,
            "commit": best["commit"] if best else None, "text": best["text"] if best else None, "front": front}


# ---- other signs of life: git and the pod --------------------------------------------------------------------------------------------------

def _git(repo: Path, *args: str, timeout: float = 20) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def repo_default() -> Path | None:
    env = os.environ.get("RESEARCH_REPO")
    if env:
        return Path(env).expanduser()
    top = _git(Path.cwd(), "rev-parse", "--show-toplevel")
    return Path(top.strip()) if top and top.strip() else None


def worktrees(repo: Path) -> dict[str, str]:
    """branch name -> worktree path, from `git worktree list --porcelain`."""
    out, cur, res = _git(repo, "worktree", "list", "--porcelain") or "", None, {}
    for line in out.splitlines():
        if line.startswith("worktree "):
            cur = line[9:]
        elif line.startswith("branch refs/heads/") and cur:
            res[line[18:]] = cur
    return res


def git_signals(repo: Path, branch: str | None, worktree: str | None, remote: bool = False) -> dict:
    """commit_at: the branch's last commit (origin's tip as last fetched with `remote`); edit_at: newest mtime among the worktree's
    uncommitted files; dirty: their count."""
    res: dict = {"commit_at": None, "edit_at": None, "dirty": None}
    ref = f"refs/remotes/origin/{branch}" if remote else f"refs/heads/{branch}"
    if branch and (ct := (_git(repo, "log", "-1", "--format=%ct", ref) or "").strip()).isdigit():
        res["commit_at"] = _ts(int(ct))
    if worktree and Path(worktree).is_dir():
        out = _git(Path(worktree), "status", "--porcelain=v1", "-z", "--untracked-files=all")
        if out is not None:
            toks, paths, i = out.split("\0"), [], 0
            while i < len(toks):
                t = toks[i]
                if len(t) > 3:
                    paths.append(t[3:])
                    if t[0] in "RC":            # rename / copy: the source path follows as its own token
                        i += 1
                i += 1
            res["dirty"] = len(paths)
            mt = [p.stat().st_mtime for p in (Path(worktree) / q for q in paths) if p.exists()]
            res["edit_at"] = _ts(max(mt)) if mt else None
    return res


def parse_pod_probe(out: str) -> dict:
    """POD_PROBE's output -> busy (work processes), newest_proc_s, gpu_util (the busiest GPU), gpu_mem_mb, file_age_s (on the pod's
    clock), and when reported gpu_apps (GPU compute processes) and serve (live verifier servers)."""
    res: dict = {"busy": 0, "newest_proc_s": None, "gpu_util": None, "gpu_mem_mb": None, "file_age_s": None}
    for line in out.splitlines():
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        if parts[0] == "GPU" and len(parts) >= 3 and parts[1].isdigit():
            res["gpu_util"], res["gpu_mem_mb"] = int(parts[1]), int(parts[2]) if parts[2].isdigit() else None
        elif parts[0] in ("GPUAPPS", "SERVE") and len(parts) >= 2 and parts[1].isdigit():
            res["gpu_apps" if parts[0] == "GPUAPPS" else "serve"] = int(parts[1])
        elif parts[0] == "PROC" and len(parts) >= 2 and parts[1].isdigit():
            res["busy"] += 1
            s = int(parts[1])
            res["newest_proc_s"] = s if res["newest_proc_s"] is None else min(res["newest_proc_s"], s)
        elif parts[0] == "FILE" and len(parts) >= 3 and parts[1] != "none":
            try:
                res["file_age_s"] = max(0, int(parts[2]) - int(float(parts[1])))
            except ValueError:
                pass
    return res


def probe_pods(pods: list[dict], timeout: float = 30) -> dict[str, dict]:
    """pod id -> POD_PROBE over ssh (read-only); an unreachable pod says why instead."""
    from .pods import connect, runpod  # noqa: PLC0415

    def one(pod: dict) -> tuple[str, dict]:
        try:
            tgt = runpod.ssh_target(pod) or runpod.ssh_target(runpod.get(pod["id"]) or {})   # the listing sometimes omits ports
            if tgt is None:
                return pod["id"], {"error": f"no ssh port mapping ({pod.get('desiredStatus')})"}
            ssh = connect.ssh_for(*tgt)
            out = subprocess.run([*ssh[:-1], "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", ssh[-1], POD_PROBE],
                                 capture_output=True, text=True, timeout=timeout)
            if "FILE" not in out.stdout:                     # ssh itself failed: no answer is not an idle pod
                return pod["id"], {"error": f"ssh rc {out.returncode}: {out.stderr.strip()[-100:]}"}
            return pod["id"], parse_pod_probe(out.stdout)
        except Exception as e:  # noqa: BLE001
            return pod["id"], {"error": str(e)[:120]}

    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(ex.map(one, pods))


# ---- pods: the account's running pods and the lanes that own them --------------------------------------------------------------------------

def pod_topic(name: str, lanes: Iterable[str]) -> str | None:
    """The lane a `vy-<topic>[<letter>][-<role>]` pod name carries: the longest lane name that is the whole topic, starts it plus
    `-`, or is it minus one letter (+ digits) -- a sibling pod such as vy-wave-h100b."""
    if not name.startswith(POD_PREFIX):
        return None
    n = strip_pod_suffix(name)[len(POD_PREFIX):]
    hits = [t for t in lanes if n == t or n.startswith(t + "-") or re.fullmatch(re.escape(t) + r"[a-z]\d*(?:-.+)?", n)]
    return max(hits, key=len) if hits else None


def pod_owner(name: str, lanes: set[str], succ: dict[str, str] | None = None, bound: dict[str, str] | None = None) -> str | None:
    """The lane that owns a pod: the latest attempt of the lane its name carries (`pod_topic`), else of the lane whose binding
    names it (`bound`: pod name without the campaign suffix -> lane; for names that carry no lane)."""
    t = pod_topic(name, lanes) or (bound or {}).get(strip_pod_suffix(name))
    return latest(t, succ or {}, lanes) if t else None


@dataclass
class Fleet:
    """One look at the account: every RUNNING pod, the lane owning each (pod id -> lane), the lane-owned pods' probes."""
    pods: list[dict]
    owner: dict[str, str]
    probes: dict[str, dict]
    error: str | None = None


def running_pods() -> list[dict]:
    from .pods import connect  # noqa: PLC0415

    return [p for p in connect._list_pods() if (p.get("desiredStatus") or "").upper() == "RUNNING"]  # noqa: SLF001


def load_fleet(root: Path, probe: bool = True, pods: Callable[[], list[dict]] = running_pods,
               prober: Callable[[list[dict]], dict[str, dict]] = probe_pods) -> Fleet:
    try:
        listing = pods()
    except Exception as e:  # noqa: BLE001  (no API key / network: the table still prints, without pods)
        return Fleet([], {}, {}, error=f"pod listing failed: {str(e)[:120]}")
    lanes = {d.name for d in (root / "lanes").glob("*/")}
    bound = {}
    for lane in sorted(lanes):
        if (pod := _binding_json(root, lane).get("pod")) and pod not in NO_POD:
            bound[strip_pod_suffix(pod)] = lane
    succ = successors(root)
    owner = {p["id"]: o for p in listing if p.get("id") and (o := pod_owner(p.get("name") or "", lanes, succ, bound))}
    mine = [p for p in listing if p.get("id") in owner]
    return Fleet(listing, owner, prober(mine) if probe and mine else {})


def _started(p: dict) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(str(p.get("lastStartedAt") or "")[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def pod_view(p: dict, probe: dict | None, now: dt.datetime) -> dict:
    """A lane's pod as status shows it: name (no campaign suffix), id, $/h, its probe, and `idle` when the probe saw no work
    process, no live verifier, no GPU compute app and 0 % GPU; `idle_min` is since its newest /workspace file, capped at uptime."""
    from .pods.guard import parse_time  # noqa: PLC0415

    v = {"name": strip_pod_suffix(p.get("name") or ""), "id": p.get("id"), "usd_h": float(p.get("costPerHr") or 0.0),
         **(probe or {})}
    if ts := [t for t in (parse_time(p.get("createdAt")), parse_time(p.get("lastStartedAt"))) if t is not None]:
        v["started_ts"] = max(ts)
    if probe and not probe.get("error"):
        v["idle"] = not (probe.get("busy") or probe.get("serve") or probe.get("gpu_apps") or probe.get("gpu_util"))
        started = _started(p)
        ages = [a for a in (probe.get("file_age_s"), (now - started).total_seconds() if started else None) if a is not None]
        if v["idle"] and ages:
            v["idle_min"] = round(max(0.0, min(ages)) / 60, 1)
    return v


def enrich(rows: list[dict], root: Path, repo: Path | None, stale_min: float, now: dt.datetime,
           fleet: Fleet | None = None) -> list[dict]:
    """Add git/pod signals to the lanes that are not done; recompute life (freshest signal) and STALE from all of them."""
    wts = worktrees(repo) if repo else {}
    succ = successors(root)
    owned: dict[str, list[dict]] = {}
    for p in fleet.pods if fleet else ():
        if o := fleet.owner.get(p.get("id")):
            owned.setdefault(o, []).append(pod_view(p, fleet.probes.get(p["id"]), now))
    for r in rows:
        b = binding(root, r["lane"], r.pop("front", ""))
        if b.get("keep_pod"):
            r["keep_pod"] = b["keep_pod"]
        if fleet is not None:
            r["pods"] = owned.get(r["lane"], [])
            if fleet.error:
                r["pods_error"] = fleet.error
        if r["state"] in DONE:
            late = [p for p in mail(root, r["lane"]) if r.get("done_ts") and file_time(root, p) > r["done_ts"]]
            r["mail"] = [] if r["lane"] in succ or r["state"] == "superseded" else [p.name for p in late]   # a successor has it
            continue
        r["branch"] = b.get("branch") or f"lane/{r['lane']}"
        r["worktree"] = b.get("worktree") or wts.get(r["branch"])
        r["pod"] = b.get("pod")
        if repo:
            from . import steward  # noqa: PLC0415

            gitl = steward.git_liveness()                   # origin's tip; no worktree signal (steward.py, TIMES)
            g = git_signals(repo, r["branch"], None if gitl else r["worktree"], remote=gitl)
            r["dirty"] = g["dirty"]
            for k in ("commit", "edit"):
                r[f"{k}_min"] = round((now - g[f"{k}_at"]).total_seconds() / 60, 1) if g[f"{k}_at"] else None
        box = inbox(root, r["lane"], succ)
        r["inbox"] = len(box)
        r["inbox_oldest_min"] = round((now.timestamp() - box[0]["mtime"]) / 60, 1) if box else None
        ages = [0.0 if p.get("busy") else p["file_age_s"] / 60 for p in r.get("pods", []) if p.get("busy") or p.get("file_age_s") is not None]
        if ages:
            r["pod_min"] = round(min(ages), 1)
        sig = {k: r.get(f"{k}_min") for k in ("report", "checkpoint", "commit", "edit", "pod")}
        sig["report"] = r.get("age_min")
        have = {k: v for k, v in sig.items() if v is not None}
        if have:
            r["life_via"], r["life_min"] = min(have.items(), key=lambda kv: kv[1])
            r["stale"] = r["life_min"] > stale_min
    order = {"stale": 0, "blocked": 1, "open": 2, "final": 3, "superseded": 4}
    return sorted(rows, key=lambda r: (order["stale"] if r["stale"] else order.get(r["state"], 2),
                                       -(r["life_min"] if r.get("life_min") is not None else r["age_min"])))


def status(root: Path, stale_min: float = 30, since_hours: float | None = 24, now: dt.datetime | None = None,
           lanes: list[str] | None = None, exclude: list[str] | None = None, repo: Path | None = None,
           fleet: Fleet | None = None, every_attempt: bool = False) -> list[dict]:
    """One row per lane in the window (and per lane owning a running pod, however old), the latest attempt of each topic only
    unless `every_attempt`."""
    from fnmatch import fnmatch  # noqa: PLC0415

    now = now or _utcnow()
    refresh_times()
    owners = set(fleet.owner.values()) if fleet else set()
    rows = []
    for d in sorted((root / "lanes").glob("*/")):
        if (lanes and not any(fnmatch(d.name, g) for g in lanes)) or any(fnmatch(d.name, g) for g in exclude or ()):
            continue
        path = report_for(root, d.name)
        bound = d / BINDING
        newest = max([file_time(root, p) for p in (path, bound) if p is not None and p.exists()], default=None)
        if newest is None or (since_hours is not None and now.timestamp() - newest > since_hours * 3600 and d.name not in owners):
            continue
        if path is not None:
            rows.append(lane_status(path, now, stale_min))
        else:                                       # bound at launch, never reported: exactly the lane that dies unseen
            age = round((now.timestamp() - file_time(root, bound)) / 60, 1)
            rows.append({"lane": d.name, "report": None, "state": "open", "age_min": age, "stale": age > stale_min,
                         "checkpoint_at": None, "checkpoint_min": None, "commit": None, "text": "(no report yet)", "front": ""})
    rows = enrich(rows, root, repo, stale_min, now, fleet)
    succ = successors(root)
    shown = {r["lane"] for r in rows}
    for r in rows:
        r["attempt"] = attempt(root, r["lane"], succ)[1]
    return rows if every_attempt else [r for r in rows if latest(r["lane"], succ, shown) == r["lane"]]


# ---- finish checks (run by `checkpoint LANE final`) ----------------------------------------------------------------------------------------

def _custody(arts: list[str]) -> tuple[bool | None, str]:
    from .store import preserved  # noqa: PLC0415
    from .store.local import LocalStore  # noqa: PLC0415

    st = LocalStore(None)
    full, unknown = [], []
    for a in arts:
        hits = st.resolve_art(f"art:{a}")
        (full.append(hits[0]) if len(hits) == 1 else unknown.append(f"art:{a}"))
    rep = preserved.check(st, full, mode="recorded")
    bad = [x["id"] for x in rep["not_preserved"]] + unknown
    if not bad:
        return True, f"{len(arts)} cited artifacts preserved (recorded remote replicas, else verified on the remote now)"
    ids_ = " ".join(b[:13] for b in bad[:8]) + (" ..." if len(bad) > 8 else "")
    return False, f"{len(bad)}/{len(arts)} cited artifacts not preserved: {ids_} -> research data push <ids> --verify head"


def _pods(lane: str, bound: str | None, lanes: set[str], succ: dict[str, str] | None = None) -> tuple[bool | None, str]:
    try:
        listing = running_pods()
    except Exception as e:  # noqa: BLE001
        return None, f"unknown (pod listing failed: {str(e)[:80]})"
    mine = [p for p in listing if p.get("name")
            and (p["name"] in {bound, *(f"{bound}{s}" for s in POD_SUFFIXES)} or pod_owner(p["name"], lanes | {lane}, succ) == lane)]
    if not mine:
        return True, "no running pod"
    return False, "; ".join(f"{p['name']} RUNNING ({p.get('costPerHr', '?')}/h) -> research pods terminate {p['id']}" for p in mine) + \
        " | --keep-pod WHY"


def finish(root: Path, lane: str, repo: Path | None, keep_pod: str | None = None,
           custody: Callable[[list[str]], tuple[bool | None, str]] = _custody,
           pods: Callable[..., tuple[bool | None, str]] = _pods, require_pushed: bool = False) -> tuple[str, int]:
    """The end-of-lane checks.  They never block the FINAL line (already written); they say what is left, exit 3 if anything is.
    `pushed` (the branch tip is on origin, as last fetched) is advisory unless `require_pushed`."""
    rep = report_for(root, lane)
    text = rep.read_text(errors="replace") if rep else ""
    b = binding(root, lane, _split_front_matter(text)[0])
    keep_pod = keep_pod or b.get("keep_pod")
    checks: list[tuple[str, bool | None, str]] = []
    arts = sorted(set(ART_RE.findall(text)))
    if arts:
        try:
            checks.append(("custody", *custody(arts)))
        except Exception as e:  # noqa: BLE001
            checks.append(("custody", None, f"not checked ({type(e).__name__}: {str(e)[:80]})"))
    else:
        checks.append(("custody", None, "the report cites no art: ids (cite what you produced)"))
    if keep_pod:
        checks.append(("pod", True, f"kept: {keep_pod}"))
    elif b.get("pod") in NO_POD:
        checks.append(("pod", True, "none bound"))
    else:
        checks.append(("pod", *pods(lane, b.get("pod"), {d.name for d in (root / "lanes").glob("*/")}, successors(root))))
    branch = b.get("branch") or f"lane/{lane}"
    wt = b.get("worktree") or (worktrees(repo).get(branch) if repo else None)
    if wt and Path(wt).is_dir():
        dirty = [x for x in (_git(Path(wt), "status", "--porcelain") or "").splitlines() if x.strip()]
        head = (_git(Path(wt), "rev-parse", "HEAD") or "").strip()
        tip = (_git(Path(wt), "rev-parse", f"refs/heads/{branch}") or "").strip()
        ok = not dirty and head == tip
        checks.append(("worktree", ok, f"{'clean' if not dirty else f'{len(dirty)} uncommitted files'}, HEAD {head[:8]}"
                       f"{' == ' if head == tip else ' != '}{branch} {tip[:8]}"))
    else:
        checks.append(("worktree", None, f"none found for {branch}"))
    if repo or wt:
        s = sync_state(Path(wt) if wt and Path(wt).is_dir() else repo, branch)
        on = s["sync"] in ("in sync", "behind")
        why = f"{branch} {'on origin' if on else s['sync']}" + ("" if on else f" -> research notes push {lane}")
        checks.append(("pushed", True if on else (False if require_pushed else None), why))
    unnamed = [f"{d}/{p.name}" if d != lane else p.name for d in chain(root, lane) for p in mail(root, d) if p.name not in text]
    checks.append(("handoffs", not unnamed, "all named in the report" if not unnamed else
                   f"{len(unnamed)} never named in the report (answer or say why not): " + ", ".join(unnamed[:6])))
    bad = [c for c in checks if c[1] is False]
    head_ = f"FINISH {lane}: " + (f"{len(bad)} problem(s) (FINAL line written; exit 3)" if bad else "ok")
    return "\n".join([head_] + [f"  {k:<9} {msg}" for k, _, msg in checks]), 3 if bad else 0


# ---- history: the notes tree is its own git repo, committed by one writer ------------------------------------------------------------------

SNAPSHOT_MAX_BYTES = 1 << 20
#: git data of any repository never goes in notes (the notes repo is public; code moves only through verity branches)
GIT_DATA_SUFFIXES = (".bundle", ".pack", ".idx", ".rev")
GIT_DATA_MAGIC = (b"# v2 git bundle", b"# v3 git bundle", b"PACK")


def is_git_data(p: Path) -> bool:
    """A git bundle, pack, pack index or `.git` file, by name or by its first bytes."""
    if p.suffix in GIT_DATA_SUFFIXES or p.name == "packed-refs" or ".git" in p.parts:
        return True
    try:
        with open(p, "rb") as f:
            head = f.read(16)
    except OSError:
        return False
    return head.startswith(GIT_DATA_MAGIC)


def snapshot(root: Path, max_bytes: int = SNAPSHOT_MAX_BYTES, now: dt.datetime | None = None) -> dict | None:
    """Commit every change in the notes repo except files over `max_bytes` (dumps belong in the store; .gitignore drops the known
    kinds), git data (`refused`) and evidence or renders (`evidence`: `notes_sync.is_evidence`, the store holds them).  Only the
    coordinator's watcher calls this, so lanes never contend for the index lock.  None: not a repo, or clean."""
    from .notes_sync import is_evidence  # noqa: PLC0415

    if not (root / ".git").exists():
        return None
    st = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if st is None:
        return None
    entries, paths, big, refused, evidence = st.split("\0"), [], [], [], []
    i = 0
    while i < len(entries):
        e = entries[i]
        i += 1
        if not e:
            continue
        code, path = e[:2], e[3:]
        if code[0] in "RC":                                   # a rename is followed by its source path
            paths.append(entries[i])
            i += 1
        if path in (STATE_FILE, f".{STATE_FILE}.tmp"):        # the watcher's own memory, rewritten every pass: not history
            continue
        p = root / path
        if p.is_file() and is_git_data(p):
            refused.append(path)
            continue
        if p.is_file() and is_evidence(path):
            evidence.append(path)
            continue
        (big if p.is_file() and p.stat().st_size > max_bytes else paths).append(path)
    left = {"skipped": big, "refused": refused, "evidence": evidence}
    if not paths:
        return {"commit": None, "paths": 0, **left} if big or refused or evidence else None
    subprocess.run(["git", "--literal-pathspecs", "-C", str(root), "add", "-A", "--pathspec-from-file=-", "--pathspec-file-nul"],
                   input="\0".join(paths), text=True, check=True, capture_output=True, timeout=120)
    if evidence:                                          # a staged copy would be committed anyway
        _git(root, "--literal-pathspecs", "reset", "-q", "--", *evidence)
    now = now or _utcnow()
    msg = (f"snapshot {now:%Y-%m-%dT%H:%MZ}: {len(paths)} paths" + (f"; {len(big)} too big, not committed" if big else "")
           + (f"; {len(refused)} git data refused" if refused else "") + (f"; {len(evidence)} evidence left out" if evidence else ""))
    if subprocess.run(["git", "-C", str(root), "commit", "-q", "--no-verify", "-m", msg], capture_output=True, timeout=60).returncode:
        return {"commit": None, "paths": 0, **left}
    return {"commit": (_git(root, "rev-parse", "--short", "HEAD") or "").strip(), "paths": len(paths), **left}


# ---- rendering and the watch loop ----------------------------------------------------------------------------------------------------------

def _m(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}m"


def _pods_cell(r: dict) -> str:
    """`N $X/h` for the lane's running pods; `-` without --pods, `?` when the listing failed."""
    if "pods" not in r:
        return "-"
    if r.get("pods_error"):
        return "?"
    return f"{len(r['pods'])} ${sum(p['usd_h'] for p in r['pods']):.2f}/h" if r["pods"] else "0"


def _pod_state(p: dict) -> str:
    if p.get("error"):
        return f"err ({p['error'][:50]})"
    if "busy" not in p:
        return ""                                   # not probed (a finished lane's pod)
    s = f"busy{p['busy']}" if p["busy"] else f"serve{p['serve']}" if p.get("serve") else "idle"
    s += f" {p['gpu_util']}%" if p.get("gpu_util") is not None else ""
    s += f" apps{p['gpu_apps']}" if p.get("gpu_apps") else ""
    return s + (f" {_m(p['idle_min'])}" if p.get("idle_min") is not None else "")


def _label(r: dict) -> str:
    return f"{r['lane']} #{r['attempt']}" if r.get("attempt", 1) > 1 else r["lane"]


def render(rows: list[dict]) -> str:
    if not rows:
        return "no lane reports in the window"
    w = max(len(_label(r)) for r in rows)
    out = [f"{'lane':<{w}}  {'state':<8} {'life':>5} {'via':<10} {'ckpt':>5} {'commit':>6} {'edit':>5} {'dirty':>5} {'pods':<11} "
           f"{'inbox':<9} checkpoint"]
    for r in rows:
        flag = "STALE" if r["stale"] else ("supersd" if r["state"] == "superseded" else r["state"])
        if r["state"] in DONE:
            box = f"MAIL {len(r['mail'])}" if r.get("mail") else ""
            out.append(f"{_label(r):<{w}}  {flag:<8} {_m(r['age_min']):>5} {'report':<10} {'':>5} {'':>6} {'':>5} {'':>5} "
                       f"{_pods_cell(r):<11} {box:<9} {(r['text'] or '')[:70]}")
        else:
            dirty = "-" if r.get("dirty") is None else str(r["dirty"])
            box = f"{r['inbox']} ({_m(r['inbox_oldest_min'])})" if r.get("inbox") else "0"
            out.append(f"{_label(r):<{w}}  {flag:<8} {_m(r.get('life_min', r['age_min'])):>5} {r.get('life_via', 'report'):<10} "
                       f"{_m(r.get('checkpoint_min')):>5} {_m(r.get('commit_min')):>6} {_m(r.get('edit_min')):>5} {dirty:>5} "
                       f"{_pods_cell(r):<11} {box:<9} {(r['text'] or '')[:70]}")
        for p in r.get("pods", []):
            out.append(f"{'':<{w}}    {p['name']:<32} {p['id']:<15} ${p['usd_h']:.2f}/h  {_pod_state(p)}".rstrip())
    return "\n".join(out)


def transitions(prev: set[str], rows: list[dict], now: dt.datetime) -> tuple[list[str], set[str]]:
    """Lines for lanes that became STALE or came back since the last pass, and the new STALE set."""
    stale = {r["lane"] for r in rows if r["stale"]}
    by = {r["lane"]: r for r in rows}
    lines = []
    for lane in sorted(stale - prev):
        r = by[lane]
        states = "; ".join(filter(None, (_pod_state(p) for p in r.get("pods", []))))
        lines.append(f"{now:%H:%MZ} STALE {lane} life={_m(r.get('life_min', r['age_min']))} via={r.get('life_via', 'report')} "
                     f"dirty={r.get('dirty', '-')} pod={_pods_cell(r)}{f' [{states}]' if states else ''} branch={r.get('branch', '-')}")
    for lane in sorted(prev - stale):
        r = by.get(lane)
        why = "gone from the window" if r is None else (r["state"] if r["state"] in DONE else f"via={r.get('life_via')}")
        lines.append(f"{now:%H:%MZ} ALIVE {lane} ({why})")
    return lines, stale


def _snapshot_line(snap: dict) -> str:
    skipped = snap["skipped"]
    tail = f"; skipped {len(skipped)} over 1 MB (put them in the store): {', '.join(skipped[:3])}{' ...' if len(skipped) > 3 else ''}" \
        if skipped else ""
    refused = snap.get("refused") or []
    if refused:
        tail += f"; REFUSED {len(refused)} git data (code moves only through verity branches): {', '.join(refused[:3])}"
    evidence = snap.get("evidence") or []
    if evidence:
        tail += f"; left out {len(evidence)} evidence/render files (the evidence store holds them): {', '.join(evidence[:3])}"
    return f"notes snapshot {snap['commit'] or '(none)'}: {snap['paths']} paths{tail}"


def mail_lines(prev: dict[str, set[str]], rows: list[dict], now: dt.datetime) -> tuple[list[str], dict[str, set[str]]]:
    """A line per finished lane that received handoffs nobody will read (no successor), once per new file."""
    cur = {r["lane"]: set(r["mail"]) for r in rows if r.get("mail")}
    lines = [f"{now:%H:%MZ} MAIL {lane} got {len(new)} handoff(s) after it finished: {', '.join(sorted(new))}"
             for lane, names in sorted(cur.items()) if (new := names - prev.get(lane, set()))]
    return lines, cur


def idle_lines(since: dict[str, float], alerted: set[str], rows: list[dict], now: dt.datetime, idle_min: float) -> list[str]:
    """IDLE-POD once per idle spell of an open lane's pod idle for >= idle_min.  `since` (pod id -> idle since, epoch s) and
    `alerted` are the watch's memory; a pod first seen idle starts at its own estimate (newest /workspace file, uptime)."""
    lines, seen, t = [], set(), now.timestamp()
    for r in rows:
        for p in r.get("pods", []) if r["state"] not in DONE else ():
            if not p.get("idle"):
                continue
            seen.add(p["id"])
            mins = (t - since.setdefault(p["id"], t - 60 * (p.get("idle_min") or 0.0))) / 60
            if mins >= idle_min and p["id"] not in alerted:
                alerted.add(p["id"])
                gpu = f"GPU {p['gpu_util']}%" if p.get("gpu_util") is not None else "no GPU"
                lines.append(f"{now:%H:%MZ} IDLE-POD {r['lane']} {p['name']} {mins:.0f}m (${p['usd_h']:.2f}/h, {gpu}, no work process)")
    for k in set(since) - seen:
        del since[k]
        alerted.discard(k)
    return lines


def account() -> dict:
    """The account's balance and spend rate (RunPod GraphQL `myself`)."""
    from .pods import runpod  # noqa: PLC0415

    me = runpod._graphql("{ myself { clientBalance currentSpendPerHr } }")["data"]["myself"]  # noqa: SLF001
    return {"balance": float(me["clientBalance"]), "usd_h": float(me.get("currentSpendPerHr") or 0.0)}


def account_line(fleet: Fleet, acct: dict | None, now: dt.datetime) -> tuple[str, float | None]:
    """The ACCOUNT line (balance, $/h, runway; lane-owned pods' $/h vs every other pod's) and the runway in hours."""
    def usd(ps: list[dict]) -> float:
        return sum(float(p.get("costPerHr") or 0.0) for p in ps)

    ours = [p for p in fleet.pods if p.get("id") in fleet.owner]
    other = [p for p in fleet.pods if p.get("id") not in fleet.owner]
    split = fleet.error or f"lane pods ${usd(ours):.2f}/h ({len(ours)}), other pods ${usd(other):.2f}/h ({len(other)})"
    if not acct:
        return f"{now:%H:%MZ} ACCOUNT balance unavailable; {split}", None
    rate = acct["usd_h"] or usd(fleet.pods)
    runway = acct["balance"] / rate if rate > 0 else None
    return (f"{now:%H:%MZ} ACCOUNT ${acct['balance']:.2f} at ${rate:.2f}/h, "
            f"{f'runway {runway:.1f}h' if runway is not None else 'no spend'}; {split}"), runway


def runway_alert(runway: float | None, under_h: float, last: float | None, now: dt.datetime) -> tuple[list[str], float | None]:
    """RUNWAY when the runway drops under `under_h`, then again each time it is half an hour shorter than at the last alert."""
    if runway is None or runway >= under_h:
        return [], None
    if last is not None and runway > last - 0.5:
        return [], last
    return [f"{now:%H:%MZ} RUNWAY {runway:.1f}h (under {under_h:g}h)"], runway


def terminate_pod(pod_id: str) -> None:
    from .pods import runpod  # noqa: PLC0415

    runpod._request("DELETE", f"/pods/{pod_id}")  # noqa: SLF001


def pod_gone(why: str) -> bool:
    """A custody check that failed because RunPod no longer knows the pod (HTTP 404): nothing to reap, nothing to report."""
    return "HTTP 404" in why


def reap_lines(root: Path, rows: list[dict], now: dt.datetime, grace_min: float, noted: set[str], reap: bool = False,
               terminate: Callable[[str], None] = terminate_pod, custody: Callable[[str, dict], tuple[bool, str]] | None = None,
               state: dict | None = None) -> list[str]:
    """Pods of a FINAL lane (rows are latest attempts: no open successor) done for >= grace_min, not kept (--keep-pod): FINAL-POD
    once per pod, or with `reap` terminate them once custody passes (`reap_custody`): REAPED, else `REAP-BLOCKED <lane> <pod>: <why>`
    (re-checked after 30 min); both go to <root>/reaper.log.  A pod started after the FINAL (or with no start time in the listing)
    means the lane was reopened: `RESUMED-POD` once, never terminated -- its reopening checkpoint may not have reached these notes
    yet.  Pods no lane owns are never in rows."""
    custody = custody or (lambda lane, pod: reap_custody(root, lane, pod["id"]))
    tried = (state if state is not None else {}).setdefault("reap_final", {})
    t, lines = now.timestamp(), []
    for r in rows:
        if r["state"] != "final" or r.get("keep_pod") or not r.get("done_ts") or t - r["done_ts"] < grace_min * 60:
            continue
        for p in r.get("pods", []):
            started = p.get("started_ts")
            if started is None or started > r["done_ts"]:
                if f"resumed:{p['id']}" not in noted:
                    at = f"{dt.datetime.fromtimestamp(started, dt.timezone.utc):%H:%MZ}" if started is not None else "unknown"
                    lines.append(f"{now:%H:%MZ} RESUMED-POD {r['lane']} {p['name']} {p['id']} (created {at}, after FINAL "
                                 f"{dt.datetime.fromtimestamp(r['done_ts'], dt.timezone.utc):%H:%MZ}): not reaped")
                    noted.add(f"resumed:{p['id']}")
                continue
            what = f"{r['lane']} {p['name']} {p['id']} ${p['usd_h']:.2f}/h (final {(t - r['done_ts']) / 60:.0f}m ago)"
            if not reap:
                line = f"{now:%H:%MZ} FINAL-POD {what}: --reap terminates it, checkpoint --keep-pod WHY keeps it"
            else:
                if t - tried.get(p["id"], 0.0) < REAP_RETRY_S:
                    continue
                tried[p["id"]] = t
                try:
                    ok, why = custody(r["lane"], p)
                except Exception as e:  # noqa: BLE001
                    ok, why = False, f"custody not checked ({type(e).__name__}: {str(e)[:80]})"
                if not ok and pod_gone(why):
                    noted.add(p["id"])
                    continue
                if not ok:
                    line = f"{now:%H:%MZ} REAP-BLOCKED {r['lane']} {p['name']}: {why}"
                else:
                    try:
                        terminate(p["id"])
                        line = f"{now:%H:%MZ} REAPED {what}; custody: {why}"
                    except Exception as e:  # noqa: BLE001
                        line = f"{now:%H:%MZ} REAP-FAILED {what}: {str(e)[:100]}"
                with (root / "reaper.log").open("a") as f:
                    f.write(line + "\n")
            if reap or p["id"] not in noted:
                lines.append(line)
            noted.add(p["id"])
    for k in [k for k, v in tried.items() if t - v > 86400]:
        del tried[k]
    return lines


# ---- the steward: duties the watcher carries out itself (stale-lane pods, deadlines, budgets, scheduled runs, guardian kills) ---------------------

STATE_FILE = "steward-state.json"
STEWARD_TOML = "steward.toml"
R2_ENV = Path.home() / ".config" / "verity" / "r2.env"
GUARDIAN_LOG = Path.home() / ".veritor" / "mem_guardian.log"
REAP_RETRY_S = 30 * 60
OVERDUE_GRACE_MIN = 30
RUN_LAUNCH_TIMEOUT_S = 600
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
NEVER_RAN = ("refused", "resolving", "shipping")        # remote.json phases of a launch that never reached the machine's runner
BUDGET_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
HHMMZ_RE = re.compile(r"(\d{1,2}):(\d{2})Z")
KILLED_RE = re.compile(r"^(?P<ts>\S+) KILLED pid=(?P<pid>\d+) .*?reason=(?P<reason>.*?)(?: cwd=(?P<cwd>.*?))? cmd=(?P<cmd>.*)$")
ON_RE = re.compile(r"--on[= ](\S+)")


def load_state(root: Path) -> dict:
    try:
        doc = json.loads((root / STATE_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def save_state(root: Path, state: dict) -> None:
    tmp = root / f".{STATE_FILE}.tmp"
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, root / STATE_FILE)


def load_env_file(path: Path, environ: dict | None = None) -> list[str]:
    """KEY=VALUE lines (`export`, quotes, # comments allowed) into the environment where the key is unset or empty: the watcher runs
    under launchd, without the R2 credentials.  -> the keys it set."""
    env = os.environ if environ is None else environ
    try:
        text = path.read_text()
    except OSError:
        return []
    done = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = (x.strip() for x in line.removeprefix("export ").split("=", 1))
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        if re.fullmatch(r"[A-Za-z_]\w*", k) and not env.get(k):
            env[k] = v
            done.append(k)
    return done


def art_custody(arts: list[str], max_readback_mb: float = 64) -> tuple[bool, str]:
    """Custody (a): every cited art is PRESERVED on the remote -- `preserved.check`, the predicate of `research data preserved`.  A
    verified remote replica the catalog already records counts (`recorded`); else the `head` check runs, but only when the blobs it
    would read back (those not held locally) come to at most max_readback_mb: the laptop's disk is nearly full."""
    from .store import preserved  # noqa: PLC0415
    from .store.local import LocalStore  # noqa: PLC0415

    st = LocalStore(None)
    for a in arts:
        hits = st.resolve_art(f"art:{a}")
        if len(hits) != 1:
            return False, f"art:{a} is {'ambiguous' if hits else 'unknown'} in the local catalog and on the remote"
        if preserved.check(st, [hits[0]], mode="recorded", head_fallback=False)["preserved"]:
            continue
        try:
            m = st.get_manifest(hits[0])
            mb = sum(n for h, n in m.blobs() if not st.object_path(h).is_file()) / 2**20
        except Exception as e:  # noqa: BLE001
            return False, f"art:{a}: {type(e).__name__}: {str(e)[:80]}"
        if mb > max_readback_mb:
            return False, f"art:{a}: no verified remote replica recorded, {mb:.0f} MB to read back -> research data preserved art:{a}"
        rep = preserved.check(st, [hits[0]], mode="head")
        if not rep["preserved"]:
            why = [e for x in rep["not_preserved"] for e in x.get("errors") or []] or [rep.get("reason") or "?"]
            return False, f"art:{a} not preserved: {why[0][:100]}"
    return True, f"{len(arts)} cited art(s) preserved"


def _store_attempt() -> Callable[[str], bool]:
    st = None

    def has(run_id: str) -> bool:
        nonlocal st
        from .store.local import LocalStore  # noqa: PLC0415

        try:
            st = st or LocalStore(None)
            return st.get_attempt(run_id) is not None
        except Exception:  # noqa: BLE001
            return False
    return has


def run_custody(pod_id: str, runs_dir: Path | None = None, registered: Callable[[str], bool] | None = None) -> tuple[bool, str]:
    """Custody (b): every local run launched on the pod (remote.json `pod_id`) that reached its runner has preserved.json (`research
    fetch --all` verified every file) or is registered in the store (an attempt)."""
    from . import steward  # noqa: PLC0415
    from .runs import runs_root  # noqa: PLC0415

    if steward.git_liveness():
        return False, ("run custody not checked: with --liveness git, runs launched from other machines are not in this host's run "
                       "records; watch --custody-r2 takes it from the store's remote and the pod guard")
    registered = registered or _store_attempt()
    n = 0
    for rj in sorted((runs_dir or runs_root()).glob("*/remote.json")):
        try:
            rec = json.loads(rj.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict) or rec.get("pod_id") != pod_id or rec.get("phase") in NEVER_RAN or (rj.parent / "refused.json").exists():
            continue
        n += 1
        if not (rj.parent / "preserved.json").exists() and not registered(rj.parent.name):
            return False, f"run {rj.parent.name} on it is neither fetched nor in the store -> research fetch --all {rj.parent.name}"
    return True, f"{n} run(s) on it fetched or registered"


def run_custody_r2(pod_id: str) -> tuple[bool, str]:
    """Custody (b) from any machine (`watch --custody-r2`): `pods.drain.pod_custody` -- the catalog refreshed from the store's remote,
    every attempt tied to the pod preserved there, and the pod guard's verdict that every run dir on it has custody (.fetched or
    .custody).  Needs neither ~/.research/runs nor the machine that launched the runs."""
    from .pods.drain import pod_custody  # noqa: PLC0415
    from .store.local import LocalStore  # noqa: PLC0415

    ok, why, _ = pod_custody(LocalStore(None), pod_id, refresh=True, probe=False)
    return ok, why


def reap_custody(root: Path, lane: str, pod_id: str, arts: Callable[[list[str]], tuple[bool, str]] = art_custody,
                 runs: Callable[[str], tuple[bool, str]] = run_custody) -> tuple[bool, str]:
    """Both custody legs for terminating a stale lane's pod: (a) the art: ids the lane's newest report cites, (b) the runs on the pod."""
    rep = report_for(root, lane)
    cited = sorted(set(ART_RE.findall(rep.read_text(errors="replace")))) if rep else []
    ok, why = arts(cited) if cited else (True, "no art: cited")
    if not ok:
        return ok, why
    ok2, why2 = runs(pod_id)
    return ok2, f"{why}; {why2}" if ok2 else why2


def stale_reap_lines(root: Path, rows: list[dict], now: dt.datetime, idle_min: float, state: dict, reap: bool = False,
                     custody: Callable[[str, dict], tuple[bool, str]] | None = None,
                     terminate: Callable[[str], None] = terminate_pod) -> list[str]:
    """With `reap`: a pod of a STALE lane (not done, no --keep-pod) that the probe finds idle (no work process, no verifier, 0 % GPU)
    for >= idle_min is terminated once custody passes (`reap_custody`): REAPED, else `REAP-BLOCKED <lane> <pod>: <first reason>`.
    A pod acted on is left alone for 30 min (custody re-checked then); every line goes to <root>/reaper.log."""
    if not reap:
        return []
    custody = custody or (lambda lane, pod: reap_custody(root, lane, pod["id"]))
    tried = state.setdefault("reap_stale", {})
    t, lines = now.timestamp(), []
    for r in rows:
        if not r.get("stale") or r["state"] in DONE or r.get("keep_pod"):
            continue
        for p in r.get("pods", []):
            if not p.get("idle") or (p.get("idle_min") or 0.0) < idle_min or t - tried.get(p["id"], 0.0) < REAP_RETRY_S:
                continue
            tried[p["id"]] = t
            try:
                ok, why = custody(r["lane"], p)
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"custody not checked ({type(e).__name__}: {str(e)[:80]})"
            if not ok and pod_gone(why):
                continue
            what = f"{r['lane']} {p['name']} {p['id']} ${p['usd_h']:.2f}/h (STALE {_m(r.get('life_min', r['age_min']))}, idle {_m(p['idle_min'])})"
            if not ok:
                line = f"{now:%H:%MZ} REAP-BLOCKED {r['lane']} {p['name']}: {why}"
            else:
                try:
                    terminate(p["id"])
                    line = f"{now:%H:%MZ} REAPED {what}; custody: {why}"
                except Exception as e:  # noqa: BLE001
                    line = f"{now:%H:%MZ} REAP-FAILED {what}: {str(e)[:100]}"
            with (root / "reaper.log").open("a") as f:
                f.write(line + "\n")
            lines.append(line)
    for k in [k for k, v in tried.items() if t - v > 86400]:
        del tried[k]
    return lines


def final_deadline(root: Path, lane: str, final: str) -> dt.datetime | None:
    """A binding's `final` (HH:MMZ) as an instant: the first such time after the binding was last written (`bind --final` at launch)."""
    m = HHMMZ_RE.fullmatch(final or "")
    p = root / "lanes" / lane / BINDING
    if not m or not p.exists():
        return None
    anchor = _ts(file_time(root, p))
    t = anchor.replace(hour=int(m[1]) % 24, minute=int(m[2]), second=0, microsecond=0)
    return t if t > anchor else t + dt.timedelta(days=1)


def overdue_lines(root: Path, rows: list[dict], now: dt.datetime, state: dict, grace_min: float = OVERDUE_GRACE_MIN) -> list[str]:
    """`OVERDUE <lane>` once per deadline: not done grace_min after its binding's `final`.  The deadline is fixed in the state when
    first seen, so a later bind that keeps `final` does not move it; a new `final` sets a new one."""
    seen = state.setdefault("overdue", {})
    lines = []
    for r in rows:
        final = _binding_json(root, r["lane"]).get("final")
        if not final:
            continue
        cur = seen.get(r["lane"])
        if not cur or cur.get("final") != final:
            if (at := final_deadline(root, r["lane"], final)) is None:
                continue
            cur = seen[r["lane"]] = {"final": final, "at": at.timestamp()}
        if r["state"] in DONE or cur.get("alerted") or now.timestamp() < cur["at"] + grace_min * 60:
            continue
        cur["alerted"] = True
        lines.append(f"{now:%H:%MZ} OVERDUE {r['lane']} FINAL {final} passed {(now.timestamp() - cur['at']) / 60:.0f}m ago: "
                     f"{'STALE' if r.get('stale') else r['state']}, life {_m(r.get('life_min', r['age_min']))} via {r.get('life_via', 'report')}")
    return lines


def budget_usd(text: str | None) -> float | None:
    """A binding's budget in dollars: its first `$<number>` ('$22: 1x A100 + ...' -> 22.0)."""
    m = BUDGET_RE.search(text or "")
    return float(m[1]) if m else None


def budget_lines(root: Path, fleet: Fleet, now: dt.datetime, state: dict) -> list[str]:
    """Each lane's pod spend, accumulated in the state each pass: every owned pod's $/h x the time since it was last counted (since
    it started, the first time it is seen or after a restart).  `OVER-BUDGET <lane>` once when it passes the binding's budget."""
    if fleet.error:                                     # an empty listing is not "every pod stopped"
        return []
    counted, spend, alerted = (state.setdefault(k, {}) for k in ("pod_counted", "spend", "over_budget"))
    t, rate = now.timestamp(), {}
    for p in fleet.pods:
        if not (lane := fleet.owner.get(p.get("id"))):
            continue
        started = _started(p)
        since = max([x for x in (counted.get(p["id"]), started.timestamp() if started else None) if x is not None], default=t)
        usd_h = float(p.get("costPerHr") or 0.0)
        spend[lane] = round(spend.get(lane, 0.0) + usd_h * max(0.0, t - since) / 3600, 4)
        rate[lane] = rate.get(lane, 0.0) + usd_h
        counted[p["id"]] = t
    for k in [k for k, v in counted.items() if t - v > 2 * 86400]:
        del counted[k]
    lines = []
    for lane, usd in sorted(spend.items()):
        text = _binding_json(root, lane).get("budget")
        cap = budget_usd(text)
        if cap is not None and usd > cap and alerted.get(lane) != cap:
            alerted[lane] = cap
            lines.append(f"{now:%H:%MZ} OVER-BUDGET {lane} ${usd:.2f} on pods > ${cap:g} (budget: {text[:60]}); "
                         f"now ${rate.get(lane, 0.0):.2f}/h")
    return lines


def _run_command(cmd: list[str], env: dict, cwd: Path, timeout: float) -> tuple[int | None, str, str]:
    try:
        r = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "", f"timeout after {timeout:.0f}s"
    except OSError as e:
        return None, "", f"{type(e).__name__}: {e}"
    return r.returncode, r.stdout, r.stderr


def steward_entries(path: Path, table: str, now: dt.datetime, state: dict, fail: str) -> tuple[list, list[str]]:
    """The `[[table]]` entries of steward.toml, re-read every pass.  A file that does not parse is one `<fail> steward.toml: <error>`
    line, said again only when the error changes."""
    if not path.exists():
        return [], []
    try:
        entries = tomllib.loads(path.read_text()).get(table) or []
        state.pop(f"{table}_config_error", None)
    except (OSError, tomllib.TOMLDecodeError) as e:
        if state.get(f"{table}_config_error") == str(e):
            return [], []
        state[f"{table}_config_error"] = str(e)
        return [], [f"{now:%H:%MZ} {fail} {path.name}: {e}"]
    return (entries if isinstance(entries, list) else []), []


def due_period(at: re.Match, now: dt.datetime, weekday: str | None = None) -> str | None:
    """The period a scheduled entry is due in: the UTC day (`2026-09-28`) once `now` is at or after `at` (an HH:MMZ match), or with
    `weekday` the ISO week (`2026-W40`) once `now` is at or after `at` on that weekday; None before."""
    start = now.replace(hour=int(at[1]) % 24, minute=int(at[2]), second=0, microsecond=0)
    if weekday is None:
        return f"{now:%Y-%m-%d}" if now >= start else None
    start -= dt.timedelta(days=now.weekday() - WEEKDAYS.index(weekday))
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}" if now >= start else None


def run_lines(root: Path, now: dt.datetime, state: dict, config: Path | None = None,
              run: Callable[[list[str], dict, Path, float], tuple[int | None, str, str]] = _run_command,
              timeout_s: float = RUN_LAUNCH_TIMEOUT_S, new_id: Callable[[], str] = new_run_id) -> list[str]:
    """Scheduled recorded runs, from `[[run]]` entries of <root>/steward.toml (re-read every pass): `name`, `at` (HH:MMZ), `weekday`
    (`mon` ... `sun`: weekly; daily without), `source` (the tree it is launched from) and `args`, the arguments of `research run`, which
    put it on a machine (`["--on", MACHINE, ..., "--", CMD...]`).  Once per UTC day (per ISO week with `weekday`) at or after `at`,
    `python -m research run --id <run id> <args>` runs in `source` with the tree's workspace on PYTHONPATH.
    It returns once the machine has started the workload detached, so the pass waits for the launch only (at most 10 min),
    never for the run: `LAUNCHED <name> <run id> on <machine>`, else `RUN-FAILED <name> <run id>: <stderr tail>`, and the next
    launch is the next period's.  The run's Attempt records the rest (`research inspect <run id>`).  An invalid entry is one
    `RUN-FAILED <name>: <why>` line."""
    entries, lines = steward_entries(config or root / STEWARD_TOML, "run", now, state, "RUN-FAILED")
    if not entries:
        return lines
    done, invalid, seen = state.setdefault("run", {}), state.setdefault("run_invalid", {}), set()
    for e in entries:
        name, args, weekday = str(e.get("name", "")), e.get("args"), e.get("weekday")
        m = HHMMZ_RE.fullmatch(str(e.get("at", "")))
        ok = name and m and e.get("source") and isinstance(args, list) and all(isinstance(a, str) for a in args)
        opts = args[:args.index("--")] if ok and "--" in args else []
        machine = remote.flag_value(opts, "--on")
        bad = ('an entry needs name, at = "HH:MMZ", source and args (a list of strings)' if not ok else
               f"weekday = {weekday!r}: one of {', '.join(WEEKDAYS)}" if weekday is not None and weekday not in WEEKDAYS else
               'args put the run on a machine: ["--on", MACHINE, ..., "--", CMD...]' if not machine else
               "args set --id: the steward names the run" if remote.flag_value(opts, "--id") is not None else
               f"name {name!r} is used by another entry" if name in seen else None)
        seen.add(name)
        if bad:
            if invalid.get(name) != bad:
                invalid[name] = bad
                lines.append(f"{now:%H:%MZ} RUN-FAILED {name or '?'}: {bad}")
            continue
        if (period := due_period(m, now, weekday)) is None or done.get(name, {}).get("period") == period:
            continue
        run_id = new_id()
        done[name] = {"period": period, "run": run_id}
        invalid.pop(name, None)
        src = Path(str(e["source"])).expanduser()
        cmd = [sys.executable, "-m", "research", "run", "--id", run_id, *args]
        rc, stdout, stderr = run(cmd, {**os.environ, "PYTHONPATH": tree_pythonpath(src)}, src, timeout_s)
        done[name]["rc"] = rc
        lines.append(f"{now:%H:%MZ} LAUNCHED {name} {run_id} on {machine}" if rc == 0 else
                     f"{now:%H:%MZ} RUN-FAILED {name} {run_id}: {' '.join((stderr or stdout).split())[-200:] or f'rc {rc}'}")
    return lines


def sync_lines(root: Path, srep: dict, now: dt.datetime) -> list[str]:
    """The watcher's line for its own `notes sync`: the report when it committed, and a failure as `SYNC-FAILED ahead A behind B:
    <report>` every pass it lasts, so a clone that stops syncing (83 commits behind on 2026-09-26) is never silent.  A lane note left
    out for a stamp ahead of the clock or broken front matter is one `SYNC-REFUSED <path>: ...` line every pass it lasts; the other
    notes synced.  A `SYNC-WARN <path>: ...` line comes once, with the commit that took the note."""
    from . import notes_sync  # noqa: PLC0415

    refused = [f"{now:%H:%MZ} {x}" for x in notes_sync.refused_lines(srep)]
    if srep["rc"] not in (0, notes_sync.FUTURE_RC):
        ab = notes_sync.lag(root)
        where = f"ahead {ab[0]} behind {ab[1]}" if ab else "lag unknown"
        return [f"{now:%H:%MZ} SYNC-FAILED {where}: {notes_sync.render_sync(srep)}", *refused]
    return ([f"{now:%H:%MZ} {notes_sync.render_sync(srep)}"] if srep.get("commit") else []) + refused


def steward_rule(name: str, fn: Callable[..., list[str]], *args, **kw) -> list[str]:
    """One rule's lines; an exception becomes a `STEWARD-ERROR <rule>` line instead of stopping the watcher (a launchd service)."""
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        return [f"{_utcnow():%H:%MZ} STEWARD-ERROR {name}: {type(e).__name__}: {str(e)[:150]}"]


def guardian_kills(path: Path, state: dict, max_bytes: int = 8 << 20) -> list[str]:
    """The guardian log's KILLED lines since the offset saved in the state.  The first look only records where the log ends; a
    rotated or truncated log starts over from its beginning."""
    try:
        st = path.stat()
    except OSError:
        return []
    g = state.get("guardian") or {}
    if g.get("path") != str(path):
        state["guardian"] = {"path": str(path), "offset": st.st_size, "ino": st.st_ino}
        return []
    off = g.get("offset", 0) if g.get("ino") == st.st_ino and st.st_size >= g.get("offset", 0) else 0
    off = max(off, st.st_size - max_bytes)
    with path.open("rb") as f:
        f.seek(off)
        data = f.read(st.st_size - off)
    end = data.rfind(b"\n") + 1
    state["guardian"] = {"path": str(path), "offset": off + end, "ino": st.st_ino}
    return [x for x in data[:end].decode(errors="replace").splitlines() if " KILLED pid=" in x]


def kill_owner(line: str, lanes: set[str], succ: dict[str, str], machines: dict[str, dict], pod_lane: dict[str, str],
               wts: dict[str, str], bound: dict[str, str] | None = None) -> str | None:
    """The lane a guardian KILLED line belongs to: `--on <machine>` in its cmd (machines.toml -> pod id -> the lane owning that pod;
    else the machine name as a pod name, `pod_owner` with the bindings' pod hints), else its cwd under a bound worktree (the
    deepest; the latest attempt)."""
    m = KILLED_RE.match(line)
    if not m:
        return None
    if on := ON_RE.search(m["cmd"]):
        pid_ = (machines.get(on[1]) or {}).get("pod_id")
        if owner := pod_lane.get(pid_) or pod_owner(on[1], lanes, succ, bound):
            return owner
    cwd = (m["cwd"] or "").strip()
    if not cwd or cwd == "?":
        return None
    c = Path(os.path.realpath(cwd))
    hits = [(len(w.parts), lane) for lane, w in ((lane, Path(os.path.realpath(w))) for lane, w in wts.items()) if c == w or w in c.parents]
    return latest(max(hits)[1], succ, lanes) if hits else None


def guardian_handoff(root: Path, lane: str, kills: list[str], now: dt.datetime, to: str | None = None) -> Path:
    """The handoff in `lanes/<to>/`: `to` is the owning lane, or the coordinator when that lane is final (contract §5)."""
    to = to or lane
    d = root / "lanes" / to
    d.mkdir(parents=True, exist_ok=True)
    p, n = d / f"{now:%Y%m%dT%H%MZ}-handoff-steward-guardian.md", 2
    while p.exists():
        p, n = d / f"{now:%Y%m%dT%H%MZ}-handoff-steward-guardian-{n}.md", n + 1
    m = KILLED_RE.match(kills[0])
    why = m["reason"] if m else "?"
    whose = lane if to == lane else f"{lane} (final)"
    p.write_text(f"---\nlane: {to}\nkind: handoff\nfrom: steward\ncreated: {now:%Y-%m-%dT%H:%MZ}\n---\n\n"
                 f"# the laptop guardian killed {len(kills)} process(es) of {whose} ({why}): rerun\n\n~~~text\n" + "\n".join(kills) +
                 "\n~~~\n\nkilled on the laptop by the guardian, not on the pod: rerun it (LANE-CONTRACT §7: heavy work runs on the pod).\n")
    return p


def guardian_lines(root: Path, rows: list[dict], fleet: Fleet | None, now: dt.datetime, state: dict, log: Path | None = GUARDIAN_LOG,
                   machines: Callable[[], dict] | None = None) -> list[str]:
    """New guardian kills to the owning lane's inbox (`kill_owner`): one handoff per lane per pass, `ROUTED <lane> <n>`; a final
    lane's kills go to the coordinator's (`ROUTED coordinator <n>: ... (<lane> is final)`); a kill no lane owns: `KILLED-UNOWNED <line>`."""
    kills = guardian_kills(log, state) if log else []
    if not kills:
        return []
    from .remote import load_machines  # noqa: PLC0415

    try:
        ms = (machines or load_machines)()
    except Exception:  # noqa: BLE001  (a broken machines.toml: the cwd still maps)
        ms = {}
    lanes, succ = {d.name for d in (root / "lanes").glob("*/")}, successors(root)
    bindings = {lane: _binding_json(root, lane) for lane in sorted(lanes)}
    wts = {lane: b["worktree"] for lane, b in bindings.items() if b.get("worktree")}
    wts.update({r["lane"]: r["worktree"] for r in rows if r.get("worktree") and r["lane"] not in wts})
    bound = {strip_pod_suffix(x): lane for lane, b in bindings.items()
             for x in (s.strip() for s in str(b.get("pod") or "").split(",")) if x and x not in NO_POD}
    by: dict[str, list[str]] = {}
    lines = []
    for k in kills:
        if lane := kill_owner(k, lanes, succ, ms, fleet.owner if fleet else {}, wts, bound):
            by.setdefault(lane, []).append(k)
        else:
            lines.append(f"{now:%H:%MZ} KILLED-UNOWNED {k[:300]}")
    final = {r["lane"] for r in rows if r["state"] in DONE}
    for lane, ks in sorted(by.items()):
        to = "coordinator" if lane in final else lane
        tail = f" ({lane} is final)" if to != lane else ""
        lines.append(f"{now:%H:%MZ} ROUTED {to} {len(ks)}: {guardian_handoff(root, lane, ks, now, to).name}{tail}")
    return lines


# ---- relaunch (contract §C) and worktree garbage collection ---------------------------------------------------------------------------------

UNMERGED = ("DD", "AU", "UD", "UA", "DU", "AA", "UU")


def _tilde(p: str | Path | None) -> str:
    s, home = str(p or ""), str(Path.home())
    return "~" + s[len(home):] if s == home or s.startswith(home + "/") else s


def save_uncommitted(wt: Path, dest: Path, stamp: str, max_file_mb: float = 100) -> dict:
    """Contract §C's copy of a worktree's uncommitted work as `dest/uncommitted-<stamp>`: .patch (git diff HEAD --binary),
    .status, -untracked.tgz (untracked, not ignored; a file over max_file_mb is only named in .status), .merge (MERGE_HEAD)."""
    dest.mkdir(parents=True, exist_ok=True)
    base = f"uncommitted-{stamp}"
    files = [dest / f"{base}.patch", dest / f"{base}.status"]
    files[0].write_bytes(subprocess.run(["git", "-C", str(wt), "diff", "HEAD", "--binary"], capture_output=True, timeout=120).stdout)
    st = _git(wt, "status", "--short") or ""
    raw = subprocess.run(["git", "-C", str(wt), "ls-files", "--others", "--exclude-standard", "-z"], capture_output=True, timeout=120).stdout
    untracked = [u for u in raw.decode("utf-8", "surrogateescape").split("\0") if u and ((wt / u).is_file() or (wt / u).is_symlink())]
    big = [u for u in untracked if (wt / u).is_file() and (wt / u).stat().st_size > max_file_mb * 2**20]
    files[1].write_text(st + "".join(f"# not in the tgz (over {max_file_mb:g} MB): {u}\n" for u in big))
    if small := [u for u in untracked if u not in big]:
        files.append(dest / f"{base}-untracked.tgz")
        with tarfile.open(files[-1], "w:gz") as tf:
            for u in small:
                tf.add(wt / u, arcname=u, recursive=False)
    g = (_git(wt, "rev-parse", "--absolute-git-dir") or "").strip()
    if g and (Path(g) / "MERGE_HEAD").is_file():
        files.append(dest / f"{base}.merge")
        files[-1].write_bytes((Path(g) / "MERGE_HEAD").read_bytes())
    lines = [x for x in st.splitlines() if x.strip()]
    return {"files": files, "dirty": len(lines), "conflicts": [x[3:] for x in lines if x[:2] in UNMERGED], "big": big}


def relaunch_dir(lane: str) -> Path:
    """Where `relaunch` saves a lane's uncommitted work: `$RESEARCH_RELAUNCH/<lane>`, else ~/.research/relaunch/<lane>.  On the
    worktree's machine, never in the notes: a copy of code is not evidence, and the notes are Git."""
    return Path(os.environ.get("RESEARCH_RELAUNCH") or Path.home() / ".research" / "relaunch").expanduser() / lane


def successor_name(root: Path, lane: str, succ: dict[str, str] | None = None) -> str:
    """`<topic>-<N+1>`: wave-4090 -> wave-4090-2 -> wave-4090-3 (a name some lane dir already has is skipped)."""
    topic, n = attempt(root, lane, succ)
    while (root / "lanes" / f"{topic}-{n + 1}").exists():
        n += 1
    return f"{topic}-{n + 1}"


def relaunch(root: Path, lane: str, successor: str | None = None, why: str = "", repo: Path | None = None,
             now: dt.datetime | None = None, fleet: Callable[[], Fleet] | None = None, brief: str | None = None,
             final: str | None = None, budget: str | None = None) -> tuple[str, list[str], str]:
    """Contract §C: save the lane's uncommitted work, checkpoint it superseded, bind the successor to the same branch, worktree
    and pod hint with `succeeds` (brief / final / budget: the lane's unless given).  -> (successor, what was done, launch message)."""
    now = now or _utcnow()
    succ = successors(root)
    if lane in succ:
        raise ValueError(f"{lane} was already taken over by {succ[lane]}; relaunch {latest(lane, succ)} instead")
    rep = report_for(root, lane)
    if rep is None and not (root / "lanes" / lane / BINDING).exists():
        raise ValueError(f"no lane {lane!r} under {root / 'lanes'} (no report, no binding)")
    s = successor or successor_name(root, lane, succ)
    if report_for(root, s) or (root / "lanes" / s / BINDING).exists():
        raise ValueError(f"lane {s} exists already (pick another --as)")
    b = binding(root, lane, _split_front_matter(rep.read_text(errors="replace"))[0] if rep else "")
    branch = b.get("branch") or f"lane/{lane}"
    wt = b.get("worktree") or (worktrees(repo).get(branch) if repo else None)
    stamp, done = f"{now:%H%MZ}", []
    saved = save_uncommitted(Path(wt), relaunch_dir(lane), stamp) if wt and Path(wt).is_dir() else None
    if saved:
        done.append(f"saved {saved['dirty']} uncommitted path(s): " + " ".join(_tilde(p) for p in saved["files"]))
    else:
        done.append(f"no worktree found for {branch}: nothing saved")
    sha = ((_git(Path(wt), "rev-parse", "--short=8", "HEAD") or "").strip() if saved else "") or "none"
    from . import steward  # noqa: PLC0415

    on_origin = False
    if not saved and repo and steward.git_liveness():       # the dead lane's machine is gone: its pushed branch is the base
        _git(repo, "fetch", "-q", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}", timeout=300)
        if tip := (_git(repo, "rev-parse", "-q", "--verify", "--short=8", f"refs/remotes/origin/{branch}^{{commit}}") or "").strip():
            sha, on_origin = tip, True
            done.append(f"base: origin/{branch} @ {sha}")
        wt = None if not (wt and Path(wt).is_dir()) else wt  # a worktree path on another machine binds nothing here
    checkpoint(root, lane, "superseded", f"by {s} (relaunch){': ' + why if why else ''}", commit=sha, now=now)
    done.append(f"checkpoint {lane} superseded by {s}")
    extra = {"brief": brief or b.get("brief"), "final": final or b.get("final"), "budget": budget or b.get("budget")}
    bind(root, s, branch=branch, worktree=wt, pod=b.get("pod"), succeeds=lane, **extra)
    done.append(f"bound {s}: {branch} {_tilde(wt) or '(no worktree)'} succeeds {lane}")
    f = fleet() if fleet else None
    pods = [pod_view(p, None, now) for p in f.pods if f.owner.get(p.get("id")) == s] if f else []
    topic, n = attempt(root, s)
    msg = [f"You are lane `{s}` (attempt {n} of {topic}), taking over `{lane}`. First read `~/.research/notes/kb/LANE-CONTRACT.md` "
           f"(it wins over everything), then run `~/.research/bin/research notes inbox {s}` (it includes {lane}'s unread handoffs).",
           f"- Goal: {'the brief `' + _tilde(extra['brief']) + '`; ' if extra['brief'] else ''}continue where {lane} stopped "
           f"(its report says what is done and what is left)."]
    keep = f"; its {saved['dirty']} uncommitted files stay yours, a copy is in `{_tilde(saved['files'][0].parent)}/uncommitted-{stamp}*`" \
        if saved and saved["dirty"] else ""
    if on_origin:
        msg.append(f"- Base: `{branch}` @ {sha} on origin: `git fetch origin {branch} && git checkout -B {branch} origin/{branch}` "
                   f"in your own checkout; push after every commit (`research notes push {s}`).")
    else:
        msg.append(f"- Base: `{branch}` @ {sha}" + (f", worktree `{_tilde(wt)}` (bound to you{keep})." if wt else "."))
    if rep:
        msg.append(f"- Read: `{_tilde(rep)}` and `git log {branch}`.")
    if pods:
        msg.append("- Pods (yours now): " + ", ".join(f"`{p['name']}` ({p['id']}, ${p['usd_h']:.2f}/h)" for p in pods)
                   + "; `research pods ssh <name>`.")
    elif f is not None and not f.error:
        msg.append("- Pods: none running (create `vy-" + s + "` if the work needs one).")
    elif b.get("pod") and b["pod"] not in NO_POD:
        msg.append(f"- Pod: `{b['pod']}` (bound; the pod listing was not checked).")
    if extra["final"] or extra["budget"]:
        msg.append("- " + "; ".join(x for x in (f"FINAL: {extra['final']} hard" if extra["final"] else "",
                                                 f"budget: {extra['budget']}" if extra["budget"] else "") if x) + ".")
    if saved and saved["conflicts"]:
        msg.append(f"- MID-MERGE: finish the merge first ({', '.join(saved['conflicts'][:6])} unmerged).")
    return s, done, "\n".join(msg)


def _state(root: Path, lane: str, now: dt.datetime) -> str:
    rep = report_for(root, lane)
    return lane_status(rep, now, 30)["state"] if rep else "open"


def gc_worktrees(root: Path, repo: Path, wt_root: Path | None = None, apply: bool = False, now: dt.datetime | None = None) -> list[dict]:
    """Worktrees under wt_root (default: the parent of `repo`'s worktree), with the latest attempts of the lanes using them (bound,
    or `lane/<lane>` checked out).  `keep` is None -- removable -- when every such lane is done, `git status` is clean and a
    remote-tracking branch of origin contains the branch tip.  `apply`: `git worktree remove` (never --force); branches stay."""
    now = now or _utcnow()
    top = Path((_git(repo, "rev-parse", "--show-toplevel") or str(repo)).strip()).resolve()
    wt_root = (wt_root or top.parent).expanduser().resolve()
    lanes, succ = {d.name for d in (root / "lanes").glob("*/")}, successors(root)
    users: dict[Path, set[str]] = {}
    for lane in lanes:
        if w := _binding_json(root, lane).get("worktree"):
            users.setdefault(Path(w).expanduser().resolve(), set()).add(lane)
    out = []
    for branch, path in sorted(worktrees(repo).items(), key=lambda kv: kv[1]):
        p = Path(path).resolve()
        if p == top or wt_root not in p.parents:
            continue
        who = users.get(p, set()) | ({branch[5:]} if branch.startswith("lane/") and branch[5:] in lanes else set())
        states = {h: _state(root, h, now) for h in sorted({latest(x, succ, lanes) for x in who})}
        live = [h for h, s in states.items() if s not in DONE]
        st = _git(p, "status", "--porcelain")
        dirty = [x for x in (st or "").splitlines() if x.strip()]
        tip = (_git(repo, "rev-parse", f"refs/heads/{branch}") or "").strip()
        pushed = bool(tip) and bool((_git(repo, "for-each-ref", "--count=1", "--contains", tip, "refs/remotes/origin") or "").strip())
        keep = ("no lane" if not states else f"lane {live[0]} is {states[live[0]]}" if live else
                "git status failed" if st is None else
                f"dirty ({len(dirty)} paths)" if dirty else None if pushed else "branch tip not on origin")
        row = {"worktree": str(p), "branch": branch, "lanes": states, "keep": keep}
        if apply and keep is None:
            r = subprocess.run(["git", "-C", str(repo), "worktree", "remove", str(p)], capture_output=True, text=True, timeout=300)
            row["removed"] = r.returncode == 0
            if r.returncode:
                row["error"] = r.stderr.strip()[-200:]
        out.append(row)
    return out


def render_gc(rows: list[dict], apply: bool) -> str:
    """The removable worktrees (or what --apply did), the done lanes' worktrees kept and why, and a count of the rest."""
    lines = []
    for r in rows:
        done = bool(r["lanes"]) and all(s in DONE for s in r["lanes"].values())
        if r["keep"] is None or done:
            verb = r["keep"] and "keep" or (("removed" if r.get("removed") else f"FAILED ({r.get('error')})") if apply else "remove")
            lanes = ", ".join(f"{k} {v}" for k, v in r["lanes"].items())
            lines.append(f"{verb:<8} {_tilde(r['worktree'])}  {r['branch']}  ({lanes}){'  ' + r['keep'] if r['keep'] else ''}")
    rm = sum(r["keep"] is None for r in rows)
    lines.append(f"{rm} {'removed' if apply else 'removable (--apply removes them)'}; kept {len(rows) - rm}: "
                 f"{sum((r['keep'] or '').startswith('lane ') for r in rows)} in use by an open lane, "
                 f"{sum(r['keep'] == 'no lane' for r in rows)} with no lane")
    return "\n".join(lines)


# ---- the push rule: a lane pushes its branch after every commit; the research coordinator merges from origin ---------------------------

NEVER_PUSH = ("main", "master")


def lane_branch(root: Path, lane: str) -> tuple[str, str | None]:
    """(branch, bound worktree or None) as `finish` resolves them: binding.json, then front matter, then `lane/<lane>`."""
    rep = report_for(root, lane)
    b = binding(root, lane, _split_front_matter(rep.read_text(errors="replace"))[0] if rep else "")
    return b.get("branch") or f"lane/{lane}", b.get("worktree")


def sync_state(repo: Path, branch: str) -> dict:
    """The local branch against `refs/remotes/origin/<branch>` as last fetched: `unpushed` (origin has no such branch), `in sync`,
    `ahead` / `behind` / `diverged` with counts, `missing` (no local branch)."""
    tip = (_git(repo, "rev-parse", "-q", "--verify", f"refs/heads/{branch}^{{commit}}") or "").strip()
    remote = (_git(repo, "rev-parse", "-q", "--verify", f"refs/remotes/origin/{branch}^{{commit}}") or "").strip()
    res: dict = {"tip": tip or None, "origin": remote or None, "ahead": None, "behind": None}
    if not tip:
        return {**res, "sync": "missing"}
    if not remote:
        return {**res, "sync": "unpushed"}
    if tip == remote:
        return {**res, "sync": "in sync", "ahead": 0, "behind": 0}
    counts = (_git(repo, "rev-list", "--left-right", "--count", f"{tip}...{remote}") or "").split()
    a, b = (int(x) for x in counts) if len(counts) == 2 else (None, None)
    return {**res, "ahead": a, "behind": b, "sync": "ahead" if a and not b else "behind" if b and not a else "diverged"}


def _same_repo(a: Path, b: Path) -> bool:
    def top(p: Path) -> str:
        p = Path(p).expanduser()
        return (_git(p, "rev-parse", "--show-toplevel") or "").strip() if p.is_dir() else ""

    ta, tb = top(a), top(b)
    return bool(ta and tb) and Path(ta).resolve() == Path(tb).resolve()


def push_branch(root: Path, lane: str, repo: Path | None) -> tuple[str, int]:
    """`git push origin <branch>` for the lane's branch, from its worktree when there is one.  Never `main` (one merger), never
    --force: a rejected push says to integrate origin's commits first.  Run in the notes clone itself, it lands `lanes/<lane>/`
    on the notes' main with `notes_sync.sync` instead (RESEARCH_NOTES_TOKEN's route): a lane branch of the notes is read by
    nobody.  -> (one line, exit code)."""
    if repo is not None and _same_repo(repo, root):
        from . import notes_sync  # noqa: PLC0415

        rep = notes_sync.sync(root, paths=[f"lanes/{lane}"], message=f"notes: {lane}")
        return "\n".join([notes_sync.render_sync(rep), *notes_sync.refused_lines(rep)]), rep["rc"]
    branch, wt = lane_branch(root, lane)
    if branch in NEVER_PUSH:
        return f"REFUSED {lane}: its branch is {branch}; lanes push lane/* and never {branch} (the research coordinator merges)", 2
    wt = wt or (worktrees(repo).get(branch) if repo else None)
    where = Path(wt).expanduser() if wt and Path(wt).expanduser().is_dir() else repo
    if where is None:
        return f"REFUSED {lane}: no worktree bound and no repo (--repo, $RESEARCH_REPO, or run inside the checkout)", 2
    tip = (_git(where, "rev-parse", "-q", "--verify", f"refs/heads/{branch}^{{commit}}") or "").strip()
    if not tip:
        cur = (_git(where, "branch", "--show-current") or "").strip()
        hint = f"; its worktree is on {cur}: `research notes bind {lane} --branch {cur} --worktree {_tilde(where)} --pod none`" \
            if cur and cur not in NEVER_PUSH else ""
        return f"REFUSED {lane}: no local branch {branch} in {_tilde(where)}{hint}", 2
    try:
        r = subprocess.run(["git", "-C", str(where), "push", "origin", f"refs/heads/{branch}:refs/heads/{branch}"],
                           capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        return f"PUSH-FAILED {branch} {tip[:8]}: {type(e).__name__}: {e}", 1
    if r.returncode:
        err = [x for x in r.stderr.strip().splitlines() if x.strip()]
        hint = (f" -> origin has commits {branch} lacks: git fetch origin {branch} and rebase or merge origin/{branch}, then push again"
                if any(k in r.stderr for k in ("non-fast-forward", "fetch first", "[rejected]")) else "")
        return f"PUSH-FAILED {branch} {tip[:8]}: {(err[-1] if err else f'git push rc {r.returncode}')[:200]}{hint}", 1
    return f"PUSHED {branch} {tip[:8]} -> origin", 0


def branches(root: Path, repo: Path, globs: list[str] | None = None, now: dt.datetime | None = None) -> list[dict]:
    """Every local `lane/*` branch and every bound branch, with the lanes using it (and their states) and its sync state against
    origin (as last fetched).  `blocking`: an open lane's branch that origin does not have in full (the push rule's check, H1)."""
    from fnmatch import fnmatch  # noqa: PLC0415

    now = now or _utcnow()
    lanes = {d.name for d in (root / "lanes").glob("*/")} if (root / "lanes").is_dir() else set()
    users: dict[str, set[str]] = {}
    for lane in lanes:
        users.setdefault(lane_branch(root, lane)[0], set()).add(lane)
    refs = (_git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/lane/") or "").split()
    names = sorted({*refs, *(b for b in users if b not in NEVER_PUSH and _git(repo, "rev-parse", "-q", "--verify", f"refs/heads/{b}"))})
    out = []
    for br in names:
        who = users.get(br, set()) | ({br[5:]} if br.startswith("lane/") and br[5:] in lanes else set())
        if globs and not any(fnmatch(x, g) for x in (br, br.removeprefix("lane/"), *who) for g in globs):
            continue
        states = {x: _state(root, x, now) for x in sorted(who)}
        s = sync_state(repo, br)
        ct = (_git(repo, "log", "-1", "--format=%ct", f"refs/heads/{br}") or "").strip()
        live = any(v not in DONE for v in states.values())
        out.append({"branch": br, "lanes": states, **s, "commit_at": _ts(int(ct)).isoformat() if ct.isdigit() else None,
                    "blocking": live and s["sync"] in ("unpushed", "ahead", "diverged")})
    return out


def render_branches(rows: list[dict], show_all: bool = False) -> str:
    """One line per branch origin lacks (all with `show_all`), blocking ones first, then the tally."""
    def cell(r: dict) -> str:
        n = {"ahead": f"ahead {r['ahead']}", "behind": f"behind {r['behind']}", "diverged": f"diverged +{r['ahead']}/-{r['behind']}"}
        return n.get(r["sync"], r["sync"])
    lines = []
    for r in sorted(rows, key=lambda r: (not r["blocking"], r["sync"] == "in sync", r["branch"])):
        if show_all or r["sync"] != "in sync":
            lanes = ", ".join(f"{k} {v}" for k, v in r["lanes"].items()) or "no lane"
            lines.append(f"{'BLOCKING' if r['blocking'] else '':<8} {r['branch']:<40} {cell(r):<16} {(r['commit_at'] or '')[:16]}  ({lanes})")
    tally: dict[str, int] = {}
    for r in rows:
        tally[r["sync"]] = tally.get(r["sync"], 0) + 1
    lines.append(f"{len(rows)} branches: " + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())) +
                 f"; {sum(r['blocking'] for r in rows)} blocking (an open lane's work origin lacks: research notes push <lane>)")
    return "\n".join(lines)


def _add_status_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--repo", type=Path, default=None, help="any worktree of the shared repo (default: $RESEARCH_REPO, else the cwd's)")
    p.add_argument("--pods", action="store_true", help="also list the account's pods by owning lane and probe them over ssh")
    p.add_argument("--stale-min", type=float, default=30)
    p.add_argument("--since-hours", type=float, default=24)
    p.add_argument("--all", action="store_true", help="every lane however old its report, and every attempt (predecessors too)")
    p.add_argument("--lane", action="append", default=[], metavar="GLOB", help="only lanes matching (repeatable)")
    p.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="skip lanes matching (repeatable)")
    p.add_argument("--liveness", choices=["mtime", "git"], default=None,
                   help="git: notes commit times and origin's lane/* tips, no worktree signal (steward.py; default $RESEARCH_LIVENESS, else mtime)")
    p.add_argument("--fetch", action="store_true", help="first fetch origin's lane/* branches (every pass, for watch)")


def fetch_lanes(repo: Path | None) -> str | None:
    """Fetch origin's lane/* into refs/remotes/origin/lane/*; None, or why it failed."""
    if repo is None:
        return "no repo (--repo, or $RESEARCH_REPO)"
    return None if _git(repo, "fetch", "-q", "origin", "+refs/heads/lane/*:refs/remotes/origin/lane/*", timeout=300) is not None \
        else "git fetch origin lane/* failed; origin as last fetched"


def main(argv: list[str]) -> int:
    """`sync` / `scan` live in notes_sync; with RESEARCH_NOTES_SYNC=1 a successful checkpoint / bind / relaunch syncs what it wrote."""
    from . import notes_sync  # noqa: PLC0415

    if line := notes_sync.ensure_clone(default_root()):
        print(line, file=sys.stderr)
    if argv and argv[0] in ("sync", "scan"):
        return notes_sync.main(argv, default_root())
    if argv and argv[0] == "archive":
        from .notes_archive import main as archive_main  # noqa: PLC0415

        return archive_main(argv[1:], default_root())
    if argv and argv[0] in ("claim", "approach", "approaches"):
        from . import approaches  # noqa: PLC0415

        return approaches.main(argv, default_root())
    if argv and argv[0] == "service":
        from .steward import service_main  # noqa: PLC0415

        return service_main(argv[1:])
    if argv and argv[0] in ("index", "mcp"):
        from . import notes_index, notes_mcp  # noqa: PLC0415

        return (notes_index if argv[0] == "index" else notes_mcp).main(argv[1:], default_root())
    rc = _main(argv)
    if rc in (0, 3) and argv and argv[0] in ("checkpoint", "bind", "relaunch"):
        root = Path(argv[argv.index("--root") + 1]).expanduser() if "--root" in argv[:-1] else default_root()
        lanes = [argv[1]] if len(argv) > 1 and not argv[1].startswith("-") else []
        if argv[0] == "relaunch" and "--as" in argv[:-1]:
            lanes.append(argv[argv.index("--as") + 1])
        line, sync_rc = notes_sync.auto_sync(root, lanes if argv[0] != "relaunch" or "--as" in argv else [])
        if line:
            print(line, file=sys.stderr)
        rc = rc or sync_rc
    return rc


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research notes", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("checkpoint", help="write a structured CHECKPOINT line at the top of the lane's report")
    cp.add_argument("lane")
    cp.add_argument("state", choices=STATES)
    cp.add_argument("text", nargs="+")
    cp.add_argument("--commit", help="default: git rev-parse --short HEAD in the current directory")
    cp.add_argument("--keep-pod", metavar="WHY", help="final: the lane's pod stays up on purpose (e.g. a shared live verifier)")
    cp.add_argument("--require-pushed", action="store_true", help="final: a branch tip not on origin fails the checks (exit 3)")
    cp.add_argument("--root", type=Path, default=None)
    cp.add_argument("--repo", type=Path, default=None)
    ib = sub.add_parser("inbox", help="the lane's handoffs it has not been shown yet (its own dir and its predecessors')")
    ib.add_argument("lane")
    ib.add_argument("--peek", action="store_true", help="do not mark them read (someone else looking)")
    ib.add_argument("--urgent", action="store_true",
                    help="only unread handoffs with `urgent` in the file name, never marked read; exit 1 when there are none "
                         "(a worker's timer runs it)")
    ib.add_argument("--root", type=Path, default=None)
    bd = sub.add_parser("bind", help="record the lane's branch / worktree / pod (a successor that took over a predecessor's)")
    bd.add_argument("lane")
    bd.add_argument("--branch")
    bd.add_argument("--worktree")
    bd.add_argument("--pod")
    bd.add_argument("--succeeds", metavar="LANE", help="the lane this one took over (its unread handoffs become this lane's)")
    bd.add_argument("--keep-pod", metavar="WHY", help="the reaper leaves this lane's pods alone ('' clears it)")
    for p in (bd, rl := sub.add_parser("relaunch", help="contract §C in one command: save the lane's uncommitted work, supersede it, "
                                                         "bind its successor to the same branch/worktree/pods, print the launch message")):
        p.add_argument("--brief", "--campaign", dest="brief", metavar="PATH", help="the campaign brief (named in a successor's launch message)")
        p.add_argument("--final", metavar="HH:MMZ", help="the lane's hard FINAL time")
        p.add_argument("--budget", metavar="TEXT", help="e.g. '$5' or '1x 4090 <= 30 min'")
        p.add_argument("--root", type=Path, default=None)
    rl.add_argument("lane")
    rl.add_argument("--as", dest="successor", metavar="LANE", help="the successor's name (default <topic>-<N+1>: wave-4090-2 -> wave-4090-3)")
    rl.add_argument("--why", default="", help="goes into the superseded checkpoint")
    rl.add_argument("--repo", type=Path, default=None)
    rl.add_argument("--no-pods", action="store_true", help="do not ask the pod API which pods the successor now owns")
    st = sub.add_parser("status", help="one row per lane topic: state, freshest sign of life and which, uncommitted files, pods, STALE")
    _add_status_args(st)
    st.add_argument("--json", action="store_true")
    wa = sub.add_parser("watch", help="re-run status every --every minutes; print a line for each thing to act on (STALE, IDLE-POD, ...)")
    _add_status_args(wa)
    wa.add_argument("--every", type=float, default=10)
    wa.add_argument("--passes", type=int, default=0, help="stop after N passes (0 = forever)")
    wa.add_argument("--snapshot", action="store_true", help="each pass, commit the notes repo (research notes snapshot)")
    wa.add_argument("--sync", action="store_true", help="each pass, `research notes sync` the notes repo (commit, rebase, push)")
    wa.add_argument("--idle-min", type=float, default=5, help="--pods: IDLE-POD when an open lane's pod is idle this long")
    wa.add_argument("--runway-h", type=float, default=3, help="--pods: RUNWAY when balance / spend rate drops under this")
    wa.add_argument("--reap", action="store_true", help="--pods: terminate the pods of FINAL lanes (no open successor, not kept) after the grace, "
                                                        "and idle pods of STALE lanes once custody passes")
    wa.add_argument("--reap-grace-min", type=float, default=10)
    wa.add_argument("--reap-stale-min", type=float, default=30, help="--reap: a STALE lane's pod idle this long is terminated (custody permitting)")
    wa.add_argument("--custody-r2", action="store_true", help="--reap: a pod's run custody comes from the store's remote and the pod guard "
                                                              "(run_custody_r2), not from ~/.research/runs on this machine")
    wa.add_argument("--guardian-log", default=str(GUARDIAN_LOG), help="the laptop guardian's log, whose KILLED lines go to the owning "
                                                                        "lane's inbox ('none': off)")
    wa.add_argument("--lease", nargs="?", const="store", default=None, metavar="store|PATH",
                    help="act only while holding the steward lease (steward.py): the store bucket's steward/lease.json, or a file")
    wa.add_argument("--standby", action="store_true", help="--lease: when another steward holds it, watch read-only and take over when it frees")
    wa.add_argument("--steward-id", default=None, help="--lease: the owner name recorded in the lease (default: the host name)")
    sv = sub.add_parser("service", help="print the steward's systemd unit or launchd plist (steward.py)")
    sv.add_argument("kind", choices=["systemd", "launchd"])
    gc = sub.add_parser("gc-worktrees", help="clean worktrees of finished lanes whose branch tip is on origin: list, or --apply to remove")
    gc.add_argument("--apply", action="store_true", help="git worktree remove them (never --force; branches stay)")
    gc.add_argument("--wt-root", type=Path, default=None, help="default: the parent of --repo's worktree")
    gc.add_argument("--root", type=Path, default=None)
    gc.add_argument("--repo", type=Path, default=None)
    ps = sub.add_parser("push", help="push the lane's branch to origin (after every commit); never main, never --force; in the notes clone, sync lanes/LANE/ to the notes' main")
    ps.add_argument("lane")
    ps.add_argument("--root", type=Path, default=None)
    ps.add_argument("--repo", type=Path, default=None)
    br = sub.add_parser("branches", help="lane branches against origin (as last fetched); exit 1 if an open lane's work is not on origin")
    br.add_argument("--lane", action="append", default=[], metavar="GLOB", help="only branches / lanes matching (repeatable)")
    br.add_argument("--fetch", action="store_true", help="first fetch origin's lane/* branches (updates remote-tracking refs only)")
    br.add_argument("--all", action="store_true", help="also list branches in sync")
    br.add_argument("--json", action="store_true")
    br.add_argument("--root", type=Path, default=None)
    br.add_argument("--repo", type=Path, default=None)
    sn = sub.add_parser("snapshot", help="commit the notes tree to its own git repo (files over 1 MB are skipped and listed)")
    sn.add_argument("--root", type=Path, default=None)
    ns = ap.parse_args(argv)
    root = ns.root or default_root()
    if getattr(ns, "liveness", None):
        from .steward import LIVENESS_ENV  # noqa: PLC0415

        os.environ[LIVENESS_ENV] = ns.liveness
    if ns.cmd in ("checkpoint", "inbox"):
        if ns.cmd == "checkpoint":
            print(checkpoint(root, ns.lane, ns.state, " ".join(ns.text), commit=ns.commit))
            if hint := wake_hint(" ".join(ns.text)):
                print(hint)
            if ns.keep_pod:                                 # the reaper reads it from the binding
                bind(root, ns.lane, keep_pod=ns.keep_pod)
            if ns.state == "superseded":                    # written by whoever relaunched it: the mail is the successor's
                return 0
        items = inbox(root, ns.lane)
        if getattr(ns, "urgent", False):
            urgent = [x for x in items if "urgent" in x["path"].name.lower()]
            print(render_inbox(ns.lane, urgent))
            return 0 if urgent else 1
        print(render_inbox(ns.lane, items))
        if not getattr(ns, "peek", False):
            mark_seen(root, ns.lane, items)
        if ns.cmd == "checkpoint" and ns.state == "final":
            text, rc = finish(root, ns.lane, ns.repo or repo_default(), keep_pod=ns.keep_pod, require_pushed=ns.require_pushed)
            print(text)
            return rc
        return 0
    if ns.cmd in ("bind", "relaunch"):
        try:
            if ns.cmd == "bind":
                print(bind(root, ns.lane, ns.branch, ns.worktree, ns.pod, ns.succeeds, ns.brief, ns.final, ns.budget, ns.keep_pod))
                return 0
            s, done, msg = relaunch(root, ns.lane, ns.successor, ns.why, ns.repo or repo_default(), brief=ns.brief, final=ns.final,
                                    budget=ns.budget, fleet=None if ns.no_pods else (lambda: load_fleet(root, probe=False)))
        except ValueError as e:
            print(f"research notes {ns.cmd}: {e}", file=sys.stderr)
            return 2
        print("\n".join(done) + f"\n\n--- launch message for {s} ---\n{msg}")
        return 0
    if ns.cmd == "snapshot":
        snap = snapshot(root)
        print(_snapshot_line(snap) if snap else f"{root}: nothing to commit (or not a git repo)")
        return 0
    repo = ns.repo or repo_default()
    if ns.cmd == "gc-worktrees":
        if repo is None:
            print("research notes gc-worktrees: no repo (--repo, or $RESEARCH_REPO)", file=sys.stderr)
            return 2
        print(render_gc(gc_worktrees(root, repo, ns.wt_root, apply=ns.apply), ns.apply))
        return 0
    if ns.cmd == "push":
        line, rc = push_branch(root, ns.lane, repo)
        print(line, file=sys.stdout if rc == 0 else sys.stderr)
        return rc
    if ns.cmd == "branches":
        if repo is None:
            print("research notes branches: no repo (--repo, or $RESEARCH_REPO)", file=sys.stderr)
            return 2
        if ns.fetch and _git(repo, "fetch", "-q", "origin", "+refs/heads/lane/*:refs/remotes/origin/lane/*", timeout=600) is None:
            print("research notes branches: git fetch origin failed; showing origin as last fetched", file=sys.stderr)
        rows = branches(root, repo, ns.lane)
        print(json.dumps(rows, indent=1) if ns.json else render_branches(rows, ns.all))
        return 1 if any(r["blocking"] for r in rows) else 0

    def run(fleet: Fleet | None) -> list[dict]:
        return status(root, ns.stale_min, None if ns.all else ns.since_hours, lanes=ns.lane, exclude=ns.exclude, repo=repo,
                      fleet=fleet, every_attempt=ns.all)

    if ns.cmd == "status":
        if ns.fetch and (err := fetch_lanes(repo)):
            print(f"research notes status: {err}", file=sys.stderr)
        rows = run(load_fleet(root) if ns.pods else None)
        print(json.dumps(rows, indent=1, default=str) if ns.json else render(rows))
        return 1 if any(r["stale"] for r in rows) else 0
    lease = None
    if ns.lease:
        from . import steward  # noqa: PLC0415

        load_env_file(R2_ENV)                               # the lease lives in the store's bucket
        try:
            lease = steward.Lease(steward.lease_backend(ns.lease), owner=ns.steward_id,
                                  ttl_s=max(steward.LEASE_TTL_MIN_S, 3 * ns.every * 60))
            held, doc = lease.ensure()
        except Exception as e:  # noqa: BLE001
            print(f"research notes watch: steward lease unavailable ({type(e).__name__}: {str(e)[:200]}); refusing to act without it",
                  file=sys.stderr)
            return 2
        if not held and not ns.standby:
            print(f"research notes watch: REFUSED: the steward lease ({lease.backend.describe()}) is held by {steward.holder(doc)}; "
                  "only the holder reaps, routes, launches and commits the notes (--standby waits for it)", file=sys.stderr)
            return 2
        import signal  # noqa: PLC0415

        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # the service manager's stop still releases the lease
    try:
        return _watch(ns, root, repo, run, lease)
    finally:
        if lease is not None:
            lease.release()


def _watch(ns: argparse.Namespace, root: Path, repo: Path | None, run: Callable[[Fleet | None], list[dict]], lease) -> int:
    """The watch loop.  With a lease, the steward rules (reaping, budget/overdue memory, guardian routing, scheduled runs, the state file,
    the notes commit) run only in a pass that renewed it; the alerts print either way."""
    prev: set[str] = set()
    prev_mail: dict[str, set[str]] | None = None
    idle_since: dict[str, float] = {}
    idle_alerted: set[str] = set()
    pods_noted: set[str] = set()
    runway_last: float | None = None
    state = load_state(root)
    if ns.reap:
        load_env_file(R2_ENV)                               # custody asks the remote; launchd gives the watcher no credentials
    guardian = None if ns.guardian_log in ("", "none") else Path(ns.guardian_log).expanduser()
    n = 0
    was_acting: bool | None = None
    while True:
        if ns.fetch and (err := fetch_lanes(repo)):
            print(f"{_utcnow():%H:%MZ} FETCH-FAILED {err}", flush=True)
        fleet = load_fleet(root) if ns.pods else None
        rows = run(fleet)
        now = _utcnow()
        acting, lease_line = True, None
        if lease is not None:
            from . import steward  # noqa: PLC0415

            try:
                acting, doc = lease.ensure()
            except Exception as e:  # noqa: BLE001  (the store is unreachable: nobody may assume the lease is still ours)
                acting, doc = False, {"owner": f"unknown ({type(e).__name__}: {str(e)[:80]})"}
            if acting and was_acting is False:
                lease_line = f"{now:%H:%MZ} LEASE-ACQUIRED {lease.owner}: this watcher is the steward now"
            elif not acting and was_acting is not False:
                lease_line = f"{now:%H:%MZ} {'LEASE-LOST' if was_acting else 'STANDBY'}: the steward lease is held by {steward.holder(doc)}"
            if not acting and not ns.standby:
                print(lease_line, flush=True)
                return 3
            was_acting = acting
        lines, prev = transitions(prev, rows, now)
        more, cur_mail = mail_lines(prev_mail or {}, rows, now)
        if prev_mail is None:                               # the first pass: mail already there is not news
            more = []
        prev_mail = cur_mail
        more += idle_lines(idle_since, idle_alerted, rows, now, ns.idle_min)
        if fleet is not None:
            try:
                acct = account()
            except Exception:  # noqa: BLE001  (the API is flaky: say so in the line, keep watching)
                acct = None
            line, runway = account_line(fleet, acct, now)
            alert, runway_last = runway_alert(runway, ns.runway_h, runway_last, now)
            custody = (lambda lane, pod: reap_custody(root, lane, pod["id"], runs=run_custody_r2)) if ns.custody_r2 else None
            more += [line, *alert, *reap_lines(root, rows, now, ns.reap_grace_min, pods_noted, reap=ns.reap and acting,
                                               custody=custody, state=state)]
            if acting:
                more += [*steward_rule("reap-stale", stale_reap_lines, root, rows, now, ns.reap_stale_min, state, reap=ns.reap,
                                       custody=custody),
                         *steward_rule("budget", budget_lines, root, fleet, now, state)]
        if acting:
            more += steward_rule("overdue", overdue_lines, root, rows, now, state)
            more += steward_rule("guardian", guardian_lines, root, rows, fleet, now, state, guardian)
            more += steward_rule("run", run_lines, root, now, state)
            from .approaches import steward_lines as approaches_lines  # noqa: PLC0415

            more += steward_rule("approaches", approaches_lines, root, now, state)
            more += steward_rule("state", lambda: save_state(root, state) or [])
        if lease_line:
            more.append(lease_line)
        if ns.sync and acting:
            from . import notes_sync  # noqa: PLC0415

            srep = notes_sync.sync(root)
            more += sync_lines(root, srep, now)
        elif ns.sync:                                       # standby: read the others' notes, write nothing
            from . import notes_sync  # noqa: PLC0415

            notes_sync.pull(root)
        elif ns.snapshot and acting and (snap := snapshot(root, now=now)) and snap["commit"]:
            more.append(f"{now:%H:%MZ} {_snapshot_line(snap)}")
        open_ = sum(r["state"] not in DONE for r in rows)
        for line in lines + more:
            print(line, flush=True)
        print(f"{now:%H:%MZ} pass {n + 1}: {open_} open, {len(prev)} stale", flush=True)
        n += 1
        if ns.passes and n >= ns.passes:
            return 1 if prev else 0
        time.sleep(ns.every * 60)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
