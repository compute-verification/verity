"""`research notes`: lane checkpoints and the coordinator's liveness table.

  research notes checkpoint LANE STATE TEXT... [--commit SHA] [--keep-pod WHY] [--root DIR]   # STATE = open | blocked | final | superseded
  research notes inbox LANE [--peek]
  research notes bind LANE [--branch B] [--worktree DIR] [--pod NAME] [--succeeds LANE] [--brief PATH] [--final HH:MMZ] [--budget TEXT]
  research notes relaunch LANE [--as SUCCESSOR] [--why TEXT] [--brief PATH] [--final HH:MMZ] [--budget TEXT]   # contract §C in one command
  research notes status [--root DIR] [--repo DIR] [--pods] [--stale-min 30] [--since-hours 24] [--all] [--lane GLOB] [--exclude GLOB] [--json]
  research notes watch [--every 10] [--idle-min 5] [--runway-h 3] [--reap] [status options]   # alert lines, below
  research notes gc-worktrees [--apply] [--wt-root DIR]     # clean worktrees of finished lanes whose branch tip is on origin

A lane's report is `<root>/lanes/<lane>/<stamp>-report-<lane>.md` (the newest one if several).  `checkpoint` writes
`CHECKPOINT <sha> (<HH:MM>Z) [<state>] <text>` directly under the front matter (newest first, where lanes already put them),
sets the front matter's `status:` to the state, and creates the report if the lane has none.  `superseded` marks a lane whose
work a successor took over: it is done, never STALE.

Handoffs are files named `*handoff*` / `*asks*` / `*reply*` in the recipient's lane dir.  `checkpoint` (and `inbox`) print the
ones changed since the lane was last shown its inbox, by mtime, and mark them read; a successor (`bind --succeeds`, or the
predecessor's `[superseded] by <lane>` checkpoint) also gets its predecessors' unread ones.  `checkpoint LANE final` then runs
the finish checks: every `art:` id the report cites is preserved on the remote (catalog-recorded), no pod of the lane is still
running (`--keep-pod WHY` when one should), the worktree is clean at the branch tip, and every handoff is named in the report.
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
the grace) or, with `--reap`, `REAPED` (that pod terminated, logged in `<root>/reaper.log`; `checkpoint ... --keep-pod WHY` keeps it).

`relaunch` is contract §C in one command: save the lane's uncommitted work to `lanes/<lane>/evidence/uncommitted-<HHMMZ>*`,
checkpoint it superseded, bind the successor (`<topic>-<N+1>` unless `--as`) to its branch, worktree and pods, print the
successor's launch message.  `gc-worktrees` lists (`--apply`: removes) clean worktrees under the coordinator's worktree root
whose lanes are all done and whose branch tip is on origin; branches are never deleted, dirty worktrees never removed.
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

STATES = ("open", "blocked", "final", "superseded")
DONE = ("final", "superseded")
CHECKPOINT_RE = re.compile(r"CHECKPOINT\s+(?P<sha>[0-9a-fA-F]{4,40}|none)?\)?[:\s]*(?P<fin>\*{0,2}FINAL\*{0,2})?\s*\(?"
                           r"(?:(?P<hh>\d{1,2}):(?P<mm>\d{2})Z)?[);]?\s*(?:\[(?P<state>open|blocked|final|superseded)\])?\s*[-—:]*\s*(?P<text>.*)")
FINAL_RE = re.compile(r"^##\s+FINAL\b.*$", re.M)
DONE_STATUSES = ("done", "final", "closed", "complete", "completed")
POD_PREFIX, POD_SUFFIX = "vy-", "-veritor-campaign"
NO_POD = ("none", "-")
BINDING = "binding.json"
MAIL_GLOBS = ("*handoff*.md", "*asks*.md", "*reply*.md")
SEEN = ".inbox-seen"
SUPERSEDED_BY_RE = re.compile(r"\[superseded\]\s+by\s+([A-Za-z0-9][\w.-]*)")
ART_RE = re.compile(r"\bart:([0-9a-f]{8,64})\b")
PROBE_MARK = "research-notes-probe"
POD_PROBE = rf"""# {PROBE_MARK}
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', *' 'NF >= 2 {{ if ($1 + 0 > u) u = $1 + 0; m += $2; n++ }} END {{ if (n) print "GPU", u + 0, m + 0; else print "GPU none" }}'
echo "GPUAPPS $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"
ps -eo etimes=,args= | grep -E 'python|cargo|rustc|pytest|ligero|nvcc|\.sh( |$)' \
  | grep -vE 'grep|{PROBE_MARK}|compile_worker|nginx|jupyter|sshd|start\.sh|live serve|sleep' | awk '{{print "PROC", $1}}'
echo "SERVE $(ps -eo args= | grep -v {PROBE_MARK} | grep -c '[l]ive serve')"
f=$(timeout 10 find /workspace -xdev \( -name venv312 -o -name .venv -o -name target -o -name __pycache__ -o -name .cache \
  -o -name node_modules \) -prune -o -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1); echo "FILE ${{f:-none}} $(date +%s)"
"""


def default_root() -> Path:
    return Path(os.environ.get("RESEARCH_NOTES", Path.home() / ".research" / "notes"))


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _ts(t: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc)


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


def bind(root: Path, lane: str, branch: str | None = None, worktree: str | None = None, pod: str | None = None,
         succeeds: str | None = None, brief: str | None = None, final: str | None = None, budget: str | None = None,
         keep_pod: str | None = None) -> Path:
    """Record (merge into) the lane's binding; an empty string clears a field.  `pod` is a hint (pods go by name, `pod_owner`);
    brief / final (HH:MMZ) / budget go into a successor's launch message; keep_pod (checkpoint --keep-pod) stops the reaper."""
    if final and not re.fullmatch(r"\d{1,2}:\d{2}Z", final):
        raise ValueError(f"--final wants HH:MMZ (UTC), not {final!r}")
    path = root / "lanes" / lane / BINDING
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(path.read_text()) if path.exists() else {}
    for k, v in (("branch", branch), ("worktree", worktree), ("pod", pod), ("succeeds", succeeds), ("brief", brief),
                 ("final", final), ("budget", budget), ("keep_pod", keep_pod)):
        if v is not None:
            if v:
                cur[k] = str(Path(v).expanduser()) if k == "worktree" else v
            else:
                cur.pop(k, None)
    path.write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n")
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
        return float(p.read_text().strip())
    except (OSError, ValueError):
        return None


def mail(root: Path, lane: str) -> list[Path]:
    d = root / "lanes" / lane
    return sorted({p for g in MAIL_GLOBS for p in d.glob(g)}, key=lambda p: p.stat().st_mtime)


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
            t = p.stat().st_mtime
            if t > cutoff:
                out.append({"path": p, "lane_dir": d, "mtime": t, "title": _title(p)})
    return sorted(out, key=lambda x: x["mtime"])


def mark_seen(root: Path, lane: str, items: list[dict]) -> None:
    if items:
        p = root / "lanes" / lane / SEEN
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{max([x['mtime'] for x in items] + [_seen(root, lane) or 0.0])}\n")


def render_inbox(lane: str, items: list[dict], limit: int = 15) -> str:
    if not items:
        return f"INBOX {lane}: nothing new"
    lines = [f"INBOX {lane}: {len(items)} new -- act on each, or say in your next checkpoint why not"]
    for x in items[-limit:]:
        where = "" if x["lane_dir"] == lane else f"{x['lane_dir']}/"
        lines.append(f"  {_ts(x['mtime']):%H:%MZ} {where}{x['path'].name}  \"{x['title']}\"")
    if len(items) > limit:
        lines.append(f"  ... and {len(items) - limit} older: research notes inbox {lane} --peek")
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
    mtime = _ts(path.stat().st_mtime)
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
    cp_state = (best or {}).get("state")
    if "superseded" in (fm_state, cp_state):
        state = "superseded"
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


def git_signals(repo: Path, branch: str | None, worktree: str | None) -> dict:
    """commit_at: the branch's last commit; edit_at: newest mtime among the worktree's uncommitted files; dirty: their count."""
    res: dict = {"commit_at": None, "edit_at": None, "dirty": None}
    if branch and (ct := (_git(repo, "log", "-1", "--format=%ct", f"refs/heads/{branch}") or "").strip()).isdigit():
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
    n = name.removesuffix(POD_SUFFIX)[len(POD_PREFIX):]
    hits = [t for t in lanes if n == t or n.startswith(t + "-") or re.fullmatch(re.escape(t) + r"[a-z]\d*(?:-.+)?", n)]
    return max(hits, key=len) if hits else None


def pod_owner(name: str, lanes: set[str], succ: dict[str, str] | None = None, bound: dict[str, str] | None = None) -> str | None:
    """The lane that owns a pod: the latest attempt of the lane its name carries (`pod_topic`), else of the lane whose binding
    names it (`bound`: pod name without the campaign suffix -> lane; for names that carry no lane)."""
    t = pod_topic(name, lanes) or (bound or {}).get(name.removesuffix(POD_SUFFIX))
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
            bound[pod.removesuffix(POD_SUFFIX)] = lane
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
    v = {"name": (p.get("name") or "").removesuffix(POD_SUFFIX), "id": p.get("id"), "usd_h": float(p.get("costPerHr") or 0.0),
         **(probe or {})}
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
            late = [p for p in mail(root, r["lane"]) if r.get("done_ts") and p.stat().st_mtime > r["done_ts"]]
            r["mail"] = [] if r["lane"] in succ or r["state"] == "superseded" else [p.name for p in late]   # a successor has it
            continue
        r["branch"] = b.get("branch") or f"lane/{r['lane']}"
        r["worktree"] = b.get("worktree") or wts.get(r["branch"])
        r["pod"] = b.get("pod")
        if repo:
            g = git_signals(repo, r["branch"], r["worktree"])
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
    owners = set(fleet.owner.values()) if fleet else set()
    rows = []
    for d in sorted((root / "lanes").glob("*/")):
        if (lanes and not any(fnmatch(d.name, g) for g in lanes)) or any(fnmatch(d.name, g) for g in exclude or ()):
            continue
        path = report_for(root, d.name)
        bound = d / BINDING
        newest = max([p.stat().st_mtime for p in (path, bound) if p is not None and p.exists()], default=None)
        if newest is None or (since_hours is not None and now.timestamp() - newest > since_hours * 3600 and d.name not in owners):
            continue
        if path is not None:
            rows.append(lane_status(path, now, stale_min))
        else:                                       # bound at launch, never reported: exactly the lane that dies unseen
            age = round((now.timestamp() - bound.stat().st_mtime) / 60, 1)
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
        hits = st.index.resolve_art_prefix(f"art:{a}")
        (full.append(hits[0]) if len(hits) == 1 else unknown.append(f"art:{a}"))
    rep = preserved.check(st, full, mode="recorded")
    bad = [x["id"] for x in rep["not_preserved"]] + unknown
    if not bad:
        return True, f"{len(arts)} cited artifacts preserved (catalog-recorded remote replicas)"
    ids_ = " ".join(b[:13] for b in bad[:8]) + (" ..." if len(bad) > 8 else "")
    return False, f"{len(bad)}/{len(arts)} cited artifacts not preserved: {ids_} -> research data push <ids> --verify head"


def _pods(lane: str, bound: str | None, lanes: set[str], succ: dict[str, str] | None = None) -> tuple[bool | None, str]:
    try:
        listing = running_pods()
    except Exception as e:  # noqa: BLE001
        return None, f"unknown (pod listing failed: {str(e)[:80]})"
    mine = [p for p in listing if p.get("name")
            and (p["name"] in (bound, f"{bound}{POD_SUFFIX}") or pod_owner(p["name"], lanes | {lane}, succ) == lane)]
    if not mine:
        return True, "no running pod"
    return False, "; ".join(f"{p['name']} RUNNING ({p.get('costPerHr', '?')}/h) -> research pods terminate {p['id']}" for p in mine) + \
        " | --keep-pod WHY"


def finish(root: Path, lane: str, repo: Path | None, keep_pod: str | None = None,
           custody: Callable[[list[str]], tuple[bool | None, str]] = _custody,
           pods: Callable[..., tuple[bool | None, str]] = _pods) -> tuple[str, int]:
    """The end-of-lane checks.  They never block the FINAL line (already written); they say what is left, exit 3 if anything is."""
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
    unnamed = [f"{d}/{p.name}" if d != lane else p.name for d in chain(root, lane) for p in mail(root, d) if p.name not in text]
    checks.append(("handoffs", not unnamed, "all named in the report" if not unnamed else
                   f"{len(unnamed)} never named in the report (answer or say why not): " + ", ".join(unnamed[:6])))
    bad = [c for c in checks if c[1] is False]
    head_ = f"FINISH {lane}: " + (f"{len(bad)} problem(s) (FINAL line written; exit 3)" if bad else "ok")
    return "\n".join([head_] + [f"  {k:<9} {msg}" for k, _, msg in checks]), 3 if bad else 0


# ---- history: the notes tree is its own git repo, committed by one writer ------------------------------------------------------------------

SNAPSHOT_MAX_BYTES = 1 << 20


def snapshot(root: Path, max_bytes: int = SNAPSHOT_MAX_BYTES, now: dt.datetime | None = None) -> dict | None:
    """Commit every change in the notes repo except files over `max_bytes` (dumps belong in the store; .gitignore drops the known
    kinds).  Only the coordinator's watcher calls this, so lanes never contend for the index lock.  None: not a repo, or clean."""
    if not (root / ".git").exists():
        return None
    st = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if st is None:
        return None
    entries, paths, big = st.split("\0"), [], []
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
        p = root / path
        (big if p.is_file() and p.stat().st_size > max_bytes else paths).append(path)
    if not paths:
        return {"commit": None, "paths": 0, "skipped": big} if big else None
    subprocess.run(["git", "--literal-pathspecs", "-C", str(root), "add", "-A", "--pathspec-from-file=-", "--pathspec-file-nul"],
                   input="\0".join(paths), text=True, check=True, capture_output=True, timeout=120)
    now = now or _utcnow()
    msg = f"snapshot {now:%Y-%m-%dT%H:%MZ}: {len(paths)} paths" + (f"; {len(big)} too big, not committed" if big else "")
    if subprocess.run(["git", "-C", str(root), "commit", "-q", "--no-verify", "-m", msg], capture_output=True, timeout=60).returncode:
        return {"commit": None, "paths": 0, "skipped": big}
    return {"commit": (_git(root, "rev-parse", "--short", "HEAD") or "").strip(), "paths": len(paths), "skipped": big}


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


def reap_lines(root: Path, rows: list[dict], now: dt.datetime, grace_min: float, noted: set[str], reap: bool = False,
               terminate: Callable[[str], None] = terminate_pod) -> list[str]:
    """Pods of a FINAL lane (rows are latest attempts: no open successor) done for >= grace_min, not kept (--keep-pod): FINAL-POD
    once per pod, or with `reap` terminate them (REAPED, appended to <root>/reaper.log).  Pods no lane owns are never in rows."""
    lines = []
    for r in rows:
        if r["state"] != "final" or r.get("keep_pod") or not r.get("done_ts") or now.timestamp() - r["done_ts"] < grace_min * 60:
            continue
        for p in r.get("pods", []):
            what = f"{r['lane']} {p['name']} {p['id']} ${p['usd_h']:.2f}/h (final {(now.timestamp() - r['done_ts']) / 60:.0f}m ago)"
            if not reap:
                line = f"{now:%H:%MZ} FINAL-POD {what}: --reap terminates it, checkpoint --keep-pod WHY keeps it"
            else:
                try:
                    terminate(p["id"])
                    line = f"{now:%H:%MZ} REAPED {what}"
                except Exception as e:  # noqa: BLE001
                    line = f"{now:%H:%MZ} REAP-FAILED {what}: {str(e)[:100]}"
                with (root / "reaper.log").open("a") as f:
                    f.write(line + "\n")
            if line.split()[1] == "REAPED" or p["id"] not in noted:
                lines.append(line)
            noted.add(p["id"])
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
    saved = save_uncommitted(Path(wt), root / "lanes" / lane / "evidence", stamp) if wt and Path(wt).is_dir() else None
    if saved:
        done.append(f"saved {saved['dirty']} uncommitted path(s): " + " ".join(_tilde(p) for p in saved["files"]))
    else:
        done.append(f"no worktree found for {branch}: nothing saved")
    sha = ((_git(Path(wt), "rev-parse", "--short=8", "HEAD") or "").strip() if saved else "") or "none"
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
        dirty = ["(git status failed)"] if st is None else [x for x in st.splitlines() if x.strip()]
        tip = (_git(repo, "rev-parse", f"refs/heads/{branch}") or "").strip()
        pushed = bool(tip) and bool((_git(repo, "for-each-ref", "--count=1", "--contains", tip, "refs/remotes/origin") or "").strip())
        keep = ("no lane" if not states else f"lane {live[0]} is {states[live[0]]}" if live else
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


def _add_status_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--repo", type=Path, default=None, help="any worktree of the shared repo (default: $RESEARCH_REPO, else the cwd's)")
    p.add_argument("--pods", action="store_true", help="also list the account's pods by owning lane and probe them over ssh")
    p.add_argument("--stale-min", type=float, default=30)
    p.add_argument("--since-hours", type=float, default=24)
    p.add_argument("--all", action="store_true", help="every lane however old its report, and every attempt (predecessors too)")
    p.add_argument("--lane", action="append", default=[], metavar="GLOB", help="only lanes matching (repeatable)")
    p.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="skip lanes matching (repeatable)")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research notes", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("checkpoint", help="write a structured CHECKPOINT line at the top of the lane's report")
    cp.add_argument("lane")
    cp.add_argument("state", choices=STATES)
    cp.add_argument("text", nargs="+")
    cp.add_argument("--commit", help="default: git rev-parse --short HEAD in the current directory")
    cp.add_argument("--keep-pod", metavar="WHY", help="final: the lane's pod stays up on purpose (e.g. a shared live verifier)")
    cp.add_argument("--root", type=Path, default=None)
    cp.add_argument("--repo", type=Path, default=None)
    ib = sub.add_parser("inbox", help="the lane's handoffs it has not been shown yet (its own dir and its predecessors')")
    ib.add_argument("lane")
    ib.add_argument("--peek", action="store_true", help="do not mark them read (someone else looking)")
    ib.add_argument("--root", type=Path, default=None)
    bd = sub.add_parser("bind", help="record the lane's branch / worktree / pod (a successor that took over a predecessor's)")
    bd.add_argument("lane")
    bd.add_argument("--branch")
    bd.add_argument("--worktree")
    bd.add_argument("--pod")
    bd.add_argument("--succeeds", metavar="LANE", help="the lane this one took over (its unread handoffs become this lane's)")
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
    wa.add_argument("--idle-min", type=float, default=5, help="--pods: IDLE-POD when an open lane's pod is idle this long")
    wa.add_argument("--runway-h", type=float, default=3, help="--pods: RUNWAY when balance / spend rate drops under this")
    wa.add_argument("--reap", action="store_true", help="--pods: terminate the pods of FINAL lanes (no open successor, not kept) after the grace")
    wa.add_argument("--reap-grace-min", type=float, default=10)
    gc = sub.add_parser("gc-worktrees", help="clean worktrees of finished lanes whose branch tip is on origin: list, or --apply to remove")
    gc.add_argument("--apply", action="store_true", help="git worktree remove them (never --force; branches stay)")
    gc.add_argument("--wt-root", type=Path, default=None, help="default: the parent of --repo's worktree")
    gc.add_argument("--root", type=Path, default=None)
    gc.add_argument("--repo", type=Path, default=None)
    sn = sub.add_parser("snapshot", help="commit the notes tree to its own git repo (files over 1 MB are skipped and listed)")
    sn.add_argument("--root", type=Path, default=None)
    ns = ap.parse_args(argv)
    root = ns.root or default_root()
    if ns.cmd in ("checkpoint", "inbox"):
        if ns.cmd == "checkpoint":
            print(checkpoint(root, ns.lane, ns.state, " ".join(ns.text), commit=ns.commit))
            if ns.keep_pod:                                 # the reaper reads it from the binding
                bind(root, ns.lane, keep_pod=ns.keep_pod)
            if ns.state == "superseded":                    # written by whoever relaunched it: the mail is the successor's
                return 0
        items = inbox(root, ns.lane)
        print(render_inbox(ns.lane, items))
        if not getattr(ns, "peek", False):
            mark_seen(root, ns.lane, items)
        if ns.cmd == "checkpoint" and ns.state == "final":
            text, rc = finish(root, ns.lane, ns.repo or repo_default(), keep_pod=ns.keep_pod)
            print(text)
            return rc
        return 0
    if ns.cmd in ("bind", "relaunch"):
        try:
            if ns.cmd == "bind":
                print(bind(root, ns.lane, ns.branch, ns.worktree, ns.pod, ns.succeeds, ns.brief, ns.final, ns.budget))
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

    def run(fleet: Fleet | None) -> list[dict]:
        return status(root, ns.stale_min, None if ns.all else ns.since_hours, lanes=ns.lane, exclude=ns.exclude, repo=repo,
                      fleet=fleet, every_attempt=ns.all)

    if ns.cmd == "status":
        rows = run(load_fleet(root) if ns.pods else None)
        print(json.dumps(rows, indent=1, default=str) if ns.json else render(rows))
        return 1 if any(r["stale"] for r in rows) else 0
    prev: set[str] = set()
    prev_mail: dict[str, set[str]] | None = None
    idle_since: dict[str, float] = {}
    idle_alerted: set[str] = set()
    pods_noted: set[str] = set()
    runway_last: float | None = None
    n = 0
    while True:
        fleet = load_fleet(root) if ns.pods else None
        rows = run(fleet)
        now = _utcnow()
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
            more += [line, *alert, *reap_lines(root, rows, now, ns.reap_grace_min, pods_noted, reap=ns.reap)]
        if ns.snapshot and (snap := snapshot(root, now=now)) and snap["commit"]:
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
