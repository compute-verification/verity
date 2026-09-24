"""`research notes`: lane checkpoints and the coordinator's liveness table.

  research notes checkpoint LANE STATE TEXT... [--commit SHA] [--keep-pod WHY] [--root DIR]   # STATE = open | blocked | final | superseded
  research notes inbox LANE [--peek]
  research notes bind LANE [--branch B] [--worktree DIR] [--pod NAME] [--succeeds LANE] [--root DIR]
  research notes status [--root DIR] [--repo DIR] [--pods] [--stale-min 30] [--since-hours 24] [--all] [--lane GLOB] [--exclude GLOB] [--json]
  research notes watch [--every 10] [status options]         # one line per lane that goes STALE, comes back, or gets MAIL after it finished

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
`lane/<lane>`, the worktree `git worktree list` shows for it, and the pod `vy-<lane>-veritor-campaign` (`--pod none`: a
lane without one); front matter `branch:` / `worktree:` / `pod:` lines are read too.

`status` reads every lane touched in the window.  Liveness is the freshest of: the report's last change, its newest checkpoint,
the lane branch's last commit, the newest uncommitted edit in its worktree, and with `--pods` the pod (a work process running
counts as now; otherwise its newest file under /workspace).  Reports alone are not enough: lanes that die mid-step leave a
fresh-looking checkpoint and a worktree full of uncommitted edits.  A lane that is not done and shows no sign of life for
`--stale-min` minutes is STALE; `dirty` is how many files it would lose.  A busy pod with every other signal old can be a
detached job that outlived its lane: the `via` column says which signal is keeping the lane alive.  `inbox` is the open lane's
unread handoff count (and the oldest one's age); `MAIL n` on a finished lane with no successor is handoffs nobody will read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

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
g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); echo "GPU ${{g:-none}}"
ps -eo etimes=,args= | grep -E 'python|cargo|rustc|pytest|ligero|nvcc|\.sh( |$)' \
  | grep -vE 'grep|{PROBE_MARK}|compile_worker|nginx|jupyter|sshd|start\.sh|live serve|sleep' | awk '{{print "PROC", $1}}'
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
         succeeds: str | None = None) -> Path:
    """Record (merge into) the lane's binding; an empty string clears a field."""
    path = root / "lanes" / lane / BINDING
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(path.read_text()) if path.exists() else {}
    for k, v in (("branch", branch), ("worktree", worktree), ("pod", pod), ("succeeds", succeeds)):
        if v is not None:
            if v:
                cur[k] = str(Path(v).expanduser()) if k == "worktree" else v
            else:
                cur.pop(k, None)
    path.write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n")
    return path


def binding(root: Path, lane: str, front: str = "") -> dict:
    """branch / worktree / pod for a lane: binding.json, then front matter; unset keys fall back at the call site."""
    out = {}
    for k in ("branch", "worktree", "pod"):
        if m := re.search(rf"^{k}:\s*`?([^\s`]+)", front, re.M):
            out[k] = str(Path(m.group(1)).expanduser()) if k == "worktree" else m.group(1)
    p = root / "lanes" / lane / BINDING
    if p.exists():
        try:
            out.update({k: v for k, v in json.loads(p.read_text()).items() if v})
        except ValueError:
            pass
    return out


# ---- inbox: handoffs a lane has not been shown yet ---------------------------------------------------------------------------------------

def successors(root: Path) -> dict[str, str]:
    """predecessor -> successor, from `[superseded] by <lane>` checkpoints and bindings' `succeeds`."""
    out: dict[str, str] = {}
    for d in (root / "lanes").glob("*/"):
        b = d / BINDING
        if b.exists():
            try:
                pred = json.loads(b.read_text()).get("succeeds")
            except ValueError:
                pred = None
            if pred:
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
    """POD_PROBE's output -> busy (work processes), newest_proc_s, gpu_util, gpu_mem_mb, file_age_s (on the pod's clock)."""
    res: dict = {"busy": 0, "newest_proc_s": None, "gpu_util": None, "gpu_mem_mb": None, "file_age_s": None}
    for line in out.splitlines():
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        if parts[0] == "GPU" and len(parts) >= 3 and parts[1].isdigit():
            res["gpu_util"], res["gpu_mem_mb"] = int(parts[1]), int(parts[2]) if parts[2].isdigit() else None
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


def probe_pods(names: dict[str, str], timeout: float = 30) -> dict[str, dict]:
    """lane -> pod probe for the pods that exist (by exact name); a pod that is missing or unreachable says so."""
    from .pods import connect, runpod  # noqa: PLC0415

    try:
        listing = {p.get("name"): p for p in connect._list_pods()}  # noqa: SLF001
    except Exception as e:  # noqa: BLE001  (no API key / network: the table still prints, without pods)
        return {lane: {"error": f"pod listing failed: {e}"} for lane in names}

    def one(item: tuple[str, str]) -> tuple[str, dict]:
        lane, name = item
        pod = listing.get(name) or listing.get(name + POD_SUFFIX)
        if pod is None:
            return lane, {"name": name, "gone": True}
        try:
            tgt = runpod.ssh_target(pod) or runpod.ssh_target(runpod.get(pod["id"]) or {})   # the listing sometimes omits ports
            if tgt is None:
                return lane, {"name": pod.get("name"), "error": f"no ssh port mapping ({pod.get('desiredStatus')})"}
            out = subprocess.run([*connect.ssh_for(*tgt), POD_PROBE], capture_output=True, text=True, timeout=timeout)
            return lane, {"name": pod.get("name"), **parse_pod_probe(out.stdout)}
        except Exception as e:  # noqa: BLE001
            return lane, {"name": pod.get("name"), "error": str(e)[:120]}

    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(ex.map(one, names.items()))


def enrich(rows: list[dict], root: Path, repo: Path | None, stale_min: float, now: dt.datetime,
           pods: Callable[[dict[str, str]], dict[str, dict]] | None = None) -> list[dict]:
    """Add git/pod signals to the lanes that are not done; recompute life (freshest signal) and STALE from all of them."""
    live = [r for r in rows if r["state"] not in DONE]
    wts = worktrees(repo) if repo else {}
    for r in live:
        b = binding(root, r["lane"], r.pop("front", ""))
        r["branch"] = b.get("branch") or f"lane/{r['lane']}"
        r["worktree"] = b.get("worktree") or wts.get(r["branch"])
        r["pod"] = b.get("pod") or f"{POD_PREFIX}{r['lane']}{POD_SUFFIX}"
        if repo:
            g = git_signals(repo, r["branch"], r["worktree"])
            r["dirty"] = g["dirty"]
            for k in ("commit", "edit"):
                r[f"{k}_min"] = round((now - g[f"{k}_at"]).total_seconds() / 60, 1) if g[f"{k}_at"] else None
    probes = pods({r["lane"]: r["pod"] for r in live if r["pod"] not in NO_POD}) if pods and live else {}
    succ = successors(root)
    for r in rows:
        r.pop("front", None)
        if r["state"] in DONE:
            late = [p for p in mail(root, r["lane"]) if r.get("done_ts") and p.stat().st_mtime > r["done_ts"]]
            r["mail"] = [] if r["lane"] in succ else [p.name for p in late]     # a successor inherits it: not lost
            continue
        box = inbox(root, r["lane"], succ)
        r["inbox"] = len(box)
        r["inbox_oldest_min"] = round((now.timestamp() - box[0]["mtime"]) / 60, 1) if box else None
        p = probes.get(r["lane"])
        if p is not None:
            r["pod_probe"] = p
            if p.get("busy"):
                r["pod_min"] = 0.0
            elif p.get("file_age_s") is not None:
                r["pod_min"] = round(p["file_age_s"] / 60, 1)
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
           pods: Callable[[dict[str, str]], dict[str, dict]] | None = None) -> list[dict]:
    from fnmatch import fnmatch  # noqa: PLC0415

    now = now or _utcnow()
    rows = []
    for d in sorted((root / "lanes").glob("*/")):
        if (lanes and not any(fnmatch(d.name, g) for g in lanes)) or any(fnmatch(d.name, g) for g in exclude or ()):
            continue
        path = report_for(root, d.name)
        bound = d / BINDING
        newest = max([p.stat().st_mtime for p in (path, bound) if p is not None and p.exists()], default=None)
        if newest is None or (since_hours is not None and now.timestamp() - newest > since_hours * 3600):
            continue
        if path is not None:
            rows.append(lane_status(path, now, stale_min))
        else:                                       # bound at launch, never reported: exactly the lane that dies unseen
            age = round((now.timestamp() - bound.stat().st_mtime) / 60, 1)
            rows.append({"lane": d.name, "report": None, "state": "open", "age_min": age, "stale": age > stale_min,
                         "checkpoint_at": None, "checkpoint_min": None, "commit": None, "text": "(no report yet)", "front": ""})
    return enrich(rows, root, repo, stale_min, now, pods)


# ---- finish checks (run by `checkpoint LANE final`) ----------------------------------------------------------------------------------------

def pod_owner(name: str, lanes: set[str]) -> str | None:
    """The lane a `vy-<...>` pod belongs to: the longest lane name it is, or starts with plus `-`."""
    n = name.removesuffix(POD_SUFFIX).removeprefix(POD_PREFIX)
    hits = [lane for lane in lanes if n == lane or n.startswith(lane + "-")]
    return max(hits, key=len) if hits else None


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


def _pods(lane: str, bound: str | None, lanes: set[str]) -> tuple[bool | None, str]:
    from .pods import connect  # noqa: PLC0415

    try:
        listing = connect._list_pods()  # noqa: SLF001
    except Exception as e:  # noqa: BLE001
        return None, f"unknown (pod listing failed: {str(e)[:80]})"
    mine = [p for p in listing if (p.get("desiredStatus") or "").upper() == "RUNNING" and p.get("name")
            and (p["name"] in (bound, f"{bound}{POD_SUFFIX}") or pod_owner(p["name"], lanes | {lane}) == lane)]
    if not mine:
        return True, "no running pod"
    return False, "; ".join(f"{p['name']} RUNNING ({p.get('costPerHr', '?')}/h) -> research pods terminate {p['id']}" for p in mine) + \
        " | --keep-pod WHY"


def finish(root: Path, lane: str, repo: Path | None, keep_pod: str | None = None,
           custody: Callable[[list[str]], tuple[bool | None, str]] = _custody,
           pods: Callable[[str, str | None, set[str]], tuple[bool | None, str]] = _pods) -> tuple[str, int]:
    """The end-of-lane checks.  They never block the FINAL line (already written); they say what is left, exit 3 if anything is."""
    rep = report_for(root, lane)
    text = rep.read_text(errors="replace") if rep else ""
    b = binding(root, lane, _split_front_matter(text)[0])
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
        checks.append(("pod", *pods(lane, b.get("pod"), {d.name for d in (root / "lanes").glob("*/")})))
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


# ---- rendering and the watch loop ----------------------------------------------------------------------------------------------------------

def _m(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}m"


def _pod_cell(r: dict) -> str:
    p = r.get("pod_probe")
    if p is None:
        return "-"
    if p.get("gone"):
        return "gone"
    if p.get("error"):
        return "err"
    gpu = f" {p['gpu_util']}%" if p.get("gpu_util") is not None else ""
    return f"busy{p['busy']}{gpu}" if p.get("busy") else f"idle{gpu}"


def render(rows: list[dict]) -> str:
    if not rows:
        return "no lane reports in the window"
    w = max(len(r["lane"]) for r in rows)
    out = [f"{'lane':<{w}}  {'state':<8} {'life':>5} {'via':<10} {'ckpt':>5} {'commit':>6} {'edit':>5} {'dirty':>5} {'pod':<9} "
           f"{'inbox':<9} checkpoint"]
    for r in rows:
        flag = "STALE" if r["stale"] else ("supersd" if r["state"] == "superseded" else r["state"])
        if r["state"] in DONE:
            box = f"MAIL {len(r['mail'])}" if r.get("mail") else ""
            out.append(f"{r['lane']:<{w}}  {flag:<8} {_m(r['age_min']):>5} {'report':<10} {'':>5} {'':>6} {'':>5} {'':>5} {'':<9} "
                       f"{box:<9} {(r['text'] or '')[:70]}")
            continue
        dirty = "-" if r.get("dirty") is None else str(r["dirty"])
        box = f"{r['inbox']} ({_m(r['inbox_oldest_min'])})" if r.get("inbox") else "0"
        out.append(f"{r['lane']:<{w}}  {flag:<8} {_m(r.get('life_min', r['age_min'])):>5} {r.get('life_via', 'report'):<10} "
                   f"{_m(r.get('checkpoint_min')):>5} {_m(r.get('commit_min')):>6} {_m(r.get('edit_min')):>5} {dirty:>5} "
                   f"{_pod_cell(r):<9} {box:<9} {(r['text'] or '')[:70]}")
    return "\n".join(out)


def transitions(prev: set[str], rows: list[dict], now: dt.datetime) -> tuple[list[str], set[str]]:
    """Lines for lanes that became STALE or came back since the last pass, and the new STALE set."""
    stale = {r["lane"] for r in rows if r["stale"]}
    by = {r["lane"]: r for r in rows}
    lines = []
    for lane in sorted(stale - prev):
        r = by[lane]
        lines.append(f"{now:%H:%MZ} STALE {lane} life={_m(r.get('life_min', r['age_min']))} via={r.get('life_via', 'report')} "
                     f"dirty={r.get('dirty', '-')} pod={_pod_cell(r)} branch={r.get('branch', '-')}")
    for lane in sorted(prev - stale):
        r = by.get(lane)
        why = "gone from the window" if r is None else (r["state"] if r["state"] in DONE else f"via={r.get('life_via')}")
        lines.append(f"{now:%H:%MZ} ALIVE {lane} ({why})")
    return lines, stale


def mail_lines(prev: dict[str, set[str]], rows: list[dict], now: dt.datetime) -> tuple[list[str], dict[str, set[str]]]:
    """A line per finished lane that received handoffs nobody will read (no successor), once per new file."""
    cur = {r["lane"]: set(r["mail"]) for r in rows if r.get("mail")}
    lines = [f"{now:%H:%MZ} MAIL {lane} got {len(new)} handoff(s) after it finished: {', '.join(sorted(new))}"
             for lane, names in sorted(cur.items()) if (new := names - prev.get(lane, set()))]
    return lines, cur


def _add_status_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--repo", type=Path, default=None, help="any worktree of the shared repo (default: $RESEARCH_REPO, else the cwd's)")
    p.add_argument("--pods", action="store_true", help="also ssh into each open lane's pod (processes, GPU, newest file)")
    p.add_argument("--stale-min", type=float, default=30)
    p.add_argument("--since-hours", type=float, default=24)
    p.add_argument("--all", action="store_true", help="every lane, however old its report")
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
    bd.add_argument("--root", type=Path, default=None)
    st = sub.add_parser("status", help="one row per lane: state, freshest sign of life and which, uncommitted files, pod, STALE")
    _add_status_args(st)
    st.add_argument("--json", action="store_true")
    wa = sub.add_parser("watch", help="re-run status every --every minutes; print a line when a lane goes STALE or comes back")
    _add_status_args(wa)
    wa.add_argument("--every", type=float, default=10)
    wa.add_argument("--passes", type=int, default=0, help="stop after N passes (0 = forever)")
    ns = ap.parse_args(argv)
    root = ns.root or default_root()
    if ns.cmd in ("checkpoint", "inbox"):
        if ns.cmd == "checkpoint":
            print(checkpoint(root, ns.lane, ns.state, " ".join(ns.text), commit=ns.commit))
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
    if ns.cmd == "bind":
        print(bind(root, ns.lane, ns.branch, ns.worktree, ns.pod, ns.succeeds))
        return 0
    repo = ns.repo or repo_default()

    def run() -> list[dict]:
        return status(root, ns.stale_min, None if ns.all else ns.since_hours, lanes=ns.lane, exclude=ns.exclude, repo=repo,
                      pods=probe_pods if ns.pods else None)

    if ns.cmd == "status":
        rows = run()
        print(json.dumps(rows, indent=1, default=str) if ns.json else render(rows))
        return 1 if any(r["stale"] for r in rows) else 0
    prev: set[str] = set()
    prev_mail: dict[str, set[str]] = {}
    n = 0
    while True:
        rows = run()
        now = _utcnow()
        lines, prev = transitions(prev, rows, now)
        more, prev_mail = mail_lines(prev_mail, rows, now)
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
