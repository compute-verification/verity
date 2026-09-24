"""`research notes`: lane checkpoints and the coordinator's liveness table.

  research notes checkpoint LANE STATE TEXT... [--commit SHA] [--root DIR]   # STATE = open | blocked | final | superseded
  research notes bind LANE [--branch B] [--worktree DIR] [--pod NAME] [--root DIR]
  research notes status [--root DIR] [--repo DIR] [--pods] [--stale-min 30] [--since-hours 24] [--all] [--lane GLOB] [--exclude GLOB] [--json]
  research notes watch [--every 10] [status options]                         # one line per lane that goes STALE or comes back

A lane's report is `<root>/lanes/<lane>/<stamp>-report-<lane>.md` (the newest one if several).  `checkpoint` writes
`CHECKPOINT <sha> (<HH:MM>Z) [<state>] <text>` directly under the front matter (newest first, where lanes already put them),
sets the front matter's `status:` to the state, and creates the report if the lane has none.  `superseded` marks a lane whose
work a successor took over: it is done, never STALE.

`bind` records where a lane works in `<root>/lanes/<lane>/binding.json` (a successor that takes over its predecessor's
worktree names it there; lanes rewrite their reports, so the binding lives beside them).  Unbound lanes default to branch
`lane/<lane>`, the worktree `git worktree list` shows for it, and the pod `vy-<lane>-veritor-campaign` (`--pod none`: a
lane without one); front matter `branch:` / `worktree:` / `pod:` lines are read too.

`status` reads every lane touched in the window.  Liveness is the freshest of: the report's last change, its newest checkpoint,
the lane branch's last commit, the newest uncommitted edit in its worktree, and with `--pods` the pod (a work process running
counts as now; otherwise its newest file under /workspace).  Reports alone are not enough: lanes that die mid-step leave a
fresh-looking checkpoint and a worktree full of uncommitted edits.  A lane that is not done and shows no sign of life for
`--stale-min` minutes is STALE; `dirty` is how many files it would lose.  A busy pod with every other signal old can be a
detached job that outlived its lane: the `via` column says which signal is keeping the lane alive.
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


def bind(root: Path, lane: str, branch: str | None = None, worktree: str | None = None, pod: str | None = None) -> Path:
    """Record (merge into) the lane's binding; an empty string clears a field."""
    path = root / "lanes" / lane / BINDING
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(path.read_text()) if path.exists() else {}
    for k, v in (("branch", branch), ("worktree", worktree), ("pod", pod)):
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
    return {"lane": path.parent.name, "report": str(path), "state": state, "age_min": round(age, 1),
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
    for r in rows:
        r.pop("front", None)
        if r["state"] in DONE:
            continue
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
    out = [f"{'lane':<{w}}  {'state':<8} {'life':>5} {'via':<10} {'ckpt':>5} {'commit':>6} {'edit':>5} {'dirty':>5} {'pod':<9} checkpoint"]
    for r in rows:
        flag = "STALE" if r["stale"] else ("supersd" if r["state"] == "superseded" else r["state"])
        if r["state"] in DONE:
            out.append(f"{r['lane']:<{w}}  {flag:<8} {_m(r['age_min']):>5} {'report':<10} {'':>5} {'':>6} {'':>5} {'':>5} {'':<9} "
                       f"{(r['text'] or '')[:70]}")
            continue
        dirty = "-" if r.get("dirty") is None else str(r["dirty"])
        out.append(f"{r['lane']:<{w}}  {flag:<8} {_m(r.get('life_min', r['age_min'])):>5} {r.get('life_via', 'report'):<10} "
                   f"{_m(r.get('checkpoint_min')):>5} {_m(r.get('commit_min')):>6} {_m(r.get('edit_min')):>5} {dirty:>5} "
                   f"{_pod_cell(r):<9} {(r['text'] or '')[:70]}")
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
    cp.add_argument("--root", type=Path, default=None)
    bd = sub.add_parser("bind", help="record the lane's branch / worktree / pod (a successor that took over a predecessor's)")
    bd.add_argument("lane")
    bd.add_argument("--branch")
    bd.add_argument("--worktree")
    bd.add_argument("--pod")
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
    if ns.cmd == "checkpoint":
        print(checkpoint(root, ns.lane, ns.state, " ".join(ns.text), commit=ns.commit))
        return 0
    if ns.cmd == "bind":
        print(bind(root, ns.lane, ns.branch, ns.worktree, ns.pod))
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
    n = 0
    while True:
        rows = run()
        now = _utcnow()
        lines, prev = transitions(prev, rows, now)
        open_ = sum(r["state"] not in DONE for r in rows)
        for line in lines:
            print(line, flush=True)
        print(f"{now:%H:%MZ} pass {n + 1}: {open_} open, {len(prev)} stale", flush=True)
        n += 1
        if ns.passes and n >= ns.passes:
            return 1 if prev else 0
        time.sleep(ns.every * 60)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
