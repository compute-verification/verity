"""Coarse per-PR landing trace from git + GitHub push activity + store labels.

For each `Queue: #N` merge on checked's first-parent history since SINCE: the PR, its landed head (parent 2), its ready and
review labels at that head, the time `checked` first contained the merge (push activity) and the time `main` first did.
"""
import os

#: the study's data directory: trace.py writes it, the rest read it (`research data fetch art:7ee7a884...` restores it)
DATA = os.environ.get("LANDING_STUDY_DIR", "/tmp/ld")
import concurrent.futures as cf
import json
import re
import subprocess
import sys
from datetime import datetime

REPO = "/agent/repos/verity"
SINCE = sys.argv[1] if len(sys.argv) > 1 else "2026-10-20"
OUT = sys.argv[2] if len(sys.argv) > 2 else DATA + "/trace.json"


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def pushes(ref):
    rows = []
    for line in open(f"{DATA}/act_{ref}.tsv"):
        t, kind, before, after, *_ = line.rstrip("\n").split("\t")
        rows.append((ts(t), kind, before, after))
    rows.sort()
    return rows


def first_contained(ref, commits):
    """commit -> earliest push time of ref whose `after` has it in first-parent history."""
    seen = {}
    want = set(commits)
    for t, kind, before, after in pushes(ref):
        if not after or set(after) == {"0"}:
            continue
        if subprocess.run(["git", "-C", REPO, "cat-file", "-e", after], capture_output=True).returncode:
            continue
        for c in git("rev-list", "--first-parent", "--max-count=400", after).split():
            if c in want and c not in seen:
                seen[c] = t
    return seen


def labels(pr, head):
    out = subprocess.run(["research", "data", "labels", f"pr:{pr}@{head}", "--history", "--json"], capture_output=True,
                         text=True).stdout
    try:
        return json.loads(out)
    except ValueError:
        return []


def lean_files(c):
    files = git("diff", "--name-only", f"{c}^1", c).split()
    return files


def main():
    log = git("log", "--first-parent", f"--since={SINCE}", "--format=%H%x09%P%x09%cI%x09%s", "origin/checked")
    rows = []
    for line in log.splitlines():
        h, parents, cdate, subj = line.split("\t", 3)
        ps = parents.split()
        m = re.match(r"Queue: #(\d+)", subj)
        rows.append({"commit": h, "parents": ps, "cdate": ts(cdate), "subject": subj[:120],
                     "pr": int(m[1]) if m else None, "head": ps[1] if m and len(ps) > 1 else None})
    commits = [r["commit"] for r in rows]
    on_checked = first_contained("checked", commits)
    on_main = first_contained("main", commits)
    with cf.ThreadPoolExecutor(16) as ex:
        labs = list(ex.map(lambda r: labels(r["pr"], r["head"]) if r["pr"] else [], rows))
        files = list(ex.map(lambda r: lean_files(r["commit"]) if r["pr"] else [], rows))
    for r, ls, fs in zip(rows, labs, files):
        r["checked_at"] = on_checked.get(r["commit"])
        r["main_at"] = on_main.get(r["commit"])
        ready = sorted(ts(x["ts"]) for x in ls if x["key"] == "ready" and x["value"] is True)
        rev = sorted((ts(x["ts"]), x["by"]) for x in ls if x["key"] == "reviewed" and x["value"] == "land")
        r["ready_at"] = ready[0] if ready else None
        r["ready_last"] = ready[-1] if ready else None
        r["reviewed_at"] = rev[0][0] if rev else None
        r["reviewer"] = rev[0][1] if rev else None
        r["ready_by"] = next((x["by"] for x in ls if x["key"] == "ready"), None)
        r["nfiles"] = len(fs)
        r["lean"] = any(f.endswith(".lean") or f.endswith("lean-audit.json") or "lakefile" in f for f in fs)
        r["lock"] = any(f.endswith("lean-audit.json") for f in fs)
        r["hot"] = sorted({f for f in fs if f.endswith(("lean-audit.json", "TRUSTED.sha256", "GLOSSARY.md"))
                           or f.endswith("README.md")})[:8]
        r["files"] = fs[:60]
    json.dump(rows, open(OUT, "w"), indent=1)
    n = sum(1 for r in rows if r["pr"])
    print(f"{len(rows)} commits, {n} PR merges; checked-time {sum(1 for r in rows if r['checked_at'])}, "
          f"main-time {sum(1 for r in rows if r['main_at'])}, ready {sum(1 for r in rows if r['ready_at'])}")


main()
