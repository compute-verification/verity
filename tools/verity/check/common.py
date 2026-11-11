"""`common`: the helpers `check.py` and `lean_audit.py` share. They live apart from `check.py` so that `lean_audit.key`, which
hashes every file the audit step runs, covers this module and not `check.py`, which only launches the step: an edit to
`check.py` alone then keeps every Lean package's verdict. Standard library only, so the key covers everything it runs."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
#: the store's attempts (an SQL condition on its index) that are recorded checks: `check.py --record`'s own, whose tool is
#: `check`, and every check launched under `slot.py` (drive's trains, Lean checks and after-landing runs), whose tool is
#: `python3`
RECORDED_CHECKS = "(tool = 'check' OR (tool = 'python3' AND manifest_json LIKE '%tools/verity/check/check.py%'))"
#: the attempts, as `RECORDED_CHECKS` selects its own, that ran the Lean audit of a ready mark's merge
#: (`tools/verity/lean/warm.py`, which the run's command names): the other runs whose built modules and Lean verdicts a
#: check takes (`lean_audit.shared_bundles`, `check.verdict_packs`)
MARK_AUDITS = "manifest_json LIKE '%tools/verity/lean/warm.py%'"
#: where such a run's files hold what its audit left, the bundle and the verdict pack: the `--out` the quick tier gives it
MARK_AUDIT_OUT = "out/lean-audit/"
#: how the verified queue's commit of one PR onto a candidate begins (`research.queue.queue_commit`: a merge, or the one
#: commit of a move it regenerated), and the trailer of its message that names the PR's head as the queue took it
QUEUE_SUBJECT, QUEUE_HEAD = "Queue: #", re.compile(r"^Head: ([0-9a-f]{40})$", re.M)
#: how many commits back `merged_heads` reads: past the most the queue holds above `main` (`research.vqueue_loop`'s `DEPTH`
#: candidates stacked, of `BATCH` members each: 48), into the PRs `main` landed just before them
MERGED = 64
#: how many of the mark audits of those heads a launch looks through (`lookups`), the newest first: the queue puts few
#: members that change Lean in a candidate (its `LEAN`), so more than these would be the marks of PRs `main` landed long ago
MEMBERS = 32
#: how far back a launch lists the store's attempts (`recent`): well past the longest a recorded check runs, its steps
#: running at once and each stopped after `check.STEP_BUDGET_S`, so the newest recorded checks all started since; listing every
#: attempt the store holds instead takes half a minute
RECENT_S = 48 * 3600


def merged_heads(root: Path = ROOT, limit: int = MERGED) -> list[str]:
    """The head of each PR the verified queue put in the last ``limit`` commits of ``root``'s history (first parents), the
    newest first: a candidate's members, those of the candidates it stands on, and the PRs `main` landed before them. Empty
    outside a checkout and in one with no commit."""
    if not (root / ".git").exists():
        return []
    r = subprocess.run(["git", "-C", str(root), "log", "--first-parent", f"--max-count={limit}", "--format=%B%x1e"],
                       capture_output=True, text=True, errors="replace")
    found = (QUEUE_HEAD.search(body) for body in map(str.lstrip, r.stdout.split("\x1e") if r.returncode == 0 else ())
             if body.startswith(QUEUE_SUBJECT))
    return list(dict.fromkeys(m[1] for m in found if m))


def lookups(runs: str, limit: int, heads: list[str] | tuple[str, ...] = ()) -> list[tuple[str, tuple, int | None]]:
    """The queries a launch makes among the store's attempts of the kind ``runs`` (an SQL condition), in order, each as (its
    condition, its arguments, the last of them the most rows it reads, and the most runs taken from it, None for every one):
    the mark audits of ``heads`` first, where ``runs`` is `MARK_AUDITS` and there are heads, then the newest of the kind, of
    which ``limit`` are taken. A mark's audit is found by its head: the quick tier of a ready mark names the head it ran for
    (`--head`, or `--merged` where it ran on that head's merge onto `main`). That condition comes after the kind's, which
    few runs meet, so few commands are searched for the heads."""
    newest = [(runs, (4 * limit,), limit)]
    if runs != MARK_AUDITS or not heads:
        return newest
    named = " OR ".join("instr(manifest_json, ?) > 0" for _ in heads)
    return [(f"{runs} AND ({named})", (*heads, MEMBERS), None), *newest]


def recent(now: float) -> str:
    """The run id prefix (`rYYYYMMDD-HHMMSS`, UTC) `RECENT_S` before ``now`` (seconds since the epoch): the ``since`` of a store
    refresh that lists only the attempts started after it (`LocalStore.refresh`)."""
    return time.strftime("r%Y%m%d-%H%M%S", time.gmtime(now - RECENT_S))


def cache_dir() -> Path:
    return Path(os.environ.get("CHECK_CACHE") or Path.home() / ".cache" / "verity-check")


def cores() -> int:
    """The CPUs this process may use, bounded by the cgroup quota (a pod's `nproc` can exceed it)."""
    n = len(os.sched_getaffinity(0))
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            n = min(n, max(1, int(q) // int(p)))
    except (OSError, ValueError):
        try:                                           # cgroup v1
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            if q > 0:
                n = min(n, max(1, q // int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())))
        except (OSError, ValueError):
            pass
    return n


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_files(root: Path = ROOT) -> list[str] | None:
    """The tree's files as git has them, tracked and untracked-unignored (a tree `research run --on` ships is a checkout); None
    outside a checkout."""
    if not (root / ".git").exists():
        return None
    r = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "-co", "--exclude-standard"], capture_output=True)
    return sorted(p.decode() for p in r.stdout.split(b"\0") if p) if r.returncode == 0 else None
