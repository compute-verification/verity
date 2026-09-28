"""`common`: the helpers `check.py` and `lean_audit.py` share. They live apart from `check.py` so that `lean_audit.key`, which
hashes every file the audit step runs, covers this module and not `check.py`, which only launches the step: an edit to
`check.py` alone then keeps every Lean package's verdict. Standard library only, so the key covers everything it runs."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


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
