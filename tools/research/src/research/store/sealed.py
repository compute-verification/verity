"""Sealed run dirs: an ended run dir that a Nebius node's vy-seal-runs.timer has made root's and read-only
(pods/nebius/lockout/seal_runs.py), so no agent can change what custody vouched for.  The two tools that free disk still remove
one, through the root helper sudoers allows research and ubuntu for exactly that: `research retention gc` / `rm` delete a sealed
run dir with `vy-seal-runs delete`, and `research data evict --runs` removes its preserved big files with `vy-seal-runs evict`.
The helper redoes the checks that matter as root (a run dir directly under a runs root, sealed, custody still vouching for it)."""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

SEALER = ("sudo", "-n", "/usr/local/sbin/vy-seal-runs")
SEALED_UID = 0


def sealed(path: Path | str) -> bool:
    """Whether `path` is a sealed run dir: a directory of SEALED_UID's with no write bit, which this process can't change itself."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and st.st_uid == SEALED_UID and not st.st_mode & 0o222 and os.geteuid() != 0


def _call(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([*SEALER, *args], capture_output=True, text=True)


def delete(path: Path | str) -> str | None:
    """Delete the sealed run dir `path` through the helper; None, or why not."""
    r = _call("delete", str(path))
    return None if r.returncode == 0 else (r.stderr or r.stdout).strip() or f"{SEALER[-1]} delete: exit {r.returncode}"


def evict(path: Path | str, min_bytes: int) -> dict[str, Any]:
    """The helper's eviction of the sealed run dir `path` (store.evict.evict_custody's report); OSError with why when it refused."""
    r = _call("evict", str(path), "--min-bytes", str(min_bytes))
    if r.returncode != 0:
        raise OSError((r.stderr or r.stdout).strip() or f"{SEALER[-1]} evict: exit {r.returncode}")
    return json.loads(r.stdout.strip().splitlines()[-1])
