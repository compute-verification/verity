"""Sealed run dirs: ended run dirs a node's timer has made root's and read-only."""
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
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and st.st_uid == SEALED_UID and not st.st_mode & 0o222 and os.geteuid() != 0


def _call(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([*SEALER, *args], capture_output=True, text=True)


def delete(path: Path | str) -> str | None:
    r = _call("delete", str(path))
    return None if r.returncode == 0 else (r.stderr or r.stdout).strip() or f"{SEALER[-1]} delete: exit {r.returncode}"


def evict(path: Path | str, min_bytes: int) -> dict[str, Any]:
    r = _call("evict", str(path), "--min-bytes", str(min_bytes))
    if r.returncode != 0:
        raise OSError((r.stderr or r.stdout).strip() or f"{SEALER[-1]} evict: exit {r.returncode}")
    return json.loads(r.stdout.strip().splitlines()[-1])
