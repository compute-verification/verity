from __future__ import annotations

import fcntl
import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def once(tmp_path_factory):
    """`once(name, make)`: the directory `make` fills, made once in the whole run: xdist's workers share their base
    directory's parent, and the first to take the lock makes it while the others wait for it."""
    base = tmp_path_factory.getbasetemp()
    root = base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base

    def made(name: str, make) -> Path:
        d = root / name
        with open(f"{d}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (d / "done").exists():
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir()
                make(d)
                (d / "done").touch()
        return d

    return made
