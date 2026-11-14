"""drive.toml's `lowrisk_excludes` keeps the code the auditor's devices and the verifiers run off the low-risk lane, where the
verified queue's loop lands a change on its ready mark alone, with no reader.

Two ways the list goes wrong without failing anything: a `device/` or `verifier/` directory it doesn't name as a whole (a
new file there would land unread, even while the files there now are caught by a name glob or a subdirectory's entry), and
an entry a move left behind, which matches nothing and so excludes nothing. Both are read through the queue's own parse
(`DriveFlags.parse`) and matcher (`research.queue.lowrisk`), against `git ls-files`.
"""
from __future__ import annotations

import functools
import subprocess
from collections.abc import Iterable
from pathlib import Path

from research.queue import DRIVE_FLAGS, DriveFlags, lowrisk

ROOT = Path(__file__).resolve().parents[1]
#: the directories whose every file, present or added later, is the auditor's devices' or a verifier's code
TRUSTED_DIRS = ("device", "verifier")
#: a file name no name glob of the list matches, standing for a file added later to a directory
NEW_FILE = "a-file-added-later"


@functools.cache
def tracked() -> tuple[str, ...]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    return tuple(sorted(p for p in out.split("\0") if p))


def excludes() -> tuple[str, ...]:
    return DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS).lowrisk_excludes


def trusted_dirs(files: Iterable[str]) -> list[str]:
    """Every directory of ``files`` named one of `TRUSTED_DIRS`, from the root, ending in `/`."""
    dirs = (f.split("/")[:-1] for f in files)
    return sorted({"/".join(parts[:i + 1]) + "/" for parts in dirs for i, part in enumerate(parts) if part in TRUSTED_DIRS})


def open_dirs(files: Iterable[str], entries: tuple[str, ...]) -> list[str]:
    """The `trusted_dirs` of ``files`` where a file added later would be low-risk under ``entries``."""
    return [d for d in trusted_dirs(files) if lowrisk([d + NEW_FILE], entries)]


def dead_entries(files: Iterable[str], entries: tuple[str, ...]) -> list[str]:
    """The ``entries`` under which no file of ``files`` is kept off the low-risk lane."""
    files = list(files)
    return [e for e in entries if lowrisk(files, [e])]


def test_every_device_and_verifier_directory_is_excluded():
    entries = excludes()
    assert trusted_dirs(tracked()), "no `device/` or `verifier/` directory is tracked: the test reads the wrong tree"
    assert not (bad := open_dirs(tracked(), entries)), (
        f"{DRIVE_FLAGS}'s `lowrisk_excludes` doesn't name these directories as a whole, so a file added to one would land on "
        f"its ready mark alone, with no reader: " + ", ".join(bad) + ". Fix: add each, as written here, to `lowrisk_excludes`.")


def test_no_excludes_entry_is_dead():
    assert not (bad := dead_entries(tracked(), excludes())), (
        f"these `lowrisk_excludes` entries of {DRIVE_FLAGS} match no tracked path, so they exclude nothing: "
        + ", ".join(bad) + ". Fix: point each at the path it was meant to cover (where a move put it), or delete it.")


def test_each_check_catches_its_failure():
    files = ["a/device/run.py", "a/device/sub/x.py", "b/verifier/registration.py", "b/verifier/lean/X.lean", "c/devices/y.py"]
    assert trusted_dirs(files) == ["a/device/", "b/verifier/"]
    assert open_dirs(files, ("a/device/", "registration.py", "*.lean")) == ["b/verifier/"]
    assert open_dirs(files, ("a/", "b/verifier/")) == []
    assert dead_entries(files, ("a/device/", "gone/", "a/device/run.py", "*.lean", "*.rs")) == ["gone/", "*.rs"]
