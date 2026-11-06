"""Every temporary directory drive's code makes ignores errors cleaning it up.

Drive runs on the control pod, whose TMPDIR is an NFS volume: removing a file a just-ended subprocess held open leaves an
`.nfsXXXX` silly-rename behind, so `TemporaryDirectory`'s cleanup raises `OSError: [Errno 39] Directory not empty` and the
whole tick fails over a leftover file."""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "research"
DRIVE_MODULES = ("queue.py", "drive.py")


def _unguarded(path: Path) -> list[int]:
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "TemporaryDirectory"
                and not any(k.arg == "ignore_cleanup_errors" and isinstance(k.value, ast.Constant) and k.value.value is True
                            for k in node.keywords)):
            out.append(node.lineno)
    return out


def test_every_temporary_directory_in_drives_code_ignores_cleanup_errors():
    bad = {m: lines for m in DRIVE_MODULES if (lines := _unguarded(SRC / m))}
    assert not bad, (f"tempfile.TemporaryDirectory without ignore_cleanup_errors=True at {bad}: pass it, since drive's TMPDIR "
                     "is NFS and a silly-renamed file fails the cleanup")
