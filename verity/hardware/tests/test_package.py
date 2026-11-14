"""The package file holds nothing: importing the record, the registration log or the Site runs no card-loading code."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "__init__.py"
READS = """
import os, sys
seen = []
sys.addaudithook(lambda event, args: seen.append(args[0]) if event in ("open", "os.scandir", "os.listdir") else None)
import verity.channel, verity.hardware.registration, verity.device as S
sheets = {os.fspath(S.CARDS), os.fspath(S.NODES)}
read = sorted({os.fspath(p) for p in seen if isinstance(p, (str, os.PathLike))
               and (os.fspath(p) in sheets or os.path.dirname(os.fspath(p)) in sheets)})
sys.exit("\\n".join(read) or None)
"""
"""What a fresh interpreter runs: the three imported under an audit hook, then each card or node file, or directory of
them, that the imports opened or listed, printed as the exit's message."""


def test_the_package_file_is_a_docstring_alone() -> None:
    """Whatever ``verity/hardware/__init__.py`` holds runs on every import of a module under it, a verifier's included:
    it re-exports nothing, and the readers of the card and node files are the Site's (``verity.device``)."""
    body = ast.parse(PACKAGE.read_text(encoding="utf-8")).body
    assert len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0].value, "value", None), str), (
        "verity/hardware/__init__.py holds more than its docstring; put the code in the module that owns it "
        "(the card and node files' readers are verity/device.py's) and import it from there")


def test_the_record_the_registration_and_the_site_read_no_sheet_on_import() -> None:
    """In an interpreter of its own, since this process's other tests read the sheets: importing the three opens no card
    file or node file and lists neither directory, so the Site's readers of them run only when called."""
    done = subprocess.run([sys.executable, "-c", READS], capture_output=True, text=True)
    assert done.returncode == 0 and not done.stderr, (
        "importing verity.channel, verity.hardware.registration and verity.device read these sheets (or failed): read a "
        f"card or a node file only where a reader is called, never at import\n{done.stderr}")
