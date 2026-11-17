"""Importing the wire, which holds the record entry, or the Site reads no sheet."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CARD_FILES = Path(__file__).resolve().parents[3] / "calibration/hardware/cards"
NODE_FILES = Path(__file__).resolve().parents[3] / "calibration/hardware/nodes"

READS = """
import os, sys
seen = []
sys.addaudithook(lambda event, args: seen.append(args[0]) if event in ("open", "os.scandir", "os.listdir") else None)
import verity.channel, verity.device
sheets = set(sys.argv[1:])
read = sorted({os.fspath(p) for p in seen if isinstance(p, (str, os.PathLike))
               and (os.fspath(p) in sheets or os.path.dirname(os.fspath(p)) in sheets)})
sys.exit("\\n".join(read) or None)
"""
"""What a fresh interpreter runs: the two imported under an audit hook, then each card or node file, or directory of
them, that the imports opened or listed, printed as the exit's message."""


def test_the_record_and_the_site_read_no_sheet_on_import() -> None:
    """In an interpreter of its own, since this process's other tests read the sheets: importing the two opens no card
    file or node file and lists neither directory, so the Site's readers of them run only when called."""
    done = subprocess.run([sys.executable, "-c", READS, str(CARD_FILES), str(NODE_FILES)], capture_output=True, text=True)
    assert done.returncode == 0 and not done.stderr, (
        "importing verity.channel and verity.device read these sheets (or failed): read a card or a node file only "
        f"where a reader is called, never at import\n{done.stderr}")
