"""The Lean executables storage's, the bandwidth audit's and the wipe's pieces build and answer with are their schemes'
declared ones (`Launch.executables`), which `declared.py` alone reads: served_piece.sh and served_wipe.sh take each
package and target from it and state none themselves, and served_piece.py and served_case.py take no executable's
path, only the tree they are built in."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from verity.protocols.run import schemes

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ("benchmarks/pous/served_piece.sh", "benchmarks/pous/streamed_wipe/served_wipe.sh")
NAMES = ("pous", "bandwidth", "wipe")


def _declared() -> list[tuple[str, str]]:
    return [(e.package, e.target) for n in NAMES for e in schemes.get(n).pieces["prover"].executables]


def test_declared_py_reads_the_declaration_and_nothing_else() -> None:
    got = subprocess.run([sys.executable, str(ROOT / "benchmarks/pous/declared.py"), *NAMES], check=True,
                         capture_output=True, text=True, env={"PYTHONPATH": str(ROOT)}).stdout
    assert [tuple(line.split(" ")) for line in got.splitlines()] == _declared()

