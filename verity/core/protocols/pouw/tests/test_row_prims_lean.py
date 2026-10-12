"""The Lean word functions of PoUW's row-unit primitives (`Security/Definitions/Pouw/PearlC/RowPrims.lean`) are
`verity.core.protocols.pouw.circuit.words`': `lean/scripts/row_prims_vectors.py`, run again, writes the vectors the kernel checks
(`RowPrimVectors.lean`, `Proofs/Pouw/PearlC/RowPrimChecks.lean`) byte for byte, each expected word the primitive's
own `evaluate`."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("verity.core.protocols.pouw.circuit.words", reason="the row primitives are `verity.core.protocols.pouw.circuit.words`'")

LEAN = Path(__file__).resolve().parents[5] / "Security"
SCRIPT = Path(__file__).resolve().parents[1] / "lean" / "scripts" / "row_prims_vectors.py"
FILES = ["Definitions/Pouw/PearlC/RowPrimVectors.lean", "Proofs/Pouw/PearlC/RowPrimChecks.lean"]


def test_the_row_prim_vectors_are_regenerated(tmp_path: Path) -> None:
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path)], check=True, capture_output=True)
    assert [f for f in FILES if (tmp_path / f).read_bytes() != (LEAN / f).read_bytes()] == []


def test_etilde_is_checked_on_its_whole_domain() -> None:
    lines = (LEAN / FILES[0]).read_text().splitlines()
    rows = [lines[i + 1] for i, line in enumerate(lines) if line.startswith("def etilde")]
    pairs = {tuple(int(x, 16) for x in v.split(", ")[:2]) for row in rows for v in row.strip(" []").strip("()").split("), (")}
    assert pairs == {(e, ae) for e in range(16) for ae in range(256)}
