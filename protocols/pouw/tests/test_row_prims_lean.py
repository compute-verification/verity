"""The Lean word functions of PoUW's row-unit primitives (`lean/Pouw/Protocol/PearlC/RowPrims.lean`) are
`verity_pouw.circuit.words`': `lean/scripts/row_prims_vectors.py`, run again, writes the vectors the kernel checks
(`RowPrimVectors.lean`, and `RowPrimChecks.lean` in `security_proofs/pouw`) byte for byte, each expected word the
primitive's own `evaluate`."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("verity_pouw.circuit.words", reason="the row primitives are `verity_pouw.circuit.words`'")

LEAN = Path(__file__).resolve().parents[1] / "lean"
PROOFS = Path(__file__).resolve().parents[3] / "security_proofs" / "pouw"
SCRIPT = LEAN / "scripts" / "row_prims_vectors.py"
FILES = [(LEAN, "Pouw/Protocol/PearlC/RowPrimVectors.lean"), (PROOFS, "PouwProofs/SecurityProofs/PearlC/RowPrimChecks.lean")]


def test_the_row_prim_vectors_are_regenerated(tmp_path: Path) -> None:
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path / "spec"), "--proofs-out", str(tmp_path / "proofs")],
                   check=True, capture_output=True)
    made = {LEAN: tmp_path / "spec", PROOFS: tmp_path / "proofs"}
    assert [f for pkg, f in FILES if (made[pkg] / f).read_bytes() != (pkg / f).read_bytes()] == []


def test_etilde_is_checked_on_its_whole_domain() -> None:
    lines = (LEAN / FILES[0][1]).read_text().splitlines()
    rows = [lines[i + 1] for i, line in enumerate(lines) if line.startswith("def etilde")]
    pairs = {tuple(int(x, 16) for x in v.split(", ")[:2]) for row in rows for v in row.strip(" []").strip("()").split("), (")}
    assert pairs == {(e, ae) for e in range(16) for ae in range(256)}
