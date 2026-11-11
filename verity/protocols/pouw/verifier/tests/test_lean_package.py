"""PoUW's Lean (in `security`) and what the rest of this directory says about it: every Lean name the README cites is
a guarantee in `security/lean-audit.json`, a named assumption or a definition one states, and its FP8 conformance data is what `verity.definitions.silicon`
computes. `lean/scripts/fp8atom_vectors.py`, run again, writes the kernel-pinned vectors in `Definitions/Pouw/Fp8Atom/` and
their checks in `Proofs/Pouw/Fp8Atom/` byte for byte, and the bulk whose SHA-256 the proofs' record holds for
`Proofs.Pouw.Bulk`. The
Lean audit runs `Proofs.Pouw.Bulk` on that bulk (`runs`); this is the half that says the data is still `verity.definitions.silicon`'s."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[3]
LEAN = ROOT / "security"
SCRIPT = HERE / "lean" / "scripts" / "fp8atom_vectors.py"
POLICY = json.loads((LEAN / "lean-audit.json").read_text())
RUNS = json.loads((LEAN / "Proofs" / "lean-audit.json").read_text())["runs"]


def test_cited_theorems_are_guarantees() -> None:
    """A qualified `Pouw.…` name in the README is a guarantee in the record (proved, or stated and frozen), a premise
    it records (a named assumption, an open premise or a domain hypothesis), or a definition a guarantee's signature
    names (a game). Not `reads`: a merged lock's are a candidate until the Lean check regenerates them."""
    guarantees = set(POLICY["guarantees"]) | set(POLICY["shape"]["frozen"])
    named = {n for k in ("assumptions", "open", "domain") for n in POLICY.get(k, {})}
    stated = {n for g in POLICY["guarantees"].values() for n in re.findall(r"[\w.']+", " ".join(g["signature"]))}
    doc = (HERE / "README.md").read_text()
    cited = set(re.findall(r"`(Pouw\.[A-Za-z0-9_.']+)`", doc))
    assert cited
    assert sorted(cited - guarantees - named - stated) == []


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("fp8atom")
    subprocess.run([sys.executable, str(SCRIPT), str(ROOT), "--out", str(out)], check=True, capture_output=True)
    return out


@pytest.mark.slow
def test_kernel_pinned_vectors_are_regenerated(generated: Path) -> None:
    made = sorted(p for p in (q.relative_to(generated) for q in generated.rglob("*") if q.is_file()) if p.parts[0] != "conformance")
    assert len(made) >= 10, made
    assert [str(p) for p in made if (generated / p).read_bytes() != (LEAN / p).read_bytes()] == []


@pytest.mark.slow
def test_bulk_is_what_the_audit_pins(generated: Path) -> None:
    pinned = RUNS["Proofs.Pouw.Bulk"]["data"]
    made = {n: hashlib.sha256(gzip.decompress((generated / f"{n}.gz").read_bytes())).hexdigest() for n in pinned}
    assert made == pinned
