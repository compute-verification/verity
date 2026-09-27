"""The Lean verifier's trust surface: a pinned toolchain, no dependencies, and no escape hatch that would let compiled code
differ from the definitions a proof is about (`native_decide`, `implemented_by`, `extern`, `unsafe`)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[1] / "verifier" / "lean"
TOOLCHAIN = "leanprover/lean4:v4.34.0"  # the soundness package's Mathlib and ArkLib pin v4.34.0
FORBIDDEN = re.compile(r"\b(native_decide|implemented_by|extern|unsafe)\b")


def sources():
    """The executable's sources (the proofs' package, `soundness/`, has its own dependencies and checks)."""
    return sorted(p for p in LEAN.rglob("*.lean") if ".lake" not in p.parts and "soundness" not in p.relative_to(LEAN).parts)


def test_toolchain_is_pinned():
    assert (LEAN / "lean-toolchain").read_text().strip() == TOOLCHAIN


def test_no_dependencies():
    assert "require" not in (LEAN / "lakefile.toml").read_text()
    assert json.loads((LEAN / "lake-manifest.json").read_text())["packages"] == []


def test_no_escape_hatches():
    assert sources()
    hits = [f"{p.relative_to(LEAN)}:{i}: {line.strip()}" for p in sources()
            for i, line in enumerate(p.read_text().splitlines(), 1) if FORBIDDEN.search(line)]
    assert not hits, hits


STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def check_axioms(root, target):
    subprocess.run(["lake", "build", target], cwd=root, check=True, capture_output=True)
    out = subprocess.run(["lake", "env", "lean", "CheckAxioms.lean"], cwd=root, check=True, capture_output=True, text=True).stdout
    lines = [l for l in out.splitlines() if "depends on axioms" in l or "does not depend on any axioms" in l]
    assert lines, out
    for l in lines:
        used = set(re.findall(r"[\w.]+", l.split(":", 1)[1])) if "depends on axioms" in l else set()
        assert used <= STANDARD_AXIOMS, l


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_proofs_use_only_standard_axioms():
    check_axioms(LEAN, "FlockProofs")


LEVEL3 = LEAN / "level3"


@pytest.mark.skipif(shutil.which("lake") is None or not (LEVEL3 / ".lake" / "packages" / "mathlib" / ".lake" / "build").is_dir(),
                    reason="level3 needs Mathlib's build cache (`lake exe cache get` in lean/level3)")
def test_field_proofs_use_only_standard_axioms():
    assert (LEVEL3 / "lean-toolchain").read_text().strip() == TOOLCHAIN
    check_axioms(LEVEL3, "FlockLevel3")


def _has_unit_cut():
    try:
        from verity.ir.partition import validate_unit_cut  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(shutil.which("lake") is None or not _has_unit_cut(), reason="no Lean toolchain or no validate_unit_cut")
def test_unit_cut_agrees_with_the_reference(tmp_path):
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(LEAN.parent / "unit_cut_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--out", str(tmp_path), "--n", "300"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
