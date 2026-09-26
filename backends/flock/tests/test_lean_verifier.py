"""The Lean verifier's trust surface: a pinned toolchain, no dependencies, and no escape hatch that would let compiled code
differ from the definitions a proof is about (`native_decide`, `implemented_by`, `extern`, `unsafe`)."""
from __future__ import annotations

import json
import re
from pathlib import Path

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
