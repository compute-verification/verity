"""The Lean transcription of `tc_dot_total` (`Security/Definitions/Core/TC/Spec.lean`) is held to the Python by
vectors: `Verity/SecurityProofs/TC/Vectors.lean` states each one as a theorem the kernel checks against the Lean, and this
test checks the same vectors against the Python. A change to either side's semantics then fails a build."""
from __future__ import annotations

import re
from pathlib import Path

from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16 as AMP
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16 as HOP, cvt_rn_bf16_f32, tc_dot_total

VECTORS = Path(__file__).resolve().parents[5] / "Security" / "Proofs" / "Core" / "TC" / "Vectors.lean"
PIPES = {"HOPPER_BF16_WGMMA_K16": HOP, "AMPERE_BF16_M16N8K16": AMP}
THEOREM = re.compile(
    r"theorem \w+ : tcDotTotal (\w+) (0x[0-9a-f]+)\s*\[([^\]]*)\]\s*\[([^\]]*)\] =\s*(0x[0-9a-f]+) :=")

CVT = re.compile(r"theorem \w+ : cvtRnBf16F32 (0x[0-9a-f]+) = (0x[0-9a-f]+) :=")


def _words(text: str) -> list[int]:
    return [int(w, 16) for w in text.split(",")]


def test_lean_vectors_are_the_python() -> None:
    vectors = THEOREM.findall(VECTORS.read_text())
    assert sum(p == "HOPPER_BF16_WGMMA_K16" for p, *_ in vectors) >= 48
    assert sum(p == "AMPERE_BF16_M16N8K16" for p, *_ in vectors) >= 24
    words = set()
    for pipe, acc, a, b, d in vectors:
        assert tc_dot_total(PIPES[pipe], int(acc, 16), _words(a), _words(b)) == int(d, 16), (pipe, acc, a, b)
        words.add(int(d, 16))
    # the vectors reach every kind of result word: NaN, both infinities, +0, and finite words of both signs
    assert {0x7FFFFFFF, 0x7F800000, 0xFF800000, 0} <= words
    assert any(0 < w < 0x7F800000 for w in words) and any(0x80000000 < w < 0xFF800000 for w in words)


def test_lean_cvt_vectors_are_the_python() -> None:
    vectors = CVT.findall(VECTORS.read_text())
    assert len(vectors) >= 24
    for word, bf16 in vectors:
        assert cvt_rn_bf16_f32(int(word, 16)) == int(bf16, 16), word
