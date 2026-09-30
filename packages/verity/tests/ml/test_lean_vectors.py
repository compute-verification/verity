"""The Lean transcription of `tc_dot_total` (`packages/verity/lean/Verity/TC/Spec.lean`) is held to the Python by vectors:
`Verity/TC/Vectors.lean` states each one as a theorem the kernel checks against the Lean, and this test checks the same
vectors against the Python. A change to either side's semantics then fails a build."""
from __future__ import annotations

import re
from pathlib import Path

from verity.ml.tc.total import HOPPER_BF16_WGMMA_K16 as HOP, tc_dot_total

VECTORS = Path(__file__).resolve().parents[2] / "lean" / "Verity" / "TC" / "Vectors.lean"
THEOREM = re.compile(
    r"theorem v\d+ : tcDotTotal HOPPER_BF16_WGMMA_K16 (0x[0-9a-f]+)\s*\[([^\]]*)\]\s*\[([^\]]*)\] =\s*(0x[0-9a-f]+) :=")


def _words(text: str) -> list[int]:
    return [int(w, 16) for w in text.split(",")]


def test_lean_vectors_are_the_python() -> None:
    vectors = THEOREM.findall(VECTORS.read_text())
    assert len(vectors) >= 48
    words = set()
    for acc, a, b, d in vectors:
        assert tc_dot_total(HOP, int(acc, 16), _words(a), _words(b)) == int(d, 16), (acc, a, b)
        words.add(int(d, 16))
    # the vectors reach every kind of result word: NaN, both infinities, +0, and finite words of both signs
    assert {0x7FFFFFFF, 0x7F800000, 0xFF800000, 0} <= words
    assert any(0 < w < 0x7F800000 for w in words) and any(0x80000000 < w < 0xFF800000 for w in words)
