"""The streamed wipe's labels, `Accounting.Wipe.deployedLabels` against `streamed_wipe.label`, on the vectors of
`Security/Proofs/Properties/Verity/WipeLabelVectors.lean`.

Each vector there is a theorem the kernel checks (`decide +kernel`): label `i` of the deployed labels under the key `K`,
at the compiled SHA-512, is a label written in hex, the key's and the label's hex read most significant bit first. These
tests read the same vectors and hold the Python's `label` to each, so the two agree on them: the Lean check fails when
`keyBytes` or `digestBits` reads bytes in another bit order, and these tests fail when `label` changes. The Lean's side is
the spec's, which imports Mathlib and which no core package may require, so it has no difftest executable; the vectors
are the shared input.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from verity.core.protocols.pous.streamed_wipe import label

ROOT = Path(__file__).resolve().parents[6]
VECTORS = ROOT / "Security" / "Proofs" / "Properties" / "Verity" / "WipeLabelVectors.lean"
THEOREM = re.compile(r'theorem (label_\d+) :\s+label "([0-9a-f]{64})" (\d+) =\s+bitsOfHex 512 \("([0-9a-f]{64})" \+\+\s+'
                     r'"([0-9a-f]{64})"\) := by\s+decide \+kernel\n')
TEXT = VECTORS.read_text()
CASES = [(name, bytes.fromhex(key), int(i), bytes.fromhex(hi + lo)) for name, key, i, hi, lo in THEOREM.findall(TEXT)]


def test_every_theorem_is_a_vector():
    """Every theorem of the file is a vector these tests read, and the four keys and indices the file names are among them."""
    assert len(CASES) == TEXT.count("\ntheorem ") >= 4
    assert {(key, i) for _, key, i, _ in CASES} >= {(bytes(range(32)), 0), (bytes(range(32)), 1),
                                                    (b"\x80" + bytes(30) + b"\x01", 0x0102030405060708),
                                                    (b"\xff" * 32, 2**64 - 1)}


@pytest.mark.parametrize("name,key,i,expected", CASES, ids=[c[0] for c in CASES])
def test_label_is_the_lean_vector(name, key, i, expected):
    assert label(key, i) == expected
