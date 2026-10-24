"""`Security/Proofs/Accounting/TS/J10RChecks.lean`'s vectors are the Python's: each `theorem NAME : LHS = RHS` the kernel
evaluates there is recomputed here from `seeds` and `h1tile`, so the Lean definitions of J10-R's noise (`J10R.lean`) and
the units' references can't drift apart."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import seeds as SD

pytestmark = pytest.mark.skipif(H.SG is None, reason="H1's receipts (circuit.segment, E2E-120) are not on this tree")
CHECKS = Path(__file__).resolve().parents[5] / "Security" / "Proofs" / "Accounting" / "TS" / "J10RChecks.lean"
SALT = bytes(range(64))
RA = bytes((7 * i + 3) % 256 for i in range(64))
RB = bytes((5 * i + 1) % 256 for i in range(64))


def _lean(v) -> str:
    if isinstance(v, tuple):
        return "(" + ", ".join(map(_lean, v)) + ")"
    if isinstance(v, list):
        return "[" + ", ".join(map(_lean, v)) + "]"
    return str(v)


def _vectors() -> dict[str, str]:
    sc = SD.call_seed(SALT, 40, RA, RB)
    lay, blocks = H.Layout(4, 8, 64), H.j10r_blocks(8, 64, (4, 8), 1, 2)
    rows, cols = H.j10r_noise(sc, lay, blocks)
    keys = H.fold_keys(SD.receipt_seed(SALT, 46, RA, RB), lay)
    return {
        "callSeed_vec": _lean(list(sc[:8])),
        "receiptSeed_vec": _lean(list(SD.receipt_seed(SALT, 46, RA, RB)[:8])),
        "seed_vec": _lean(list(SD.seed(SALT, 46, RA, RB)[:8])),
        "noiseBlock_vec": _lean(list(SD.noise_block(sc, 24)[:8])),
        "unitBlocks_vec": _lean(blocks),
        "row_vec": _lean([(H._value(rows[0][0][:32]), [H._value(z) for z in rows[0][1]])]),
        "col_vec": _lean([(H._value(cols[3][0][:32]), cols[3][1][:8])]),
        "keys_vec": _lean([k[:4] for k in keys]),
    }


def _theorems() -> dict[str, str]:
    text = CHECKS.read_text()
    return {m[1]: m[2].strip() for m in re.finditer(r"^theorem (\w+_vec) :.*= (\[.*\]|\(.*\)) := by decide \+kernel$",
                                                     text, re.M)}


def test_every_vector_is_the_python_s():
    assert _theorems() == _vectors()


def test_the_compression_checks_are_the_python_s():
    text = CHECKS.read_text()
    for m, n, k in ((32, 4096, 4096), (256, 4096, 4096), (256, 4096, 14336)):
        units = (m // 32) * (n // 32)
        num = H.j10r_compressions(m, n, k) * units
        assert num.denominator == 1 and f"callCompressionsNum {m} {n} {k} 32 32 = {num.numerator}" in text
    lay = H.Layout(32, 32, 4096)
    assert f"rb 4096 = {lay.rb} ∧ cb 4096 = {lay.cb} ∧ (Layout.mk 32 32 4096).blocks = {lay.blocks}" in text
    assert f"unitCompressions ⟨32, 32, 4096⟩ = {H.unit_compressions(4096)}" in text
    assert f"unitCompressions ⟨32, 32, 14336⟩ = {H.unit_compressions(14336)}" in text
