"""Core's library list (`verity.ml.circuits.library`) and the backend's table pins name the same tables: the Lean verifier's
`MUFU_TABLES` (its archive keys), the lowering's SHA-256 pins, and the derived tables that stay `program`."""
from __future__ import annotations

import re
from pathlib import Path

from verity.ml.circuits import library as L
from verity_flock import ir_lower as IL
from verity_flock import tail_pieces as TP

LOOKUP_LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock" / "Flock" / "Lookup.lean"


def _lean_tables() -> dict[str, tuple[str, str, int]]:
    text = LOOKUP_LEAN.read_text()
    body = text[text.index("def MUFU_TABLES"):]
    body = body[:body.index("namespace Lookup")]
    rows = re.findall(r'⟨"(\w+)",\s*"([0-9a-f]{64})",\s*"([0-9a-f]{128})",\s*2 \^ (\d+)⟩', body)
    return {name: (sha256, sha512, int(bits)) for name, sha256, sha512, bits in rows}


def test_the_lean_verifiers_tables_are_library_tables():
    lean = _lean_tables()
    assert set(lean) == {"rsq", "sqrt", "rcp", "ex2"}
    for name, (sha256, sha512, bits) in lean.items():
        t = L.table(name)
        assert (t.sha256, t.sha512, t.index_bits) == (sha256, sha512, bits)
        assert L.table_label(sha512) == L.LIBRARY


def test_the_lowerings_pins_agree_with_the_library():
    for name, sha256 in IL.TABLES.items():
        assert L.table(name).sha256 == sha256
    for name, bits in TP.TABLE_BITS.items():
        if name in {t.name for t in L.TABLES}:
            assert L.table(name).index_bits == bits
            assert TP.table_sha256(name) == L.table(name).sha256


def test_the_lean_library_list_is_cores():
    """`Flock/Library.lean`'s `LIBRARY_TABLES`, which labels tables in `Flock/CircuitType.lean`, is core's list."""
    text = (LOOKUP_LEAN.parent / "Library.lean").read_text()
    rows = re.findall(r'⟨"(\w+)",\s*"([0-9a-f]{128})",\s*(\d+),\s*(\d+)⟩', text)
    assert [(n, s, int(i), int(v)) for n, s, i, v in rows] == [(t.name, t.sha512, t.index_bits, t.value_bits) for t in L.TABLES]


def test_derived_tables_stay_program():
    assert {"tanh_rn", "gelu_tanh_bf16"} <= set(TP.TABLE_SHA256)
    for name in ("tanh_rn", "gelu_tanh_bf16"):
        assert name not in {t.name for t in L.TABLES}
