"""The library list (`verity.circuits.library`): the measured MUFU tables and core's tensor-core steps and casts are `library`,
everything else `program`."""
from __future__ import annotations

import re

import pytest

from verity.circuits.defs import REGISTRY
from verity.circuits import library as L
from verity.circuits.ops import prims  # noqa: F401  (registers the listed Definitions)

def test_tables_are_library_by_their_sha512():
    for t in L.TABLES:
        assert L.table_label(t.sha512) == L.LIBRARY
        assert L.table(t.sha512) is t and L.table(t.name) is t
        assert re.fullmatch(r"[0-9a-f]{128}", t.sha512) and re.fullmatch(r"[0-9a-f]{64}", t.sha256)
        assert t.words == 1 << t.index_bits and t.value_bits == 31
    assert L.table_label("0" * 128) == L.PROGRAM
    with pytest.raises(KeyError):
        L.table("gelu_tanh_bf16")


def test_definitions_are_registered_core_primitives():
    for fid in L.DEFINITIONS:
        assert fid in REGISTRY, fid
        assert L.definition_label(fid) == L.LIBRARY
    assert L.definition_label("MufuEx2Ftz_v1") == L.PROGRAM
    assert L.definition_label("AmpereBF16TcDot16_v1") == L.PROGRAM


def test_every_core_primitive_declares_only_library_tables():
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.ops import mufu  # noqa: F401  (registers the MUFU primitives)
    core = "verity.circuits."

    names = {t.name for t in L.TABLES}
    declared = {d.id: d.tables for d in REGISTRY.defs.values()
                if isinstance(d, PrimitiveDefinition) and d.evaluate.__module__.startswith(core) and d.tables}
    assert set(declared) >= {"MufuEx2Ftz_v1", "MufuRcpFtz_v1", "MufuTanh_v1"}
    for fid, tables in declared.items():
        assert all(kind == L.LIBRARY and n in names for n, kind in tables), fid
