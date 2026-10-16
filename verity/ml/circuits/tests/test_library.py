"""The library list (`verity.ml.circuits.library`): the measured MUFU tables and core's tensor-core steps and casts are `library`,
everything else `program`; the list's digest is pinned, so an edit without a version bump fails here."""
from __future__ import annotations

import re

import pytest

from verity.ml.circuits.defs import REGISTRY
from verity.ml.circuits import library as L
from verity.ml.circuits.ops import prims  # noqa: F401  (registers the listed Definitions)

#: `digest()` of VERSION 2. A change to the list needs a new VERSION, then this pin.
DIGEST_V2 = ("353600e311538d244317fbb43634d934a24edd7594029bf33e537f64bd3c0c77"
             "e8cbd4527dfa3a0503f781c64b1e6396957c06c66718178d70359c6dff8a65ba")


def test_tables_are_library_by_their_sha512():
    for t in L.TABLES:
        assert L.table_label(t.sha512) == L.LIBRARY
        assert L.table(t.sha512) is t and L.table(t.name) is t
        assert re.fullmatch(r"[0-9a-f]{128}", t.sha512) and re.fullmatch(r"[0-9a-f]{64}", t.sha256)
        assert t.words == 1 << t.index_bits and t.value_bits == 31
    assert L.table_label("0" * 128) == L.PROGRAM
    with pytest.raises(KeyError):
        L.table("gelu_tanh_bf16")


def test_the_measured_mufu_tables_are_listed():
    assert {t.name: t.index_bits for t in L.TABLES} == {"rsq": 24, "sqrt": 24, "rcp": 23, "ex2": 23, "tanh_mufu": 27, "lg2": 23}


def test_definitions_are_registered_core_primitives():
    for fid in L.DEFINITIONS:
        assert fid in REGISTRY, fid
        assert L.definition_label(fid) == L.LIBRARY
    assert L.definition_label("MufuEx2Ftz_v1") == L.PROGRAM
    assert L.definition_label("AmpereBF16TcDot16_v1") == L.PROGRAM


def test_the_digest_is_pinned():
    assert L.digest() == DIGEST_V2
    assert L.spec()["version"] == L.VERSION == 2


def test_every_core_primitive_declares_only_library_tables():
    from verity.ml.circuits.defs import PrimitiveDefinition
    from verity.ml.circuits.ops import mufu  # noqa: F401  (registers the MUFU primitives)

    names = {t.name for t in L.TABLES}
    declared = {d.id: d.tables for d in REGISTRY.defs.values()
                if isinstance(d, PrimitiveDefinition) and d.evaluate.__module__.startswith(("verity.", "verity.ml.circuits.")) and d.tables}
    assert set(declared) >= {"MufuEx2Ftz_v1", "MufuRcpFtz_v1", "MufuTanh_v1"}
    for fid, tables in declared.items():
        assert all(kind == L.LIBRARY and n in names for n, kind in tables), fid
