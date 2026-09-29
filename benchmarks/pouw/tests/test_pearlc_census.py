"""Pearl-C's census: the vectorized chained atom is verity.ml.tc's HOPPER_E4M3_K32, and a generic tile has nothing to skip."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("pouw_pearlc_census", Path(__file__).parents[1] / "pearlc_census.py")
C = importlib.util.module_from_spec(spec)
sys.modules["pouw_pearlc_census"] = C
spec.loader.exec_module(C)


def test_chained_atom_matches_verity():
    assert C.selftest(400) == 0


def test_generic_tile_has_nothing_to_skip():
    for row in C.cell("gaussian", 1024, 1.0, 8, 4, run=1):
        assert row["in_domain"] and row["skippable_share"] == 0.0


def test_domain_rule():
    assert C.in_domain(4, 65536, 1.0) and C.in_domain(0, 32768, 1.0)
    assert not C.in_domain(0, 65536, 1.0) and not C.in_domain(0, 16384, 0.5)
