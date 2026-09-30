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


def test_s5_forming_rules():
    rows = {r["family"]: r for fam in ("gaussian", "zero-row") for r in C.cell(fam, 1024, 1.0, 4, 2, run=1, forming="s5")}
    assert rows["gaussian"]["rules_ok"] and rows["gaussian"]["collision_A"] < 0.5
    assert not rows["zero-row"]["rules_ok"] and "noise-floor-A" in rows["zero-row"]["rejected_by"]


def test_debit_counts_atoms_skippable_alone():
    import numpy as np
    a, b = np.zeros((1, 128), dtype=np.uint8), np.zeros((1, 128), dtype=np.uint8)
    a[0, 0], b[0, 0] = 0x38, 0x38            # 1 * 1
    a[0, 32], b[0, 32] = 0x30, 0x38          # 0.5 * 1
    a[0, 64:66], b[0, 64:66] = [0x7E, 0xFE], [0x7E, 0x7E]   # 448^2 - 448^2 sets the grid to 2^3 and truncates to +0
    a[0, 96], b[0, 96] = 0x38, 0x38          # 1 * 1
    eF, extra = C.debit(a, b, 4)
    assert eF.tolist() == [0] and extra.tolist() == [2]


def test_aligned_spikes_pass_the_rules_and_freeze_words():
    g4 = [r for r in C.cell("aligned-spikes-r300", 1024, 1.0, 4, 2, run=1, forming="s5", line_norm=16.0) if r["layout"] == "G4"]
    assert g4[0]["rules_ok"] and g4[0]["word_identity"] > 0.5
