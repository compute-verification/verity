"""Pearl-C's census: the vectorized chained atom is verity.ml.circuits.silicon's HOPPER_E4M3_K32, and a generic tile has nothing to skip."""

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


def test_sm120_atom_matches_verity():
    C.set_atom("sm120")
    try:
        assert C.selftest(400, "sm120") == 0
    finally:
        C.set_atom("hopper")


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


def test_debit_flags_joint_skips_inside_a_group():
    import numpy as np
    a, b = np.zeros((1, 128), dtype=np.uint8), np.zeros((1, 128), dtype=np.uint8)
    a[0, 0], b[0, 0] = 0x38, 0x38            # 1 * 1
    a[0, 32], b[0, 32] = 0x30, 0x38          # 0.5 * 1
    a[0, 64:66], b[0, 64:66] = [0x7E, 0xFE], [0x7E, 0x7E]   # 448^2 - 448^2 sets the grid to 2^3 and truncates to +0
    a[0, 96], b[0, 96] = 0x38, 0x38          # 1 * 1
    flags, eF = C.debit(a, b, 4)
    # No atom is an identity, but skipping {0, 1}, {0, 1, 2} or {1, 2, 3} jointly leaves the group's word 1: all four flagged.
    assert eF.tolist() == [0] and flags.tolist() == [[True, True, True, True]]


def test_aligned_spikes_pass_the_rules_and_freeze_words():
    g4 = [r for r in C.cell("aligned-spikes-r300", 1024, 1.0, 4, 2, run=1, forming="s5", line_norm=16.0) if r["layout"] == "G4"]
    assert g4[0]["rules_ok"] and g4[0]["word_identity"] > 0.5


def test_pm1_background_skips_whole_atoms_and_the_debit_covers_them():
    rows = [r for r in C.cell("aligned-spikes-pm1-r444", 1024, 1.0, 4, 2, run=1, forming="s5", line_norm=16.0, with_debit=True)
            if r["layout"] == "G4"]
    assert rows[0]["rules_ok"] and rows[0]["tile_identity_atoms"] > 0
    assert rows[0]["debit_share"] >= rows[0]["tile_identity_atoms"] / rows[0]["atoms"]


def test_cross_group_finds_cancelling_groups():
    import numpy as np
    a, b = np.zeros((1, 256), dtype=np.uint8), np.zeros((1, 256), dtype=np.uint8)
    a[0, 0:128:32], b[0, 0:128:32] = 0x38, 0x38      # +1 in each atom of group 0
    a[0, 128:256:32], b[0, 128:256:32] = 0xB8, 0x38  # -1 in each atom of group 1: the output is 0 (joint_cross_group)
    flags, _ = C.debit(a, b, 4)
    size, beyond = C.cross_group(a, b, flags, np.array([0]), 4)
    assert not flags.any() and size[0] == 8 and beyond[0] == 8


def test_transfer_test_runs_and_reports():
    rows = [r for r in C.cell("outliers-groupstart", 1024, 1.0, 4, 2, run=1, forming="s5", line_norm=16.0, with_debit=True,
                              cross_words=2, transfer_words=2) if r["layout"] == "G4"]
    r = rows[0]
    assert r["transfer_words"] == 2 and 0 <= r["transfer_hits"] <= r["transfer_nonempty_sim_sets"] <= 2
