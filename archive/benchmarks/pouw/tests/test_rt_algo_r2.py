"""The round-2 algebra guard: the credited-word algebra of attention, FP-native design A and the compiled H1 unit's
prefix layout.

`rt_algo_r2/rt_algo_r2_check.py` checks, bit-exact against main's `h1tile` and `segment`, that the compiled unit's final
words (T_(KS-1), O', D_d) are order-invariant while T_0 .. T_(KS-2) are not, that the unit's gates checkpoint every set
with T_s extending T_(s-1), that H1 has no attention unit, that a near-zero P row's residue collapse saves nothing a
running-partial word, that a two-coin entry is a single square mod 2^m iff v₂(c) >= ⌈(m + 1)/2⌉, that design A's grids
meet `AdmWrap` at s32, and that a shift pairs design A's entries. It would fail if a change made another prefix word
final-only, gave H1 an attention unit, or moved a grid's valuations or the two-coin threshold
."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parents[1] / "rt_algo_r2" / "rt_algo_r2_check.py"
_spec = importlib.util.spec_from_file_location("rt_algo_r2_check", CHECK)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_round_two_algebra_holds():
    passed, failed = _mod.collect(gates=False)
    assert not failed, f"round-2 credited-word algebra no longer holds: {failed}"
    assert len(passed) == 12


@pytest.mark.slow
def test_compiled_unit_gates_are_cp_w():
    """c5 reads the unit's gates rather than `reference`, so it catches a body that drops a set's checkpoint or computes a
    prefix word from anything but the previous partial, which the reference-level checks can't see. Slow: building the
    unit's body (its seeds' SHA-512 in gates) takes about 30 s."""
    ok, detail = _mod.check_compiled_unit_is_cp_w()
    assert ok, detail
