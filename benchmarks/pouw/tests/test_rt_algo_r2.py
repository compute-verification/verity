"""RT-ALGO round 2's guard: the credited-word algebra of attention, the FP-native two-coin design and the served prefix layout.

`rt_algo_r2/rt_algo_r2_check.py` checks, bit-exact against main's `h1tile` and `segment`, that the served receipt's final
words (T_(KS-1), O', D_d) are order-invariant while the prefix partials are not, that H1 has no attention unit, that a
near-zero P row's residue collapse saves nothing a running-partial word, and that a two-coin entry is a single square mod 2^m
iff v₂(c) >= ⌈(m + 1)/2⌉. It would fail if a change made a credited word final-only, gave H1 an attention unit, or moved the
two-coin threshold (note:20261019T0618Z-draft-rt-algo-r2)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

CHECK = Path(__file__).resolve().parents[1] / "rt_algo_r2" / "rt_algo_r2_check.py"
_spec = importlib.util.spec_from_file_location("rt_algo_r2_check", CHECK)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_round_two_algebra_holds():
    passed, failed = _mod.collect()
    assert not failed, f"round-2 credited-word algebra no longer holds: {failed}"
    assert len(passed) == 10
