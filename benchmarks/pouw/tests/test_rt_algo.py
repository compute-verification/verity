"""The floor-tightness guard: no algebraic route produces H1 Dither's credited words below the floor.

`rt_algo/rt_algo_check.py` checks, bit-exact against `h1tile.reference`, that the served AB cannot be reused (the
credited words are the running prefix partials, not the final product), that U's linear structure is already the digit
words, and that the fold is non-additive. It would fail if a change made a credited word peelable from the served output
."""
from __future__ import annotations

import importlib.util
from pathlib import Path

CHECK = Path(__file__).resolve().parents[1] / "rt_algo" / "rt_algo_check.py"
_spec = importlib.util.spec_from_file_location("rt_algo_check", CHECK)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_no_algebraic_route_beats_the_floor():
    passed, failed = _mod.collect()
    assert not failed, f"credited-word algebra no longer holds: {failed}"
    assert len(passed) == 9
