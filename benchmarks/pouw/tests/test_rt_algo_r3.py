"""The round-3 algebra guard: J10's shared-noise forms against H1 + fx1's after-coin work.

`rt_algo_r3/rt_algo_r3_check.py` builds a 64 × 64 call's units from injected noise with `segment.call_words`, as
`h1tile.reference` does (and checks that equality on one unit), runs each form's cheapest after-coin route from the AB
running partials bit-exact against those words and receipts, and checks the route counts: J10-R (a stream a call row and
a call column) keeps the record's products and the op count at the honest unit's, every shared or structured form loses
the record's products, forms structured on both sides lose the op count, and window-seeded columns lose the record's
whole count. It would fail if a change to the unit's words broke the decomposition the routes use, or if J10-R's noise
lost rank."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

CHECK = Path(__file__).resolve().parents[1] / "rt_algo_r3" / "rt_algo_r3_check.py"
_spec = importlib.util.spec_from_file_location("rt_algo_r3_check", CHECK)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod  # its dataclasses resolve their annotations through the module
_spec.loader.exec_module(_mod)


def test_round_three_routes_hold():
    passed, failed = _mod.collect(quick=True)
    assert not failed, f"round-3 shared-noise algebra no longer holds: {failed}"
    assert len(passed) == len(_mod.CHECKS)
