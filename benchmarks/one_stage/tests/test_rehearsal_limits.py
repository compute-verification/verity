"""`rehearsal_limits.py`'s record form's pair, from a made-up class's numbers alone (no circuit, no Lean)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("one_stage_rehearsal_limits", HERE.parent / "rehearsal_limits.py")
RL = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = RL
_path = sys.path[:]
_spec.loader.exec_module(RL)  # it puts the tree's roots ahead on sys.path for its own run, not this suite's
sys.path[:] = _path


class Refused(Exception):
    pass


def test_the_held_pair_is_a_block_a_slot_and_refused_where_the_slots_do_not_divide_the_profiles_gamma():
    """Four slots, k = 6 and a profile's gamma of 100: a block a slot has 25 gates, and alpha is 4 × 6 fresh bits over
    100 gates, 6/25 in lowest terms. Three slots don't divide 100, so there is no held pair."""
    assert RL.held(4, 6, 100, Refused) == {"alpha": [6, 25], "gamma": 25}
    with pytest.raises(Refused, match="the class's 3 slots don't divide the profile's gamma 100"):
        RL.held(3, 6, 100, Refused)
