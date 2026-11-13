"""`engine.arrivals`' read of the scheduler's requests gives None, never raises, on a snapshot it can't read: the arrivals loop
then counts forwards by the lag rule instead."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from verity_vllm.engine.arrivals import _snapshot_computed


@pytest.mark.parametrize("sched, want", [
    (NS(requests={"a": NS(num_computed_tokens=3), 7: NS(num_computed_tokens="4"), "c": NS()}), {"a": 3, "7": 4}),
    (NS(requests={"a": NS(num_computed_tokens=None)}), {}),
    (NS(), None),
    (NS(requests=[NS(num_computed_tokens=1)]), None),
    (NS(requests={"a": NS(num_computed_tokens="n/a")}), None),
    (NS(requests={"a": NS(num_computed_tokens=[1])}), None),
])
def test_snapshot_computed_or_none(sched, want):
    assert _snapshot_computed(sched) == want
