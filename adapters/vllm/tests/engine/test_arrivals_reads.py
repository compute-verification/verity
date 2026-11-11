"""`engine.arrivals`' scheduler reads give None, never raise, on an engine that doesn't expose the in-process V1 scheduler and on
a snapshot they can't read: the arrivals loop then counts forwards by the lag rule instead."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from verity_vllm.engine.arrivals import _snapshot_computed, scheduler_of


def test_scheduler_of_reads_the_in_process_scheduler_or_none():
    sched = NS(requests={})
    assert scheduler_of(NS(llm_engine=NS(engine_core=NS(scheduler=sched)))) is sched
    assert scheduler_of(NS(llm_engine=NS(engine_core=NS(engine_core=NS(scheduler=sched))))) is sched
    assert scheduler_of(NS()) is None
    assert scheduler_of(NS(llm_engine=None)) is None
    assert scheduler_of(NS(llm_engine=NS(engine_core=NS(scheduler=NS(requests=[]))))) is None


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
