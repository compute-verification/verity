"""`trace` runs its builder with the cyclic collector off, and leaves the collector as it found it: on after a trace that
returns or raises, off after one its caller ran with it off."""
from __future__ import annotations

import gc

import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.types import Array

PARAMS, RET = [Array(2, BL.BIT)], Array(1, BL.BIT)


@pytest.fixture
def collector():
    was = gc.isenabled()
    gc.enable()
    yield
    (gc.enable if was else gc.disable)()


def test_a_trace_runs_with_the_collector_off_and_leaves_it_as_it_was(collector) -> None:
    seen = []

    def build(C, x):
        seen.append(gc.isenabled())
        return x[:1]

    T = TR.trace(build, PARAMS, RET)
    assert (seen, gc.isenabled(), T.n_inputs, T.gates, T.outputs) == ([False], True, 2, [], [2])
    gc.disable()
    TR.trace(build, PARAMS, RET)
    assert (seen, gc.isenabled()) == ([False, False], False)


def test_a_trace_that_raises_turns_the_collector_back_on(collector) -> None:
    def build(C, x):
        raise RuntimeError("the builder failed")

    with pytest.raises(RuntimeError, match="the builder failed"):
        TR.trace(build, PARAMS, RET)
    assert gc.isenabled()
