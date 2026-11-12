"""Where a Pearl-C pass's seconds go (`pearl_c.parts`): one part current at a time on the thread that runs the pass, what
other threads spend landed behind it, and a window's summary of its passes.  The clock is a count of its readings, so a
part's seconds are how many times its thread left it."""
from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor

import pytest

from verity_vllm.protocols.pouw.pearl_c.parts import Parts, summary


@pytest.fixture(autouse=True)
def _ticks(monkeypatch):
    ticks = itertools.count()
    monkeypatch.setattr(Parts, "clock", staticmethod(lambda: float(next(ticks))))


def test_a_threads_seconds_go_to_one_part_at_a_time():
    """A call inside a forward takes its seconds out of the forward's and hands the forward back; a part entered twice
    adds up; nothing is charged while no part is current, so the parts sum to the readings between the first switch and
    the last, less those gaps."""
    parts = Parts()
    assert parts.switch("forward") is None and parts.seconds == {}
    for _ in range(2):
        was = parts.switch("stage")
        assert was == "forward" and parts.switch("launch") == "stage"
        inner = parts.switch("retain")
        assert parts.switch("y_tree") == "retain" and parts.switch("retain") == "y_tree" and inner == "launch"
        assert parts.switch(was) == "retain"
    assert parts.switch(None) == "forward"
    assert parts.seconds == {"forward": 3.0, "stage": 2.0, "launch": 2.0, "retain": 4.0, "y_tree": 2.0}
    assert parts.switch("next") is None and parts.switch(None) == "next"
    assert parts.seconds["next"] == 1.0 and sum(parts.seconds.values()) == 14.0 and Parts.clock() == 16.0


def test_what_other_threads_spend_lands_behind_the_pass():
    """`Parts.behind` ends the other thread's parts and adds each as ``behind_<part>``, exactly, from any number of
    threads at once, beside the pass's own parts and in none of them."""
    parts = Parts()
    parts.switch("forward")
    own = Parts()
    own.switch("copies")
    own.switch("writes")
    parts.behind(own)
    assert own.part is None and own.seconds == {"copies": 1.0, "writes": 1.0}
    assert parts.seconds == {"behind_copies": 1.0, "behind_writes": 1.0} and parts.part == "forward"

    def writer(_):
        for _ in range(250):
            landed = Parts()
            landed.seconds = {"copies": 0.25, "writes": 0.5}
            parts.behind(landed)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(writer, range(8)))
    assert parts.seconds == {"behind_copies": 501.0, "behind_writes": 1001.0}


def test_a_windows_summary_is_its_first_pass_and_the_later_ones_summed():
    first = {"forward": 0.4, "weights": 0.25, "behind_writes": 1.5}
    later = [{"forward": 0.125, "y_tree": 0.0000004}, {"forward": 0.25, "behind_writes": 2.0}]
    assert summary([]) == {"count": 0}
    assert summary([first]) == {"count": 1, "first": first}
    assert summary([first, *later]) == {"count": 3, "first": first,
                                        "later": {"forward": 0.375, "y_tree": 0.0, "behind_writes": 2.0}}
