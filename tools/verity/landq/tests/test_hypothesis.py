"""Soundness and liveness again, over scenarios Hypothesis draws field by field, so a counterexample shrinks to a few PRs and
tasks.  Derandomized and without a database: the same examples every run, and nothing written to the tree."""
from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

from harness import Scenario, Spec  # noqa: E402
from test_properties import _live, drive  # noqa: E402
from verity_landq import model as lq  # noqa: E402

PATHS = [f"p{i}/f.py" for i in range(5)]


@st.composite
def scenarios(draw) -> Scenario:
    tasks = {f"t{i}": frozenset(draw(st.sets(st.sampled_from(PATHS), min_size=1, max_size=3)))
             for i in range(draw(st.integers(1, 4)))}
    prs = []
    for n in range(1, draw(st.integers(1, 9)) + 1):
        paths = frozenset(draw(st.sets(st.sampled_from(PATHS), min_size=1, max_size=2)))
        touched = sorted(t for t, ins in tasks.items() if paths & ins)
        breaks = frozenset(draw(st.sets(st.sampled_from(touched), max_size=1))) if touched else frozenset()
        prs.append(Spec(n, paths, breaks, late_review=draw(st.booleans()), rehead=draw(st.booleans())))
    policy = lq.Policy(batch=draw(st.integers(1, 4)), depth=draw(st.integers(1, 3)))
    return Scenario(tasks, tuple(prs), policy, draw(st.integers(0, 2**32 - 1)), infra=draw(st.sampled_from([0.0, 0.3])))


@settings(derandomize=True, database=None, max_examples=300, deadline=None)
@given(scenarios())
def test_sound_and_live_on_drawn_scenarios(s: Scenario) -> None:
    r = drive(s)
    assert not r.unsound, r.unsound[:2]
    assert not _live(r, exact=False), _live(r, exact=False)[:3]
