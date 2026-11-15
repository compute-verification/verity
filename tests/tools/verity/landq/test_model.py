"""The shared vocabulary: how a change's paths touch a task's inputs, and the values it refuses by name."""
from __future__ import annotations

import pytest

from verity_landq import model as lq


def test_touches_reads_an_input_as_a_file_a_directory_the_tree_or_a_glob() -> None:
    assert lq.touches(frozenset({"a/b/c.py"}), frozenset({"a/b"}))
    assert lq.touches(frozenset({"a/b/c.py"}), frozenset({"a/b/"}))
    assert lq.touches(frozenset({"a/b/c.py"}), frozenset({"a/b/c.py"}))
    assert not lq.touches(frozenset({"a/bc/d.py"}), frozenset({"a/b"}))
    assert not lq.touches(frozenset({"a/b/c.py"}), frozenset({"a/b/c.pyi"}))
    assert lq.touches(frozenset({"x.md"}), frozenset({"."}))
    assert not lq.touches(frozenset(), frozenset({"."}))
    assert lq.touches(frozenset({"S/P/A.lean"}), frozenset({"S/**/*.lean"}))
    assert lq.touches(frozenset({"S/A.lean"}), frozenset({"S/*.lean"}))
    assert not lq.touches(frozenset({"S/A.py"}), frozenset({"S/**/*.lean"}))


def test_bad_policies_and_results_are_refused_by_name() -> None:
    with pytest.raises(ValueError, match="Policy.batch"):
        lq.Policy(batch=0)
    with pytest.raises(ValueError, match="Policy.depth"):
        lq.Policy(depth=0)
    with pytest.raises(ValueError, match="Result.status"):
        lq.Result("T1", "unit:a", "error")
