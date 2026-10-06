"""`cli.parallel`, the worker pool behind `circuit-check --jobs`, on its own: `test_circuit_check` builds every target at
collection, which this needs none of."""
from __future__ import annotations

import os
from concurrent.futures.process import BrokenProcessPool

import pytest

from circuit_check import cli


def test_every_item_is_yielded_once():
    assert sorted(cli.parallel(abs, [-1, -2, -3], jobs=2)) == [1, 2, 3]


def test_a_worker_that_dies_fails_the_run_instead_of_hanging_it():
    with pytest.raises(BrokenProcessPool):
        list(cli.parallel(os._exit, [3, 3], jobs=2))
