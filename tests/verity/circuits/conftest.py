from __future__ import annotations

import gc

import pytest

import stand_in  # noqa: F401  (registers the suite's Boolean lookup)


@pytest.fixture
def collector():
    """The cyclic collector on for the test, and restored to how the test found it after."""
    was = gc.isenabled()
    gc.enable()
    yield
    (gc.enable if was else gc.disable)()
