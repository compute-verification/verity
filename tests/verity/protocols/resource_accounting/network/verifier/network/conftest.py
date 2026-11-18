from __future__ import annotations

import pytest

from tests.verity.circuits import stand_in


@pytest.fixture(autouse=True)
def _hand_primitive_bodies(request):
    """The inference-only scripts' hand primitives' stand-in bodies for the inference-only tests, which `period_case`'s
    lookup, registered as it loads, would refuse."""
    if not request.module.__name__.rpartition(".")[2].startswith("test_inference_only_"):
        yield
        return
    with stand_in.registered():
        yield
