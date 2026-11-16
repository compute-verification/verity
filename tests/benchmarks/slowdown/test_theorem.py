import math

import pytest
from slowdown.theorem import multiple_bound


def test_the_overhead_is_the_hardware_multiple_at_the_references_mfu_floored_at_1():
    """The unregistered leaves train like the reference and the registered share converts at `a`: 1 under reference
    efficiency, `1/MFU₀` without it. The slowdown is at least 1."""
    s, x = 0.01 / 2.0, 0.99 * 0.01 / 2.0
    assert multiple_bound(0.4, s, x) == pytest.approx(1 / (s + x))
    assert multiple_bound(0.4, s, x, reference_efficiency=False) == pytest.approx(1 / (s + x / 0.4))
    assert multiple_bound(0.4, 0.0, 1.0) == 1.0 and multiple_bound(0.4, 0.0, 0.0) == math.inf
    assert multiple_bound(0.4, 0.0, 0.5) == 2.0 and multiple_bound(0.4, 0.0, 0.5, reference_efficiency=False) == 1.0
