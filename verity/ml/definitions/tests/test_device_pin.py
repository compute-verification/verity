"""The device pins (`registry/device_pins/`): the cc 9.0 device's probed cuBLAS routes are the eager Build's default routes off the
GLM-5.3 capture's shapes, a route is never read for an M the probe did not run, and a pin is found by its device's reported name."""
from __future__ import annotations

import pytest

from verity.ml.definitions.registry import bf16_gemm_check as BC


def test_the_probed_routes_are_the_default_routes_at_their_m_only():
    assert BC.default_cublas_route("linear", 256, 2048, 5120) == ("non_split", {})
    assert BC.default_cublas_route("linear", 1, 4096, 2048) == ("splitk", {"SPLIT_K": 3})
    assert BC.default_cublas_route("linear", 5, 2048, 5120) is None
    assert BC.default_cublas_route("router", 1, 2048, 128) is None


def test_the_capture_pins_come_first():
    assert BC.default_cublas_route("linear", 1, 6144, 160) == ("splitk", {"SPLIT_K": 4})


def test_a_pin_is_found_by_the_name_its_device_reports():
    pin = BC.device_pin(BC.capability_pin((9, 0))["device"])
    assert tuple(pin["compute_capability"]) == (9, 0) and pin["sms"] > 0
    with pytest.raises(KeyError):
        BC.device_pin("no such device")
    with pytest.raises(KeyError):
        BC.capability_pin((1, 0))
