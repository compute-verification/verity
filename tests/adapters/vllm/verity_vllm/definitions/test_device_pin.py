"""The device pins (`adapters/vllm/verity_vllm/definitions/registry/device_pins/`): the cc 9.0 device's probed cuBLAS routes are the eager Build's default routes off the
GLM-5.3 capture's shapes, a route is never read for an M the probe did not run, a pin is found by its device's reported name, a named
device reads its own pin only, and a process is held to the route conditions of the device its Build was made for."""
from __future__ import annotations

import json
import shutil

import pytest

from verity_vllm.definitions.registry import bf16_gemm_check as BC


def test_the_probed_routes_are_the_default_routes_at_their_m_only():
    assert BC.default_cublas_route("linear", 256, 2048, 5120) == ("non_split", {})
    assert BC.default_cublas_route("linear", 1, 4096, 2048) == ("splitk", {"SPLIT_K": 3})
    assert BC.default_cublas_route("linear", 5, 2048, 5120) is None
    assert BC.default_cublas_route("router", 1, 2048, 128) is None


def test_the_capture_pins_come_first():
    assert BC.default_cublas_route("linear", 1, 6144, 160) == ("splitk", {"SPLIT_K": 4})


def test_a_pin_is_found_by_the_name_its_device_reports():
    pin = BC.device_pin(BC.ROUTE_CONDITIONS["device"])
    assert pin["device"] == BC.ROUTE_CONDITIONS["device"] and tuple(pin["compute_capability"]) == (9, 0) and pin["sms"] > 0
    with pytest.raises(KeyError):
        BC.device_pin("no such device")


H100 = "NVIDIA H100 80GB HBM3"


@pytest.fixture
def h100_pin(tmp_path, monkeypatch):
    """The pins with a cc 9.0 H100 pin beside the H200's, its routes and conditions differing from the H200's."""
    pins = tmp_path / "device_pins"
    shutil.copytree(BC.DEVICE_PINS, pins)
    h200 = BC.device_pin(BC.ROUTE_CONDITIONS["device"])
    conditions = {**BC.ROUTE_CONDITIONS, "device": H100, "cublas_version": 120900}
    pin = {**h200, "device": H100, "conditions": conditions, "cublas": {"1,4096,2048": {"kernel": "nosplit"}}}
    (pins / f"{H100.replace(' ', '_')}.json").write_text(json.dumps(pin))
    monkeypatch.setattr(BC, "DEVICE_PINS", pins)
    BC.device_pin.cache_clear()
    yield pin
    BC.device_pin.cache_clear()


def _todays_route(pin: dict, phase: str, M: int, K: int, N: int) -> tuple[str, dict[str, int]] | None:
    """`default_cublas_route` as it reads before a Build names its device: the capture's route, else the cc 9.0 pin's."""
    run = BC._route_run(phase, M, K, N)
    if run is not None:
        return ("splitk", {"SPLIT_K": run[2]}) if run[2] is not None else ("non_split", {})
    if phase == "linear" and (r := pin.get(f"{M},{K},{N}")) is not None:
        return ("splitk", {"SPLIT_K": r["SPLIT_K"]}) if r["kernel"] == "splitk" else ("non_split", {})
    return None


def _every_recorded_launch(pin: dict):
    for phase, K, N in BC._ROUTES:
        for M in range(1, BC.CUBLAS_ROUTES["m_max"] + 1):
            yield phase, M, K, N
    for key in pin:
        M, K, N = map(int, key.split(","))
        for phase in ("linear", "router", "f32out"):
            yield phase, M, K, N


def _assert_todays(pin: dict) -> int:
    n = 0
    for launch in _every_recorded_launch(pin):
        want = _todays_route(pin, *launch)
        assert BC.default_cublas_route(*launch) == want == BC.default_cublas_route(*launch, BC.ROUTE_CONDITIONS["device"]), launch
        n += want is not None
    assert BC.pinned_routes() == BC.pinned_routes(BC.ROUTE_CONDITIONS["device"]) == pin
    for device in (None, BC.ROUTE_CONDITIONS["device"]):
        assert BC.route_pin_of(device) == (BC.CUBLAS_ROUTES["run_record"], BC.ROUTE_CONDITIONS)
    return n


def test_the_h200_and_a_build_naming_no_device_read_todays_route_at_every_recorded_launch():
    """Every M up to the scan's m_max at every captured shape, and every pinned shape: the route an H200 or a device-less
    Build binds, and the route conditions its process is held to, are one and the same, and those of the scan and of the scan
    device's pin, the one cc 9.0 pin a device-less Build read before it named its device."""
    pin = BC.device_pin(BC.ROUTE_CONDITIONS["device"])["cublas"]
    assert _assert_todays(pin) > len(pin)


def test_another_cc_9_0_pin_moves_neither(h100_pin):
    _assert_todays(BC.device_pin(BC.ROUTE_CONDITIONS["device"])["cublas"])


def test_a_named_device_reads_its_own_pin_and_never_the_scan(h100_pin):
    assert BC.default_cublas_route("linear", 1, 4096, 2048, H100) == ("non_split", {})
    assert BC.default_cublas_route("linear", 1, 6144, 160, H100) is None
    for phase, M, K, N in (("linear", 1, 4096, 2048), ("router", 17, 6144, 256), ("f32out", 1, 7168, 1024)):
        with pytest.raises(KeyError, match="no device pin NVIDIA_B300"):
            BC.default_cublas_route(phase, M, K, N, "NVIDIA B300")


def _engine(**cublas):
    env = cublas.pop("env", BC.ROUTE_CONDITIONS["env"])
    return {"cublas": {**{k: v for k, v in BC.ROUTE_CONDITIONS.items() if k != "env"}, "env": dict(env), **cublas}}


def test_a_process_is_held_to_the_route_conditions_of_the_device_its_build_was_made_for(h100_pin):
    h100 = h100_pin["conditions"]
    assert BC.route_departures(_engine()) == BC.route_departures(_engine(), BC.ROUTE_CONDITIONS["device"]) == []
    assert BC.route_departures(_engine(**h100), H100) == []
    assert BC.route_departures(_engine(device=H100)) == [f"device {H100!r} (pin: 'NVIDIA H200')"]
    assert BC.route_departures(_engine(**h100)) == [f"device {H100!r} (pin: 'NVIDIA H200')", "cublas_version 120900 (pin: 120901)"]
    assert BC.route_departures(_engine(), H100) == [f"device 'NVIDIA H200' (pin: {H100!r})", "cublas_version 120901 (pin: 120900)"]


def test_a_build_for_a_device_with_no_measured_conditions_departs_by_name(h100_pin):
    [why] = BC.route_departures(_engine(device="NVIDIA B300"), "NVIDIA B300")
    assert "built for 'NVIDIA B300', on which no route pin is measured" in why and "verity-vllm device-pin probe" in why
    del h100_pin["conditions"]
    (BC.DEVICE_PINS / f"{H100.replace(' ', '_')}.json").write_text(json.dumps(h100_pin))
    BC.device_pin.cache_clear()
    [why] = BC.route_departures(_engine(device=H100), H100)
    assert "on which no route pin is measured" in why
