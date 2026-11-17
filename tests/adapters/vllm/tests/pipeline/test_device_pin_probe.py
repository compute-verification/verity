"""`verity-vllm device-pin probe`: a device's pin is its probed routes at the like device's shapes and the route conditions read in
the probing process, which `bf16_gemm_check` then reads for that device; a condition not read or another device is refused."""
from __future__ import annotations

import json

import pytest

from verity_vllm.pipeline import device_pin as DP
from verity_vllm.acquire import served as SV
from verity_vllm.check.replay import driver
from verity_vllm.config import UsageError
from verity_vllm.definitions.registry import bf16_gemm_check as BC

H100 = "NVIDIA H100 80GB HBM3"
H100_FACTS = {**{k: v for k, v in BC.ROUTE_CONDITIONS.items() if k != "env"}, "device": H100, "cublas_version": 120900,
              "cublaslt_version": 120900, "env": {"CUBLAS_WORKSPACE_CONFIG": None, "CUBLASLT_WORKSPACE_SIZE": None}}


@pytest.fixture
def on_h100(monkeypatch):
    probed: list = []

    def probe_routes(shapes):
        probed.extend(shapes)
        return {"device": H100, "versions": {"torch": "2.13.0+cu129", "cuda": "12.9"},
                "routes": {f"{M},{K},{N}": {"kernel": "nosplit"} for M, K, N in shapes}}

    facts = {"cublas": dict(H100_FACTS)}
    monkeypatch.setattr(SV, "probe_routes", probe_routes)
    monkeypatch.setattr(driver, "engine_facts_of_record", lambda: facts)
    return probed, facts


def test_the_pin_is_the_routes_at_the_like_devices_shapes_and_the_conditions_read_here(on_h100, tmp_path, monkeypatch):
    probed, _ = on_h100
    out = tmp_path / "device_pins" / f"{H100.replace(' ', '_')}.json"
    out.parent.mkdir()
    assert DP.main(DP.ProbeOptions(out=str(out), evidence="r-probe")) == 0
    pin = json.loads(out.read_text())
    like = BC.device_pin("NVIDIA H200")["cublas"]
    assert [f"{M},{K},{N}" for M, K, N in probed] == list(like) == list(pin["cublas"])
    assert (pin["device"], pin["compute_capability"], pin["sms"], pin["evidence"]) == (H100, [9, 0], 132, {"cublas": "r-probe"})
    assert pin["conditions"] == {k: H100_FACTS[k] for k in BC.ROUTE_CONDITIONS}
    monkeypatch.setattr(BC, "DEVICE_PINS", out.parent)
    BC.device_pin.cache_clear()
    try:
        assert BC.default_cublas_route("linear", 1, 4096, 2048, H100) == ("non_split", {})
        assert BC.route_departures({"cublas": H100_FACTS}) == []
    finally:
        BC.device_pin.cache_clear()


def test_a_condition_the_process_could_not_read_is_refused(on_h100):
    _, facts = on_h100
    facts["cublas"]["torch"] = "unavailable: ImportError"
    with pytest.raises(UsageError, match="route conditions not read in this process: {'torch'"):
        DP.probe("NVIDIA H200", "r-probe")


def test_routes_and_conditions_of_two_devices_are_refused(on_h100):
    _, facts = on_h100
    facts["cublas"]["device"] = "NVIDIA H200"
    with pytest.raises(UsageError, match="probed on 'NVIDIA H100 80GB HBM3', the conditions read on 'NVIDIA H200'"):
        DP.probe("NVIDIA H200", "r-probe")
