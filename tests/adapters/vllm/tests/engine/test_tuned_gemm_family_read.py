"""The tuned-GEMM arch family a GPU record carries: `verity_vllm.engine.engine_profile._tuned_gemm_arch_family` asks vLLM's own
function for a device's compute capability, and `gpu_info` records the answer.  vLLM is the stand-in of ``tests/fake_vllm.py``
and CUDA's device properties are faked, so no GPU and no vLLM is needed."""
from __future__ import annotations

import logging
import sys
from types import SimpleNamespace as NS
from typing import Any

import pytest

from tests import fake_vllm
from verity_vllm.engine import engine_profile as EP

RULE = "vllm.model_executor.determinism.batch_invariant_configs:_get_tuned_matmul_arch_family"
CAPABILITY = "vllm.platforms.interface:DeviceCapability"


class _Capability:
    def __init__(self, major: Any, minor: Any) -> None:
        self.major, self.minor = major, minor


@pytest.fixture
def vllm(monkeypatch: pytest.MonkeyPatch) -> fake_vllm.FakeVllm:
    fake = fake_vllm.install(monkeypatch)
    fake.set(CAPABILITY, _Capability)
    return fake


def test_the_family_is_vllms_answer_for_the_capability_as_two_ints(vllm: fake_vllm.FakeVllm) -> None:
    asked: list[tuple[Any, Any]] = []

    def rule(device: _Capability) -> str | None:
        asked.append((device.major, device.minor))
        return {(8, 9): "ada", (9, 0): "hopper"}.get((device.major, device.minor))

    vllm.set(RULE, rule)
    assert EP._tuned_gemm_arch_family(8, 9) == "ada"
    assert EP._tuned_gemm_arch_family(9, 0) == "hopper"
    assert EP._tuned_gemm_arch_family(8, 0) is None, "no table for the device is vLLM's None, as vLLM gives it"
    as_text: Any = "8"
    as_float: Any = 9.0
    assert EP._tuned_gemm_arch_family(as_text, as_float) == "ada"
    assert asked == [(8, 9), (9, 0), (8, 0), (8, 9)] and all(type(v) is int for pair in asked for v in pair)


def test_a_vllm_that_cannot_say_leaves_the_field_unavailable_and_says_why_at_debug(
        vllm: fake_vllm.FakeVllm, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def fails(device: _Capability) -> str:
        raise KeyError("no such device")

    no_number: Any = "eight"
    with caplog.at_level(logging.DEBUG, logger="verity_vllm"):
        vllm.set(RULE, fails)
        assert EP._tuned_gemm_arch_family(8, 9) == "<unavailable>"
        vllm.set(RULE, lambda device: "ada")
        assert EP._tuned_gemm_arch_family(no_number, 9) == "<unavailable>", "a capability that is no number"
        monkeypatch.setitem(sys.modules, "vllm.model_executor.determinism.batch_invariant_configs", None)
        assert EP._tuned_gemm_arch_family(8, 9) == "<unavailable>", "another vLLM layout: no such module"
    mine = [r for r in caplog.records if r.name == "verity_vllm.engine.engine_profile"]
    said = [r.getMessage() for r in mine]
    assert len(said) == 3 and {r.levelno for r in mine} == {logging.DEBUG}
    assert "capability 8.9" in said[0] and "KeyError('no such device')" in said[0]
    assert "capability 'eight'.9" in said[1] and "ValueError" in said[1]
    assert "capability 8.9" in said[2] and "ModuleNotFoundError" in said[2]


def test_a_gpu_record_carries_the_family_of_its_first_device(vllm: fake_vllm.FakeVllm, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    vllm.set(RULE, lambda device: f"family-of-{device.major}.{device.minor}")
    asked: list[int] = []

    def properties(index: int) -> NS:
        asked.append(index)
        return NS(name="F", major=9, minor=0, total_memory=3 << 30, multi_processor_count=132)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(EP, "_run", lambda argv: "550.1")
    doc = EP.gpu_info()
    assert asked == [0]
    assert doc["tuned_gemm_arch_family"] == "family-of-9.0" and doc["compute_capability"] == "9.0"
    assert {k: doc[k] for k in ("name", "count", "driver", "total_memory_gb", "multi_processor_count")} == {
        "name": "F", "count": 2, "driver": "550.1", "total_memory_gb": 3.0, "multi_processor_count": 132}
    assert list(doc) == ["name", "count", "driver", "compute_capability", "pod", "total_memory_gb", "multi_processor_count",
                         "tuned_gemm_arch_family"]
