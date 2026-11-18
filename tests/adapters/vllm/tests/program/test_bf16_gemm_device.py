"""The bf16 GEMM rules read the pin of the device the Build answers for: a device with no pin is refused by name, an error out
of the rule's `applies` that ends the Build, never a reason that lets another rule or another card's pin bind the launch."""
from __future__ import annotations

import pytest

from verity_vllm.definitions.registry import bf16_gemm_check as BC
from verity_vllm.program.frontend.rules.vllm_bindings import bf16_gemm
from verity_vllm.program.frontend.rules.vllm_bindings.observations import clear_observations, observe_device_name

#: a captured f32-output launch: the scan's split-K route at M 17
LAUNCH = (17, 6144, 256)


@pytest.fixture(autouse=True)
def _clean():
    clear_observations()
    yield
    clear_observations()


def test_the_h200_and_a_build_naming_no_device_bind_the_captured_route():
    clear_observations()
    unnamed = bf16_gemm.f32out_route(*LAUNCH)
    observe_device_name(BC.ROUTE_CONDITIONS["device"])
    assert bf16_gemm.f32out_route(*LAUNCH) == unnamed == ("router", "cublas_splitk_gemm_f32out", {"SPLIT_K": 4})


@pytest.mark.parametrize("device", ["NVIDIA H100 80GB HBM3", "NVIDIA B300"])
def test_a_device_with_no_pin_is_refused_by_name(device):
    observe_device_name(device)
    with pytest.raises(KeyError, match=f"no device pin {device.replace(' ', '_')}.json in device_pins/"):
        bf16_gemm.f32out_route(*LAUNCH)
