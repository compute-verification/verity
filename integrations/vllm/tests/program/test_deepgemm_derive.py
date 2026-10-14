"""The default sm_90 block-FP8 linear binds the custom op's two served branches."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from verity_vllm.program.frontend.export_compat import default_fp8_linear_leaf
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules import vllm_bindings as VB
from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset
from verity_vllm.program.frontend.rules.vllm_iface import iface
from verity_vllm.program.frontend.target_profile import TargetProfile

K, N = 128, 2
HOPPER = TargetProfile(compute_capability=(9, 0), num_sms=132, batch_invariant=False)


class _Linear(nn.Module):
    def __init__(self, *, group: int = 128, ue8m0: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(N, K, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.empty(1, 1, dtype=torch.float32), requires_grad=False)
        self.group, self.ue8m0 = group, ue8m0

    def forward(self, x):
        return iface.dynamic_flashinfer_deepgemm_blockscale_gemm(
            x, self.weight, self.weight_scale, self.group, self.ue8m0
        )


@pytest.fixture(autouse=True)
def _target():
    saved = dict(VB._OBSERVED)
    VB.observe_target_profile(HOPPER)
    yield
    VB._OBSERVED.clear()
    VB._OBSERVED.update(saved)


def _derive(m: int, **kw):
    return derive_report(
        _Linear(**kw),
        (torch.empty(m, K, dtype=torch.bfloat16),),
        mode="profile",
        profile="b1-eager-v3",
        name="DefaultFp8Linear",
        rules=vllm_ruleset(),
    )


@pytest.mark.parametrize(
    ("m", "branch", "definition"),
    [(1, "flashinfer", "FlashInferFp8Linear_v1"), (31, "flashinfer", "FlashInferFp8Linear_v1"),
     (32, "deepgemm", "DeepGemmFp8Linear_v1")],
)
def test_default_sm90_linear_binds_the_branch_selected_by_m(m: int, branch: str, definition: str) -> None:
    rep = _derive(m)
    assert rep.unsupported is None
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "DynamicDeepGemmFp8Linear[vllm-default]_v1"]
    assert len(apps) == 1
    assert apps[0]["preconditions_checked"] == {
        "M": m, "K": K, "N": N, "G": 128, "branch": branch, "use_deep_gemm_e8m0": True
    }
    assert apps[0]["emitted"] == f"{definition}{{K={K},N={N}}} x{m}"
    assert rep.provenance["weight_roles"] == {
        "weight": {"rule": "DynamicDeepGemmFp8Linear[vllm-default]_v1", "role": "fp8_w_requant_ue8m0", "K": K, "N": N},
        "weight_scale": {"rule": "DynamicDeepGemmFp8Linear[vllm-default]_v1", "role": "fp8_ws_requant_ue8m0",
                         "K": K, "N": N, "G": 128},
    }


@pytest.mark.parametrize(("kw", "reason"), [
    ({"group": 64}, "expected group_size=128"),
    ({"ue8m0": False}, "use_deep_gemm_e8m0=True"),
])
def test_default_sm90_linear_refuses_another_kernel_configuration(kw: dict, reason: str) -> None:
    rep = _derive(4, **kw)
    assert rep.program is None and rep.unsupported is not None
    assert reason in rep.unsupported.hint


def test_default_kernel_export_keeps_the_dynamic_custom_op_as_a_leaf() -> None:
    kernel_type = type("FlashInferFp8DeepGEMMDynamicBlockScaledKernel", (), {})
    model = nn.Module()
    model.quant_method = SimpleNamespace(fp8_linear=kernel_type())
    original = object()
    site = SimpleNamespace(dynamic_flashinfer_deepgemm_blockscale_gemm=original)

    with default_fp8_linear_leaf(model, namespace=site):
        assert site.dynamic_flashinfer_deepgemm_blockscale_gemm is iface.dynamic_flashinfer_deepgemm_blockscale_gemm
    assert site.dynamic_flashinfer_deepgemm_blockscale_gemm is original

    model.quant_method.fp8_linear = object()
    with default_fp8_linear_leaf(model, namespace=site):
        assert site.dynamic_flashinfer_deepgemm_blockscale_gemm is original
