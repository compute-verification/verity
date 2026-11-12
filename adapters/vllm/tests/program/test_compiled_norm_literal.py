"""compiled_norm_literal: the literal Triton transcription of Inductor's RMSNorm kernels is the compiled workloads' norm relation of record.

GPU tests (skipped without CUDA + Triton): on 2048-wide rows (Llama: N a power of two, so torch's IEEE division and Triton's div.full.f32
agree) the literal at the default executed config must equal the torch model's pinned order `r0=2048 warps=8 chunk=8 sep` bit for bit
(both match the real generated kernels on 12.6 M elements each); on 1536-wide rows (Qwen) the literal must be
a variant of `relation()` and be labelled with the executed/assumed config.  CPU: no CUDA -> no literal variant, torch variants stand."""
from __future__ import annotations

import pytest
import torch

from verity_vllm.program import compiled_norm_literal as lit
from verity_vllm.program.compiled_relations import relation

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="literal Triton kernels need CUDA")


def _heavy(M, n, sc, g, device):
    z = torch.randn(M, n, device=device, generator=g)
    t = z / torch.sqrt(torch.rand(M, n, device=device, generator=g).clamp_min(0.02))
    return (t * sc).to(torch.bfloat16)


def test_cpu_inputs_yield_no_literal_variant_and_the_torch_variants_stand():
    g = torch.Generator().manual_seed(0)
    x, r = _heavy(4, 256, 0.7, g, "cpu"), _heavy(4, 256, 0.5, g, "cpu")
    w = torch.ones(256, dtype=torch.bfloat16)
    assert lit.literal_variants("add1_norm", "triton_red_fused_fused_add_rms_norm_t_0", [x, r, w], 1e-5) == {}
    rel = relation("triton_red_fused_fused_add_rms_norm_t_0", [x, r, w], [{"shape": [4, 256]}], eps=1e-5)
    assert rel["kind"] == "add1_norm" and rel["variants"] and not any(k.startswith("triton-literal") for k in rel["variants"])


def test_executed_configs_is_empty_for_an_unknown_kernel():
    assert lit.executed_configs("no_such_kernel_" + "x" * 8) == []


@cuda
def test_literal_equals_the_pinned_torch_order_on_power_of_two_rows_and_is_the_first_variant():
    if lit._build() is None:
        pytest.skip("Triton unavailable")
    g = torch.Generator(device="cuda").manual_seed(7)
    M, N = 512, 2048
    x, r = _heavy(M, N, 0.7, g, "cuda"), _heavy(M, N, 0.5, g, "cuda")
    w = (torch.randn(N, device="cuda", generator=g) * 0.8 + 1.0).to(torch.bfloat16)
    rel = relation("triton_red_fused_fused_add_rms_norm_t_0", [x, r, w], [{"shape": [M, N]}], eps=1e-5)
    labels = list(rel["variants"])
    assert labels[0].startswith("triton-literal XBLOCK=2 R0_BLOCK=2048 warps=16"), labels[0]
    lit_out = rel["variants"][labels[0]][0]
    pinned = rel["variants"]["r0=2048 warps=8 chunk=8 sep"][0]
    assert lit_out.shape == (M, N) and lit_out.dtype == torch.bfloat16
    assert torch.equal(lit_out, pinned)
    # add2 form: two outputs (the bf16 residual store and the normed rows), both from the literal
    r2 = _heavy(M, N, 0.4, g, "cuda")
    rel2 = relation("triton_red_fused_fused_add_rms_norm_t_2", [x, r, r2, w], [{"shape": [M, N]}, {"shape": [M, N]}], eps=1e-5)
    l2 = next(k for k in rel2["variants"] if k.startswith("triton-literal"))
    assert len(rel2["variants"][l2]) == 2 and torch.equal(rel2["variants"][l2][0], (x.float() + (r.float() + r2.float())).to(torch.bfloat16))


@cuda
def test_literal_runs_on_qwen_width_with_the_masked_tail_and_names_its_config():
    if lit._build() is None:
        pytest.skip("Triton unavailable")
    g = torch.Generator(device="cuda").manual_seed(3)
    M, N = 55, 1536
    x, r = _heavy(M, N, 0.7, g, "cuda"), _heavy(M, N, 0.5, g, "cuda")
    w = (torch.randn(N, device="cuda", generator=g) * 0.8 + 1.0).to(torch.bfloat16)
    v = lit.literal_variants("add1_norm", "triton_red_fused_fused_add_rms_norm_t_0", [x, r, w], 1e-6)
    assert len(v) == 1
    (label, outs), = v.items()
    assert label == "triton-literal XBLOCK=2 R0_BLOCK=2048 warps=16 (assumed)"
    o = outs[0]
    assert o.shape == (M, N) and torch.isfinite(o.float()).all()
    # a bf16-ulp neighbourhood of the plain f32 formula (the literal differs from torch only by rounding of the division / rsqrt / order)
    s = (x.float() + r.float())
    ref = ((s * torch.rsqrt((s * s).sum(-1, keepdim=True) / N + 1e-6)) * w.float()).to(torch.bfloat16)
    assert ((o.float() - ref.float()).abs() <= 2 * (ref.float().abs() * 2 ** -7 + 1e-6)).all()
