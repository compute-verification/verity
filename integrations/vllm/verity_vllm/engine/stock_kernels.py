"""vLLM's own kernels at the pin, as typed baselines: what kernels and compute accounting time against.

Each function takes tensors, calls one vLLM kernel the way vLLM's serving path calls it, and names that call. Each imports
vLLM when called, so this module imports without it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["cutlass_scaled_mm", "flash_attn_varlen", "fp8_gemm_nt", "fp8_m_grouped_gemm_nt_masked",
           "m_grouped_fp8_gemm_nt_contiguous", "per_token_group_quant_fp8", "scaled_int8_quant", "triton_scaled_mm"]


def scaled_int8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's dynamic per-token symmetric int8 quantization (``vllm._custom_ops.scaled_int8_quant``): ``x`` in int8, and
    its fp32 scales, one a row (``[rows, 1]``)."""
    from vllm import _custom_ops as ops
    q, scales, _ = ops.scaled_int8_quant(x)
    return q, scales


def triton_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor,
                     out_dtype: torch.dtype | None = None, bias: torch.Tensor | None = None) -> torch.Tensor:
    """vLLM's Triton scaled GEMM with its own tile heuristic
    (``vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm.triton_scaled_mm``, which
    ``_custom_ops.cutlass_scaled_mm`` falls back to): ``(scale_a * a) @ (scale_b * b)`` for ``a`` ``[m, k]`` and ``b``
    ``[k, n]``, in ``out_dtype`` (bf16 unless given)."""
    import torch
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm as mm
    return mm(a, b, scale_a, scale_b, out_dtype or torch.bfloat16, bias)


def cutlass_scaled_mm(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor,
                      scale_b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """vLLM's CUTLASS scaled GEMM, the op itself (``torch.ops._C.cutlass_scaled_mm``, which importing
    ``vllm._custom_ops`` registers), written into ``out`` ``[m, n]`` and returned. Unlike ``_custom_ops.cutlass_scaled_mm``
    it checks no shape and never falls back to Triton. ``a`` is ``[m, k]`` row-major and ``b`` ``[k, n]`` column-major,
    int8 or FP8, with per-tensor, per-row and per-column, or 128-block scales."""
    import torch
    from vllm import _custom_ops  # noqa: F401
    torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)
    return out


def per_token_group_quant_fp8(x: torch.Tensor, group_size: int = 128, *, column_major_scales: bool = False,
                              tma_aligned_scales: bool = False,
                              use_ue8m0: bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's per-token-group FP8 quantization
    (``vllm.model_executor.layers.quantization.utils.fp8_utils.per_token_group_quant_fp8``): ``x`` in FP8 E4M3, and
    one fp32 scale for each row and ``group_size`` columns. ``use_ue8m0`` rounds the scales to powers of two, as
    DeepGEMM reads them; None leaves it to vLLM's choice for the device."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8 as quant
    return quant(x, group_size, column_major_scales=column_major_scales, tma_aligned_scales=tma_aligned_scales,
                 use_ue8m0=use_ue8m0)


def fp8_gemm_nt(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor], out: torch.Tensor) -> None:
    """DeepGEMM's FP8 GEMM as vLLM calls it (``vllm.utils.deep_gemm.fp8_gemm_nt``, with UE8M0 scales where vLLM uses
    them): ``out = a @ b.T`` into ``out`` ``[m, n]``, for ``a`` an FP8 ``[m, k]`` with its per-token-group scales and
    ``b`` an FP8 ``[n, k]`` with its 128 x 128 block scales."""
    from vllm.utils import deep_gemm
    deep_gemm.fp8_gemm_nt(a, b, out)


def m_grouped_fp8_gemm_nt_contiguous(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor],
                                     out: torch.Tensor, m_indices: torch.Tensor) -> None:
    """DeepGEMM's grouped FP8 GEMM over contiguous groups, as vLLM's MoE calls it
    (``vllm.utils.deep_gemm.m_grouped_fp8_gemm_nt_contiguous``): each row of ``a`` times the transpose of its group's
    ``b`` (``b`` ``[groups, n, k]``), the group of each row in ``m_indices`` (-1 for a padding row)."""
    from vllm.utils import deep_gemm
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, out, m_indices)


def fp8_m_grouped_gemm_nt_masked(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor],
                                 out: torch.Tensor, masked_m: torch.Tensor, expected_m: int) -> None:
    """DeepGEMM's grouped FP8 GEMM over masked groups (``vllm.utils.deep_gemm.fp8_m_grouped_gemm_nt_masked``): for
    each group ``g`` of ``a`` ``[groups, m, k]``, its first ``masked_m[g]`` rows times the transpose of ``b[g]``, with
    ``expected_m`` the typical count, which picks the tiles."""
    from vllm.utils import deep_gemm
    deep_gemm.fp8_m_grouped_gemm_nt_masked(a, b, out, masked_m, expected_m)


def flash_attn_varlen(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, out: torch.Tensor,
                      cu_seqlens_q: torch.Tensor, max_seqlen_q: int, seqused_k: torch.Tensor, max_seqlen_k: int,
                      block_table: torch.Tensor, fa_version: int, softmax_scale: float | None = None,
                      causal: bool = True, num_splits: int = 0) -> torch.Tensor:
    """vLLM's FlashAttention over a paged KV cache (``vllm.vllm_flash_attn.flash_attn_varlen_func``), as
    ``FlashAttentionImpl`` calls it for a prefill: ``q`` ``[tokens, heads, d]`` against the ``k`` and ``v`` caches'
    blocks that ``block_table`` names, written into ``out`` and returned. ``fa_version`` is 2 or 3, and ``softmax_scale``
    is ``d ** -0.5`` unless given."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func(q=q, k=k, v=v, out=out, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
                                  seqused_k=seqused_k, max_seqlen_k=max_seqlen_k, block_table=block_table,
                                  softmax_scale=q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale,
                                  causal=causal, fa_version=fa_version, num_splits=num_splits)
