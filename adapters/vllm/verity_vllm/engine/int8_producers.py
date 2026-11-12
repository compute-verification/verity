"""The int8 linears' producers fused with their quantizer: on each decoder layer that runs vLLM's Llama forwards
(`LlamaDecoderLayer`'s, `LlamaAttention`'s and `LlamaMLP`'s, which hand each of these producers' outputs straight to its
linear) and whose linears run the plain int8 arm (`int8_experts.PlainGemm`), the kernel that writes a linear's input also
writes its codes and scales, so the quantizer never reads x again:

    input_layernorm, with a residual (layers 1 on)  -> self_attn.qkv_proj    `int8_fused.add_norm_quant`
    post_attention_layernorm                        -> mlp.gate_up_proj      `int8_fused.add_norm_quant`
    mlp.act_fn                                      -> mlp.down_proj         `int8_fused.silu_mul_quant`

Each producer's forward (the instance's `CustomOp._forward_method`, patched through `engine.hooks`) writes the tensors vLLM's
op writes, bit for bit (`tests/engine/test_int8_producers.py`), and hands its codes and scales to the next
``verity_int8::row_quant`` (`int8_ops.hand_off`), whose CUDA body returns them instead of quantizing when its x is the
producer's output: the same storage, shape, strides and dtype.  The hand-off holds that output, so no other tensor takes
its memory, and the next ``row_quant`` of the device takes the hand-off either way.  Nothing writes a producer's output
before its linear quantizes it (those forwards pass it on directly, and a forward hook only reads it), so its bytes are
the ones the codes were made from.  ``row_quant`` is still called: the int8 source, a capture and ``observe`` see the same
calls with the same values.

A producer runs vLLM's own op instead when its call isn't the served one it reproduces: a dispatch mode or a compile is
active, a CUDA graph is capturing, the norm has no residual, vLLM's op is patched (the norm-scale tap replaces
``_C.fused_add_rms_norm``), a shape, dtype or layout differs, or the device's architecture has no measured win at the
row width and M (`int8_fused.producer_warps`; Llama-3.1-8B's silu at I 14,336 isn't faster fused).  Nothing is installed unless
the engine runs vLLM's batch-invariant CUDA forwards (the norm's sum is then its generic kernel's at min(H, 1024) threads,
`int8_fused.norm_threads`) and ``VERITY_INT8_FUSED_PRODUCERS`` is on.  o_proj's input (attention's output) and layer 0's
input norm (vLLM's Triton `rms_norm` with no residual) keep the separate quantizer.
"""
from __future__ import annotations

from typing import Any

import torch

from verity_vllm import settings
from verity_vllm.engine import hooks

ENV = "VERITY_INT8_FUSED_PRODUCERS"
TAG = "int8_producers"

def _served(*ts: torch.Tensor) -> bool:
    return (torch._C._len_torch_dispatch_stack() == 0 and not torch.compiler.is_compiling()
            and not torch.cuda.is_current_stream_capturing()
            and all(t.is_cuda and t.dtype == torch.bfloat16 and t.dim() == 2 and t.stride(1) == 1 for t in ts))


def _norm(norm: Any, original: Any) -> Any:
    from verity_vllm.engine import int8_fused as F
    from verity_vllm.engine import int8_ops

    def forward(x: torch.Tensor, residual: torch.Tensor | None = None):
        w = norm.weight.data
        if (residual is None or not _served(x, residual) or residual.shape != x.shape or not residual.is_contiguous()
                or not isinstance(torch.ops._C.fused_add_rms_norm, torch._ops.OpOverloadPacket)
                or F.norm_threads(x.shape[1]) is None
                or F.producer_warps("norm", x.shape[0], x.shape[1], F.arch(x.device.index)) is None
                or w.dtype != torch.bfloat16 or not w.is_contiguous()):
            return original(x, residual)
        xq = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        s = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        if x.shape[0]:
            F.add_norm_quant(x, residual, w, norm.variance_epsilon, xq, s)
        int8_ops.hand_off(x, xq, s)
        return x, residual
    return forward


def _act(act: Any, original: Any) -> Any:
    from verity_vllm.engine import int8_fused as F
    from verity_vllm.engine import int8_ops

    def forward(x: torch.Tensor):
        if (not _served(x) or not x.is_contiguous() or x.shape[1] % 2
                or F.producer_warps("silu", x.shape[0], x.shape[1] // 2, F.arch(x.device.index)) is None
                or not isinstance(torch.ops._C.silu_and_mul, torch._ops.OpOverloadPacket)
                or act.op is not torch.ops._C.silu_and_mul):
            return original(x)
        m, i = x.shape[0], x.shape[1] // 2
        h = torch.empty((m, i), dtype=x.dtype, device=x.device)
        xq = torch.empty((m, i), dtype=torch.int8, device=x.device)
        s = torch.empty(m, dtype=torch.float32, device=x.device)
        if m:
            F.silu_mul_quant(x, h, xq, s)
        int8_ops.hand_off(h, xq, s)
        return h
    return forward


def install(model: Any) -> dict | None:
    """Install the fused producers on `model`'s decoder layers that run vLLM's Llama forwards and whose consumer linears run
    the plain int8 arm; a record of what was installed, or None when nothing was (``VERITY_INT8_FUSED_PRODUCERS`` off, vLLM
    not batch invariant, no such layer).  The patches carry `TAG` (`hooks.live(TAG)`)."""
    if not settings.get(ENV):
        return None
    from vllm import envs
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP

    from verity_vllm.engine.int8_experts import PlainGemm
    if not envs.VLLM_BATCH_INVARIANT:
        return None
    h = hooks.Hooks(TAG)
    sites: list[tuple[Any, type, Any]] = []
    for _, layer in model.named_modules():
        attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
        if (type(layer).forward is not LlamaDecoderLayer.forward or type(attn).forward is not LlamaAttention.forward
                or type(mlp).forward is not LlamaMLP.forward):
            continue
        for producer, consumer, kind in ((getattr(layer, "input_layernorm", None), getattr(attn, "qkv_proj", None), RMSNorm),
                                         (getattr(layer, "post_attention_layernorm", None), getattr(mlp, "gate_up_proj", None),
                                          RMSNorm),
                                         (getattr(mlp, "act_fn", None), getattr(mlp, "down_proj", None), SiluAndMul)):
            if (type(producer) is kind and isinstance(getattr(consumer, "quant_method", None), PlainGemm)
                    and producer._forward_method == producer.forward_cuda
                    and (kind is SiluAndMul or (producer.variance_size_override is None and producer.has_weight))):
                sites.append((producer, kind, producer._forward_method))
    if not sites:
        return None
    for producer, kind, original in sites:
        h.patch(producer, "_forward_method", (_norm if kind is RMSNorm else _act)(producer, original))
    _INSTALLED.append(h)
    return {"norms": sum(kind is RMSNorm for _, kind, _ in sites), "acts": sum(kind is SiluAndMul for _, kind, _ in sites)}


def uninstall() -> None:
    """Put every producer's forward back and drop the hand-offs."""
    while _INSTALLED:
        _INSTALLED.pop().uninstall()
    from verity_vllm.engine import int8_ops
    int8_ops.drop_hand_offs()


_INSTALLED: list[hooks.Hooks] = []

__all__ = ["ENV", "TAG", "install", "uninstall"]
