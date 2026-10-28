"""The int8 linears' producers fused with their quantizer: on each Llama decoder layer whose linears run the plain int8 arm
(`int8_experts.PlainGemm`), the kernel that writes a linear's input also writes its codes and scales, so the quantizer
never reads x again:

    input_layernorm, with a residual (layers 1 on)  -> self_attn.qkv_proj    `int8_fused.add_norm_quant`
    post_attention_layernorm                        -> mlp.gate_up_proj      `int8_fused.add_norm_quant`
    mlp.act_fn                                      -> mlp.down_proj         `int8_fused.silu_mul_quant`

Each producer's forward (the instance's `CustomOp._forward_method`, patched through `engine.hooks`) writes the tensors vLLM's
op writes, bit for bit (`tests/engine/test_int8_producers.py`), and hands its codes and scales to the next
``verity_int8::row_quant`` (`take`), whose CUDA body returns them instead of quantizing when its x is the producer's output:
the same storage, shape, strides and dtype.  The hand-off holds that output, so no other tensor takes its memory, and the
next ``row_quant`` of the device takes the hand-off either way.  Nothing writes a producer's output before its linear
quantizes it (a forward hook only reads it), so its bytes are the ones the codes were made from.  The op still runs: the
int8 source, a capture and ``observe`` see the same calls with the same values.

A producer runs vLLM's own op instead when its call isn't the served one it reproduces: a dispatch mode or a compile is
active, a CUDA graph is capturing, the norm has no residual, vLLM's op is patched (the norm-scale tap replaces
``_C.fused_add_rms_norm``), or a shape, dtype or layout differs.  Nothing is installed unless the engine runs vLLM's
batch-invariant CUDA forwards (the norm's sum is then its generic kernel's at min(H, 1024) threads, `int8_fused.norm_threads`)
and ``VERITY_INT8_FUSED_PRODUCERS`` is on.  o_proj's input (attention's output) and layer 0's input norm (vLLM's Triton
`rms_norm` with no residual) keep the separate quantizer.
"""
from __future__ import annotations

from typing import Any

import torch

from verity_vllm import settings
from verity_vllm.engine import hooks

ENV = "VERITY_INT8_FUSED_PRODUCERS"
TAG = "int8_producers"
#: the decoder layers whose forward this reads (each producer's output goes straight to its linear)
LAYERS = ("LlamaDecoderLayer",)

_HANDOFF: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def take(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The codes and scales a producer handed off for x (its output, or a view of it of the same layout), else None; the
    device's hand-off is gone after this call either way."""
    got = _HANDOFF.pop(x.device, None)
    if got is None:
        return None
    src, xq, s = got
    if x.data_ptr() == src.data_ptr() and x.shape == src.shape and x.stride() == src.stride() and x.dtype == src.dtype:
        return xq, s
    return None


def _served(*ts: torch.Tensor) -> bool:
    return (torch._C._len_torch_dispatch_stack() == 0 and not torch.compiler.is_compiling()
            and not torch.cuda.is_current_stream_capturing()
            and all(t.is_cuda and t.dtype == torch.bfloat16 and t.dim() == 2 and t.stride(1) == 1 for t in ts))


def _norm(norm: Any, original: Any) -> Any:
    from verity_vllm.engine import int8_fused as F

    def forward(x: torch.Tensor, residual: torch.Tensor | None = None):
        w = norm.weight.data
        if (residual is None or not _served(x, residual) or residual.shape != x.shape or not residual.is_contiguous()
                or not isinstance(torch.ops._C.fused_add_rms_norm, torch._ops.OpOverloadPacket)
                or F.norm_threads(x.shape[1]) is None or w.dtype != torch.bfloat16 or not w.is_contiguous()):
            return original(x, residual)
        xq = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        s = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        if x.shape[0]:
            F.add_norm_quant(x, residual, w, norm.variance_epsilon, xq, s)
        _HANDOFF[x.device] = (x, xq, s)
        return x, residual
    return forward


def _act(act: Any, original: Any) -> Any:
    from verity_vllm.engine import int8_fused as F

    def forward(x: torch.Tensor):
        if (not _served(x) or not x.is_contiguous() or x.shape[1] % 2
                or not isinstance(torch.ops._C.silu_and_mul, torch._ops.OpOverloadPacket)
                or act.op is not torch.ops._C.silu_and_mul):
            return original(x)
        m, i = x.shape[0], x.shape[1] // 2
        h = torch.empty((m, i), dtype=x.dtype, device=x.device)
        xq = torch.empty((m, i), dtype=torch.int8, device=x.device)
        s = torch.empty(m, dtype=torch.float32, device=x.device)
        if m:
            F.silu_mul_quant(x, h, xq, s)
        _HANDOFF[h.device] = (h, xq, s)
        return h
    return forward


def install(model: Any) -> dict | None:
    """Install the fused producers on `model`'s decoder layers (`LAYERS`) whose consumer linears run the plain int8 arm; a
    record of what was installed, or None when nothing was (``VERITY_INT8_FUSED_PRODUCERS`` off, vLLM not batch invariant,
    no such layer).  The patches carry `TAG` (`hooks.live(TAG)`)."""
    if not settings.get(ENV):
        return None
    from vllm import envs
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm

    from verity_vllm.engine.int8_experts import PlainGemm
    if not envs.VLLM_BATCH_INVARIANT:
        return None
    h = hooks.Hooks(TAG)
    sites: list[tuple[Any, Any, Any]] = []
    for _, layer in model.named_modules():
        if type(layer).__name__ not in LAYERS:
            continue
        attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
        for producer, consumer, kind in ((getattr(layer, "input_layernorm", None), getattr(attn, "qkv_proj", None), RMSNorm),
                                         (getattr(layer, "post_attention_layernorm", None), getattr(mlp, "gate_up_proj", None),
                                          RMSNorm),
                                         (getattr(mlp, "act_fn", None), getattr(mlp, "down_proj", None), SiluAndMul)):
            if (type(producer) is kind and isinstance(getattr(consumer, "quant_method", None), PlainGemm)
                    and producer._forward_method == producer.forward_cuda
                    and (kind is SiluAndMul or (producer.variance_size_override is None and producer.has_weight))):
                sites.append((producer, kind, producer._forward_method))
    for producer, kind, original in sites:
        h.patch(producer, "_forward_method", (_norm if kind is RMSNorm else _act)(producer, original))
    if not sites:
        return None
    return {"norms": sum(kind is RMSNorm for _, kind, _ in sites), "acts": sum(kind is SiluAndMul for _, kind, _ in sites)}


__all__ = ["ENV", "LAYERS", "TAG", "install", "take"]
