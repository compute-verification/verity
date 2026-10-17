"""The exact-int8 attention inside the pinned vLLM: the attention backend in vLLM's `CUSTOM` slot
(`TargetProfile.INT8_EXACT_ATTENTION_BACKEND`).

`register()` fills `AttentionBackendEnum.CUSTOM` with `Int8AttentionBackend` (vLLM's own `register_backend`), in this process;
`engine.build` calls it before `LLM()` when the target's `int8_exact` covers attention, whose engine args then select
`attention_backend = "CUSTOM"`.  The engine core is in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), so the selector sees it.

The backend is vLLM's FlashAttention backend in all but the forward: its validation, metadata builder and paged KV cache (layout,
block sizes, `reshape_and_cache_flash` writing K after RoPE and V as bf16).  The cache stays bf16 because the step quantizes from
bf16 on every window: K's per-token scales could be stored with an int8 K (a cache spec of its own, more than a backend), but V's
per-channel scale reads the whole window, so an int8 V would have to be requantized as the window grows.

`Int8AttentionImpl.forward` runs `engine.int8_attention.paged_attention` on the cache, with its layer's K scales (the served
checkpoint's `engine.int8_k_scales` table, looked up by the model's revision when the impl is built and by the layer's name in each
call), and counts its calls (`int8_arm` and `calls`, read back by `kernel_readback.int8_exact_of` and `int8_attention.applied_record`).  It refuses what the step does not define:
ALiBi, a sliding window, a logit soft cap, sinks, non-decoder attention, a quantized KV cache, cascade attention, decode context
parallelism, non-causal metadata, fused output quantization, and queries that are not bf16.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path

import torch
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from verity_vllm.engine import int8_attention as A
from verity_vllm.engine import int8_k_scales as K
from verity_vllm.program.frontend.target_profile import TargetProfile

NAME = TargetProfile.INT8_EXACT_ATTENTION_BACKEND


class Int8AttentionImpl(FlashAttentionImpl):
    """One attention layer on `engine.int8_attention` (FlashAttention's construction, cache and metadata)."""

    int8_arm = A.ARM

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0
        bad = [why for why, hit in (("ALiBi", self.alibi_slopes is not None), ("a sliding window", self.sliding_window != (-1, -1)),
                                    ("a logit soft cap", bool(self.logits_soft_cap)), ("sinks", self.sinks is not None),
                                    (f"attention type {self.attn_type}", self.attn_type != AttentionType.DECODER),
                                    (f"KV cache {self.kv_cache_dtype}", is_quantized_kv_cache(self.kv_cache_dtype))) if hit]
        if bad:
            raise ValueError(f"int8 attention does not define {', '.join(bad)}")
        self.k_table = torch.from_numpy(K.table(served_revision()))
        if self.k_table.shape[1] != self.num_kv_heads:
            raise ValueError(f"int8 attention: K scales for {self.k_table.shape[1]} KV heads, the layer has {self.num_kv_heads}")
        self._k_on: dict[tuple[str, int], torch.Tensor] = {}

    def k_scales(self, layer_name: str, device: torch.device) -> torch.Tensor:
        """The layer's K scales [G] (float32) on `device`."""
        key = (str(device), K.layer_index(layer_name))
        if key not in self._k_on:
            self._k_on[key] = self.k_table[key[1]].to(device)
        return self._k_on[key]

    def forward(self, layer: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor,
                attn_metadata, output: torch.Tensor, output_scale: torch.Tensor | None = None,
                output_block_scale: torch.Tensor | None = None) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("int8 attention: fused output quantization")
        if attn_metadata is None:
            return output.fill_(0)
        if attn_metadata.use_cascade or getattr(self, "dcp_world_size", 1) > 1 or attn_metadata.causal is not True:
            raise ValueError("int8 attention: cascade attention, decode context parallelism and non-causal metadata are refused")
        if query.dtype != torch.bfloat16:
            raise ValueError(f"int8 attention: queries are {query.dtype}, the step quantizes bf16")
        n = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        self.calls += 1
        A.paged_attention(query[:n], key_cache, value_cache, attn_metadata.query_start_loc, attn_metadata.seq_lens,
                          attn_metadata.block_table, self.k_scales(layer.layer_name, query.device), self.scale, output[:n])
        return output


def served_revision() -> str:
    """The revision of the model vLLM is building: its model config's `revision`, else the snapshot directory's name."""
    from vllm.config import get_current_vllm_config
    mc = get_current_vllm_config().model_config
    return mc.revision or Path(mc.model).name


class Int8AttentionBackend(FlashAttentionBackend):
    """FlashAttention's backend with `Int8AttentionImpl` in the `CUSTOM` slot."""

    supported_kv_cache_dtypes = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return NAME

    @staticmethod
    def get_impl_cls() -> type[Int8AttentionImpl]:
        return Int8AttentionImpl


@cache
def register() -> str:
    """Put the backend in this process's `AttentionBackendEnum.CUSTOM` (once); returns its name."""
    register_backend(AttentionBackendEnum[NAME], f"{Int8AttentionBackend.__module__}.{Int8AttentionBackend.__qualname__}")
    return NAME
