"""The exact-int8 linears inside the pinned vLLM: quantization method `TargetProfile.INT8_EXACT_QUANTIZATION`.

`register()` registers `Int8ExactConfig` under that name with vLLM's own `register_quantization_config` (and its linear method
with `register_weight_loader_v2_supported_method`), in this process; `engine.build` calls it before `LLM()` when the target
declares `int8_exact`.  The engine core is in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), so the registration reaches the
model loader.  The checkpoint is `engine.int8_checkpoint`'s: `quantization_config.quant_method` names the method, every linear
weight is int8 [N, K] with a float32 `weight_scale` [N, 1].

Every linear layer (`LinearBase`) gets an `Int8PlainLinearMethod` with its own prefix; nothing else is quantized (the embedding
and `lm_head` are vocabulary embeddings, attention keeps its KV cache unquantized).  The method computes
`engine.int8_exact.plain_linear` and counts its calls, so `int8_exact.applied_record` shows which linear ran what.  Loading
refuses a weight outside [-127, 127] and a scale that is not finite and positive; a layer refuses tensor parallelism (a row-parallel
shard would take each row's activation scale over its own K slice, not the row's), a bias, K or N not a multiple of 8, K > MAX_K, and
activations that are not bf16.
"""
from __future__ import annotations

from functools import cache
from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import ChannelQuantScaleParameter, ModelWeightParameter

from verity_vllm.engine import int8_exact as E
from verity_vllm.program.frontend.target_profile import TargetProfile

NAME = TargetProfile.INT8_EXACT_QUANTIZATION


class Int8PlainLinearMethod(LinearMethodBase):
    """One linear layer's plain arm (`engine.int8_exact`)."""

    int8_arm = E.ARM

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.calls = 0

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs: Any) -> None:
        n, k = sum(output_partition_sizes), input_size_per_partition
        if getattr(layer, "tp_size", 1) != 1:
            raise ValueError(f"{self.prefix}: the int8 plain arm is single-GPU (tp_size {layer.tp_size})")
        if n % 8 or k % 8 or k > E.MAX_K:
            raise ValueError(f"{self.prefix}: [{n}, {k}] needs N and K multiples of 8 and K <= {E.MAX_K}")
        loader = extra_weight_attrs["weight_loader"]
        layer.register_parameter("weight", ModelWeightParameter(data=torch.empty(n, k, dtype=torch.int8), input_dim=1, output_dim=0,
                                                                weight_loader=loader))
        layer.register_parameter("weight_scale", ChannelQuantScaleParameter(data=torch.empty(n, 1, dtype=torch.float32), output_dim=0,
                                                                            weight_loader=loader))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w, s = layer.weight.data, layer.weight_scale.data.reshape(-1).contiguous()
        if int(w.to(torch.int16).abs().max()) > E.W_MAX:
            raise ValueError(f"{self.prefix}: an int8 weight outside [-{E.W_MAX}, {E.W_MAX}]")
        if not bool((torch.isfinite(s) & (s > 0)).all()):
            raise ValueError(f"{self.prefix}: a weight scale that is not finite and positive")
        layer.weight = torch.nn.Parameter(w, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if bias is not None:
            raise ValueError(f"{self.prefix}: the int8 plain arm has no bias")
        if x.dtype != torch.bfloat16:
            raise ValueError(f"{self.prefix}: activations are {x.dtype}, the plain arm quantizes bf16")
        self.calls += 1
        return E.plain_linear(x, layer.weight, layer.weight_scale)


class Int8ExactConfig(QuantizationConfig):
    """`quantization_config` {quant_method: NAME, weight_max: 127}: every linear layer on `Int8PlainLinearMethod`."""

    def __init__(self, weight_max: int):
        super().__init__()
        if weight_max != E.W_MAX:
            raise ValueError(f"{NAME}: weight_max {weight_max!r} is not {E.W_MAX}")

    def get_name(self) -> str:
        return NAME

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Int8ExactConfig":
        return cls(int(cls.get_from_keys(config, ["weight_max"])))

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Int8PlainLinearMethod | None:
        return Int8PlainLinearMethod(prefix) if isinstance(layer, LinearBase) else None


@cache
def register() -> str:
    """Register the method with this process's vLLM (once); returns its name."""
    register_weight_loader_v2_supported_method(Int8PlainLinearMethod)
    register_quantization_config(NAME)(Int8ExactConfig)
    return NAME
