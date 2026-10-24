"""The exact-int8 linears inside the pinned vLLM: quantization method `TargetProfile.INT8_EXACT_QUANTIZATION`.

`register()` registers `Int8ExactConfig` under that name with vLLM's own `register_quantization_config` (and its linear method
with `register_weight_loader_v2_supported_method`), in this process; `engine.build` calls it before `LLM()` when the target
declares `int8_exact`.  The engine core is in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), so the registration reaches the
model loader.  The checkpoint is `engine.int8_checkpoint`'s: `quantization_config.quant_method` names the method, every linear
weight is int8 [N, K] with a float32 `weight_scale` [N, 1].

Every linear layer (`LinearBase`) gets an `Int8PlainLinearMethod` with its own prefix, but one `modules_to_not_convert` names,
which stays bf16 (`Int8KeptLinearMethod`), and so does the LM head (`ParallelLMHead`) when the list doesn't name `lm_head`
(`[]`, recipe v2; v1's `["lm_head"]` keeps it bf16): the logits processor calls the method's `apply` and takes its bf16 logits.  A MoE layer's routed experts (`RoutedExperts`) get an `Int8ExpertsMethod`: every
expert GEMM, gate_up and down, is the same plain call on that expert's routed rows (`engine.int8_experts`, which states the
routing, the activation and the combine), and its shared experts are linear layers like any other.  The router is not quantized
(the model builds it with no quantization config, and it stays bf16, as the LM head does under v1).  Nothing else is quantized
(the embedding is a gather, attention keeps its KV cache unquantized).  The method computes
`engine.int8_exact.plain_linear` as a ``verity_int8::row_quant`` then a ``verity_int8::scaled_mm`` call (`engine.int8_ops.linear`) and counts its calls, so `int8_exact.applied_record` shows which linear ran what.  After
both ops it hands the call's served values to `Int8PlainLinearMethod.observe`, a no-op a protocol option wraps (PoUW's H1,
`protocol_options.pouw_h1`): the layer it was handed, the s8 rows and per-token scales (what the Commit taps as
``Int8RowQuant_v1``) and y.  With ``zy_rows`` set (H1's T, which the option sets on the layers it commits) the product is
``verity_int8::scaled_mm_zy`` over that many slots instead (`int8_ops.linear_zy`): the same y, and from the same accumulator
the call's z/y records, which ``observe`` gets as its z; a call of more rows than T serves through ``scaled_mm``.  Loading
refuses a weight outside [-127, 127] and a scale that is not finite and positive; a layer refuses a bias, K or N (its shard's) not a
multiple of 8, K > MAX_K, and activations that are not bf16.

Under tensor parallelism each rank runs the plain step on its shard, as vLLM's own linear methods do: a column-parallel shard (qkv,
gate_up) is its rows of W and of ``weight_scale``, the whole activation row; a row-parallel shard (o_proj, down_proj) is its K slice
of W with every row's ``weight_scale``, and quantizes its K slice of the activation, so its activation scale is that slice's max,
not the whole row's.  The ranks' bf16 outputs are then summed by vLLM's all-reduce.  The LM head's shard is its vocabulary rows, as
a column-parallel one, and the logits processor gathers them.
"""
from __future__ import annotations

from functools import cache
from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, FusedMoeWeightScaleSupported, RoutedExperts
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase, UnquantizedLinearMethod,
                                               register_weight_loader_v2_supported_method)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_experts as X
from verity_vllm.program.frontend.target_profile import TargetProfile

NAME = TargetProfile.INT8_EXACT_QUANTIZATION


class Int8PlainLinearMethod(X.PlainGemm, LinearMethodBase):
    """One linear layer's plain arm (`engine.int8_exact`): `int8_experts.PlainGemm`'s ``apply`` and ``observe``."""

    def __init__(self, prefix: str):
        X.PlainGemm.__init__(self, prefix)

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs: Any) -> None:
        n, k = sum(output_partition_sizes), input_size_per_partition
        if n % 8 or k % 8 or k > E.MAX_K:
            raise ValueError(f"{self.prefix}: [{n}, {k}] needs N and K multiples of 8 and K <= {E.MAX_K}")
        loader = extra_weight_attrs["weight_loader"]
        if isinstance(layer, ParallelLMHead):
            # the head loads through `VocabParallelEmbedding.weight_loader(param, w)`, which reads `output_dim` off plain Parameters
            for name, t in (("weight", torch.empty(n, k, dtype=torch.int8)), ("weight_scale", torch.empty(n, 1, dtype=torch.float32))):
                p = torch.nn.Parameter(t, requires_grad=False)
                set_weight_attrs(p, {"output_dim": 0, "weight_loader": loader, **({"input_dim": 1} if name == "weight" else {})})
                layer.register_parameter(name, p)
            return
        layer.register_parameter("weight", ModelWeightParameter(data=torch.empty(n, k, dtype=torch.int8), input_dim=1, output_dim=0,
                                                                weight_loader=loader))
        layer.register_parameter("weight_scale", ChannelQuantScaleParameter(data=torch.empty(n, 1, dtype=torch.float32), output_dim=0,
                                                                            weight_loader=loader))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w, s = layer.weight.data, layer.weight_scale.data.contiguous()
        _checked(self.prefix, w, s)
        layer.weight = torch.nn.Parameter(w, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)


def _checked(prefix: str, w: torch.Tensor, s: torch.Tensor) -> None:
    if int(w.to(torch.int16).abs().max()) > E.W_MAX:
        raise ValueError(f"{prefix}: an int8 weight outside [-{E.W_MAX}, {E.W_MAX}]")
    if not bool((torch.isfinite(s) & (s > 0)).all()):
        raise ValueError(f"{prefix}: a weight scale that is not finite and positive")


class Int8ExpertsMethod(FusedMoEMethodBase):
    """One MoE layer's routed experts (`RoutedExperts`) on the plain arm: `int8_experts.forward` over its sites
    (`int8_experts.sites`, made after loading), every expert GEMM an `int8_experts.ExpertGemm`.  It refuses what that
    forward does not define: an activation other than SiLU-and-mul, expert biases, router weights applied on the input, and
    experts not all on this rank (expert parallelism)."""

    int8_arm = E.ARM

    def __init__(self, moe: Any, prefix: str):
        super().__init__(moe)
        self.prefix = prefix
        self.calls = 0
        #: the top-k slots routed to no expert (`int8_experts.UNROUTED`), over every call
        self.unrouted = 0
        self.sites: list[X.ExpertSite] = []
        if getattr(moe.activation, "value", moe.activation) != "silu":
            raise ValueError(f"{prefix}: the int8 experts run SiLU-and-mul, not {moe.activation!r}")
        if moe.has_bias:
            raise ValueError(f"{prefix}: the int8 experts have no bias")
        if moe.num_local_experts != moe.num_experts:
            raise ValueError(f"{prefix}: {moe.num_local_experts} of {moe.num_experts} experts on this rank; the int8 experts "
                             "run every expert (no expert parallelism)")

    def create_weights(self, layer: torch.nn.Module, num_experts: int, hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs: Any) -> None:
        i, h = intermediate_size_per_partition, hidden_size
        if i % 8 or h % 8 or max(i, h) > E.MAX_K:
            raise ValueError(f"{self.prefix}: hidden {h} and intermediate {i} need multiples of 8 and at most {E.MAX_K}")
        for name, shape in (("w13_weight", (num_experts, 2 * i, h)), ("w2_weight", (num_experts, h, i))):
            p = torch.nn.Parameter(torch.empty(*shape, dtype=torch.int8), requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, extra_weight_attrs)
        scale_attrs = {**extra_weight_attrs, "quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        for name, rows in (("w13_weight_scale", 2 * i), ("w2_weight_scale", h)):
            p = torch.nn.Parameter(torch.ones(num_experts, rows, 1, dtype=torch.float32), requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, scale_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for w, s in (("w13_weight", "w13_weight_scale"), ("w2_weight", "w2_weight_scale")):
            _checked(f"{self.prefix}.{w}", getattr(layer, w).data, getattr(layer, s).data)
        self.sites = X.sites(self.prefix, layer.w13_weight.data, layer.w13_weight_scale.data.contiguous(),
                             layer.w2_weight.data, layer.w2_weight_scale.data.contiguous())

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> None:
        return None

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
              shared_experts: Any = None, shared_experts_input: torch.Tensor | None = None) -> torch.Tensor:
        if getattr(layer, "apply_router_weight_on_input", False):
            raise ValueError(f"{self.prefix}: the int8 experts weight each expert's output, not its input")
        self.calls += 1
        self.unrouted += int((topk_ids == X.UNROUTED).sum())
        return X.forward(self.sites, x, topk_weights, topk_ids)


class Int8KeptLinearMethod(UnquantizedLinearMethod):
    """A linear the checkpoint's ``modules_to_not_convert`` keeps bf16: vLLM's own unquantized method, marked
    (``int8_kept``) so the readback and H1 leave it out as they leave out a router (`kernel_readback.kept`)."""

    int8_kept = True


class Int8ExactConfig(QuantizationConfig):
    """`quantization_config` {quant_method: NAME, weight_max: 127, modules_to_not_convert: [...]}: every linear layer on
    `Int8PlainLinearMethod` but those the list names (vLLM's `is_layer_skipped`, exact names, a fused layer by its shards),
    which stay bf16 on `Int8KeptLinearMethod`; the LM head too unless the list names ``lm_head`` (absent, the list is
    ``["lm_head"]``); every MoE layer's routed experts on `Int8ExpertsMethod`, refused if the list names them.  The list
    may name non-linear modules (norms, biases), which nothing here reads."""

    def __init__(self, weight_max: int, modules_to_not_convert: list[str] | None = None):
        super().__init__()
        if weight_max != E.W_MAX:
            raise ValueError(f"{NAME}: weight_max {weight_max!r} is not {E.W_MAX}")
        self.kept = ["lm_head"] if modules_to_not_convert is None else list(modules_to_not_convert)
        self.lm_head = "lm_head" not in self.kept

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
        return cls(int(cls.get_from_keys(config, ["weight_max"])), cls.get_from_keys_or(config, ["modules_to_not_convert"], None))

    def get_quant_method(self, layer: torch.nn.Module, prefix: str
                         ) -> Int8PlainLinearMethod | Int8KeptLinearMethod | Int8ExpertsMethod | None:
        skipped = isinstance(layer, (LinearBase, RoutedExperts)) and is_layer_skipped(prefix, self.kept,
                                                                                      self.packed_modules_mapping)
        if isinstance(layer, LinearBase):
            return Int8KeptLinearMethod() if skipped else Int8PlainLinearMethod(prefix)
        if self.lm_head and isinstance(layer, ParallelLMHead):
            return Int8PlainLinearMethod(prefix)
        if isinstance(layer, RoutedExperts):
            if skipped:
                raise ValueError(f"{NAME}: modules_to_not_convert names {prefix}; the int8 experts have no bf16 variant")
            return Int8ExpertsMethod(layer.moe_config, prefix)
        return None


@cache
def register() -> str:
    """Register the method with this process's vLLM (once); returns its name."""
    register_weight_loader_v2_supported_method(Int8PlainLinearMethod)
    register_weight_loader_v2_supported_method(Int8KeptLinearMethod)
    register_quantization_config(NAME)(Int8ExactConfig)
    return NAME
