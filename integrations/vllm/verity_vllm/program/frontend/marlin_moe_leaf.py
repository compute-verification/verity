"""12. vLLM's MXFP4 Marlin MoE (`Mxfp4MoEMethod` on the MARLIN backend: `MarlinExperts` -> `fused_marlin_moe`) with its two expert
GEMM launches as leaf ops over the layer's logical weights.

The engine builds the layer's kernel object in `process_weights_after_loading`, after repacking the experts' e2m1 codes into
Marlin's tile layout and permuting their scales (`convert_weight_to_mxfp4_moe_kernel_format`), which a meta instantiation never
runs; `fused_marlin_moe` asserts that repacked layout before its first launch.  The Definitions (`registry.marlin_moe`) state each
launch over the checkpoint's [E][N][K/2] packed e2m1 bytes and [E][N][K/32] scale bytes, so for the duration of
`mxfp4_marlin_moe_leaf` each such layer has the kernel object `_setup_kernel` builds without the conversion (pure Python
construction: `MarlinExperts` has no post-load step), and `fused_marlin_moe` is its own steps with each `ops.moe_wna16_marlin_gemm`
as `vllm_iface.marlin_moe_mxfp4_gemm` on those weights: `moe_align_block_size` at the block size `fused_marlin_moe` picks as
`vllm_iface.marlin_moe_align`, the w13 launch, the experts' activation, the w2 launch (top_k 1, the router weights multiplied in)
and the experts' `moe_sum`.  The align kernel ranks each expert's pairs in atomicAdd's order, so `marlin_moe_align` reads a key
per pair, the TP rank Program's `schedule` input (`CollectiveBus.schedule_key`): a TP1 export has no such input and refuses.
At a launch that also routes other requests' tokens (`traced_launch_tokens`) the block size is the launch's and the align sorts
every request's pairs, so the call takes its pairs' blocks and the launch's block count from the `schedule` input instead
(`CollectiveBus.launch_schedule`), words the workload Program links to that launch's unit.  Only W4A16 float4_e2m1f weights without bias, zero points, act order, global scales or an expert map are modelled; anything else
refuses at export.
"""

from __future__ import annotations

import importlib
import math
import sys
from contextlib import contextmanager

import torch

from verity_vllm.engine import hooks

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

_MXFP4 = "vllm.model_executor.layers.quantization.mxfp4"
_MARLIN_MOE = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
_ORACLE = "vllm.model_executor.layers.fused_moe.oracle.mxfp4"
_PARALLEL_STATE = "vllm.distributed.parallel_state"

#: the operands of `fused_marlin_moe` the leaf does not model, each refused unless None
_UNMODELLED = ("bias1", "bias2", "global_scale1", "global_scale2", "input_global_scale1", "input_global_scale2", "w1_zeros", "w2_zeros",
               "g_idx1", "g_idx2", "sort_indices1", "sort_indices2", "expert_map", "input_dtype")


def marlin_block_size_m(M: int, topk: int, E: int, global_num_experts: int) -> int:
    """`fused_marlin_moe`'s moe block size: the first of 8 .. 64 at which the estimated rows per expert per block fall below 0.9."""
    if global_num_experts != -1:
        M = math.ceil(M * E / global_num_experts)
    for block_size_m in (8, 16, 32, 48, 64):
        if M * topk / E / block_size_m < 0.9:
            break
    return block_size_m


def mxfp4_marlin_layers(model) -> dict[str, torch.nn.Module]:
    """The model's MXFP4 MoE layers on the Marlin backend, by qualified name."""
    mxfp4 = sys.modules.get(_MXFP4)
    if model is None or mxfp4 is None:
        return {}
    backend = importlib.import_module(_ORACLE).Mxfp4MoeBackend.MARLIN
    return {n: m for n, m in model.named_modules() if isinstance(getattr(m, "quant_method", None), mxfp4.Mxfp4MoEMethod)
            and m.quant_method.mxfp4_backend == backend and hasattr(m, "w13_weight")}


def _marlin_layers(model) -> list:
    return [m for m in mxfp4_marlin_layers(model).values() if m.quant_method.moe_kernel is None]


@contextmanager
def mxfp4_marlin_moe_leaf(model=None):
    """See the module docstring.  A model without an MXFP4 Marlin MoE layer is unchanged."""
    layers = _marlin_layers(model)
    if not layers:
        yield
        return
    from verity_vllm.program.frontend.rules.vllm_bindings.observations import traced_launch_step, traced_launch_tokens
    from verity_vllm.program.frontend.rules.vllm_iface import iface
    oracle, site, parallel_state = (importlib.import_module(m) for m in (_ORACLE, _MARLIN_MOE, _PARALLEL_STATE))
    e2m1 = importlib.import_module("vllm.scalar_type").scalar_types.float4_e2m1f

    def fused_marlin_moe(hidden_states, w1, w2, bias1, bias2, w1_scale, w2_scale, topk_weights, topk_ids, quant_type_id,
                         apply_router_weight_on_input=False, global_num_experts=-1, activation=None, activation_func=None,
                         moe_sum=None, output=None, **rest):
        given = sorted(k for k in _UNMODELLED if rest.get(k) is not None) + (["bias1"] if bias1 is not None else []) + (
            ["bias2"] if bias2 is not None else [])
        if quant_type_id != e2m1.id or given or apply_router_weight_on_input or activation_func is None or moe_sum is None:
            raise NotImplementedError(f"mxfp4_marlin_moe_leaf: fused_marlin_moe beyond W4A16 float4_e2m1f with the experts' activation "
                                      f"and moe_sum (given {given}, router weight on input {apply_router_weight_on_input})")
        bus = getattr(parallel_state.get_tp_group(), "bus", None)
        if bus is None:
            raise NotImplementedError("mxfp4_marlin_moe_leaf: the moe schedule's key per pair is a TP rank Program's `schedule` input "
                                      "(CollectiveBus.schedule_key); this export has no TP bus")
        M, K = hidden_states.shape
        E, topk, N = w1.shape[0], topk_ids.shape[1], w2.shape[2] * 2
        L, experts = traced_launch_tokens(M), E if global_num_experts == -1 else global_num_experts
        block_m = marlin_block_size_m(L, topk, E, global_num_experts)
        if L == M:
            blk, par = iface.marlin_moe_align(topk_ids, bus.schedule_key(topk_ids), experts, block_m)
        else:
            blk, par = bus.launch_schedule(topk_ids, step=traced_launch_step(), launch_tokens=L, E=experts, BM=block_m)
        up = iface.marlin_moe_mxfp4_gemm(hidden_states, w1, w1_scale, topk_ids, blk, par, topk_weights, block_m, topk, False)
        act = torch.empty((M * topk, N), dtype=hidden_states.dtype, device=hidden_states.device)
        activation_func(activation, act, up, topk_ids=topk_ids, expert_map=None)
        down = iface.marlin_moe_mxfp4_gemm(act, w2, w2_scale, topk_ids, blk, par, topk_weights, block_m, 1, True)
        if output is None:
            output = torch.empty_like(hidden_states)
        moe_sum(down.view(-1, topk, K), output, topk_ids, None)
        return output

    with hooks.Hooks() as h:
        for m in layers:
            qm = m.quant_method
            config = qm.get_fused_moe_quant_config(m)
            h.patch(qm, "moe_quant_config", config)
            h.patch(qm, "moe_kernel", oracle.make_mxfp4_moe_kernel(moe_quant_config=config, moe_config=qm.moe, mxfp4_backend=qm.mxfp4_backend,
                                                                   experts_cls=qm.experts_cls, routing_tables=m._expert_routing_tables()))
        h.patch(site, "fused_marlin_moe", fused_marlin_moe)
        yield


__all__ = ["marlin_block_size_m", "mxfp4_marlin_layers", "mxfp4_marlin_moe_leaf"]
