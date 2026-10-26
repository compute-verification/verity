"""sbench's stock-percall arm: stock vLLM int8 (TritonInt8ScaledMMLinearKernel) with its `triton_scaled_mm` call behind a
custom op, `exactlin_stock::scaled_mm`, opaque to Dynamo as the exact arms' ops are.  Traced, `triton_scaled_mm`'s tile
heuristic runs once at the traced M (max_num_batched_tokens) and every M then runs that tile; behind the op it runs on each
call's own M, at each CUDA graph's capture and on every call outside a graph.  The GEMM's values are stock's: the int32
sums don't depend on the tile, and the epilogue is the same per element.  The rest of the step is stock's arithmetic, but
its reduction configs are this process's own: the op changes Inductor's metadata around it (tiling scores, byte estimates,
a buffer made in place), which keys Inductor's autotune cache, so the arm times its reductions afresh, and the fused add +
RMSNorm + int8 quant kernel's two configs (R0_BLOCK 4096 and 1024, a near tie) sum in another order.  Its tokens equal
stock's where both processes pick alike (sbench gates them; `percall_diag.py` pins the picks).
`CALLS` counts the op's calls by (M, N), so a run shows which M the heuristic saw."""
from __future__ import annotations

import torch

CALLS: dict[tuple[int, int], int] = {}


@torch.library.custom_op("exactlin_stock::scaled_mm", mutates_args=())
def scaled_mm(x_q: torch.Tensor, w_q: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm,
    )

    key = (x_q.shape[0], w_q.shape[1])
    CALLS[key] = CALLS.get(key, 0) + 1
    return triton_scaled_mm(x_q, w_q, scale_a, scale_b, torch.bfloat16)


@scaled_mm.register_fake
def _(x_q, w_q, scale_a, scale_b):
    return x_q.new_empty((x_q.shape[0], w_q.shape[1]), dtype=torch.bfloat16)


def _call(input, weight, scale_a, scale_b, out_dtype, bias=None, **kw):
    if bias is not None or kw or out_dtype != torch.bfloat16:
        raise NotImplementedError(f"stock-percall: bf16 out, no bias only (bias {bias is not None}, {out_dtype}, {kw})")
    return torch.ops.exactlin_stock.scaled_mm(input, weight, scale_a, scale_b)


def install() -> None:
    """Route TritonInt8ScaledMMLinearKernel.apply_weights's `triton_scaled_mm` (a global of its module) through the op."""
    from vllm.model_executor.kernels.linear.scaled_mm import triton as T

    T.triton_scaled_mm = _call


def summary() -> dict:
    ms = sorted({m for m, _ in CALLS})
    return {"calls": sum(CALLS.values()), "distinct_m": len(ms), "m": ms}
