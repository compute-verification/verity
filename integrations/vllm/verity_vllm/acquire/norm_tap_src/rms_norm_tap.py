"""The norm-scale tap's Triton kernel: vLLM's batch-invariant `_rms_norm_kernel` (`vllm/model_executor/determinism/batch_invariant.py` at
d9105ea80; Apache-2.0, the vLLM project) with two lines added, the parameter `scale_ptr` and, after `inv_rms` is computed,
`tl.store(scale_ptr + row_idx + tl.arange(0, 1), inv_rms)`: the row's scale (`RMSNormTriton_v1`'s `inv` gate), stored once per row.
Every other line of `_rms_norm_kernel` below is the pinned kernel's, in order (`TAP_LINES` are the two added; the norm-tap exactness
property diffs the two sources and compares their compiled arithmetic).

`PINNED_SHA256` is the sha256 of the installed kernel's source (`inspect.getsource` of the jitted function): the norm-scale source
launches this copy in its place only after checking it.  Importing this module needs Triton.
"""
from __future__ import annotations

import triton
import triton.language as tl

PINNED_SHA256 = "909ef1f6357183e691dba930fd736f480b116f085308bee4bf27b501e3630d0d"
TAP_LINES = ("    scale_ptr,\n", "    tl.store(scale_ptr + row_idx + tl.arange(0, 1), inv_rms)\n")


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    scale_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    """
    Compute RMS normalization along the last dimension of a 2D tensor.
    RMS Norm: y = x / sqrt(mean(x^2) + eps) * weight
    Each block handles one row of the input tensor.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Compute sum of squares in float32 to avoid overflow
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        # Convert to float32 for accumulation to prevent overflow
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))

    # Step 2: Compute RMS (root mean square) in float32
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms
    tl.store(scale_ptr + row_idx + tl.arange(0, 1), inv_rms)

    # Step 3: Normalize and apply weight
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        # Compute in float32 then convert back to input dtype
        vals_f32 = vals.to(tl.float32)
        output_f32 = vals_f32 * inv_rms
        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
            output_f32 = output_f32 * weight.to(tl.float32)
        output = output_f32.to(vals.dtype)
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)
