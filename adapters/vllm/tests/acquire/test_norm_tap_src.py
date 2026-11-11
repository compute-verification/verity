"""The norm-scale tap's sources without a GPU: `build_ext.tapped_kernel` cuts the pinned generic `fused_add_rms_norm_kernel` and makes
exactly its three edits (refusing any other text), and the Triton copy is the pinned `_rms_norm_kernel` plus exactly its two tap lines.
The pinned texts below are vLLM d9105ea80's (Apache-2.0, the vLLM project)."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from verity_vllm.acquire.norm_tap_src import build_ext as BE
from verity_vllm.properties import norm_tap_exactness as X

SRC = Path(BE.__file__).resolve().parent

GENERIC = """/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width, bool HasWeight>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size], null if !HasWeight
    const float epsilon, const int num_tokens, const int hidden_size,
    const int64_t residual_stride) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * residual_stride + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * residual_stride + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * residual_stride + idx];
    if constexpr (HasWeight) {
      float w = (float)weight[idx];
      input[blockIdx.x * input_stride + idx] = (scalar_t)(x * s_variance * w);
    } else {
      input[blockIdx.x * input_stride + idx] = (scalar_t)(x * s_variance);
    }
  }
}
"""

TRITON = '''@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
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
'''

FILE = "#include <numeric>\n\nnamespace vllm {\n\n// the width-8 kernel above\n\n" + GENERIC + "\n}  // namespace vllm\n\nvoid rms_norm() {}\n"


def test_the_pinned_generic_kernel():
    assert hashlib.sha256(GENERIC.encode()).hexdigest() == BE.BLOCK_SHA256
    assert BE.generic_block(FILE) == GENERIC


def test_tapped_kernel_makes_exactly_the_three_edits():
    out = BE.tapped_kernel(FILE)
    block = out[out.index(BE.BLOCK_START): out.index("}  // namespace verity_norm_tap")]
    for _old, new in BE.EDITS:
        assert block.count(new) == 1
    undone = block.replace(BE.TAP_STORE, "")
    for old, new in BE.EDITS[:2]:
        undone = undone.replace(new, old)
    assert undone.rstrip("\n") == GENERIC.rstrip("\n"), "with the tap edits undone the kernel is the pinned text"
    assert "namespace vllm {\nnamespace verity_norm_tap {" in out and "scale_out[blockIdx.x] = s_variance;" in out


@pytest.mark.parametrize("text,why", [
    (FILE.replace("rsqrtf(", "rsqrt("), "not the pinned one"),
    (FILE + GENERIC, "occurs 2 times"),
    (FILE.replace(BE.BLOCK_START, "/* another kernel\n"), "occurs 0 times"),
])
def test_tapped_kernel_refuses_any_other_text(text, why):
    with pytest.raises(ValueError, match=why):
        BE.tapped_kernel(text)


def test_pinned_sources_are_checked(tmp_path):
    for rel in BE.PINNED_SOURCES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("not the pinned file\n")
    with pytest.raises(ValueError, match="not the pinned"):
        BE.pinned_sources(str(tmp_path))


def test_the_launcher_registers_the_ops_and_instantiates_the_tap():
    cu = (SRC / "norm_tap.cu").read_text()
    for s in ('m.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, float epsilon) -> ()")',
              "fused_add_rms_norm_scale(Tensor! input, Tensor! residual, Tensor? weight, float epsilon, Tensor! scale)",
              "fused_add_rms_norm_kernel<scalar_t, 0, true, Tap>", "fused_add_rms_norm_kernel<scalar_t, 0, false, Tap>",
              "dim3 block(std::min(hidden_size, 1024));", '#include "norm_tap_kernel.cuh"'):
        assert s in cu, s


def _module_constant(src: str, name: str):
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def _function_text(src: str, name: str) -> str:
    lines = src.splitlines(keepends=True)
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            return "".join(lines[start - 1: node.end_lineno])
    raise KeyError(name)


def test_the_triton_copy_is_the_pinned_kernel_plus_the_tap_lines():
    src = (SRC / "rms_norm_tap.py").read_text()
    assert hashlib.sha256(TRITON.encode()).hexdigest() == _module_constant(src, "PINNED_SHA256")
    tap_lines = _module_constant(src, "TAP_LINES")
    assert tap_lines == ("    scale_ptr,\n", "    tl.store(scale_ptr + row_idx + tl.arange(0, 1), inv_rms)\n")
    d = X.source_diff(TRITON, _function_text(src, "_rms_norm_kernel"), tap_lines)
    assert d["ok"] and d["inserted"] == list(tap_lines)
