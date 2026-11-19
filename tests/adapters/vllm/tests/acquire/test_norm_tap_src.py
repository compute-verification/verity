"""The norm-scale tap's sources without a GPU: `build_ext.tapped_kernel` cuts the pinned generic `fused_add_rms_norm_kernel` and makes
exactly its three edits (refusing any other text).
The pinned texts below are vLLM d9105ea80's (Apache-2.0, the vLLM project)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from verity_vllm.acquire.norm_tap_src import build_ext as BE

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
