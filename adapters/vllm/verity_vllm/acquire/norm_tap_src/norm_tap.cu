// The norm-scale tap op (verity_vllm.acquire.norm_tap_src.build_ext): vLLM d9105ea80's generic fused_add_rms_norm_kernel, cut from the
// pinned csrc/libtorch_stable/layernorm_kernels.cu into norm_tap_kernel.cuh (build dir), launched the way the installed
// _C.fused_add_rms_norm launches it under VLLM_BATCH_INVARIANT=1: the host checks of the pinned fused_add_rms_norm, grid = rows,
// block = min(hidden, 1024), the width-0 kernel (batch invariance never takes the vectorized width-8 one), the current stream.
//
//   fused_add_rms_norm(input, residual, weight?, epsilon)                 the kernel with the tap off (Tap = false)
//   fused_add_rms_norm_scale(input, residual, weight?, epsilon, scale)    the same launch, thread 0 of row r storing s_variance to scale[r]
//   tap_version()                                                         VERITY_NORM_TAP_VERSION
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <optional>

#include "cub_helpers.h"
#include "type_convert.cuh"
#include "norm_tap_kernel.cuh"

namespace {

template <bool Tap>
void launch(torch::Tensor& input, torch::Tensor& residual, const std::optional<torch::Tensor>& weight, double epsilon, float* scale_out) {
  TORCH_CHECK(input.scalar_type() == residual.scalar_type());
  TORCH_CHECK(residual.stride(-1) == 1);
  if (weight.has_value()) {
    TORCH_CHECK(weight->scalar_type() == input.scalar_type());
    TORCH_CHECK(weight->is_contiguous());
  }
  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int64_t residual_stride = residual.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "verity_norm_tap", [&] {
    if (weight.has_value()) {
      vllm::verity_norm_tap::fused_add_rms_norm_kernel<scalar_t, 0, true, Tap><<<grid, block, 0, stream>>>(
          input.data_ptr<scalar_t>(), input_stride, residual.data_ptr<scalar_t>(), weight->data_ptr<scalar_t>(), epsilon, num_tokens,
          hidden_size, residual_stride, scale_out);
    } else {
      vllm::verity_norm_tap::fused_add_rms_norm_kernel<scalar_t, 0, false, Tap><<<grid, block, 0, stream>>>(
          input.data_ptr<scalar_t>(), input_stride, residual.data_ptr<scalar_t>(), nullptr, epsilon, num_tokens, hidden_size,
          residual_stride, scale_out);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual, std::optional<torch::Tensor> weight, double epsilon) {
  launch<false>(input, residual, weight, epsilon, nullptr);
}

void fused_add_rms_norm_scale(torch::Tensor& input, torch::Tensor& residual, std::optional<torch::Tensor> weight, double epsilon,
                              torch::Tensor& scale) {
  TORCH_CHECK(scale.is_cuda() && scale.device() == input.device(), "verity_norm_tap: scale must be on the input's device");
  TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.is_contiguous(), "verity_norm_tap: scale must be a contiguous float32 tensor");
  TORCH_CHECK(scale.numel() == input.numel() / input.size(-1), "verity_norm_tap: scale must hold one word per row (", input.numel() / input.size(-1),
              "), got ", scale.numel());
  launch<true>(input, residual, weight, epsilon, scale.data_ptr<float>());
}

int64_t tap_version() { return VERITY_NORM_TAP_VERSION; }

}  // namespace

TORCH_LIBRARY(verity_norm_tap, m) {
  m.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, float epsilon) -> ()");
  m.def("fused_add_rms_norm_scale(Tensor! input, Tensor! residual, Tensor? weight, float epsilon, Tensor! scale) -> ()");
  m.def("tap_version() -> int");
}

TORCH_LIBRARY_IMPL(verity_norm_tap, CUDA, m) {
  m.impl("fused_add_rms_norm", &fused_add_rms_norm);
  m.impl("fused_add_rms_norm_scale", &fused_add_rms_norm_scale);
}

TORCH_LIBRARY_IMPL(verity_norm_tap, CompositeExplicitAutograd, m) {
  m.impl("tap_version", &tap_version);
}
