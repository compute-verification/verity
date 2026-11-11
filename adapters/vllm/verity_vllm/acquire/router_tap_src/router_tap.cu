// The router-softmax tap op (verity_vllm.acquire.router_tap_src.build_ext): vLLM d9105ea80's topkGating kernel, cut from the pinned
// csrc/libtorch_stable/moe/topk_softmax_kernels.cu into router_tap_kernel.cuh (build dir), launched the way the installed
// _moe_C.topk_softmax launches it (topkGatingKernelLauncher -> topkGatingLauncherHelper: the same expert-count cases and bytes per load,
// grid = ceil(rows / (ROWS_PER_WARP * 4)), block = (WARP_SIZE, 4), start_expert 0, end_expert E, routed_scaling_factor 1.0, the current
// stream).  An expert count the installed op sends to moeSoftmax + moeTopK (the workspace path) is refused.
//
//   topk_softmax(w, ids, tei, gating, renormalize, bias?, is_padding?)          the kernel with the tap off (Tap = false)
//   topk_softmax_tap(w, ids, tei, gating, renormalize, bias?, is_padding?, tap) the same launch, storing each row's softmax to tap[row]
//   tap_version()                                                               VERITY_ROUTER_TAP_VERSION
//
// tap is float32 [rows, 2E + 2] (renormalize: [rows, 2E + 3 + k]): max, E exponentials, 1 / sum, E probabilities (+ k selected, scale).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <optional>

#include "router_tap_kernel.cuh"

namespace {

using vllm::verity_router_tap::SCORING_SOFTMAX;

struct Launch {
  const void* gating;
  float* weights;
  void* indices;
  int* token_expert_indices;
  int num_rows;
  int k;
  bool renormalize;
  const float* bias;
  const bool* is_padding;
  float* tap;
  int tap_stride;
  cudaStream_t stream;
};

template <int EXPERTS, int MAX_BYTES_PER_LDG, bool Tap, typename IndType, typename InputType>
void launch_experts(const Launch& a) {
  static constexpr int WARPS_PER_TB = 4;
  static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(InputType) * EXPERTS);
  using Constants = vllm::verity_router_tap::detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE, InputType>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (a.num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;
  dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
  vllm::verity_router_tap::topkGating<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG, WARP_SIZE, IndType, InputType, SCORING_SOFTMAX, Tap>
      <<<num_blocks, block_dim, 0, a.stream>>>(
          static_cast<const InputType*>(a.gating), nullptr, a.weights, a.num_rows, static_cast<IndType*>(a.indices),
          a.token_expert_indices, a.k, 0, EXPERTS, a.renormalize, a.bias, 1.0, a.is_padding, a.tap, a.tap_stride);
}

template <bool Tap, typename IndType, typename InputType>
void launch_typed(const Launch& a, int num_experts) {
  static constexpr int P2 = 16;
  static constexpr int M64 = (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>) ? 4 : 8;
  switch (num_experts) {
    case 1: launch_experts<1, P2, Tap, IndType, InputType>(a); break;
    case 2: launch_experts<2, P2, Tap, IndType, InputType>(a); break;
    case 4: launch_experts<4, P2, Tap, IndType, InputType>(a); break;
    case 8: launch_experts<8, P2, Tap, IndType, InputType>(a); break;
    case 16: launch_experts<16, P2, Tap, IndType, InputType>(a); break;
    case 32: launch_experts<32, P2, Tap, IndType, InputType>(a); break;
    case 64: launch_experts<64, P2, Tap, IndType, InputType>(a); break;
    case 128: launch_experts<128, P2, Tap, IndType, InputType>(a); break;
    case 256: launch_experts<256, P2, Tap, IndType, InputType>(a); break;
    case 512: launch_experts<512, P2, Tap, IndType, InputType>(a); break;
    case 192: launch_experts<192, M64, Tap, IndType, InputType>(a); break;
    case 320: launch_experts<320, M64, Tap, IndType, InputType>(a); break;
    case 384: launch_experts<384, M64, Tap, IndType, InputType>(a); break;
    case 448: launch_experts<448, M64, Tap, IndType, InputType>(a); break;
    case 576: launch_experts<576, M64, Tap, IndType, InputType>(a); break;
    default:
      TORCH_CHECK(false, "verity_router_tap: ", num_experts, " experts take the installed op's moeSoftmax + moeTopK path, which the tap does not cover");
  }
}

template <bool Tap, typename InputType>
void launch_ind(const Launch& a, int num_experts, at::ScalarType ind) {
  if (ind == at::kInt) {
    launch_typed<Tap, int, InputType>(a, num_experts);
  } else if (ind == at::kUInt32) {
    launch_typed<Tap, uint32_t, InputType>(a, num_experts);
  } else {
    TORCH_CHECK(ind == at::kLong, "verity_router_tap: topk_indices must be int32, uint32 or int64");
    launch_typed<Tap, int64_t, InputType>(a, num_experts);
  }
}

template <bool Tap>
void run(torch::Tensor& topk_weights, torch::Tensor& topk_indices, torch::Tensor& token_expert_indices, const torch::Tensor& gating_output,
         bool renormalize, const std::optional<torch::Tensor>& bias, const std::optional<torch::Tensor>& is_padding, float* tap, int tap_stride) {
  const int num_experts = gating_output.size(-1);
  const int64_t num_tokens = gating_output.numel() / num_experts;
  const int topk = topk_weights.size(-1);
  const float* bias_ptr = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kFloat, "bias tensor must be float32");
    TORCH_CHECK(bias->dim() == 1, "bias tensor must be 1D");
    TORCH_CHECK(bias->size(0) == num_experts, "bias size mismatch, expected: ", num_experts);
    TORCH_CHECK(bias->is_contiguous(), "bias tensor must be contiguous");
    bias_ptr = bias->data_ptr<float>();
  }
  const bool* is_padding_ptr = nullptr;
  if (is_padding.has_value()) {
    TORCH_CHECK(is_padding->scalar_type() == at::kBool, "is_padding tensor must be bool");
    TORCH_CHECK(is_padding->dim() == 1, "is_padding tensor must be 1D");
    TORCH_CHECK(is_padding->size(0) == num_tokens, "is_padding size mismatch, expected: ", num_tokens);
    TORCH_CHECK(is_padding->is_contiguous(), "is_padding tensor must be contiguous");
    is_padding_ptr = is_padding->data_ptr<bool>();
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
  Launch a{gating_output.data_ptr(), topk_weights.data_ptr<float>(), topk_indices.data_ptr(), token_expert_indices.data_ptr<int>(),
           static_cast<int>(num_tokens), topk, renormalize, bias_ptr, is_padding_ptr, tap, tap_stride, at::cuda::getCurrentCUDAStream()};
  const auto ind = topk_indices.scalar_type();
  switch (gating_output.scalar_type()) {
    case at::kFloat: launch_ind<Tap, float>(a, num_experts, ind); break;
    case at::kHalf: launch_ind<Tap, __half>(a, num_experts, ind); break;
    case at::kBFloat16: launch_ind<Tap, __nv_bfloat16>(a, num_experts, ind); break;
    default: TORCH_CHECK(false, "Unsupported gating_output data type: ", gating_output.scalar_type());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices, torch::Tensor& token_expert_indices, torch::Tensor gating_output,
                  bool renormalize, std::optional<torch::Tensor> bias, std::optional<torch::Tensor> is_padding) {
  run<false>(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize, bias, is_padding, nullptr, 0);
}

void topk_softmax_tap(torch::Tensor& topk_weights, torch::Tensor& topk_indices, torch::Tensor& token_expert_indices, torch::Tensor gating_output,
                      bool renormalize, std::optional<torch::Tensor> bias, std::optional<torch::Tensor> is_padding, torch::Tensor& tap) {
  const int64_t num_experts = gating_output.size(-1);
  const int64_t rows = gating_output.numel() / num_experts;
  const int64_t stride = 2 * num_experts + 2 + (renormalize ? topk_weights.size(-1) + 1 : 0);
  TORCH_CHECK(tap.is_cuda() && tap.device() == gating_output.device(), "verity_router_tap: tap must be on the gating output's device");
  TORCH_CHECK(tap.scalar_type() == at::kFloat && tap.is_contiguous(), "verity_router_tap: tap must be a contiguous float32 tensor");
  TORCH_CHECK(tap.numel() == rows * stride, "verity_router_tap: tap must hold ", stride, " words per row (", rows, " rows), got ", tap.numel());
  run<true>(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize, bias, is_padding, tap.data_ptr<float>(),
            static_cast<int>(stride));
}

int64_t tap_version() { return VERITY_ROUTER_TAP_VERSION; }

}  // namespace

TORCH_LIBRARY(verity_router_tap, m) {
  m.def("topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! token_expert_indices, Tensor gating_output, bool renormalize, "
        "Tensor? bias, Tensor? is_padding) -> ()");
  m.def("topk_softmax_tap(Tensor! topk_weights, Tensor! topk_indices, Tensor! token_expert_indices, Tensor gating_output, bool renormalize, "
        "Tensor? bias, Tensor? is_padding, Tensor! tap) -> ()");
  m.def("tap_version() -> int");
}

TORCH_LIBRARY_IMPL(verity_router_tap, CUDA, m) {
  m.impl("topk_softmax", &topk_softmax);
  m.impl("topk_softmax_tap", &topk_softmax_tap);
}

TORCH_LIBRARY_IMPL(verity_router_tap, CompositeExplicitAutograd, m) {
  m.impl("tap_version", &tap_version);
}
