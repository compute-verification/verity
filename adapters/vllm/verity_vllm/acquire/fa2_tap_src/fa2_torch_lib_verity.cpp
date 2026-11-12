// Verity: stable-ABI torch library registration for the
// rebuilt FA2 forward.  Mirrors the pinned csrc/flash_attn/flash_api_torch_lib.cpp
// (vllm-project/flash-attention @ 506341a1) for `varlen_fwd` only (identical
// schema), plus the lane's tap-configuration op (only when VERITY_TAP > 0).
#include "namespace_config.h"

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>

#include "registration.h"

#include <optional>
#include <vector>
#include <cstdint>

#if defined(VERITY_TAP) && VERITY_TAP > 0
#include <cutlass/numeric_types.h>
#include "flash.h"   // Verity_tap + extern g_verity_tap (tapped tree only)
#endif

using torch::stable::Tensor;

namespace FLASH_NAMESPACE {

std::vector<Tensor>
mha_varlen_fwd(Tensor q,
               const Tensor &k,
               const Tensor &v,
               std::optional<Tensor> out_,
               const Tensor &cu_seqlens_q,
               const Tensor &cu_seqlens_k,
               std::optional<Tensor> seqused_k,
               const std::optional<Tensor> &leftpad_k_,
               std::optional<Tensor> block_table_,
               std::optional<Tensor> alibi_slopes_,
               int64_t max_seqlen_q,
               const int64_t max_seqlen_k,
               const double p_dropout,
               const double softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int64_t window_size_left,
               int64_t window_size_right,
               const double softcap,
               const bool return_softmax,
               int64_t num_splits,
               std::optional<Tensor> unused_generator_compat);

#if defined(VERITY_TAP) && VERITY_TAP > 0
// Defined (extern) in flash.h of the tapped tree; the launch template copies it
// into Flash_fwd_params right before the kernel launch.
Verity_tap g_verity_tap = {};
int64_t g_verity_tap_skipped = 0;

// mode: 0 = off, 1 = P1 accumulator, 2 = P0 diagnostic (+P1).
// acc:  uint32 buffer for the P1 per-(b,h,m_block,warp) words (may be None).
// diag: float32 flat buffer for P0 (may be None).  H/M/N/NB are the P0 caps.
void verity_set_tap(int64_t mode, std::optional<Tensor> acc, std::optional<Tensor> diag,
                    int64_t H, int64_t M, int64_t N, int64_t NB) {
    Verity_tap t = {};
    t.mode = static_cast<int>(mode);
    if (acc.has_value()) {
        t.acc_ptr = acc.value().data_ptr();
        t.acc_cap_words = acc.value().numel();
    }
    if (diag.has_value()) {
        t.diag_ptr = reinterpret_cast<float *>(diag.value().data_ptr());
    }
    t.H = static_cast<int>(H); t.M = static_cast<int>(M); t.N = static_cast<int>(N); t.NB = static_cast<int>(NB);
    t.BN = 128; t.D = 64;     // legacy (v7) declaration: the hdim-64 align kernel's layout
    g_verity_tap = t;
}

// v8: one call per launch declaring the FULL stream geometry the host committed to -- slabs HB, rows M, key blocks NB,
// AND the kernel tile the layout is arithmetic in: BN = kBlockN (128 for hdim 64, 64 for hdim 96/128/256 on the num_splits==1 align
// kernel) and D = head_dim (O plane width) -- plus the per-launch tag.  The launch template arms the tap only if the selected kernel's
// (kBlockN, kHeadDim) equals (BN, D); otherwise the launch is counted in verity_tap_skipped() and nothing is written (fail closed).
void verity_set_tap_v8(int64_t mode, std::optional<Tensor> diag, int64_t HB, int64_t M, int64_t NB, int64_t BN, int64_t D, int64_t tag) {
    Verity_tap t = {};
    t.mode = static_cast<int>(mode);
    if (diag.has_value()) {
        t.diag_ptr = reinterpret_cast<float *>(diag.value().data_ptr());
    }
    t.H = static_cast<int>(HB); t.M = static_cast<int>(M); t.N = static_cast<int>(NB * BN); t.NB = static_cast<int>(NB);
    t.BN = static_cast<int>(BN); t.D = static_cast<int>(D); t.tag = static_cast<int>(tag);
    g_verity_tap = t;
}

// v8: bitmask of the bf16 head dims whose split-KV kernels are compiled into this build (1<<(hd/32)): 64 -> 4, 96 -> 8, 128 -> 16, 256 -> 256.
int64_t verity_tap_hdims() {
    int64_t m = 0;
#ifdef VERITY_HAVE_SPLIT_HDIM64_BF16
    m |= 1 << 2;
#endif
#ifdef VERITY_HAVE_SPLIT_HDIM96_BF16
    m |= 1 << 3;
#endif
#ifdef VERITY_HAVE_SPLIT_HDIM128_BF16
    m |= 1 << 4;
#endif
#ifdef VERITY_HAVE_SPLIT_HDIM256_BF16
    m |= 1 << 8;
#endif
    return m;
}
// v8: the tap layout version compiled in (8 = parametric (BN, D) layout; the v7 build has no such op).
int64_t verity_tap_layout_version() { return 8; }
// What word 3 of a ROW entry holds: 0 = never written, 1 = GuardNegInfZero(row_max) after a row's first key block (VERITY_ROW_GUARD=1).
#ifndef VERITY_ROW_GUARD
#define VERITY_ROW_GUARD 0
#endif
int64_t verity_tap_row3() { return VERITY_ROW_GUARD; }

// Number of launches on which the tap had to be disabled (num_splits > 1 or capacity).
int64_t verity_tap_skipped() { return g_verity_tap_skipped; }

// v7: per-launch tag (layer / step id) bound into every SHA leaf header and kept alongside the M1 stream.
void verity_set_tap_tag(int64_t tag) { g_verity_tap.tag = static_cast<int>(tag); }
// v7: which build this is (VERITY_TAP) and which classes it consumes (the compile-time bitmask).
#ifndef VERITY_MAT_SRC
#define VERITY_MAT_SRC 63
#endif
#ifndef VERITY_SHA_SRC
#define VERITY_SHA_SRC 31
#endif
#ifndef VERITY_P1_SRC
#define VERITY_P1_SRC 1
#endif
int64_t verity_tap_mode() { return VERITY_TAP; }
int64_t verity_tap_src() { return VERITY_TAP == 3 ? VERITY_MAT_SRC : (VERITY_TAP == 4 ? VERITY_SHA_SRC : VERITY_P1_SRC); }
// v7: point the tap at another buffer without touching the caps (per-launch base pointer for the M1 stream).
void verity_set_tap_buffers(std::optional<Tensor> acc, std::optional<Tensor> diag) {
    g_verity_tap.acc_ptr = acc.has_value() ? acc.value().data_ptr() : nullptr;
    g_verity_tap.acc_cap_words = acc.has_value() ? acc.value().numel() : 0;
    g_verity_tap.diag_ptr = diag.has_value() ? reinterpret_cast<float *>(diag.value().data_ptr()) : nullptr;
}
#endif

} // namespace FLASH_NAMESPACE

STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("varlen_fwd(Tensor! q, Tensor k, Tensor v, Tensor!? out, Tensor cu_seqlens_q, "
            "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, "
            "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
            "bool is_causal, int window_size_left, int window_size_right, float softcap, bool return_softmax, "
            "int num_splits, Tensor? gen) -> Tensor[]");
#if defined(VERITY_TAP) && VERITY_TAP > 0
    ops.def("verity_set_tap(int mode, Tensor? acc, Tensor? diag, int H, int M, int N, int NB) -> ()");
    ops.def("verity_tap_skipped() -> int");
    ops.def("verity_set_tap_tag(int tag) -> ()");
    ops.def("verity_tap_mode() -> int");
    ops.def("verity_tap_src() -> int");
    ops.def("verity_set_tap_buffers(Tensor? acc, Tensor? diag) -> ()");
    ops.def("verity_set_tap_v8(int mode, Tensor? diag, int HB, int M, int NB, int BN, int D, int tag) -> ()");
    ops.def("verity_tap_hdims() -> int");
    ops.def("verity_tap_layout_version() -> int");
    ops.def("verity_tap_row3() -> int");
#endif
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
    ops.impl("varlen_fwd", TORCH_BOX(&FLASH_NAMESPACE::mha_varlen_fwd));
}

#if defined(VERITY_TAP) && VERITY_TAP > 0
STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CompositeExplicitAutograd, ops) {
    ops.impl("verity_set_tap", TORCH_BOX(&FLASH_NAMESPACE::verity_set_tap));
    ops.impl("verity_tap_skipped", TORCH_BOX(&FLASH_NAMESPACE::verity_tap_skipped));
    ops.impl("verity_set_tap_tag", TORCH_BOX(&FLASH_NAMESPACE::verity_set_tap_tag));
    ops.impl("verity_tap_mode", TORCH_BOX(&FLASH_NAMESPACE::verity_tap_mode));
    ops.impl("verity_tap_src", TORCH_BOX(&FLASH_NAMESPACE::verity_tap_src));
    ops.impl("verity_set_tap_buffers", TORCH_BOX(&FLASH_NAMESPACE::verity_set_tap_buffers));
    ops.impl("verity_set_tap_v8", TORCH_BOX(&FLASH_NAMESPACE::verity_set_tap_v8));
    ops.impl("verity_tap_hdims", TORCH_BOX(&FLASH_NAMESPACE::verity_tap_hdims));
    ops.impl("verity_tap_layout_version", TORCH_BOX(&FLASH_NAMESPACE::verity_tap_layout_version));
    ops.impl("verity_tap_row3", TORCH_BOX(&FLASH_NAMESPACE::verity_tap_row3));
}
#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
