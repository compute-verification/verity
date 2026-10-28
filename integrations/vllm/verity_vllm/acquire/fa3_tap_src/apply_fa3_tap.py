#!/usr/bin/env python3
"""Turn a pinned vllm-flash-attn tree (506341a1) into the FA3-tapped tree (M1 export tap, layout v9 = FA2 v8 word
layout keyed by packed PackGQA rows).

  apply_fa3_tap.py <tree>/hopper <this src dir>

Idempotent text patches (each guarded by a marker), all under `#if VERITY_TAP == 3` so a VERITY_TAP=0 build of the same tree is
the pinned kernel up to the (unused) trailing `vt` parameter:
 1. copies verity_tap_params.h / verity_tap_fa3.h into hopper/;
 2. softmax.h: `Softmax` gains `tap_sum, tap_inv_sum`, stashed in `finalize` (the FA3 epilogue overwrites row_sum with the lse);
 2b. softmax.h, only with `VERITY_MAT_SRC & 64` (the MS class): `Softmax::tap_max_scaled`, kept by `scale_apply_exp2_ms` (the pinned
    `scale_apply_exp2`'s arithmetic, plus each row's `max_scaled`, the register its exp2 FMAs read), which online_softmax then calls;
 3. mainloop_fwd_sm90_tma_gmma_ws.hpp: Arguments/Params gain a trailing `Verity_tap vt`; `mma()`'s IntraWGOverlap path (the only
    schedule of the bf16 hdim 64/128 tiles) materialises S (after masking), P + row state (after online_softmax), rP (after the bf16
    convert), acc_o after each P.V wgmma completes (plane index = the block whose P.V it was) and the epilogue after finalize;
 3b. the same P site stores `tap_max_scaled` into the MS class (`VERITY_FA3_MS`, only with `VERITY_MAT_SRC & 64`);
 4. flash_fwd_launch_template.h: arms the tap from g_verity_tap only when the selected kernel is exactly the declared tile
    (kBlockN == BN, kHeadDim == D), PackGQA, !Split/local/softcap/appendKV/Qv/FP8, and the declared caps cover the launch; else
    g_verity_tap_skipped++ and nothing is written;
 5. writes fa3_torch_lib_verity.cpp = the pinned flash_api_torch_lib.cpp + the verity ops (namespace = TORCH_EXTENSION_NAME).
"""
import os
import re
import shutil
import sys

hop, mine = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])


def patch(path, pairs, marker):
    s = open(path).read()
    if marker in s:
        print(f"[fa3_tap] {os.path.basename(path)}: already patched")
        return
    for old, new, count in pairs:
        n = s.count(old)
        assert n == count, f"{path}: anchor found {n}x (expected {count}):\n{old[:200]}"
        s = s.replace(old, new)
    assert marker in s
    open(path, "w").write(s)
    print(f"[fa3_tap] {os.path.basename(path)}: patched")


# 1. headers
for h in ("verity_tap_params.h", "verity_tap_fa3.h"):
    shutil.copy(os.path.join(mine, h), os.path.join(hop, h))

# 2. softmax.h
patch(os.path.join(hop, "softmax.h"), [
    ("    TensorT row_max, row_sum;\n",
     "    TensorT row_max, row_sum;\n"
     "#if defined(VERITY_TAP) && VERITY_TAP == 3\n"
     "    TensorT tap_sum, tap_inv_sum;   // [verity fa3 tap] finalize's quad-reduced sum and inv_sum (row_sum becomes the lse)\n"
     "#endif\n", 1),
    ("            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;\n            scores_scale(mi) = inv_sum * final_scale;\n"
     "            // For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount.\n"
     "            if constexpr (Max_offset != 0) {\n                static constexpr float sum_scale = 1.f / float(1 << Max_offset);\n"
     "                sum *= sum_scale;\n            }\n            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);\n"
     "        }\n        return scores_scale;\n    };\n\n    __forceinline__ __device__ TensorT finalize_aux(",
     "            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;\n            scores_scale(mi) = inv_sum * final_scale;\n"
     "#if defined(VERITY_TAP) && VERITY_TAP == 3\n            tap_sum(mi) = sum; tap_inv_sum(mi) = inv_sum;\n#endif\n"
     "            // For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount.\n"
     "            if constexpr (Max_offset != 0) {\n                static constexpr float sum_scale = 1.f / float(1 << Max_offset);\n"
     "                sum *= sum_scale;\n            }\n            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);\n"
     "        }\n        return scores_scale;\n    };\n\n    __forceinline__ __device__ TensorT finalize_aux(", 1),
], "tap_inv_sum")

# 2b. softmax.h, the MS class (compiled only with VERITY_MAT_SRC & 64): `Softmax::tap_max_scaled` and `scale_apply_exp2_ms`, the pinned
# `scale_apply_exp2` with the same arithmetic that also keeps each row's max_scaled (the register its exp2 FMAs read); online_softmax calls
# it only under the bit
MS_ON = "#if defined(VERITY_MAT_SRC) && (VERITY_MAT_SRC & 64)\n"
patch(os.path.join(hop, "softmax.h"), [
    ("    TensorT row_max, row_sum;\n",
     "    TensorT row_max, row_sum;\n" + MS_ON + "    TensorT tap_max_scaled;   // [verity MS] each row's max * scale this block's exp2 FMAs read\n#endif\n", 1),
    ("template <int kNRows, int Max_offset=0>\nstruct Softmax {\n", MS_ON + r'''// [verity MS] scale_apply_exp2 with the same arithmetic that also keeps each row's max_scaled, the register its exp2 FMAs read, in `ms`
template <bool Scale_max=true, bool Check_inf=true, int Max_offset=0,
        typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void scale_apply_exp2_ms(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale,
                                                    Tensor<Engine2, Layout2> &ms) {
    static constexpr float max_offset = float(Max_offset);
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = Check_inf
            ? (max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset)
            : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
        ms(mi) = max_scaled;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}
#endif

template <int kNRows, int Max_offset=0>
struct Softmax {
''', 1),
    ("        flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);\n",
     MS_ON + "        flash::template scale_apply_exp2_ms</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2, tap_max_scaled);\n"
     "#else\n        flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);\n#endif\n", 1),
], "scale_apply_exp2_ms")

# 3. mainloop
ml = os.path.join(hop, "mainloop_fwd_sm90_tma_gmma_ws.hpp")
s = open(ml).read()
if "Verity_mat_ctx_sm90" not in s:
    s = s.replace('#include "utils.h"\n#include "sm90_pipeline_no_cluster.hpp"\n',
                  '#include "utils.h"\n#include "sm90_pipeline_no_cluster.hpp"\n#include "verity_tap_fa3.h"\n', 1)
    # Arguments / Params trailing member
    a = "        int const* const cp_tot_seqused_k = nullptr;\n    };\n\n    // Device side kernel params\n    struct Params {"
    assert s.count(a) == 1
    s = s.replace(a, "        int const* const cp_tot_seqused_k = nullptr;\n        Verity_tap vt = {};   // [verity fa3 tap]\n    };\n\n    // Device side kernel params\n    struct Params {")
    b = "        int const* const cp_tot_seqused_k = nullptr;\n    };\n\n    static Params\n    to_underlying_arguments(Arguments const& args) {"
    assert s.count(b) == 1
    s = s.replace(b, "        int const* const cp_tot_seqused_k = nullptr;\n        Verity_tap vt = {};   // [verity fa3 tap]\n    };\n\n    static Params\n    to_underlying_arguments(Arguments const& args) {")
    c = "                args.cp_world_size, args.cp_rank, args.cp_tot_seqused_k};\n    }"
    assert s.count(c) == 1
    s = s.replace(c, "                args.cp_world_size, args.cp_rank, args.cp_tot_seqused_k, args.vt};\n    }")
    # mma(): only the IntraWGOverlap branch
    i0 = s.index("        if constexpr (IntraWGOverlap) {\n            Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));")
    i1 = s.index("        } else {  // No intra-WG overlap", i0)
    head, mid, tail = s[:i0], s[i0:i1], s[i1:]
    # ctx init (before the branch): after `int n_block = n_block_max - 1;` of mma()
    anchor = "        int const seqlen_k = seqlen_info.seqlen_k - n_offset;\n        int n_block = n_block_max - 1;\n"
    assert head.count(anchor) == 1
    head = head.replace(anchor, anchor + r'''
#if defined(VERITY_TAP) && VERITY_TAP == 3
        // [verity fa3 tap] M1 materialisation context: slab = bidb * h_k + bidh (PackGQA: bidh is the KV head), rows = packed
        // (q_row, head-in-group) indices, valid while < seqlen_q * qheads_per_kvhead.  Armed (vt.mode == 3) only by the launch template.
        int const vt_qh = !PackGQA ? 1 : params.qhead_per_khead_divmod.divisor;
        Verity_mat_ctx_sm90<kBlockM, kBlockN, kHeadDim> vmat(params.vt, bidb, get<2>(params.shape_K), bidh, m_block, thread_idx, seqlen_q * vt_qh);
        int vt_step = 0; (void)vt_step;
  #define VERITY_MAT(...) if (params.vt.mode != 0 && params.vt.diag_ptr != nullptr) { __VA_ARGS__; }
  #define VERITY_FA3_S(tSrS_, nb) VERITY_MAT(if constexpr (VERITY_MAT_SRC & 1) { vmat.mat_f32(vmat.L.off_s, tSrS_, (nb) * kBlockN); })
  #define VERITY_FA3_P(tSrS_, sc_, nb) VERITY_MAT( \
        if constexpr (VERITY_MAT_SRC & 2) { vmat.mat_f32(vmat.L.off_p, tSrS_, (nb) * kBlockN); } \
        if constexpr (VERITY_MAT_SRC & 8) { vmat.rowstat(softmax.row_max, sc_, softmax.row_sum, (nb), vt_step); } \
        vt_step++; )
  #define VERITY_FA3_RP(tOrP_, tSrS_, nb) VERITY_MAT(if constexpr (VERITY_MAT_SRC & 4) { vmat.mat_bf16(tOrP_, tSrS_.layout(), (nb) * kBlockN); })
  #define VERITY_FA3_O(tOrO_, nb) VERITY_MAT(if constexpr (VERITY_MAT_SRC & 16) { vmat.o(tOrO_, (nb)); })
  #define VERITY_FA3_FIN() VERITY_MAT(if constexpr (VERITY_MAT_SRC & 32) { vmat.fin(softmax.tap_sum, softmax.tap_inv_sum, softmax.row_max, softmax.row_sum); })
#else
  #define VERITY_FA3_S(tSrS_, nb)
  #define VERITY_FA3_P(tSrS_, sc_, nb)
  #define VERITY_FA3_RP(tOrP_, tSrS_, nb)
  #define VERITY_FA3_O(tOrO_, nb)
  #define VERITY_FA3_FIN()
#endif
''')
    reps = [
        # first block
        ("            mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block);\n\n            Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(tSrS);\n",
         "            mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block);\n            VERITY_FA3_S(tSrS, n_block)\n\n            Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(tSrS);\n"),
        ("            softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);\n            if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }\n            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));\n            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);\n            convert_type_out(tOrP_acc, tOrP);\n",
         "            softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);\n            VERITY_FA3_P(tSrS, scores_scale, n_block)\n            if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }\n            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));\n            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);\n            convert_type_out(tOrP_acc, tOrP);\n            VERITY_FA3_RP(tOrP, tSrS, n_block)\n"),
        # fwd_step
        ("                scoremod_premask_fn(tSrS);\n                mask_fn(tSrS, n_block);\n                cute::copy(softmax.template max_get_scale</*Is_first=*/false, Check_inf>(tSrS), scores_scale);\n",
         "                scoremod_premask_fn(tSrS);\n                mask_fn(tSrS, n_block);\n                VERITY_FA3_S(tSrS, n_block)\n                cute::copy(softmax.template max_get_scale</*Is_first=*/false, Check_inf>(tSrS), scores_scale);\n"),
        ("                softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);\n                if constexpr (!HasQv) {\n                    warpgroup_wait<0>();\n                    pipeline_v.consumer_release(smem_pipe_read_v);  // release V\n                }\n                if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }\n                convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);\n",
         "                softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);\n                VERITY_FA3_P(tSrS, scores_scale, n_block)\n                if constexpr (!HasQv) {\n                    warpgroup_wait<0>();\n                    pipeline_v.consumer_release(smem_pipe_read_v);  // release V\n                }\n                VERITY_FA3_O(tOrO, n_block + 1)   // the P.V wgmma of block n_block + 1 has completed: acc_o after that block\n                if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }\n                convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);\n                VERITY_FA3_RP(tOrP, tSrS, n_block)\n"),
        # tail
        ("            warpgroup_wait<0>();\n            pipeline_v.consumer_release(smem_pipe_read);  // release V, otherwise producers will hang\n            softmax.rescale_o(tOrO, scores_scale);\n",
         "            warpgroup_wait<0>();\n            pipeline_v.consumer_release(smem_pipe_read);  // release V, otherwise producers will hang\n            VERITY_FA3_O(tOrO, n_block + 1)   // final P.V (block n_block + 1 == n_block_min) complete\n            VERITY_FA3_FIN()\n            softmax.rescale_o(tOrO, scores_scale);\n"),
    ]
    for old, new in reps:
        assert mid.count(old) == 1, f"mainloop anchor found {mid.count(old)}x:\n{old[:160]}"
        mid = mid.replace(old, new)
    s = head + mid + tail
    open(ml, "w").write(s)
    print("[fa3_tap] mainloop_fwd_sm90_tma_gmma_ws.hpp: patched")
else:
    print("[fa3_tap] mainloop_fwd_sm90_tma_gmma_ws.hpp: already patched")

# 3b. mainloop, the MS class: VERITY_FA3_P also stores softmax.tap_max_scaled (VERITY_MAT_SRC & 64)
patch(ml, [
    ("        if constexpr (VERITY_MAT_SRC & 8) { vmat.rowstat(softmax.row_max, sc_, softmax.row_sum, (nb), vt_step); } \\\n        vt_step++; )\n",
     "        if constexpr (VERITY_MAT_SRC & 8) { vmat.rowstat(softmax.row_max, sc_, softmax.row_sum, (nb), vt_step); } \\\n"
     "        VERITY_FA3_MS(nb) \\\n        vt_step++; )\n", 1),
    ("  #define VERITY_FA3_FIN() VERITY_MAT(if constexpr (VERITY_MAT_SRC & 32) { vmat.fin(softmax.tap_sum, softmax.tap_inv_sum, softmax.row_max, softmax.row_sum); })\n",
     "  #define VERITY_FA3_FIN() VERITY_MAT(if constexpr (VERITY_MAT_SRC & 32) { vmat.fin(softmax.tap_sum, softmax.tap_inv_sum, softmax.row_max, softmax.row_sum); })\n"
     "  #if (VERITY_MAT_SRC & 64)\n  #define VERITY_FA3_MS(nb) vmat.ms(softmax.tap_max_scaled, (nb));\n  #else\n  #define VERITY_FA3_MS(nb)\n  #endif\n", 1),
], "VERITY_FA3_MS")

# 4. launch template
patch(os.path.join(hop, "flash_fwd_launch_template.h"), [
    ("    typename CollectiveMainloop::Arguments mainloop_args {\n",
     r'''#if defined(VERITY_TAP) && VERITY_TAP == 3
    // [verity fa3 tap] arm only when the selected kernel IS the declared one; any other launch is counted and left untapped.
    Verity_tap vt = g_verity_tap;
    if (vt.mode != 0) {
        int const vt_qh = !PackGQA ? 1 : cutlass::ceil_div(params.h, params.h_k);
        int const need_nb = cutlass::ceil_div(params.seqlen_k, kBlockN);
        bool const ok = Arch >= 90 && !Split && params.num_splits <= 1 && PackGQA && !Is_local && !Has_softcap && !AppendKV && !HasQv
            && !Is_FP8 && kHeadDimV == kHeadDim && IntraWGOverlap && MmaPV_is_RS && kBlockN == vt.BN && kHeadDim == vt.D
            && vt.diag_ptr != nullptr && params.b * params.h_k <= vt.H && params.seqlen_q * vt_qh <= vt.M && need_nb <= vt.NB;
        if (!ok) { vt.mode = 0; vt.diag_ptr = nullptr; g_verity_tap_skipped++; }
    }
#else
    Verity_tap vt = {};
#endif
    typename CollectiveMainloop::Arguments mainloop_args {
''', 1),
    ("        params.cp_world_size, params.cp_rank, params.cp_tot_seqused_k\n    };\n    typename CollectiveEpilogue::Arguments epilogue_args {",
     "        params.cp_world_size, params.cp_rank, params.cp_tot_seqused_k,\n        vt   // [verity fa3 tap]\n    };\n    typename CollectiveEpilogue::Arguments epilogue_args {", 1),
], "g_verity_tap_skipped++")

# 5. torch lib with the verity ops
src = open(os.path.join(hop, "flash_api_torch_lib.cpp")).read()
assert "verity_set_tap_v9" not in src
ops_cpp = r'''
// ---- [verity fa3 tap] host-side tap configuration (mirrors the FA2 tap's v8 ops, layout version 9)
#include "verity_tap_params.h"
Verity_tap g_verity_tap = {};
int64_t g_verity_tap_skipped = 0;
#ifndef VERITY_TAP
#define VERITY_TAP 0
#endif
#ifndef VERITY_MAT_SRC
#define VERITY_MAT_SRC 59
#endif
namespace {
// One call per launch: the FULL stream geometry the host committed to -- slabs HB = b * h_k, packed rows M = max_seqlen_q * qheads_per_kvhead,
// key blocks NB, the kernel tile (BN = kBlockN, D = head dim) the layout is arithmetic in, and the per-launch tag.
void verity_set_tap_v9(int64_t mode, std::optional<Tensor> diag, int64_t HB, int64_t M, int64_t NB, int64_t BN, int64_t D, int64_t tag) {
    Verity_tap t = {};
    t.mode = static_cast<int>(mode);
    if (diag.has_value()) { t.diag_ptr = reinterpret_cast<float *>(diag.value().data_ptr()); }
    t.H = static_cast<int>(HB); t.M = static_cast<int>(M); t.N = static_cast<int>(NB * BN); t.NB = static_cast<int>(NB);
    t.BN = static_cast<int>(BN); t.D = static_cast<int>(D); t.tag = static_cast<int>(tag);
    g_verity_tap = t;
}
void verity_set_tap_off() { g_verity_tap = Verity_tap{}; }
int64_t verity_tap_skipped() { return g_verity_tap_skipped; }
int64_t verity_tap_mode() { return VERITY_TAP; }
int64_t verity_tap_src() { return VERITY_MAT_SRC; }
int64_t verity_tap_layout_version() { return 9; }
// what word 3 of a ROW entry holds: 0 = never written, 1 = GuardNegInfZero(row_max) after a row's first key block (VERITY_ROW_GUARD=1)
#ifndef VERITY_ROW_GUARD
#define VERITY_ROW_GUARD 0
#endif
int64_t verity_tap_row3() { return VERITY_ROW_GUARD; }
// bitmask 1 << (hd / 32) of the bf16 paged PackGQA FA3 forward kernels compiled into this build
int64_t verity_tap_hdims() {
    int64_t m = 0;
#ifdef VERITY_HAVE_FA3_HDIM64_BF16
    m |= 1 << 2;
#endif
#ifdef VERITY_HAVE_FA3_HDIM128_BF16
    m |= 1 << 4;
#endif
    return m;
}
}  // namespace
'''
defs = r'''    ops.def("verity_set_tap_v9(int mode, Tensor? diag, int HB, int M, int NB, int BN, int D, int tag) -> ()");
    ops.def("verity_set_tap_off() -> ()");
    ops.def("verity_tap_skipped() -> int");
    ops.def("verity_tap_mode() -> int");
    ops.def("verity_tap_src() -> int");
    ops.def("verity_tap_layout_version() -> int");
    ops.def("verity_tap_hdims() -> int");
    ops.def("verity_tap_row3() -> int");
}
'''
impls = r'''
STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CompositeExplicitAutograd, ops) {
    ops.impl("verity_set_tap_v9", TORCH_BOX(&verity_set_tap_v9));
    ops.impl("verity_set_tap_off", TORCH_BOX(&verity_set_tap_off));
    ops.impl("verity_tap_skipped", TORCH_BOX(&verity_tap_skipped));
    ops.impl("verity_tap_mode", TORCH_BOX(&verity_tap_mode));
    ops.impl("verity_tap_src", TORCH_BOX(&verity_tap_src));
    ops.impl("verity_tap_layout_version", TORCH_BOX(&verity_tap_layout_version));
    ops.impl("verity_tap_hdims", TORCH_BOX(&verity_tap_hdims));
    ops.impl("verity_tap_row3", TORCH_BOX(&verity_tap_row3));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME);
'''
a = "/**\n *  Torch Library Registration\n */"
assert src.count(a) == 1
src = src.replace(a, ops_cpp + "\n" + a)
b = '            "    int      sm_margin) -> Tensor");\n}\n'
assert src.count(b) == 1
src = src.replace(b, '            "    int      sm_margin) -> Tensor");\n' + defs)
c = "REGISTER_EXTENSION(TORCH_EXTENSION_NAME);\n"
assert src.count(c) == 1
src = src.replace(c, impls.lstrip("\n"))
open(os.path.join(hop, "fa3_torch_lib_verity.cpp"), "w").write(src)
print("[fa3_tap] fa3_torch_lib_verity.cpp: written")
print("[fa3_tap] OK")
