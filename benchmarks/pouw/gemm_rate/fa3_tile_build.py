"""Build the pinned vllm-flash-attn's FA3 forward (hopper/, 506341a1) for bf16 head dim 64 as a torch extension, its hdim-64
causal tile optionally changed, so attn_knobs.py can time it beside the served wheel's `_vllm_fa3_C`.

    python fa3_tile_build.py --tree TREE --name NS --build-dir DIR [--tile-m 128] [--exp2-emu NUM/DEN --exp2-deg 3]
                              [--lazy-max [checked]]

TREE is a copy of the pinned tree with `integrations/vllm/verity_vllm/acquire/fa3_tap_src/apply_fa3_tap.py` applied (its
`fa3_torch_lib_verity.cpp`), built here with VERITY_TAP=0, which is the pinned kernel. The flags are that builder's
(`build_fa3_ext.py`, after the pinned CMake build of `_vllm_fa3_C`), narrowed to head dim 64: the paged-KV and the
contiguous-KV PackGQA forwards (`flash_fwd_hdim64_bf16_paged_sm90.cu`, `flash_fwd_hdim64_bf16_packgqa_sm90.cu`).

--tile-m M rewrites `tile_size_fwd_sm90`'s hdim-64 causal tile from 192 x 128 to M x 128. kBlockN, MmaPV_is_RS and
IntraWGOverlap stay, so every query row still meets the same 128-key blocks, last to first, through the same
online-softmax arithmetic; only which rows share an M tile changes. The op is `torch.ops.NS.fwd`, with
`_vllm_fa3_C.fwd`'s signature.

--exp2-emu NUM/DEN computes NUM of every DEN of each thread's score columns' exp2 on the FMA pipe (`exp2_fma`, a
degree --exp2-deg polynomial) instead of MUFU, which at head dim 64 does one exp2 a score in the time the tensor cores
take for its 128 MACs. It is not byte-equal: the other columns keep FA3's exact exp2f, and an emulated one differs by at
most the polynomial's relative error (`COEFFS`), under bf16's half-ulp, before P is rounded to bf16.

--lazy-max keeps each warp's running max after a tile's first key block, so a later block skips the max's FMNMX per
score, its shuffles and the output's rescale (`lazy_max_keeps`), which are a fifth of the hdim-64 mainloop's issued
instructions; P may then exceed 1, and the sum and lse use the kept max. Bare, it is a measurement screen: it has no
overflow recovery. `--lazy-max checked` is safe for every input: each exp2 is biased by 2^-32 (FA3's Max_offset, which
the bf16 path leaves at 0; it divides out of the output, and lse adds it back), so a kept block's P stays below 1, as
FA3's, exactly when every biased exponent is negative. Three-input ANDs of the sign bits (half an instruction a score)
test that before any exp2, and a warp that fails takes the exact step on the exponents themselves (`lazy_exp2`). The
bias costs range only below 2^-94 of the kept max, which FA3 also loses to bf16 at 2^-126. Every exp2 then waits on
that warp's vote. `thresh` (FlashAttention-4's rule) still computes each block's max, keeps the old one where a row
rose by at most 8 in log2 units (P up to 2^8), and skips the output's rescale when a warp kept every row. `held` runs a
kept block's exp2 unchecked, tests each row's block sum after it (below 2^16, so every P is; inf and NaN fail), and
reduces that over the warpgroup with one bar.red.or where its warps meet anyway; on a failure the warpgroup recomputes
S from K, which the mainloop then releases only after that test, and redoes the block exactly from the state saved
before it (`redo_softmax`). `held-redo` fails every kept block's test, so its bytes must equal FA3's: the check of the
redo path, which random inputs never take. Without either flag softmax.h is untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

PINNED_HD64 = "return {192, is_causal ? 128 : is_local || paged_kv_non_TMA ? 160 : 192, is_causal || is_local, !is_local};"
EXP_ANCHOR = "// Apply the exp to all the elements."
EXP_LINE = "            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);"
# 2^f on [-0.5, 0.5] with p(0) = 1 pinned, fitted for the least most-relative error: c1, c2, ... (degree 3: 4.6e-4,
# degree 4: 1.2e-5, both evaluated in fp32 by Horner; bf16's half-ulp is 1.95e-3).
COEFFS = {3: (0.6935547590255737, 0.2412298321723938, 0.05179294943809509),
          4: (0.6931335926055908, 0.24021488428115845, 0.05581282824277878, 0.009678369387984276)}


def retile(tile_size_h: Path, m: int) -> None:
    s = tile_size_h.read_text()
    if s.count(PINNED_HD64) != 1:
        raise SystemExit(f"{tile_size_h}: the pinned hdim-64 tile line is not there exactly once")
    tiled = f"return {{is_causal ? {m} : 192, is_causal ? 128 : is_local || paged_kv_non_TMA ? 160 : 192, is_causal || is_local, !is_local}};"
    tile_size_h.write_text(s.replace(PINNED_HD64, tiled))


def exp2_fma(deg: int) -> str:
    c = COEFFS[deg]
    p = f"{c[-1]!r}f"
    for k in reversed(c[:-1]):
        p = f"fmaf({p}, f, {k!r}f)"
    return f"""// 2^x for x < 128 on the FMA pipe: x rounded to n by the 1.5 * 2^23 shifter, the polynomial for 2^(x - n) on
// [-0.5, 0.5], and n added to its exponent. x is clamped at -125 so that exponent stays normal: a masked key then weighs
// 2^-125 rather than 0, below every causal row's sum and output, since a valid row always keeps its own key.
__forceinline__ __device__ float exp2_fma(float x) {{
    x = fmaxf(x, -125.f);
    float const r = __fadd_rn(x, 12582912.f);
    float const f = __fsub_rn(x, __fsub_rn(r, 12582912.f));
    float const p = fmaf({p}, f, 1.f);
    return __int_as_float(__float_as_int(p) + (__float_as_int(r) << 23));
}}

"""


def emulate_exp2(softmax_h: Path, num: int, den: int, deg: int) -> None:
    """Every copy of the pinned exp2f line is split, so the tap's `scale_apply_exp2_ms` keeps scale_apply_exp2's arithmetic."""
    s = softmax_h.read_text()
    if s.count(EXP_ANCHOR) != 1 or s.find(EXP_LINE) < s.find(EXP_ANCHOR):
        raise SystemExit(f"{softmax_h}: scale_apply_exp2's anchor is not there once, or no pinned exp2f line follows it")
    if not 0 < num <= den or deg not in COEFFS:
        raise SystemExit(f"--exp2-emu {num}/{den} --exp2-deg {deg}: need 0 < NUM <= DEN and a degree in {sorted(COEFFS)}")
    split = ("            float const x = tensor(mi, ni) * scale - max_scaled;\n"
             f"            tensor(mi, ni) = ni % {den} < {num} ? exp2_fma(x) : exp2f(x);")
    softmax_h.write_text(s.replace(EXP_ANCHOR, exp2_fma(deg) + EXP_ANCHOR).replace(EXP_LINE, split))


LAZY_HELPER = """// [lazy max] Whether every row of this warp already has a finite running max, which a later key block then keeps:
// its exp2 run against that max, scores_scale is 1, and rescale_o skips the output's multiply. Any row still at -inf
// (all its keys so far masked) sends the whole warp through the exact path. No overflow recovery: a key block whose
// scores exceed the kept max by 128 in log2 units gives inf.
template <typename TensorT>
__forceinline__ __device__ bool lazy_max_keeps(TensorT const& row_max) {
    bool finite = true;
    #pragma unroll
    for (int mi = 0; mi < size(row_max); ++mi) { finite &= row_max(mi) != -INFINITY; }
    return __all_sync(0xffffffffu, finite);
}

"""
LAZY_PATCHES = (
    ("template <int kNRows, int Max_offset=0>\nstruct Softmax {\n", LAZY_HELPER + "template <int kNRows, int Max_offset=0>\nstruct Softmax {\n"),
    ("    float const softmax_scale_log2;\n", "    float const softmax_scale_log2;\n    bool skip_rescale = false;\n"),
    ("        } else {\n            Tensor scores_max_prev = make_fragment_like(row_max);\n",
     "        } else if (lazy_max_keeps(row_max)) {\n            cute::fill(scores_scale, 1.f);\n            skip_rescale = true;\n"
     "        } else {\n            skip_rescale = false;\n            Tensor scores_max_prev = make_fragment_like(row_max);\n"),
    ("    __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {\n",
     "    __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {\n        skip_rescale = false;\n"),
    ("TensorT finalize_aux(TensorT const& tSrSAux, float const final_scale=1.f) {\n",
     "TensorT finalize_aux(TensorT const& tSrSAux, float const final_scale=1.f) {\n        skip_rescale = false;\n"),
    ("    __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {\n",
     "    __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {\n        if (skip_rescale) { return; }\n"),
)


LAZY_BIAS = 32
CHECKED_EXP2 = """    // [lazy max, checked] Every exp2 is biased by 2^Max_offset (Max_offset < 0), so a kept block is as safe as FA3's
    // exact one exactly when each x = s * scale - (max * scale - Max_offset) is negative, i.e. P < 1. The sign bits,
    // three-input ANDs over the scores, decide that before any exp2; a warp with a nonnegative x takes the exact step
    // on x itself, so no recovery needs S again.
    template<typename Tensor0>
    __forceinline__ __device__ void lazy_exp2(Tensor0 &scores) {
        uint32_t neg = 0xffffffffu;
        #pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            float const max_scaled = row_max(mi) * softmax_scale_log2 - float(Max_offset);
            #pragma unroll
            for (int ni = 0; ni < size<1>(scores); ni += 2) {
                scores(mi, ni) = scores(mi, ni) * softmax_scale_log2 - max_scaled;
                scores(mi, ni + 1) = scores(mi, ni + 1) * softmax_scale_log2 - max_scaled;
                neg &= __float_as_uint(scores(mi, ni)) & __float_as_uint(scores(mi, ni + 1));
            }
        }
        lazy_pending = !__all_sync(0xffffffffu, neg >> 31);
        if (!lazy_pending) {
            #pragma unroll
            for (int i = 0; i < size(scores); ++i) { scores(i) = exp2f(scores(i)); }
            return;
        }
        TensorT x_max;
        flash::template reduce_max</*zero_init=*/true>(scores, x_max);
        #pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            float const d = fmaxf(x_max(mi) - float(Max_offset), 0.f);
            lazy_scale(mi) = exp2f(-d);
            row_max(mi) += d / softmax_scale_log2;
            row_sum(mi) *= lazy_scale(mi);
            #pragma unroll
            for (int ni = 0; ni < size<1>(scores); ++ni) { scores(mi, ni) = exp2f(scores(mi, ni) - d); }
        }
    }

"""
SUM_SCALE = "static constexpr float sum_scale = 1.f / float(1 << Max_offset);"
CHECKED_PATCHES = (
    ("    bool skip_rescale = false;\n", "    bool skip_rescale = false;\n    bool lazy_pending = false;\n    TensorT lazy_scale;\n"),
    ("            cute::fill(scores_scale, 1.f);\n            skip_rescale = true;\n",
     "            cute::fill(scores_scale, 1.f);\n            skip_rescale = true;\n            lazy_pending = false;\n"),
    ("    template<bool Is_first, bool Check_inf=false, typename Tensor0>\n    __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {\n",
     CHECKED_EXP2 + "    template<bool Is_first, bool Check_inf=false, typename Tensor0>\n    __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {\n"),
    ("        flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);\n",
     "        if (!Is_first && skip_rescale) {\n            lazy_exp2(scores);\n        } else {\n"
     "            flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);\n        }\n"),
    ("        if (skip_rescale) { return; }\n", "        if (skip_rescale && !lazy_pending) { return; }\n"),
    ("            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }\n",
     "            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= skip_rescale ? lazy_scale(mi) : scores_scale(mi); }\n"),
)
SOFTMAX_PINNED = "flash::Softmax<!LargeHeadDimV ? 2 * (2 * kBlockM / NumMmaThreads) : 2, /*Max_offset=*/!Is_FP8 ? 0 : 8> softmax(softmax_scale_log2);"


THRESH_TAU = 8
THRESH_KEEP = f"""            bool kept = true;
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {{
                // [lazy max, thresh] A row whose max rose by at most {THRESH_TAU} in log2 units keeps the old one (P up to
                // 2^{THRESH_TAU}); false for -inf on both sides (NaN) and for a first finite max, which take FA3's step.
                if ((row_max(mi) - scores_max_prev(mi)) * softmax_scale_log2 <= {THRESH_TAU}.f) {{
                    row_max(mi) = scores_max_prev(mi);
                    scores_scale(mi) = 1.f;
                    continue;
                }}
                kept = false;
"""
THRESH_PATCHES = (
    ("    float const softmax_scale_log2;\n", "    float const softmax_scale_log2;\n    bool skip_rescale = false;\n"),
    ("            flash::template reduce_max</*zero_init=*/false>(scores, row_max);\n            #pragma unroll\n"
     "            for (int mi = 0; mi < size(row_max); ++mi) {\n",
     "            flash::template reduce_max</*zero_init=*/false>(scores, row_max);\n" + THRESH_KEEP),
    ("                row_sum(mi) *= scores_scale(mi);\n            }\n        }\n        return scores_scale;\n",
     "                row_sum(mi) *= scores_scale(mi);\n            }\n            skip_rescale = __all_sync(0xffffffffu, kept);\n"
     "        }\n        return scores_scale;\n"),
) + LAZY_PATCHES[3:]


HELD_LIMIT = 65536
HELD_FNS = f"""    // [lazy max, held] A kept block's exp2 run against the kept max unchecked; each thread then tests its rows' block
    // sums (below {HELD_LIMIT}, so every P is, and inf or NaN fails), and one bar.red.or a warpgroup says whether any
    // failed. Then the warpgroup recomputes S from K, which the mainloop holds until here, and every warp redoes the
    // block exactly from the state it had before it.
    __forceinline__ __device__ bool lazy_redo(int barrier_id) {{
        int r;
        asm volatile("{{\\n .reg .pred p, q;\\n setp.ne.b32 p, %1, 0;\\n bar.red.or.pred q, %2, 128, p;\\n selp.s32 %0, 1, 0, q;\\n}}\\n"
                     : "=r"(r) : "r"(int(lazy_bad)), "r"(barrier_id));
        return r != 0;
    }}

    template<typename Tensor0>
    __forceinline__ __device__ void redo_softmax(Tensor0 &acc_s) {{
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        cute::copy(saved_max, row_max);
        cute::copy(saved_sum, row_sum);
        flash::template reduce_max</*zero_init=*/false>(scores, row_max);
        #pragma unroll
        for (int mi = 0; mi < size(row_max); ++mi) {{
            float const cur = row_max(mi) == -INFINITY ? 0.f : row_max(mi);
            lazy_scale(mi) = exp2f((saved_max(mi) - cur) * softmax_scale_log2);
            row_sum(mi) *= lazy_scale(mi);
        }}
        flash::template scale_apply_exp2</*Scale_max=*/true, /*Check_inf=*/true, Max_offset>(scores, row_max, softmax_scale_log2);
        flash::reduce_sum</*zero_init=*/false, /*warp_reduce=*/false>(scores, row_sum);
        skip_rescale = true;
        lazy_pending = true;
        lazy_bad = false;
    }}

"""
HELD_PATCHES = (
    ("    bool skip_rescale = false;\n",
     "    bool skip_rescale = false;\n    bool lazy_pending = false;\n    bool lazy_bad = false;\n    TensorT lazy_scale, saved_max, saved_sum;\n"),
    ("            cute::fill(scores_scale, 1.f);\n            skip_rescale = true;\n",
     "            cute::fill(scores_scale, 1.f);\n            skip_rescale = true;\n            lazy_pending = false;\n"),
    ("    __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {\n",
     "    __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {\n        cute::copy(row_max, saved_max);\n"
     "        cute::copy(row_sum, saved_sum);\n"),
    ("    template<bool Is_first, bool Check_inf=false, typename Tensor0>\n    __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {\n",
     HELD_FNS + "    template<bool Is_first, bool Check_inf=false, typename Tensor0>\n    __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {\n"),
    ("        flash::reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, row_sum);\n    };\n",
     "        flash::reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, row_sum);\n        lazy_bad = false;\n"
     "        if (!Is_first && skip_rescale) {\n            #pragma unroll\n"
     f"            for (int mi = 0; mi < size(row_sum); ++mi) {{ lazy_bad |= !(row_sum(mi) - saved_sum(mi) < {HELD_LIMIT}.f); }}\n"
     "        }\n    };\n"),
) + CHECKED_PATCHES[4:]
HELD_MAINLOOP = (
    ("                warpgroup_wait<1>();\n                pipeline_k.consumer_release(smem_pipe_read);  // release K\n"
     "                if constexpr (HasQv) {\n",
     "                warpgroup_wait<1>();\n                if constexpr (HasQv) {\n"),
    ("                    pipeline_v.consumer_release(smem_pipe_read_v);  // release V\n                }\n"
     "                VERITY_FA3_O(tOrO, n_block + 1)",
     "                    pipeline_v.consumer_release(smem_pipe_read_v);  // release V\n                }\n"
     "                static_assert(!Transpose_V && !AppendKV,\n"
     "                              \"the held lazy max's barriers are FA3's Transpose, Transform and AppendKV ones\");\n"
     "                if (softmax.lazy_redo(warp_group_idx == 0 ? int(cutlass::arch::ReservedNamedBarriers::TransposeBarrier)\n"
     "                                      : warp_group_idx == 1 ? int(cutlass::arch::ReservedNamedBarriers::TransformBarrier)\n"
     "                                      : int(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier) + int(FwdNamedBarriers::AppendKV))) {\n"
     "                    flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);\n"
     "                    scoremod_premask_fn(tSrS);\n                    mask_fn(tSrS, n_block);\n                    softmax.redo_softmax(tSrS);\n"
     "                }\n                pipeline_k.consumer_release(smem_pipe_read);  // release K\n"
     "                VERITY_FA3_O(tOrO, n_block + 1)"),
)


def patch_once(path: Path, s: str, patches) -> str:
    for old, new in patches:
        if s.count(old) != 1:
            raise SystemExit(f"{path}: the lazy-max anchor {old.strip()[:60]!r} is not there exactly once")
        s = s.replace(old, new)
    return s


def lazy_max(softmax_h: Path, checked: bool = False, thresh: bool = False, held: bool = False, redo: bool = False) -> None:
    """`redo` (with `held`) fails every kept block's test, so each is recomputed and redone exactly: FA3's bytes."""
    if thresh:
        softmax_h.write_text(patch_once(softmax_h, softmax_h.read_text(), THRESH_PATCHES))
        return
    s = patch_once(softmax_h, softmax_h.read_text(), LAZY_PATCHES)
    if held:
        mainloop_h = softmax_h.parent / "mainloop_fwd_sm90_tma_gmma_ws.hpp"
        mainloop_h.write_text(patch_once(mainloop_h, mainloop_h.read_text(), HELD_MAINLOOP))
        s = patch_once(softmax_h, s, HELD_PATCHES)
        softmax_h.write_text(s.replace(f"< {HELD_LIMIT}.f", "< -1.f") if redo else s)
        return
    if checked:
        s = patch_once(softmax_h, s, CHECKED_PATCHES)
        if s.count(SUM_SCALE) != 2:
            raise SystemExit(f"{softmax_h}: finalize's and finalize_aux's sum_scale lines are not there twice")
        s = s.replace(SUM_SCALE, "static constexpr float sum_scale = Max_offset > 0 ? 1.f / float(1 << Max_offset) "
                                 ": float(1ull << -Max_offset);")
        kernel_h = softmax_h.parent / "flash_fwd_kernel_sm90.h"
        k = kernel_h.read_text()
        if k.count(SOFTMAX_PINNED) != 1:
            raise SystemExit(f"{kernel_h}: the pinned Softmax instantiation is not there exactly once")
        kernel_h.write_text(k.replace(SOFTMAX_PINNED, SOFTMAX_PINNED.replace("!Is_FP8 ? 0 : 8", f"!Is_FP8 ? -{LAZY_BIAS} : 8")))
    softmax_h.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--build-dir", type=Path, required=True)
    ap.add_argument("--tile-m", type=int)
    ap.add_argument("--exp2-emu", help="NUM/DEN: that share of each thread's score columns' exp2 on the FMA pipe")
    ap.add_argument("--exp2-deg", type=int, default=3, help="the FMA exp2's polynomial degree (3 or 4)")
    ap.add_argument("--lazy-max", nargs="?", const="screen", choices=("screen", "checked", "thresh", "held", "held-redo"),
                    help="keep a warp's running max after the first key block (checked: biased exp2, sign-tested; "
                         "thresh: the max computed, kept while it rises by at most 8 in log2 units; held: block sums "
                         "tested after exp2, K held for a recompute; held-redo: every kept block recomputed, a test)")
    ap.add_argument("--jobs", default="8")
    ap.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"), help="the CUDA toolkit whose libcuda stub links")
    args = ap.parse_args()
    hop = args.tree / "hopper"
    if args.tile_m:
        retile(hop / "tile_size.h", args.tile_m)
    emu = None
    if args.exp2_emu:
        num, den = (int(x) for x in args.exp2_emu.split("/"))
        emulate_exp2(hop / "softmax.h", num, den, args.exp2_deg)
        emu = {"num": num, "den": den, "deg": args.exp2_deg, "max_rel": {3: 4.6e-4, 4: 1.2e-5}[args.exp2_deg]}
    if args.lazy_max:
        lazy_max(hop / "softmax.h", checked=args.lazy_max == "checked", thresh=args.lazy_max == "thresh",
                 held=args.lazy_max.startswith("held"), redo=args.lazy_max == "held-redo")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    os.environ["MAX_JOBS"] = args.jobs
    from torch.utils.cpp_extension import load

    kernels = [hop / "instantiations" / f for f in ("flash_fwd_hdim64_bf16_paged_sm90.cu", "flash_fwd_hdim64_bf16_packgqa_sm90.cu")]
    sources = [hop / "flash_api.cpp", hop / "fa3_torch_lib_verity.cpp", hop / "flash_prepare_scheduler.cu"] + kernels
    defines = ["FLASHATTENTION_DISABLE_BACKWARD", "FLASHATTENTION_DISABLE_DROPOUT", "FLASHATTENTION_DISABLE_UNEVEN_K",
               "FLASHATTENTION_DISABLE_PYBIND", "FLASHATTENTION_VARLEN_ONLY", "FLASHATTENTION_PACKGQA_ONLY",
               "FLASHATTENTION_DISABLE_CLUSTER", "FLASHATTENTION_DISABLE_SM8x", "FLASHATTENTION_DISABLE_APPENDKV",
               "CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "CUTLASS_ENABLE_GDC_FOR_SM90", "TORCH_TARGET_VERSION=0x020A000000000000ULL",
               "USE_CUDA", "CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1", "FLASHATTENTION_DISABLE_FP16", "FLASHATTENTION_DISABLE_FP8",
               "FLASHATTENTION_DISABLE_SOFTCAP", "FLASHATTENTION_DISABLE_LOCAL", "FLASHATTENTION_DISABLE_SPLIT",
               "FLASHATTENTION_DISABLE_HDIM96", "FLASHATTENTION_DISABLE_HDIM128", "FLASHATTENTION_DISABLE_HDIM192",
               "FLASHATTENTION_DISABLE_HDIM256", "FLASHATTENTION_DISABLE_HDIMDIFF64", "FLASHATTENTION_DISABLE_HDIMDIFF192",
               "VERITY_TAP=0", "VERITY_HAVE_FA3_HDIM64_BF16"]
    dflags = ["-D" + d for d in defines]
    nvcc = ["-O3", "-DNDEBUG", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
            "-Xptxas", "-v"] + dflags
    args.build_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    load(name=args.name, sources=[str(s) for s in sources], extra_cflags=["-O3", "-DNDEBUG", "-std=c++20"] + dflags,
         extra_cuda_cflags=nvcc, extra_ldflags=[f"-L{args.cuda}/lib64/stubs", "-lcuda"],
         extra_include_paths=[str(hop), str(args.tree / "csrc" / "common"), str(args.tree / "csrc" / "cutlass" / "include")],
         build_directory=str(args.build_dir), verbose=True, is_python_module=False, with_cuda=True)
    tile = [ln.strip() for ln in (hop / "tile_size.h").read_text().splitlines()
            if "is_causal ? 128 : is_local || paged_kv_non_TMA ? 160 : 192" in ln and not ln.strip().startswith("//")]
    info = {"name": args.name, "tile_m": args.tile_m, "exp2_emu": emu, "lazy_max": args.lazy_max, "seconds": round(time.time() - t0, 1),
            "so": str(args.build_dir / f"{args.name}.so"), "hd64_tile_line": tile}
    (args.build_dir / "build_info.json").write_text(json.dumps(info, indent=1))
    print("BUILD-OK", json.dumps(info))


if __name__ == "__main__":
    main()
