/******************************************************************************
 * Verity: M1 hidden-value tap for the FlashAttention-3 forward (hopper/, vllm-flash-attn 506341a1).
 *
 * The FA2 "export-and-commit" tap (csrc/flash_attn/src/verity_tap.h, VERITY_TAP=3) ported to the sm90 warp-specialised
 * kernel `flash::FlashAttnFwdSm90`: the hidden classes selected by VERITY_MAT_SRC are written, in canonical positions, to a
 * preallocated flat f32/u32 buffer (params.vt.diag_ptr) whose layout is `Verity_mat_layout(HB, M, NB, BN, D)` -- the SAME
 * word layout as the FA2 tap (v8), so the host mirror (`hidden_source.mat_total_words` / `hidden_stream.StreamLayout`) is reused
 * verbatim.  No in-kernel hashing: the commitment is computed outside the kernel.
 *
 *   VERITY_MAT_SRC bitmask: 1 S (masked fp32 scores, before softmax), 2 P fp32 (after exp2, un-normalised), 4 rP bf16 (the
 *   packed bf16 P fed to the P.V wgmma; u32 words of column pairs), 8 per-block row state (row_max, rescale, step, lane sums
 *   l0..l3), 16 acc_o after the P.V wgmma of each block (fp32, un-normalised), 32 epilogue (sum, inv_sum, final row_max, lse).
 *   VERITY_ROW_GUARD 1 = the guarded-max build: word 3 of the row state holds GuardNegInfZero(row_max) at every key block after the first a
 *   row visits (0 at the first); 0 (default) = word 3 is never written.  `verity_tap_row3()` reports it.  VERITY_MAT_SRC bit 64 (the same
 *   build) appends the MS class after FIN: per (slab, row, block) the max * scale the block's exp2 FMAs read (softmax.h `tap_max_scaled`,
 *   apply_fa3_tap.py steps 2b / 3b), at every visited block.
 *
 * What differs from FA2 and how it is handled:
 *   * accumulator fragments are wgmma m64nNk16 C fragments ((2, 2, N/8), MMA_M=1, MMA_N=1) instead of mma.sync m16n8k16; the
 *     per-thread ownership is the same (row = warp_in_wg*16 + lane/4 (+8), col = 8n + 2*(lane%4) + j) and hopper/utils.h
 *     `convert_layout_acc_rowcol` exposes it as (2 rows, N/4 cols) -- the store loops below run over that view, so they follow
 *     the tile (kBlockN 128 or 192, kHeadDim 64 or 128) automatically;
 *   * the M tile is kBlockM = 64 (one MMA warpgroup, decode) / 128 (hdim 128) / 192 (hdim 64 prefill): row0 additionally has the
 *     warpgroup offset wg*64 (thread_idx = the consumer thread index, 0-based);
 *   * PackGQA (always on for the paged non-TMA launches vLLM issues): the tile rows are PACKED (query row, q-head-in-group)
 *     pairs, m_packed = q_row * qheads_per_kvhead + head_in_group (mask.h / pack_gqa.h: divmod by qhead_per_khead); the kernel's
 *     bidh is the KV head.  The layout therefore keys slab = bidb * h_k + bidh_kv and row = m_packed (M cap = max_seqlen_q *
 *     qheads_per_kvhead); the row predicate is m_packed < seqlen_q * qheads_per_kvhead.  The host mirror un-packs
 *     (q_row, q_head) = (m // g, bidh_kv * g + m % g);
 *   * IntraWGOverlap schedule (all bf16 hdim 64/128 tiles): the P.V wgmma of block n+1 is issued in the step of block n and
 *     waited for there, so "acc_o after the P.V of block b" is read at that wait (plane index b = n_block + 1) and after the
 *     final P.V; the arithmetic per row is FA2's (rescale-then-add), only the issue order differs;
 *   * the P.V A-operand tOrP is a reshaped view of the SAME registers as tSrS (convert_layout_acc_Aregs is a pure reshape), so
 *     the bf16 plane is read through the accumulator layout of tSrS (no dependency on the A-fragment ordering);
 *   * the epilogue reads `sum` / `inv_sum` from the softmax object (hopper/softmax.h::finalize stashes them under VERITY_TAP;
 *     FA3's inv_sum is (sum == 0 || sum != sum) ? 0 : 1/sum -- Fa3InvSum) and lse from row_sum after finalize.
 *
 * Arming (flash_fwd_launch_template.h): the launch template copies g_verity_tap into the mainloop arguments only if the selected
 * kernel's (kBlockN, kHeadDim) equals the host-declared (BN, D), num_splits == 1 (!Split), no local/softcap/append-KV/Qv/FP8, and
 * the declared caps (HB, M, NB) cover the launch; otherwise the launch is counted in g_verity_tap_skipped and NOTHING is written
 * (the host turns a skip into a collector error: X-03 fails the run closed).
 ******************************************************************************/
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include "utils.h"
#include "verity_tap_params.h"

#ifndef VERITY_TAP
#define VERITY_TAP 0
#endif

namespace flash {

using namespace cute;

using ::Verity_tap;   // verity_tap_params.h (host-declared configuration, also a member of Flash_fwd_params)

struct Verity_mat_layout {
    // word offsets into the flat u32/f32 buffer; identical to the FA2 v8 layout (hidden_source.mat_total_words).
    long off_s, off_p, off_rp, off_row, off_o, off_fin, total;
#if defined(VERITY_MAT_SRC) && (VERITY_MAT_SRC & 64)
    long off_ms;          // MS class (VERITY_MAT_SRC & 64): one word per (slab, row, key block), appended after FIN (as FA2's)
#endif
    int HB, M, N, NB, BN, D;
    __host__ __device__ Verity_mat_layout(int HB_, int M_, int NB_, int BN_, int D_)
        : HB(HB_), M(M_), N(NB_ * BN_), NB(NB_), BN(BN_), D(D_) {
        off_s   = 0;
        off_p   = off_s   + 1L * HB * M * N;
        off_rp  = off_p   + 1L * HB * M * N;
        off_row = off_rp  + 1L * HB * M * (N / 2);
        off_o   = off_row + 1L * HB * M * NB * 8;
        off_fin = off_o   + 1L * HB * M * NB * D;
        total   = off_fin + 1L * HB * M * 4;
#if defined(VERITY_MAT_SRC) && (VERITY_MAT_SRC & 64)
        off_ms  = total;
        total   = off_ms  + 1L * HB * M * NB;
#endif
    }
};

#if defined(VERITY_TAP) && VERITY_TAP == 3
#ifndef VERITY_MAT_SRC
#define VERITY_MAT_SRC 59
#endif

template <int kBlockM_, int kBlockN_, int kHeadDim_>
struct Verity_mat_ctx_sm90 {
    static constexpr int kBlockM = kBlockM_, kBlockN = kBlockN_, kHeadDim = kHeadDim_;
    static_assert(kBlockM % 64 == 0, "verity FA3 tap: kBlockM is a whole number of MMA warpgroups");
    float *buf;
    Verity_mat_layout L;
    int slab, row0, row1, lane, rows_valid;
    // thread_idx: consumer (MMA) thread index 0..NumMmaThreads-1; rows_valid = seqlen_q * qheads_per_kvhead (packed rows)
    __device__ __forceinline__ Verity_mat_ctx_sm90(const Verity_tap &vt, int bidb, int h_k, int bidh_kv, int m_block, int thread_idx, int rows_valid_)
        : buf(vt.diag_ptr), L(vt.H, vt.M, vt.NB, kBlockN, kHeadDim) {
        slab = bidb * h_k + bidh_kv;
        lane = thread_idx % 32;
        const int wg = thread_idx / 128, warp_in_wg = (thread_idx % 128) / 32;
        row0 = m_block * kBlockM + wg * 64 + warp_in_wg * 16 + lane / 4;
        row1 = row0 + 8;
        rows_valid = rows_valid_;
    }
    // fp32 wgmma C fragment -> plane [slab][row][col], adjacent column pairs as float2 (col = col_base + 8n + 2*(lane%4) + j)
    template <typename Tensor0>
    __device__ __forceinline__ void mat_f32(long plane_off, Tensor0 const &acc_s, const int col_base) {
        Tensor sc = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(sc))::value == 2, "verity FA3 tap: 2 rows per thread");
        static_assert(decltype(size<1>(sc))::value == kBlockN / 4, "verity FA3 tap: acc_s fragment width != kBlockN/4 per thread");
        float *plane = buf + plane_off + (long)slab * L.M * L.N;
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int row = mi == 0 ? row0 : row1;
            if (row < rows_valid) {
                float2 *dst = reinterpret_cast<float2 *>(plane + (long)row * L.N + col_base + (lane % 4) * 2);
                #pragma unroll
                for (int n = 0; n < size<1>(sc) / 2; ++n) {
                    dst[n * 4] = make_float2(sc(mi, 2 * n), sc(mi, 2 * n + 1));
                }
            }
        }
    }
    // bf16 P (the P.V A operand): read through the accumulator layout `acc_layout` of the tSrS it was converted from (same
    // registers, same linear order) -> plane [slab][row][col/2] of u32 words (word = cols (c, c+1), c in the low half)
    template <typename TensorP, typename AccLayout>
    __device__ __forceinline__ void mat_bf16(TensorP const &tOrP, AccLayout const &acc_layout, const int col_base) {
        Tensor pv = make_tensor(tOrP.data(), acc_layout);
        Tensor sc = make_tensor(pv.data(), flash::convert_layout_acc_rowcol(pv.layout()));
        static_assert(decltype(size<1>(sc))::value == kBlockN / 4, "verity FA3 tap: P fragment width != kBlockN/4 per thread");
        uint32_t *plane = reinterpret_cast<uint32_t *>(buf) + L.off_rp + (long)slab * L.M * (L.N / 2);
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int row = mi == 0 ? row0 : row1;
            if (row < rows_valid) {
                uint32_t *dst = plane + (long)row * (L.N / 2) + (col_base / 2) + (lane % 4);
                #pragma unroll
                for (int n = 0; n < size<1>(sc) / 2; ++n) {
                    const uint32_t lo = static_cast<uint32_t>(sc(mi, 2 * n).raw());
                    const uint32_t hi = static_cast<uint32_t>(sc(mi, 2 * n + 1).raw());
                    dst[n * 4] = lo | (hi << 16);
                }
            }
        }
    }
    // per-block row state: [slab][row][nblk][8] = row_max, rescale, step, word 3, l0, l1, l2, l3 (this lane's partial sum in its slot).
    // Word 3 as in FA2 v8: never written, except by the guarded-max build (VERITY_ROW_GUARD=1): GuardNegInfZero(row_max) at every block
    // after the first (step > 0), 0 at the first.
    template <typename TensorR>
    __device__ __forceinline__ void rowstat(TensorR const &row_max, TensorR const &scale, TensorR const &row_sum, const int nblk, const int step) {
        float *plane = buf + L.off_row + (long)slab * L.M * L.NB * 8;
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int row = mi == 0 ? row0 : row1;
            if (row < rows_valid) {
                float *dst = plane + ((long)row * L.NB + nblk) * 8;
                if (lane % 4 == 0) { dst[0] = row_max(mi); dst[1] = scale(mi); dst[2] = __int_as_float(step); }
#if defined(VERITY_ROW_GUARD) && VERITY_ROW_GUARD
                if (lane % 4 == 0) { dst[3] = step > 0 && row_max(mi) != -INFINITY ? row_max(mi) : 0.f; }
#endif
                dst[4 + (lane % 4)] = row_sum(mi);
            }
        }
    }
    // acc_o after the P.V wgmma of key block nblk: [slab][row][nblk][kHeadDim]
    template <typename Tensor0>
    __device__ __forceinline__ void o(Tensor0 const &acc_o, const int nblk) {
        Tensor oo = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<1>(oo))::value == kHeadDim / 4, "verity FA3 tap: acc_o fragment width != kHeadDim/4 per thread");
        float *plane = buf + L.off_o + (long)slab * L.M * L.NB * kHeadDim;
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int row = mi == 0 ? row0 : row1;
            if (row < rows_valid) {
                float2 *dst = reinterpret_cast<float2 *>(plane + ((long)row * L.NB + nblk) * kHeadDim + (lane % 4) * 2);
                #pragma unroll
                for (int n = 0; n < size<1>(oo) / 2; ++n) { dst[n * 4] = make_float2(oo(mi, 2 * n), oo(mi, 2 * n + 1)); }
            }
        }
    }
    // epilogue: [slab][row][4] = sum, inv_sum, final row_max, lse
    template <typename TensorR>
    __device__ __forceinline__ void fin(TensorR const &row_sum, TensorR const &inv_sum, TensorR const &row_max, TensorR const &lse) {
        if (lane % 4 != 0) return;
        float *plane = buf + L.off_fin + (long)slab * L.M * 4;
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int row = mi == 0 ? row0 : row1;
            if (row < rows_valid) {
                float4 *dst = reinterpret_cast<float4 *>(plane + (long)row * 4);
                *dst = make_float4(row_sum(mi), inv_sum(mi), row_max(mi), lse(mi));
            }
        }
    }
#if (VERITY_MAT_SRC & 64)
    // MS class: [slab][row][nblk] = the max * scale this block's exp2 FMAs read (Softmax::tap_max_scaled), at every visited block
    template <typename TensorR>
    __device__ __forceinline__ void ms(TensorR const &max_scaled, const int nblk) {
        if (lane % 4 != 0) return;
        float *plane = buf + L.off_ms + (long)slab * L.M * L.NB;
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int row = mi == 0 ? row0 : row1;
            if (row < rows_valid) { plane[(long)row * L.NB + nblk] = max_scaled(mi); }
        }
    }
#endif
};
#endif  // VERITY_TAP == 3

}  // namespace flash
