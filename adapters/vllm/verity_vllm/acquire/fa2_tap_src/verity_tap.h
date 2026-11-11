/******************************************************************************
 * Verity: in-kernel tap for the FA2 split-KV forward.
 *
 *   VERITY_TAP          0 = compiled out (should be SASS-identical to pinned kernel
 *                           up to kernel-parameter offsets)
 *                       1 = P1 accumulator (non-cryptographic, cannot be optimized away)
 *                       2 = P0 diagnostic writes of live values (+ P1) -- tiny inputs only
 *   VERITY_P1_SRC       bitmask of live values consumed by P1:
 *                         1 = P fp32 (acc_s after exp2, i.e. the un-normalised probabilities)
 *                         2 = s fp32 (acc_s after masking, before softmax; raw QK^T fp32)
 *                         4 = rP bf16 (the packed bf16 P actually fed to the PV MMA)
 *                         8 = per-block row statistics (running row_max, row_sum partial)
 *   VERITY_TAP_RUNTIME  1 = P1 code guarded by a uniform `params.vt.mode != 0` branch
 *                       0 = P1 unconditional
 *   VERITY_P1_HASH      1 = replace the 3-op accumulator by a ChaCha-style ARX quarter round per word ("P2",
 *                           first commitment-primitive implementation, no security claim); 0 = P1 accumulator;
 *                       2 = "P3": absorb + 3 extra mixing quarter rounds (48 ops/word, real-compression op density)
 *                       3 = "P4 poly": keyed Horner accumulator mod 2^32, 1 IMAD per word (universal-hash-style leaf probe)
 *   VERITY_EPI_HASH     1 = hierarchical leaf: keep the cheap in-loop fold, compress the per-warp 4-word digest once at
 *                           kernel end with the ChaCha20 permutation (960 ops per warp per launch); 0 = store the raw digest;
 *                       2 = same with ChaCha8 (4 double rounds, 384 ops); 3 = ChaCha20 issued right after the softmax
 *                           epilogue (before the O smem/gmem copies) so its dependency chain can overlap the O stores
 *
 * P1 output: per (bidb, bidh, m_block, warp) 4 x uint32 = 16 bytes, i.e. 64 bytes per
 * (batch, head, 64-query block) for kNWarps = 4, written once at kernel end.
 * P0 output: flat fp32 buffer, planes described in verity_p0_layout() (bench_fa2_tap.py mirrors it).
 *
 * All helpers are __forceinline__ and only touch registers already live in the kernel.
 ******************************************************************************/
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include "namespace_config.h"
#include "utils.h"

#ifndef VERITY_TAP
#define VERITY_TAP 0
#endif
#ifndef VERITY_P1_SRC
#define VERITY_P1_SRC 1
#endif
#ifndef VERITY_TAP_RUNTIME
#define VERITY_TAP_RUNTIME 0
#endif
#ifndef VERITY_P1_HASH
#define VERITY_P1_HASH 0
#endif
#ifndef VERITY_EPI_HASH
#define VERITY_EPI_HASH 0
#endif

namespace FLASH_NAMESPACE {

using namespace cute;

// P0 plane layout (all fp32), sizes in elements; H/M/N/NB are the caps in Verity_tap.
//   mat  [4][H][M][N]        stages: 0 s_raw (post QK^T MMA), 1 s_masked, 2 P fp32 (post exp2), 3 rP bf16 (widened)
//   row  [4][H][M][NB]       stages: 0 running row_max after block, 1 rescale factor applied this block,
//                                    2 processing step index, 3 (unused)
//   rowq [2][H][M][NB][4]    stages: 0 thread-partial running row_sum after block (per quad lane), 1 (unused)
//   o    [H][M][NB][64]      acc_o after the PV MMA of this block (fp32, un-normalised)
//   fin  [4][H][M]           0 row_sum (quad-reduced), 1 inv_sum, 2 lse, 3 final row_max
struct Verity_p0_layout {
    long off_mat, off_row, off_rowq, off_o, off_fin, total;
    __host__ __device__ Verity_p0_layout(int H, int M, int N, int NB) {
        off_mat = 0;
        off_row = off_mat + 4L * H * M * N;
        off_rowq = off_row + 4L * H * M * NB;
        off_o = off_rowq + 2L * H * M * NB * 4;
        off_fin = off_o + 1L * H * M * NB * 64;
        total = off_fin + 4L * H * M;
    }
};

#if VERITY_P1_HASH == 3
// "P4 poly" (VERITY_P1_HASH=3): keyed polynomial (Horner) accumulator mod 2^32 -- 1 IMAD per absorbed word, position bound
// by the Horner order.  The key is derived from the block index here only to keep it in a register (cost probe of a
// universal-hash-style leaf); a deployment would take per-launch key material from params.  No security claim.
struct Verity_acc {
    uint32_t a, k;
    __device__ __forceinline__ void init(uint32_t seed) { k = (0x9E3779B1u ^ (blockIdx.x * 0x85EBCA6Bu)) | 1u; a = seed; }
    __device__ __forceinline__ void fold(uint32_t w, uint32_t /*col*/) { a = a * k + w; }
    __device__ __forceinline__ uint32_t digest() const { return a; }
};
#elif VERITY_P1_HASH
// "P2" (VERITY_P1_HASH=1): first commitment-primitive implementation -- one ChaCha-style ARX quarter round per absorbed
// 32-bit word (4 add, 4 xor, 4 rotate = 12 ops), 4-word state per row per thread.  Position is bound by absorption
// order (fixed per thread) and by the lane mixing in verity_p1_finish; `col` is not absorbed.  No security claim.
// "P3" (VERITY_P1_HASH=2): the same absorb followed by 3 further mixing quarter rounds (48 ops/word) -- the op density
// of a real ARX compression (BLAKE3/ChaCha: ~3.5 G functions ~ 50 ops per message word); a cost probe, no security claim.
#ifndef VERITY_HASH_ROUNDS
#define VERITY_HASH_ROUNDS (VERITY_P1_HASH >= 2 ? 4 : 1)
#endif
struct Verity_acc {
    uint32_t a, b, c, d;
    __device__ __forceinline__ void init(uint32_t seed) { a = 0x61707865u ^ seed; b = 0x3320646Eu; c = 0x79622D32u; d = 0x6B206574u; }
    __device__ __forceinline__ void qr(uint32_t w) {
        a += w; d ^= a; d = __funnelshift_l(d, d, 16);
        c += d; b ^= c; b = __funnelshift_l(b, b, 12);
        a += b; d ^= a; d = __funnelshift_l(d, d, 8);
        c += d; b ^= c; b = __funnelshift_l(b, b, 7);
    }
    __device__ __forceinline__ void fold(uint32_t w, uint32_t /*col*/) {
        qr(w);                                       // P2: exactly the previous single round (w absorbed into a)
        #pragma unroll
        for (int r = 1; r < VERITY_HASH_ROUNDS; ++r) { qr(b); }   // P3: 3 more mixing rounds (ChaCha's a += b form)
    }
    __device__ __forceinline__ uint32_t digest() const { return (a ^ c) * 0x9E3779B1u ^ (b ^ d); }
};
#else
// P1: deliberately non-cryptographic rotate-xor-add accumulator (3 ops per absorbed word).
struct Verity_acc {
    uint32_t a;
    __device__ __forceinline__ void init(uint32_t seed) { a = seed; }
    __device__ __forceinline__ void fold(uint32_t w, uint32_t col) { a = (__funnelshift_l(a, a, 5) ^ w) + col; }
    __device__ __forceinline__ uint32_t digest() const { return a; }
};
#endif

struct Verity_p1_state {
    Verity_acc accP[2];
    Verity_acc accS[2];
    Verity_acc accR[2];
    uint32_t nblocks;
    __device__ __forceinline__ void init() {
        accP[0].init(0x243F6A88u); accP[1].init(0x243F6A88u);   // arbitrary non-zero seeds
        accS[0].init(0x85A308D3u); accS[1].init(0x85A308D3u);
        accR[0].init(0x13198A2Eu); accR[1].init(0x13198A2Eu);
        nblocks = 0;
    }
};

// Fold an fp32 (MMA=4, MMA_M, MMA_N) accumulator fragment, row-wise, into acc[2*MMA_M].
template <typename Tensor0, int NR>
__device__ __forceinline__ void verity_p1_fold_f32(Verity_acc (&acc)[NR], Tensor0 const &acc_s, const int col_base) {
    Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(decltype(size<0>(scores))::value == NR);
    const int lane = threadIdx.x % 32;
    #pragma unroll
    for (int mi = 0; mi < size<0>(scores); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(scores); ++ni) {
            const int col = col_base + (ni >> 1) * 8 + (lane % 4) * 2 + (ni & 1);
            acc[mi].fold(__float_as_uint(scores(mi, ni)), (uint32_t)col);
        }
    }
}

// Fold the packed bf16 P fragment (layout (4, MMA_M, MMA_N) of bf16 -> (2, MMA_M, MMA_N) of u32 words).
template <typename Tensor0, int NR>
__device__ __forceinline__ void verity_p1_fold_bf16x2(Verity_acc (&acc)[NR], Tensor0 const &rP, const int col_base) {
    Tensor words = recast<uint32_t>(rP);   // (2, MMA_M, MMA_N): word k -> row (k%2)*8, cols (c, c+1)
    const int lane = threadIdx.x % 32;
    #pragma unroll
    for (int mi = 0; mi < size<1>(words); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<2>(words); ++ni) {
            const int col = col_base + ni * 8 + (lane % 4) * 2;
            #pragma unroll
            for (int k = 0; k < 2; ++k) {
                acc[k + 2 * mi].fold(words(k, mi, ni), (uint32_t)col);
            }
        }
    }
}

template <typename TensorR, int NR>
__device__ __forceinline__ void verity_p1_fold_rowstats(Verity_acc (&acc)[NR], TensorR const &row_max, TensorR const &row_sum, const int n_block) {
    #pragma unroll
    for (int mi = 0; mi < NR; ++mi) {
        acc[mi].fold(__float_as_uint(row_max(mi)), (uint32_t)n_block);
        acc[mi].fold(__float_as_uint(row_sum(mi)), (uint32_t)n_block);
    }
}

// Warp-reduce the per-thread state and store 4 words per warp.
// Rows >= actual_seqlen_q (padding rows of the last 64-row block) are excluded: FA2 does not clear the OOB rows of
// the Q smem tile ("we'll only write out the valid outputs"), so their acc_s / P values are uninitialised smem
// garbage and differ from launch to launch (observed: non-deterministic digests before this guard).
#if VERITY_EPI_HASH
#define VERITY_FINISH_RET uint4
#else
#define VERITY_FINISH_RET void
#endif
template <typename Params>
__device__ __forceinline__ VERITY_FINISH_RET verity_p1_finish(const Params &params, const Verity_p1_state &st,
                                                 const int bidb, const int bidh, const int m_block, const int kNWarps,
                                                 const int row0, const int actual_seqlen_q) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const bool ok0 = row0 < actual_seqlen_q, ok1 = row0 + 8 < actual_seqlen_q;
    uint32_t w0 = ((ok0 ? st.accP[0].digest() : 0u) * 0x9E3779B1u) ^ (ok1 ? st.accP[1].digest() : 0u);
    uint32_t w1 = ((ok0 ? st.accS[0].digest() : 0u) * 0x9E3779B1u) ^ (ok1 ? st.accS[1].digest() : 0u);
    uint32_t w2 = ((ok0 ? st.accR[0].digest() : 0u) * 0x9E3779B1u) ^ (ok1 ? st.accR[1].digest() : 0u);
    uint32_t w3 = st.nblocks;
    // lane-dependent mixing so that a permutation of rows across lanes changes the result
    w0 = __funnelshift_l(w0, w0, lane) ^ (lane * 0x85EBCA6Bu);
    w1 = __funnelshift_l(w1, w1, lane) ^ (lane * 0xC2B2AE35u);
    w2 = __funnelshift_l(w2, w2, lane) ^ (lane * 0x27D4EB2Fu);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        w0 ^= __shfl_xor_sync(0xffffffffu, w0, off);
        w1 ^= __shfl_xor_sync(0xffffffffu, w1, off);
        w2 ^= __shfl_xor_sync(0xffffffffu, w2, off);
        w3 += __shfl_xor_sync(0xffffffffu, w3, off);
    }
#if VERITY_EPI_HASH
    return make_uint4(w0, w1, w2, w3);   // uniform across the warp; hashed + stored by verity_p1_store at kernel end
#else
    if (lane == 0 && params.vt.acc_ptr != nullptr) {
        const long idx = ((((long)bidb * params.h + bidh) * gridDim.x + m_block) * kNWarps + warp) * 4;
        uint4 *dst = reinterpret_cast<uint4 *>(reinterpret_cast<uint32_t *>(params.vt.acc_ptr) + idx);
        *dst = make_uint4(w0, w1, w2, w3);
    }
#endif
}

#if VERITY_EPI_HASH
#define VERITY_EPI_ROUNDS (VERITY_EPI_HASH == 2 ? 4 : 10)
// "P1+epi" (VERITY_EPI_HASH=1): hierarchical leaf -- the cheap per-word fold stays in the block loop; once per warp per
// launch, after the O/lse stores, the reduced 4-word digest is compressed with the ChaCha20 permutation (16-word state:
// 4 constants, the 4 digest words, (bidb, bidh, m_block, warp), 4 constants; 10 double rounds; feed-forward; 4-word
// xor-fold output).  Every lane holds the same words, so all 32 lanes compute it uniformly (no divergence, same latency
// as one lane).  First implementation, cost probe: no security claim (not a certified hash construction).
__device__ __forceinline__ void verity_chacha_qr(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
    a += b; d ^= a; d = __funnelshift_l(d, d, 16);
    c += d; b ^= c; b = __funnelshift_l(b, b, 12);
    a += b; d ^= a; d = __funnelshift_l(d, d, 8);
    c += d; b ^= c; b = __funnelshift_l(b, b, 7);
}
template <typename Params>
__device__ __forceinline__ void verity_p1_store(const Params &params, const uint4 w, const int bidb, const int bidh,
                                                const int m_block, const int kNWarps) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    uint32_t x[16] = {0x61707865u, 0x3320646Eu, 0x79622D32u, 0x6B206574u, w.x, w.y, w.z, w.w,
                      (uint32_t)bidb, (uint32_t)bidh, (uint32_t)m_block, (uint32_t)warp,
                      0x9E3779B9u, 0x243F6A88u, 0xB7E15162u, 0x6A09E667u};
    uint32_t in[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) { in[i] = x[i]; }
    #pragma unroll
    for (int r = 0; r < VERITY_EPI_ROUNDS; ++r) {   // ChaCha20: 10 double rounds = 80 quarter rounds = 960 int ops (ChaCha8: 4)
        verity_chacha_qr(x[0], x[4], x[8],  x[12]); verity_chacha_qr(x[1], x[5], x[9],  x[13]);
        verity_chacha_qr(x[2], x[6], x[10], x[14]); verity_chacha_qr(x[3], x[7], x[11], x[15]);
        verity_chacha_qr(x[0], x[5], x[10], x[15]); verity_chacha_qr(x[1], x[6], x[11], x[12]);
        verity_chacha_qr(x[2], x[7], x[8],  x[13]); verity_chacha_qr(x[3], x[4], x[9],  x[14]);
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) { x[i] += in[i]; }
    if (lane == 0 && params.vt.acc_ptr != nullptr) {
        const long idx = ((((long)bidb * params.h + bidh) * gridDim.x + m_block) * kNWarps + warp) * 4;
        uint4 *dst = reinterpret_cast<uint4 *>(reinterpret_cast<uint32_t *>(params.vt.acc_ptr) + idx);
        *dst = make_uint4(x[0] ^ x[4] ^ x[8] ^ x[12], x[1] ^ x[5] ^ x[9] ^ x[13],
                          x[2] ^ x[6] ^ x[10] ^ x[14], x[3] ^ x[7] ^ x[11] ^ x[15]);
    }
}
#endif

// ================================================================ v7 additions: the hidden classes
//
//   VERITY_TAP == 3  "M1" materialising tap: the hidden classes selected by VERITY_MAT_SRC are written, in canonical
//                    (slab = bidb*h + bidh, row, col | block, lane | d) positions, to a preallocated flat u32 buffer
//                    (params.vt.diag_ptr, layout Verity_mat_layout(HB, M, NB) below; N = NB*128 so no column predicate,
//                    only the row predicate `row < actual_seqlen_q` -- padding rows are uninitialised smem garbage, K1).
//                    Vectorised float2 stores (each thread owns adjacent column pairs).  Reference acquisition path; the
//                    commitment is computed outside the kernel (host or a separate GPU hash kernel).
//   VERITY_MAT_SRC   bitmask: 1 S (masked fp32 scores), 2 P fp32, 4 rP bf16 (packed u32, N/2 words per row),
//                    8 per-block row state (row_max, rescale, step, lane sums l0..l3), 16 acc_o after each block,
//                    32 epilogue (sum, inv_sum, final row_max, lse)
//   VERITY_ROW_GUARD 1 = the guarded-max build: word 3 of the row state holds GuardNegInfZero(row_max) at every key block after the
//                    first a row visits (0 at the first); 0 (default) = word 3 is never written.  `verity_tap_row3()` reports it.
//                    VERITY_MAT_SRC bit 64 (the same build) appends the MS class after FIN: per (slab, row, block) the max * scale the
//                    block's exp2 FMAs read (softmax.h `tap_max_scaled`, apply_v8.py step 6), at every visited block.
//   VERITY_TAP == 4  "H1" per-thread SHA-256 leaf: every thread hashes, per visited 128-key block, exactly its own
//                    register words of the classes in VERITY_SHA_SRC with a real SHA-256 (FIPS 180-4, verified on the
//                    host with hashlib on the M1-materialised words), preceded by one 16-word position header
//                    (tag, bidb, bidh, m_block, n_block, tidx, seqlens, launch tag, n_block_max, row validity); only the
//                    32-byte digest goes to HBM (params.vt.acc_ptr, layout [slab][m_block][NB][128 threads][8 words]).
//                    Leaf = SHA-256(header || S words || row-state block || P words || rP words || O words), message order =
//                    the kernel's emission order.  Padding rows are zeroed before hashing so every leaf is deterministic.
//   VERITY_SHA_SRC   same bit meaning as VERITY_MAT_SRC (1 S, 2 P, 4 rP, 8 row state, 16 O); 32 not used (epilogue values
//                    are bound by the block-0 leaf's O and row state -- inv_sum/sum are derived).
//   VERITY_SHA_NOINLINE  1 = the SHA-256 compression is a __noinline__ device function (one code copy; message via local
//                    memory), 0 = fully inlined at every absorb site (code size x 3 loop sites).

struct Verity_mat_layout {
    // word offsets into the flat u32/f32 buffer; HB = b*h slabs, M = seqlen_q cap (rows), NB = key-block cap, N = NB*BN.
    // v8: parametric in the kernel's key-block width BN (= Kernel_traits::kBlockN: 128 for hdim 64, 64 for hdim
    // 96/128/256 on the num_splits==1 "align" kernel) and head dim D (= Kernel_traits::kHeadDim: the O plane row width).
    // (BN, D) = (128, 64) reproduces the v7 layout word for word.  Host mirror: hidden_stream.StreamLayout(HB, M, NB, BN, D).
    long off_s, off_p, off_rp, off_row, off_o, off_fin, total;
#if defined(VERITY_MAT_SRC) && (VERITY_MAT_SRC & 64)
    long off_ms;          // MS class (VERITY_MAT_SRC & 64): one word per (slab, row, key block), appended after FIN
#endif
    int HB, M, N, NB, BN, D;
    __host__ __device__ Verity_mat_layout(int HB_, int M_, int NB_, int BN_ = 128, int D_ = 64)
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
#define VERITY_MAT_SRC 63
#endif
// v8: the context is instantiated with the kernel's compile-time tile (kBlockM, kBlockN, kHeadDim); the launch template
// only arms the tap when the host declared exactly this (kBlockN, kHeadDim) pair (params.vt.BN / vt.D), so the layout the kernel
// writes is the layout the host committed to -- a wrong declaration skips the launch (counted) instead of writing a different layout.
// Fragment -> position mapping (SM80 16x8x16 C fragments, 4 warps along M, one 16-row MMA tile per warp since kBlockM == 64):
//   row  = m_block*kBlockM + warp*16 + lane/4 (+8 for the second row of the tile)
//   col  = 8*n + 2*(lane%4) + j   for n < kBlockN/8 (S/P/RP: n_block*kBlockN + col) and n < kHeadDim/8 (O: d = col)
// -- the loops below run over size<1>(rowcol) = kBlockN/8 resp. kHeadDim/8 pairs, i.e. they follow the tile automatically.
template <int kBlockM_, int kBlockN_, int kHeadDim_>
struct Verity_mat_ctx {
    static constexpr int kBlockM = kBlockM_, kBlockN = kBlockN_, kHeadDim = kHeadDim_;
    static_assert(kBlockM == 64, "verity M1 tap: row mapping assumes kBlockM == 64 (4 warps x one 16-row MMA tile)");
    float *buf;
    Verity_mat_layout L;
    int slab, row0, row1, lane, seqlen_q;
    __device__ __forceinline__ Verity_mat_ctx(const Verity_tap &vt, int bidb, int h, int bidh, int m_block, int tidx, int kNWarps, int actual_seqlen_q)
        : buf(vt.diag_ptr), L(vt.H, vt.M, vt.NB, kBlockN, kHeadDim) {
        slab = bidb * h + bidh;
        lane = tidx % 32;
        row0 = m_block * kBlockM + (tidx / 32) * 16 + lane / 4;
        row1 = row0 + 8;
        seqlen_q = actual_seqlen_q;
    }
    // fp32 (MMA=4, MMA_M, MMA_N) fragment -> plane [slab][row][col], adjacent column pairs as float2
    template <typename Tensor0>
    __device__ __forceinline__ void mat_f32(long plane_off, Tensor0 const &acc_s, const int col_base) {
        Tensor sc = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<1>(sc))::value == kBlockN / 4, "verity M1 tap: acc_s fragment width != kBlockN/4 per thread");
        float *plane = buf + plane_off + (long)slab * L.M * L.N;
        #pragma unroll
        for (int mi = 0; mi < size<0>(sc); ++mi) {
            const int row = mi == 0 ? row0 : row1;
            if (row < seqlen_q) {
                float2 *dst = reinterpret_cast<float2 *>(plane + (long)row * L.N + col_base + (lane % 4) * 2);
                #pragma unroll
                for (int n = 0; n < size<1>(sc) / 2; ++n) {
                    dst[n * 4] = make_float2(sc(mi, 2 * n), sc(mi, 2 * n + 1));   // cols 8n + 2t, +1
                }
            }
        }
    }
    // packed bf16 P fragment -> plane [slab][row][col/2] of u32 words (word = cols (c, c+1))
    template <typename Tensor0>
    __device__ __forceinline__ void mat_bf16(Tensor0 const &rP, const int col_base) {
        Tensor words = recast<uint32_t>(rP);   // (2, MMA_M, MMA_N): word k -> row k*8, cols (8n+2t, +1)
        uint32_t *plane = reinterpret_cast<uint32_t *>(buf) + L.off_rp + (long)slab * L.M * (L.N / 2);
        #pragma unroll
        for (int mi = 0; mi < size<1>(words); ++mi) {
            #pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int row = k == 0 ? row0 : row1;
                if (row < seqlen_q) {
                    uint32_t *dst = plane + (long)row * (L.N / 2) + (col_base / 2) + (lane % 4);
                    #pragma unroll
                    for (int n = 0; n < size<2>(words); ++n) { dst[n * 4] = words(k, mi, n); }
                }
            }
        }
    }
    // per-block row state: [slab][row][nblk][8] = row_max, rescale, step, word 3, l0, l1, l2, l3.  Word 3 is never written, except by the
    // guarded-max build (VERITY_ROW_GUARD=1): GuardNegInfZero(row_max), the max this block's rescale reads, at every block after the first
    // (step > 0), and 0 at the first.
    template <typename TensorR>
    __device__ __forceinline__ void rowstat(TensorR const &row_max, TensorR const &scale, TensorR const &row_sum, const int nblk, const int step) {
        float *plane = buf + L.off_row + (long)slab * L.M * L.NB * 8;
        #pragma unroll
        for (int mi = 0; mi < size(row_max); ++mi) {
            const int row = mi == 0 ? row0 : row1;
            if (row < seqlen_q) {
                float *dst = plane + ((long)row * L.NB + nblk) * 8;
                if (lane % 4 == 0) { dst[0] = row_max(mi); dst[1] = scale(mi); dst[2] = __int_as_float(step); }
#if defined(VERITY_ROW_GUARD) && VERITY_ROW_GUARD
                if (lane % 4 == 0) { dst[3] = step > 0 && row_max(mi) != -INFINITY ? row_max(mi) : 0.f; }
#endif
                dst[4 + (lane % 4)] = row_sum(mi);
            }
        }
    }
    // acc_o after the PV MMA of this block: [slab][row][nblk][kHeadDim]
    template <typename Tensor0>
    __device__ __forceinline__ void o(Tensor0 const &acc_o, const int nblk) {
        Tensor oo = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));   // (2, kHeadDim/4)
        static_assert(decltype(size<1>(oo))::value == kHeadDim / 4, "verity M1 tap: acc_o fragment width != kHeadDim/4 per thread");
        float *plane = buf + L.off_o + (long)slab * L.M * L.NB * kHeadDim;
        #pragma unroll
        for (int mi = 0; mi < size<0>(oo); ++mi) {
            const int row = mi == 0 ? row0 : row1;
            if (row < seqlen_q) {
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
        for (int mi = 0; mi < size(row_sum); ++mi) {
            const int row = mi == 0 ? row0 : row1;
            if (row < seqlen_q) {
                float4 *dst = reinterpret_cast<float4 *>(plane + (long)row * 4);
                *dst = make_float4(row_sum(mi), inv_sum(mi), row_max(mi), lse(mi));
            }
        }
    }
#if defined(VERITY_MAT_SRC) && (VERITY_MAT_SRC & 64)
    // MS class: [slab][row][nblk] = the max * scale this block's exp2 FMAs read (Softmax::tap_max_scaled), at every visited block
    template <typename TensorR>
    __device__ __forceinline__ void ms(TensorR const &max_scaled, const int nblk) {
        if (lane % 4 != 0) return;
        float *plane = buf + L.off_ms + (long)slab * L.M * L.NB;
        #pragma unroll
        for (int mi = 0; mi < size(max_scaled); ++mi) {
            const int row = mi == 0 ? row0 : row1;
            if (row < seqlen_q) { plane[(long)row * L.NB + nblk] = max_scaled(mi); }
        }
    }
#endif
};
#if (VERITY_MAT_SRC & 64)
#define VERITY_MAT_MS(vmat_, softmax_, nblk_) VERITY_MAT(vmat_.ms(softmax_.tap_max_scaled, nblk_))
#else
#define VERITY_MAT_MS(vmat_, softmax_, nblk_)
#endif
#endif  // VERITY_TAP == 3

#if defined(VERITY_TAP) && VERITY_TAP == 4
#ifndef VERITY_SHA_SRC
#define VERITY_SHA_SRC 31
#endif
#ifndef VERITY_SHA_NOINLINE
#define VERITY_SHA_NOINLINE 1
#endif
__device__ __constant__ static const uint32_t VERITY_SHA_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

__device__ __forceinline__ uint32_t vsha_rotr(uint32_t x, int n) { return __funnelshift_r(x, x, n); }
// message words are consumed as little-endian u32 memory words; SHA-256 reads big-endian, hence the byte swap
__device__ __forceinline__ uint32_t vsha_bswap(uint32_t x) { return __byte_perm(x, 0u, 0x0123); }

#if VERITY_SHA_NOINLINE
#define VERITY_SHA_INLINE __noinline__
#else
#define VERITY_SHA_INLINE __forceinline__
#endif
// One SHA-256 compression of the 16 big-endian message words m[] into state h[] (FIPS 180-4 §6.2.2).
static __device__ VERITY_SHA_INLINE void vsha_compress(uint32_t (&h)[8], const uint32_t (&m)[16]) {
    uint32_t w[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = m[i];
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        if (i >= 16) {
            const uint32_t w15 = w[(i - 15) & 15], w2 = w[(i - 2) & 15];
            const uint32_t s0 = vsha_rotr(w15, 7) ^ vsha_rotr(w15, 18) ^ (w15 >> 3);
            const uint32_t s1 = vsha_rotr(w2, 17) ^ vsha_rotr(w2, 19) ^ (w2 >> 10);
            w[i & 15] = w[i & 15] + s0 + w[(i - 7) & 15] + s1;
        }
        const uint32_t S1 = vsha_rotr(e, 6) ^ vsha_rotr(e, 11) ^ vsha_rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = hh + S1 + ch + VERITY_SHA_K[i] + w[i & 15];
        const uint32_t S0 = vsha_rotr(a, 2) ^ vsha_rotr(a, 13) ^ vsha_rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

struct Verity_sha_ctx {
    uint32_t h[8];
    uint32_t m[16];
    int nblocks;          // 64-byte blocks compressed so far (for the final length field)
    int slab, row0, lane, seqlen_q, seqlen_k, m_block, tidx, nbmax, tag;
    bool ok0, ok1;
    uint32_t *acc; int NB;
    __device__ __forceinline__ Verity_sha_ctx(const Verity_tap &vt_, int bidb, int h_, int bidh, int m_block_, int tidx_, int kNWarps,
                                              int actual_seqlen_q, int actual_seqlen_k, int n_block_max) {
        acc = reinterpret_cast<uint32_t *>(vt_.acc_ptr); NB = vt_.NB;
        slab = bidb * h_ + bidh; lane = tidx_ % 32; tidx = tidx_; m_block = m_block_;
        row0 = m_block * 64 + (tidx / 32) * 16 + lane / 4;
        seqlen_q = actual_seqlen_q; seqlen_k = actual_seqlen_k; nbmax = n_block_max; tag = vt_.tag;
        ok0 = row0 < seqlen_q; ok1 = row0 + 8 < seqlen_q;
        nblocks = 0;
    }
    __device__ __forceinline__ void init() {
        h[0] = 0x6a09e667u; h[1] = 0xbb67ae85u; h[2] = 0x3c6ef372u; h[3] = 0xa54ff53au;
        h[4] = 0x510e527fu; h[5] = 0x9b05688cu; h[6] = 0x1f83d9abu; h[7] = 0x5be0cd19u;
        nblocks = 0;
    }
    __device__ __forceinline__ void compress_m() { vsha_compress(h, m); ++nblocks; }
    // leaf header: 16 little-endian u32 words (as the host sees them), byte-swapped into the message
    __device__ __forceinline__ void header(const int n_block) {
        uint32_t x[16] = {0x68326166u /* 'fa2h' LE */, 0x00070000u | (uint32_t)VERITY_SHA_SRC, 0u /* reserved */,
                          (uint32_t)slab, (uint32_t)m_block, (uint32_t)n_block, (uint32_t)tidx, (uint32_t)seqlen_q,
                          (uint32_t)seqlen_k, (uint32_t)tag, (uint32_t)nbmax, (uint32_t)((ok0 ? 1u : 0u) | (ok1 ? 2u : 0u)),
                          0u, 0u, 0u, 0u};
        #pragma unroll
        for (int i = 0; i < 16; ++i) m[i] = vsha_bswap(x[i]);
        compress_m();
    }
    // fp32 fragment (2 rows x 32 cols per thread) -> 4 blocks of 16 words, row 0 then row 1, ascending column
    template <typename Tensor0>
    __device__ __forceinline__ void absorb_f32(Tensor0 const &acc_s) {
        Tensor sc = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<1>(sc))::value % 16 == 0);
        #pragma unroll
        for (int mi = 0; mi < size<0>(sc); ++mi) {
            const bool ok = mi == 0 ? ok0 : ok1;
            #pragma unroll
            for (int blk = 0; blk < size<1>(sc) / 16; ++blk) {
                #pragma unroll
                for (int i = 0; i < 16; ++i) m[i] = vsha_bswap(ok ? __float_as_uint(sc(mi, blk * 16 + i)) : 0u);
                compress_m();
            }
        }
    }
    // packed bf16 fragment: (2, MMA_M, MMA_N) u32 words -> per row MMA_N words = MMA_N/16 blocks (16 for kBlockN=128,
    // 32 for the kBlockN=256 kernel that shares the TU but is never launched by the accepted profile)
    template <typename Tensor0>
    __device__ __forceinline__ void absorb_bf16x2(Tensor0 const &rP) {
        Tensor words = recast<uint32_t>(rP);
        static_assert(decltype(size<2>(words))::value % 16 == 0);
        #pragma unroll
        for (int mi = 0; mi < size<1>(words); ++mi) {
            #pragma unroll
            for (int k = 0; k < 2; ++k) {
                const bool ok = k == 0 ? ok0 : ok1;
                #pragma unroll
                for (int blk = 0; blk < size<2>(words) / 16; ++blk) {
                    #pragma unroll
                    for (int n = 0; n < 16; ++n) m[n] = vsha_bswap(ok ? words(k, mi, blk * 16 + n) : 0u);
                    compress_m();
                }
            }
        }
    }
    // per-block row state, one block: row_max[2], rescale[2], row_sum[2] (this lane's partial), n_block, step, zeros
    template <typename TensorR>
    __device__ __forceinline__ void absorb_rowstat(TensorR const &row_max, TensorR const &scale, TensorR const &row_sum, const int n_block, const int step) {
        uint32_t x[16] = {ok0 ? __float_as_uint(row_max(0)) : 0u, ok1 ? __float_as_uint(row_max(1)) : 0u,
                          ok0 ? __float_as_uint(scale(0)) : 0u,   ok1 ? __float_as_uint(scale(1)) : 0u,
                          ok0 ? __float_as_uint(row_sum(0)) : 0u, ok1 ? __float_as_uint(row_sum(1)) : 0u,
                          (uint32_t)n_block, (uint32_t)step, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        #pragma unroll
        for (int i = 0; i < 16; ++i) m[i] = vsha_bswap(x[i]);
        compress_m();
    }
    // acc_o fragment (2 rows x 16 d-values per thread... 2*MMA_M x 2*MMA_N = (2, 16)) -> 2 blocks
    template <typename Tensor0>
    __device__ __forceinline__ void absorb_o(Tensor0 const &acc_o) {
        Tensor oo = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<1>(oo))::value == 16);
        #pragma unroll
        for (int mi = 0; mi < size<0>(oo); ++mi) {
            const bool ok = mi == 0 ? ok0 : ok1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) m[i] = vsha_bswap(ok ? __float_as_uint(oo(mi, i)) : 0u);
            compress_m();
        }
    }
    // FIPS padding for a message of nblocks*64 bytes, then store the 8-word digest (as big-endian bytes = host digest())
    __device__ __forceinline__ void finish_store(const int n_block) {
        const uint64_t bits = (uint64_t)nblocks * 512ull;
        #pragma unroll
        for (int i = 0; i < 16; ++i) m[i] = 0u;
        m[0] = 0x80000000u; m[14] = (uint32_t)(bits >> 32); m[15] = (uint32_t)bits;
        compress_m();
        if (acc != nullptr) {
            const long idx = ((((long)slab * gridDim.x + m_block) * NB + n_block) * 128 + tidx) * 8;
            uint4 *dst = reinterpret_cast<uint4 *>(acc + idx);
            dst[0] = make_uint4(vsha_bswap(h[0]), vsha_bswap(h[1]), vsha_bswap(h[2]), vsha_bswap(h[3]));
            dst[1] = make_uint4(vsha_bswap(h[4]), vsha_bswap(h[5]), vsha_bswap(h[6]), vsha_bswap(h[7]));
        }
    }
};
#endif  // VERITY_TAP == 4

// ---------------------------------------------------------------- P0 diagnostic writes
// row_idx_offset = m_block*kBlockM + warp*16 + lane/4 (query index of this thread's first row within the sequence)
template <typename Tensor0>
__device__ __forceinline__ void verity_p0_mat_f32(const Verity_tap &vt, const int stage, Tensor0 const &acc_s,
                                                  const int bidh, const int row_idx_offset, const int warp_row_stride, const int col_base) {
    if (vt.diag_ptr == nullptr) return;
    Verity_p0_layout L(vt.H, vt.M, vt.N, vt.NB);
    Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
    const int lane = threadIdx.x % 32;
    float *plane = vt.diag_ptr + L.off_mat + ((long)stage * vt.H + bidh) * vt.M * vt.N;
    #pragma unroll
    for (int mi = 0; mi < size<0>(scores); ++mi) {
        const int row = row_idx_offset + (mi >> 1) * warp_row_stride + (mi & 1) * 8;
        #pragma unroll
        for (int ni = 0; ni < size<1>(scores); ++ni) {
            const int col = col_base + (ni >> 1) * 8 + (lane % 4) * 2 + (ni & 1);
            if (row < vt.M && col < vt.N) { plane[(long)row * vt.N + col] = scores(mi, ni); }
        }
    }
}

template <typename Tensor0>
__device__ __forceinline__ void verity_p0_mat_bf16(const Verity_tap &vt, const int stage, Tensor0 const &rP,
                                                   const int bidh, const int row_idx_offset, const int warp_row_stride, const int col_base) {
    if (vt.diag_ptr == nullptr) return;
    Verity_p0_layout L(vt.H, vt.M, vt.N, vt.NB);
    Tensor scores = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(rP.layout()));
    const int lane = threadIdx.x % 32;
    float *plane = vt.diag_ptr + L.off_mat + ((long)stage * vt.H + bidh) * vt.M * vt.N;
    #pragma unroll
    for (int mi = 0; mi < size<0>(scores); ++mi) {
        const int row = row_idx_offset + (mi >> 1) * warp_row_stride + (mi & 1) * 8;
        #pragma unroll
        for (int ni = 0; ni < size<1>(scores); ++ni) {
            const int col = col_base + (ni >> 1) * 8 + (lane % 4) * 2 + (ni & 1);
            if (row < vt.M && col < vt.N) { plane[(long)row * vt.N + col] = float(scores(mi, ni)); }
        }
    }
}

template <typename TensorR>
__device__ __forceinline__ void verity_p0_row(const Verity_tap &vt, const int stage, TensorR const &vals,
                                              const int bidh, const int row_idx_offset, const int warp_row_stride, const int nblk) {
    if (vt.diag_ptr == nullptr) return;
    Verity_p0_layout L(vt.H, vt.M, vt.N, vt.NB);
    const int lane = threadIdx.x % 32;
    if (lane % 4 != 0 || nblk >= vt.NB) return;
    float *plane = vt.diag_ptr + L.off_row + ((long)stage * vt.H + bidh) * vt.M * vt.NB;
    #pragma unroll
    for (int mi = 0; mi < size(vals); ++mi) {
        const int row = row_idx_offset + (mi >> 1) * warp_row_stride + (mi & 1) * 8;
        if (row < vt.M) { plane[(long)row * vt.NB + nblk] = vals(mi); }
    }
}

template <typename TensorR>
__device__ __forceinline__ void verity_p0_rowq(const Verity_tap &vt, const int stage, TensorR const &vals,
                                               const int bidh, const int row_idx_offset, const int warp_row_stride, const int nblk) {
    if (vt.diag_ptr == nullptr) return;
    Verity_p0_layout L(vt.H, vt.M, vt.N, vt.NB);
    const int lane = threadIdx.x % 32;
    if (nblk >= vt.NB) return;
    float *plane = vt.diag_ptr + L.off_rowq + ((long)stage * vt.H + bidh) * vt.M * vt.NB * 4;
    #pragma unroll
    for (int mi = 0; mi < size(vals); ++mi) {
        const int row = row_idx_offset + (mi >> 1) * warp_row_stride + (mi & 1) * 8;
        if (row < vt.M) { plane[((long)row * vt.NB + nblk) * 4 + (lane % 4)] = vals(mi); }
    }
}

template <typename Tensor0>
__device__ __forceinline__ void verity_p0_o(const Verity_tap &vt, Tensor0 const &acc_o,
                                            const int bidh, const int row_idx_offset, const int warp_row_stride, const int nblk) {
    if (vt.diag_ptr == nullptr) return;
    Verity_p0_layout L(vt.H, vt.M, vt.N, vt.NB);
    if (nblk >= vt.NB) return;
    Tensor o = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));  // (2*MMA_M, 2*MMA_K)
    const int lane = threadIdx.x % 32;
    float *plane = vt.diag_ptr + L.off_o + (long)bidh * vt.M * vt.NB * 64;
    #pragma unroll
    for (int mi = 0; mi < size<0>(o); ++mi) {
        const int row = row_idx_offset + (mi >> 1) * warp_row_stride + (mi & 1) * 8;
        #pragma unroll
        for (int ki = 0; ki < size<1>(o); ++ki) {
            const int d = (ki >> 1) * 8 + (lane % 4) * 2 + (ki & 1);
            if (row < vt.M && d < 64) { plane[((long)row * vt.NB + nblk) * 64 + d] = o(mi, ki); }
        }
    }
}

template <typename TensorR>
__device__ __forceinline__ void verity_p0_fin(const Verity_tap &vt, const int stage, TensorR const &vals,
                                              const int bidh, const int row_idx_offset, const int warp_row_stride) {
    if (vt.diag_ptr == nullptr) return;
    Verity_p0_layout L(vt.H, vt.M, vt.N, vt.NB);
    const int lane = threadIdx.x % 32;
    if (lane % 4 != 0) return;
    float *plane = vt.diag_ptr + L.off_fin + ((long)stage * vt.H + bidh) * vt.M;
    #pragma unroll
    for (int mi = 0; mi < size(vals); ++mi) {
        const int row = row_idx_offset + (mi >> 1) * warp_row_stride + (mi & 1) * 8;
        if (row < vt.M) { plane[row] = vals(mi); }
    }
}

}  // namespace FLASH_NAMESPACE
