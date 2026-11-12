// The zerocheck's round 2 without its tables. Round 2's message and the tail's first lookahead each fold the packed witness at
// z from its bytes (cuda-ghash/zerocheck_round2.cuh: a_mlv[row] = Σ_j ft[j·256 + byte j of the row]), so round 2's tables,
// 2^(m-6) words each for a and b, are never written: the first lookahead writes its quarter of them as it did. Every message
// and every folded word is the field element zerocheck_second_round_fold_with_lookahead and
// zerocheck_tail_lookahead_fold_and_message compute: the same products, XOR-summed (in any order: GF(2^128) is exact).
#pragma once
#include <cstdlib>
#include <type_traits>
#include "zc_challenger_device.cuh"   // ZT_FINISH_OP
#include "zerocheck_round2.cuh"

// Where a fold reads the table: 0 its 8 x 256 words in global memory, as round 2 does; 1 the same words in shared memory;
// 2 shared memory as C-bit chunks of the row. The table is GF(2)-linear in its byte (entry v is the XOR of the weights of v's
// set bits), so a row's fold is the XOR over its chunks of the XOR of the byte entries each chunk's bits fall in: ceil(64 / C)
// tables of 2^C words, each held 8 times so that the 8 lanes of a 128-bit load's phase read 8 different bank groups, one
// 128-bit load a chunk. C is FC_ZF_CHUNK (4 to 7, else ZF_CHUNK), narrowed until the tables fit a block's shared memory.
#ifndef ZF_TABLE
#define ZF_TABLE 2
#endif
#ifndef ZF_CHUNK
#define ZF_CHUNK 7
#endif
static const int ZF_WORDS = 8 * 256;
__host__ __device__ constexpr int zf_tab_words(int T, int C) { return T == 1 ? ZF_WORDS : T == 2 ? (63 / C + 1) * (8 << C) : 0; }

// The packed path: round 2's tables only exist when the tail's first lookahead doesn't run (prove_chunk's tail loop at k = 2).
inline bool zf_packed(int m) { return m - 7 >= 3 && (1LL << (m - 6)) / 16 > ZT_FINISH_OP; }

template <int T, int C>
__device__ __forceinline__ void zf_load_table(F128* tab, const F128* __restrict__ ft) {
    if constexpr (T != 0) {
        for (int e = threadIdx.x; e < zf_tab_words(T, C); e += blockDim.x) {
            if constexpr (T == 1) {
                tab[e] = ft[e];
            } else {
                const int b = (e >> (C + 3)) * C;
                const uint64_t v = (uint64_t)((e >> 3) & ((1 << C) - 1)) << b;
                F128 w{0, 0};
                for (int p = b >> 3; p <= (b + C - 1) >> 3 && p < 8; p++) w = f128_add(w, ft[p * 256 + ((v >> (8 * p)) & 0xff)]);
                tab[e] = w;
            }
        }
        __syncthreads();
    }
}

// One row's fold at z from its 8 bytes (bytes 0-3 in lo, 4-7 in hi, little-endian); g = the lane's copy of the chunk tables.
template <int T, int C>
__device__ __forceinline__ F128 zf_row(const F128* __restrict__ ft, const F128* tab, uint32_t lo, uint32_t hi, int g) {
    F128 acc{0, 0};
    if constexpr (T == 2) {
        const uint64_t r = (uint64_t)hi << 32 | lo;
#pragma unroll
        for (int j = 0; j * C < 64; j++) acc = f128_add(acc, tab[((j << C) + (int)((r >> (j * C)) & ((1u << C) - 1))) * 8 + g]);
    } else {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint32_t v = ((j < 4 ? lo : hi) >> (8 * (j & 3))) & 0xff;
            acc = f128_add(acc, T == 0 ? ft[j * 256 + v] : tab[j * 256 + v]);
        }
    }
    return acc;
}

// Round 2's eight lookahead slots into part[8][gridDim.x], as zerocheck_second_round_fold_with_lookahead(_high_bits) leaves
// them, without writing a_mlv/b_mlv. HB: the block's quads share one eqhi entry (plan_zerocheck_high_bits, shift 1), multiplied
// in once at the end.
template <int T, int C, bool HB>
__global__ void zerocheck_second_round_message(const uint8_t* __restrict__ a_packed, const uint8_t* __restrict__ b_packed,
                                               const F128* __restrict__ ft, const F128* __restrict__ eqlo,
                                               const F128* __restrict__ eqhi, int lobits, long long out_quads, long long chunk,
                                               F128* __restrict__ part) {
    __shared__ F128 sh[ZT_TPB];
    extern __shared__ F128 zf_tab[];
    F128* tab = zf_tab;
    zf_load_table<T, C>(tab, ft);
    const int g = threadIdx.x & 7;
    F256 acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = F256{0, 0, 0, 0};
    const long long start = (long long)blockIdx.x * chunk;
    const long long z0 = HB ? start + threadIdx.x : (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long z1 = HB ? (start + chunk < out_quads ? start + chunk : out_quads) : out_quads;
    const long long step = HB ? blockDim.x : (long long)gridDim.x * blockDim.x;
    const long long mask = (1LL << lobits) - 1;
    for (long long z = z0; z < z1; z += step) {
        const uint4* pa = (const uint4*)(a_packed + z * 32);
        const uint4* pb = (const uint4*)(b_packed + z * 32);
        const uint4 a0 = __ldg(pa), a1 = __ldg(pa + 1), b0 = __ldg(pb), b1 = __ldg(pb + 1);
        F128 w[4] = {zf_row<T, C>(ft, tab, a0.x, a0.y, g), zf_row<T, C>(ft, tab, a0.z, a0.w, g),
                     zf_row<T, C>(ft, tab, a1.x, a1.y, g), zf_row<T, C>(ft, tab, a1.z, a1.w, g)};
        F128 x[4] = {zf_row<T, C>(ft, tab, b0.x, b0.y, g), zf_row<T, C>(ft, tab, b0.z, b0.w, g),
                     zf_row<T, C>(ft, tab, b1.x, b1.y, g), zf_row<T, C>(ft, tab, b1.z, b1.w, g)};
        const long long ez = z << 1;
        if (HB) zt_accum_quad(acc, w, x, eqlo[ez & mask], eqlo[(ez + 1) & mask]);
        else zt_accum_quad(acc, w, x, evaluate_zerocheck_split_equality(eqlo, eqhi, ez, lobits),
                           evaluate_zerocheck_split_equality(eqlo, eqhi, ez + 1, lobits));
    }
    const F128 blockmul = !HB ? F128{1, 0} : start < out_quads ? eqhi[(start << 1) >> lobits] : F128{0, 0};
    zt_reduce8(acc, sh, part, blockmul);
}

// The tail's first lookahead (zerocheck_tail_lookahead_fold_and_message over a_mlv/b_mlv) with its input words folded at z
// from the packed witness: output quad z folds rows 16z..16z+15, by r1 and then r2.
template <int T, int C>
__global__ void zerocheck_tail_lookahead_packed(const uint8_t* __restrict__ a_packed, const uint8_t* __restrict__ b_packed,
                                                const F128* __restrict__ ft, F128* __restrict__ Ao, F128* __restrict__ Bo,
                                                const F128* __restrict__ eqlo, const F128* __restrict__ eqhi, int shift,
                                                int lobits, long long out_quads, F128 r1, F128 r2, F128* __restrict__ part) {
    __shared__ F128 sh[ZT_TPB];
    extern __shared__ F128 zf_tab[];
    F128* tab = zf_tab;
    zf_load_table<T, C>(tab, ft);
    const int g = threadIdx.x & 7;
    F256 acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = F256{0, 0, 0, 0};
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long z = (long long)blockIdx.x * blockDim.x + threadIdx.x; z < out_quads; z += stride) {
        const long long o = 4 * z;
        F128 w[4], x[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint4* pa = (const uint4*)(a_packed + (o + j) * 32);
            const uint4* pb = (const uint4*)(b_packed + (o + j) * 32);
            const uint4 a0 = __ldg(pa), a1 = __ldg(pa + 1), b0 = __ldg(pb), b1 = __ldg(pb + 1);
            w[j] = zt_fold1(zt_fold1(zf_row<T, C>(ft, tab, a0.x, a0.y, g), zf_row<T, C>(ft, tab, a0.z, a0.w, g), r1),
                            zt_fold1(zf_row<T, C>(ft, tab, a1.x, a1.y, g), zf_row<T, C>(ft, tab, a1.z, a1.w, g), r1), r2);
            x[j] = zt_fold1(zt_fold1(zf_row<T, C>(ft, tab, b0.x, b0.y, g), zf_row<T, C>(ft, tab, b0.z, b0.w, g), r1),
                            zt_fold1(zf_row<T, C>(ft, tab, b1.x, b1.y, g), zf_row<T, C>(ft, tab, b1.z, b1.w, g), r1), r2);
            Ao[o + j] = w[j]; Bo[o + j] = x[j];
        }
        zt_accum_quad(acc, w, x, evaluate_zerocheck_split_equality(eqlo, eqhi, (z << 1) << shift, lobits),
                      evaluate_zerocheck_split_equality(eqlo, eqhi, ((z << 1) + 1) << shift, lobits));
    }
    zt_reduce8(acc, sh, part, F128{1, 0});
}

inline int zf_chunk() {
    static const int c = [] {
        const char* s = std::getenv("FC_ZF_CHUNK");
        int c = s && *s ? std::atoi(s) : ZF_CHUNK, dev = 0, optin = 0;
        c = c < 4 ? 4 : c > 7 ? 7 : c;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        while (c > 4 && (size_t)(zf_tab_words(2, c) + ZT_TPB) * sizeof(F128) > (size_t)optin) c--;
        return c;
    }();
    return c;
}

// f(std::integral_constant<int, C>) with C the chunk width: the template's, else zf_chunk()'s.
template <int T, int C, class F>
inline void zf_dispatch(F f) {
    if constexpr (T != 2 || C != 0) f(std::integral_constant<int, T == 2 ? C : 4>{});
    else switch (zf_chunk()) {
        case 5: f(std::integral_constant<int, 5>{}); break;
        case 6: f(std::integral_constant<int, 6>{}); break;
        case 7: f(std::integral_constant<int, 7>{}); break;
        default: f(std::integral_constant<int, 4>{});
    }
}

// The dynamic shared memory kernel k's tables take, set as its limit past the 48 KB default.
template <class K>
inline size_t zf_smem(K* k, int words) {
    const size_t bytes = (size_t)words * sizeof(F128);
    if (bytes > (48u << 10)) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
    return bytes;
}

// launch_zerocheck_second_round_fold_with_lookahead's message into d_out (8 F128), reading the packed witness only.
template <int T = ZF_TABLE, int C = 0>
inline void launch_zerocheck_second_round_message(const uint8_t* d_a, const uint8_t* d_b, const F128* d_ft, const F128* d_eqlo,
                                                  const F128* d_eqhi, int lobits, long long n_out, F128 scale_1, F128* d_part,
                                                  F128* d_out) {
    const long long out_quads = n_out / 4;
    const int blocks = zt_blocks(out_quads);
    long long chunk = 0;
    const bool hb = plan_zerocheck_high_bits(out_quads, 1, lobits, blocks, chunk);
    zf_dispatch<T, C>([&](auto c) {
        constexpr int W = decltype(c)::value;
        if (hb) {
            auto k = zerocheck_second_round_message<T, W, true>;
            k<<<blocks, ZT_TPB, zf_smem(k, zf_tab_words(T, W))>>>(d_a, d_b, d_ft, d_eqlo, d_eqhi, lobits, out_quads, chunk, d_part);
        } else {
            auto k = zerocheck_second_round_message<T, W, false>;
            k<<<blocks, ZT_TPB, zf_smem(k, zf_tab_words(T, W))>>>(d_a, d_b, d_ft, d_eqlo, d_eqhi, lobits, out_quads, 0, d_part);
        }
    });
    combine_zerocheck_tail_lookahead_message<<<8, ZT_TPB>>>(d_part, blocks, F128{1, 0}, scale_1, d_out);
}

// launch_zerocheck_tail_lookahead over round 2's tables, from the packed witness: dAo/dBo get out_quads·4 words each.
template <int T = ZF_TABLE, int C = 0>
inline void launch_zerocheck_tail_lookahead_packed(const uint8_t* d_a, const uint8_t* d_b, const F128* d_ft, F128* dAo, F128* dBo,
                                                   const F128* dEqLo, const F128* dEqHi, int shift, int lobits, long long out_quads,
                                                   F128 r1, F128 r2, F128 scale_k, F128 scale_k1, F128* d_part, F128* d_out) {
    const int blocks = zt_blocks(out_quads);
    zf_dispatch<T, C>([&](auto c) {
        constexpr int W = decltype(c)::value;
        auto k = zerocheck_tail_lookahead_packed<T, W>;
        k<<<blocks, ZT_TPB, zf_smem(k, zf_tab_words(T, W))>>>(d_a, d_b, d_ft, dAo, dBo, dEqLo, dEqHi, shift, lobits, out_quads, r1,
                                                              r2, d_part);
    });
    combine_zerocheck_tail_lookahead_message<<<8, ZT_TPB>>>(d_part, blocks, scale_k, scale_k1, d_out);
}
