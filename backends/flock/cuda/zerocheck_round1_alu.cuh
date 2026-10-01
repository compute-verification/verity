// Zerocheck round 1 with its GF(2^8) products on the ALU (prover change 1a): zerocheck_first_round_cpu_structured<14>
// (cuda-ghash/zerocheck_round1_cpustyle.cuh) with only its k loop changed. Its 16 log/antilog lookups per lane per j become two
// SWAR words of four products each (xtime by 0x1b, byte masks from prmt's sign replication); the index into s_phi is the same
// residue, so round 1's 64 + 64 outputs are the same bytes. Measured beside the tables at m = 35 (zc1a, the microbenchmark on
// cursor/zc-round1-swar-bench-1d95): 52.9 ms per launch against 64.6 ms, outputs identical at m = 13, 20, 28, 32 and 35.
// Included from prove_chunk.cuh, after prove_ffi.cu's zerocheck_round1_cpustyle.cuh.
// FC_ZC_TABLES=1 launches the table kernel instead (the same-job control); FC_ZC_ALU_BREAK=1 reduces by 0x1d (the negative
// control: round 1's outputs, so the proof, change).
#pragma once
#include <cstdlib>

// each byte to 0xff if its top bit is set, else 0x00
__device__ __forceinline__ uint32_t zc_alu_top_mask(uint32_t x) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, 0xBA98;" : "=r"(r) : "r"(x), "r"(0u));
    return r;
}

// four GF(2^8) products mod x^8 + x^4 + x^3 + x + 1 (POLY = 0x1b), byte by byte: b's bit i (moved to its byte's top) selects a·x^i
template <uint32_t POLY>
__device__ __forceinline__ uint32_t zc_alu_mul4(uint32_t a, uint32_t b) {
    uint32_t p = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        p ^= a & zc_alu_top_mask(b << (7 - i));
        if (i < 7) a = ((a << 1) & 0xfefefefeu) ^ (zc_alu_top_mask(a) & (POLY * 0x01010101u));
    }
    return p;
}

// byte 0 of each of four words, in that order
__device__ __forceinline__ uint32_t zc_alu_pack(uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3) {
    return __byte_perm(__byte_perm(e0, e1, 0x0040), __byte_perm(e2, e3, 0x0040), 0x5410);
}

// XOR over k of p_k << k, for p_0..p_3 the bytes of lo and p_4..p_7 those of hi (the table kernel's aacc)
__device__ __forceinline__ uint16_t zc_alu_shift_sum(uint32_t lo, uint32_t hi) {
    return (uint16_t)((lo & 0xffu) ^ ((lo >> 7) & 0x1feu) ^ ((lo >> 14) & 0x3fcu) ^ ((lo >> 21) & 0x7f8u)
                      ^ ((hi << 4) & 0xff0u) ^ ((hi >> 3) & 0x1fe0u) ^ ((hi >> 10) & 0x3fc0u) ^ ((hi >> 17) & 0x7f80u));
}

// one lane's s_phi index at one j: reduce(XOR_k x_k·y_k << k), from byte 0 of each x_k and y_k
template <uint32_t POLY>
__device__ __forceinline__ uint32_t zc_alu_index(const uint32_t* x, const uint32_t* y) {
    const uint32_t lo = zc_alu_mul4<POLY>(zc_alu_pack(x[0], x[1], x[2], x[3]), zc_alu_pack(y[0], y[1], y[2], y[3]));
    const uint32_t hi = zc_alu_mul4<POLY>(zc_alu_pack(x[4], x[5], x[6], x[7]), zc_alu_pack(y[4], y[5], y[6], y[7]));
    return reduce_zerocheck_gf8_polynomial(zc_alu_shift_sum(lo, hi));
}

// The bounds keep two 448-thread blocks per SM (72 registers), as the table kernel's launcher requires of W = 14
template <int W, uint32_t POLY>
__global__ void __launch_bounds__(W * 32, 2)
zerocheck_first_round_alu(const uint8_t* __restrict__ a_packed, const uint8_t* __restrict__ b_packed,
                          const uint8_t* __restrict__ c_packed, const F128* __restrict__ eq_out, const u64* __restrict__ t0,
                          const F128* __restrict__ phi, long long n_out, long long G, F128* __restrict__ raw_ab,
                          F128* __restrict__ raw_c) {
    __shared__ u64 s_t0[256 * 8];
    __shared__ F128 s_phi[256];
    __shared__ u64 s_scol[W * 24];
    __shared__ u64 s_wbuf[W * 192];
    for (int i = threadIdx.x; i < 256 * 8; i += blockDim.x) s_t0[i] = t0[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_phi[i] = phi[i];
    __syncthreads();

    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u64* scol = s_scol + wid * 24;
    u64* wbuf = s_wbuf + wid * 192;
    long long warp_id = (long long)blockIdx.x * W + wid;
    long long o0 = warp_id * G, o1 = o0 + G;
    if (o1 > n_out) o1 = n_out;
    if (o0 >= o1) return;

    int wlo = lane >> 3, whi = wlo + 4, sh = (lane & 7) * 8;
    const uint32_t sel = lane & 7;
    F128 pab0{0, 0}, pab1{0, 0}, pc0{0, 0}, pc1{0, 0};

    for (long long o = o0; o < o1; o++) {
        F128 eq_o = eq_out[o];
        F128 chunk_ab0{0, 0}, chunk_ab1{0, 0}, chunk_c0{0, 0}, chunk_c1{0, 0};
        for (int j = 15; j >= 0; j--) {
            long long col0 = o * 128 + j * 8;
            long long base = col0 * 8;
            if (lane < 24) {
                const uint8_t* Wt = (lane < 8) ? a_packed : (lane < 16) ? b_packed : c_packed;
                int cl = lane & 7;
                scol[lane] = *(const u64*)(Wt + base + cl * 8);
            }
            __syncwarp();
#pragma unroll
            for (int r = 0; r < 6; r++) {
                int bld = lane + 32 * r;
                int cl = bld / 24, win = bld - cl * 24, wt = win >> 3, wcol = win & 7;
                u64 src = scol[wt * 8 + cl];
                u64 word = 0;
#pragma unroll
                for (int b = 0; b < 8; b++)
                    word ^= s_t0[(int)((src >> (8 * b)) & 0xff) * 8 + (wcol ^ b)];
                wbuf[cl * 24 + win] = word;
            }
            __syncwarp();
            uint32_t eaL[8], eaH[8], ebL[8], ebH[8];
            uint16_t cacc0 = 0, cacc1 = 0;
            auto ex = [&](u64 w) -> uint32_t { return __byte_perm((uint32_t)w, (uint32_t)(w >> 32), sel); };
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const u64* wc = wbuf + k * 24;
                eaL[k] = ex(wc[wlo]);     eaH[k] = ex(wc[whi]);
                ebL[k] = ex(wc[8 + wlo]); ebH[k] = ex(wc[8 + whi]);
                uint8_t cL = (uint8_t)(wc[16 + wlo] >> sh), cH = (uint8_t)(wc[16 + whi] >> sh);
                cacc0 ^= (uint16_t)cL << k;
                cacc1 ^= (uint16_t)cH << k;
            }
            chunk_ab0 = f128_add(multiply_zerocheck_value_by_generator(chunk_ab0), s_phi[zc_alu_index<POLY>(eaL, ebL)]);
            chunk_ab1 = f128_add(multiply_zerocheck_value_by_generator(chunk_ab1), s_phi[zc_alu_index<POLY>(eaH, ebH)]);
            chunk_c0  = f128_add(multiply_zerocheck_value_by_generator(chunk_c0),  s_phi[reduce_zerocheck_gf8_polynomial(cacc0)]);
            chunk_c1  = f128_add(multiply_zerocheck_value_by_generator(chunk_c1),  s_phi[reduce_zerocheck_gf8_polynomial(cacc1)]);
            __syncwarp();
        }
        pab0 = f128_add(pab0, ghash_mul_karatsuba(eq_o, chunk_ab0));
        pab1 = f128_add(pab1, ghash_mul_karatsuba(eq_o, chunk_ab1));
        pc0  = f128_add(pc0,  ghash_mul_karatsuba(eq_o, chunk_c0));
        pc1  = f128_add(pc1,  ghash_mul_karatsuba(eq_o, chunk_c1));
    }
    atomicXor((unsigned long long*)&raw_ab[lane].lo, pab0.lo);
    atomicXor((unsigned long long*)&raw_ab[lane].hi, pab0.hi);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].lo, pab1.lo);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].hi, pab1.hi);
    atomicXor((unsigned long long*)&raw_c[lane].lo, pc0.lo);
    atomicXor((unsigned long long*)&raw_c[lane].hi, pc0.hi);
    atomicXor((unsigned long long*)&raw_c[lane + 32].lo, pc1.lo);
    atomicXor((unsigned long long*)&raw_c[lane + 32].hi, pc1.hi);
}

// launch_zerocheck_first_round_cpu_structured's grid, memsets and scale, with the ALU kernel unless FC_ZC_TABLES=1
inline void launch_zerocheck_first_round(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c, const F128* d_eq_out,
                                         long long n_out, F128 scale, F128* d_round1_ab, F128* d_round1_c) {
    static const bool tables = [] { const char* e = std::getenv("FC_ZC_TABLES"); return e && e[0] == '1'; }();
    static const bool broken = [] { const char* e = std::getenv("FC_ZC_ALU_BREAK"); return e && e[0] == '1'; }();
    if (tables) {
        launch_zerocheck_first_round_cpu_structured(d_a, d_b, d_c, d_eq_out, n_out, scale, d_round1_ab, d_round1_c);
        return;
    }
    constexpr int W = 14;
    constexpr long long G = ZC_OUTER_CHUNKS_PER_WARP;
    long long warps = (n_out + G - 1) / G;
    int blocks = (int)((warps + W - 1) / W);
    cudaMemset(d_round1_ab, 0, 64 * sizeof(F128));
    cudaMemset(d_round1_c, 0, 64 * sizeof(F128));
    if (broken)
        zerocheck_first_round_alu<W, 0x1du><<<blocks, W * 32>>>(d_a, d_b, d_c, d_eq_out, g_zc_t0, g_zc_phi, n_out, G,
                                                               d_round1_ab, d_round1_c);
    else
        zerocheck_first_round_alu<W, 0x1bu><<<blocks, W * 32>>>(d_a, d_b, d_c, d_eq_out, g_zc_t0, g_zc_phi, n_out, G,
                                                               d_round1_ab, d_round1_c);
    scale_zerocheck_first_round<<<1, 64>>>(d_round1_ab, d_round1_c, scale);
}
