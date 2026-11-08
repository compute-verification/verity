// verity flock-gpu-link: zerocheck round 1 from the packed witness (launch_zc_round1). Reads the cuda-ghash TU's round-1
// tables and helpers (zerocheck_round1_cpustyle.cuh: g_zc_t0, g_zc_phi, the log tables and the table kernel).
#pragma once
#include <cstdlib>
#include "zerocheck_round1_cpustyle.cuh"

// an 8x8 bit matrix, row r in byte r (bit c its column c), transposed
__device__ __forceinline__ u64 zlin_t8(u64 x) {
    u64 t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAULL;
    x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL;
    x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL;
    return x ^ t ^ (t << 28);
}

// Zerocheck round 1 (zerocheck_first_round_cpu_structured's outputs) with z folded before its extension. The table kernel's
// c output is Σ_o eq_o Σ_j γ^j φ8(Σ_k x^k ẑ_λ(o, j, k)), ẑ_λ = Σ_s z_s M[s][λ] the extension of a column's 64 bits, where
// M[8b + i][λ] = byte λ & 7 of t0[1 << i][(λ >> 3) ^ b]. φ8 embeds GF(2^8) in F128 as a ring, so that is Σ_s φ8(M[s][λ])·w_s
// with w_s = Σ_o eq_o Σ_j γ^j φ8(v_s), v_s the byte whose bit k is bit s of column k, and zc_round1_combine applies M to the
// 64 sums once. Only a and b go through the t0 tables.
// Lane (w, q) = (lane & 7, lane >> 3) of a warp step's eight columns takes columns 2q and 2q + 1: it builds word w of a's and
// b's extensions there (λ = 8w + e for the word's bytes e) and multiplies them byte by byte, so no extension word leaves its
// registers. Its sums for λ = 8w + e, Σ_k x^k a_k b_k over its two columns, are then added across the four lanes of w in two
// exchanges, after which lane (w, q) holds λ = 8w + e0 and 8w + e0 + 1, e0 = 4(q & 1) + 2(q >> 1), for the reduction, φ8 and
// the fold over j; and w_s for s = λ, whose bytes v_s come from byte w of each column of z, gathered by two exchanges. Each
// lane loads its two columns of a, b and z 16 bytes at a time, the next step's before this step's products.
// a·b is the table kernel's: clmad on sm_90 (two products a carry-less product: bytes e and e + 4), the log tables elsewhere
// (ZC_R1_TABLES forces them). No table read conflicts: a half-warp's lanes read t0's entry (v, w ^ b) from copy q & 1, sixteen
// bank pairs, and a quarter-warp's eight φ8 reads lane w's copy, on bank quad w. The grid is resident (zc_round1_smem bytes a
// block, ZC_R1_MINB blocks an SM) and its warps stride over the outer chunks, so each block loads the copies once.
#ifndef ZC_R1_W
#define ZC_R1_W 8
#endif
#ifndef ZC_R1_MINB
#define ZC_R1_MINB 2
#endif
constexpr size_t zc_round1_smem() { return 256 * 8 * sizeof(F128) + 256 * 16 * 8 + 768; }

// p < 2^22 modulo the AES polynomial P (reduce_zerocheck_gf8_wide's residue): with μ = ⌊x^22 / P⌋ = 0x4696,
// q = ⌊⌊p / x^8⌋·μ / x^14⌋ is ⌊p / P⌋ (Barrett, exact over GF(2)[x]) and p - q·P's low byte the residue
__device__ __forceinline__ uint8_t zc_gf8_barrett(uint32_t p) {
    const u64 q = clmad_lo((u64)(p >> 8), 0x4696ull, 0) >> 14;
    return (uint8_t)(p ^ (uint32_t)clmad_lo(q, 0x11bull, 0));
}

// bit e of each byte of x (bytes 0-3 in lo, 4-7 in hi) as one byte, bit k from byte k
__device__ __forceinline__ uint32_t zc_byte_bits(uint32_t lo, uint32_t hi, int e) {
    const uint32_t l = ((lo >> e) & 0x01010101u) * 0x01020408u >> 24, h = ((hi >> e) & 0x01010101u) * 0x01020408u >> 24;
    return (l & 0xF) | (h & 0xF) << 4;
}

template <int W>
__global__ void __launch_bounds__(W * 32, ZC_R1_MINB) zc_round1_zlin(const uint8_t* __restrict__ a_packed,
                                                                     const uint8_t* __restrict__ b_packed,
                                                                     const uint8_t* __restrict__ z_packed,
                                                                     const F128* __restrict__ eq_out, const u64* __restrict__ t0,
                                                                     const F128* __restrict__ phi, long long n_out,
                                                                     F128* __restrict__ raw_ab, F128* __restrict__ raw_w) {
    extern __shared__ __align__(16) unsigned char zc_smem[];
    F128* s_phi = reinterpret_cast<F128*>(zc_smem);   // entry r's copy q at 8r + q
    u64* s_t0 = reinterpret_cast<u64*>(s_phi + 256 * 8);   // entry (v, w)'s copy c at 16v + 8c + w
    uint8_t* s_log = reinterpret_cast<uint8_t*>(s_t0 + 256 * 16);
    uint8_t* s_antilog = s_log + 256;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000 && !defined(ZC_R1_TABLES)
    constexpr bool cl = true;
#else
    constexpr bool cl = false;
#endif
    for (int i = threadIdx.x; !cl && i < 256; i += blockDim.x) s_log[i] = d_zc_log[i];
    for (int i = threadIdx.x; !cl && i < 512; i += blockDim.x) s_antilog[i] = d_zc_antilog[i];
    for (int i = threadIdx.x; i < 256 * 16; i += blockDim.x) s_t0[i] = t0[(i >> 4) * 8 + (i & 7)];
    for (int i = threadIdx.x; i < 256 * 8; i += blockDim.x) s_phi[i] = phi[i >> 3];
    __syncthreads();

    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, w = lane & 7, q = lane >> 3, e0 = 4 * (q & 1) + 2 * (q >> 1);
    const unsigned char* t0b = reinterpret_cast<const unsigned char*>(s_t0);
    const uint32_t tw = (uint32_t)((q & 1) * 8 + w) * 8;   // entry (0, w)'s copy q & 1; entry (v, w ^ b)'s is (128v | tw) ^ 8b
    // word w of a column's extension
    auto ext = [&](uint32_t lo, uint32_t hi) {
        u64 word = 0;
#pragma unroll
        for (int b = 0; b < 8; b++) {
            const uint32_t h = b < 4 ? lo : hi;
            const uint32_t v128 = ((b & 3) == 0 ? h << 7 : h >> (8 * (b & 3) - 7)) & 0x7f80u;
            word ^= *reinterpret_cast<const u64*>(t0b + ((v128 | tw) ^ (uint32_t)(8 * b)));
        }
        return word;
    };
    const long long stride = (long long)gridDim.x * W;
    // the lane's 16 bytes of step (o, j): columns 2q and 2q + 1
    auto at = [&](long long o, int j) { return (o * 128 + j * 8) * 8 + 16 * q; };
    long long o = (long long)blockIdx.x * W + wid;
    uint4 na = make_uint4(0, 0, 0, 0), nb = na, nz = na;
    if (o < n_out) {
        na = __ldg(reinterpret_cast<const uint4*>(a_packed + at(o, 15)));
        nb = __ldg(reinterpret_cast<const uint4*>(b_packed + at(o, 15)));
        nz = __ldg(reinterpret_cast<const uint4*>(z_packed + at(o, 15)));
    }
    F128 pab0{0, 0}, pab1{0, 0}, pw0{0, 0}, pw1{0, 0};
    for (; o < n_out; o += stride) {
        const F128 eq_o = eq_out[o];
        F128 chunk_ab0{0, 0}, chunk_ab1{0, 0}, chunk_w0{0, 0}, chunk_w1{0, 0};
        for (int j = 15; j >= 0; j--) {
            const uint4 ca = na, cb = nb, cz = nz;
            const long long no = j ? o : o + stride;
            if (no < n_out) {
                const int nj = j ? j - 1 : 15;
                na = __ldg(reinterpret_cast<const uint4*>(a_packed + at(no, nj)));
                nb = __ldg(reinterpret_cast<const uint4*>(b_packed + at(no, nj)));
                nz = __ldg(reinterpret_cast<const uint4*>(z_packed + at(no, nj)));
            }
            const u64 a0 = ext(ca.x, ca.y), a1 = ext(ca.z, ca.w), b0 = ext(cb.x, cb.y), b1 = ext(cb.z, cb.w);
            // p[e]: Σ_k x^k a_k(λ) b_k(λ) over the lane's columns k = 2q + t, x^(2q) applied after
            uint32_t p[8];
            if constexpr (cl) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const u64 m = 0x000000ff000000ffull;
                    const u64 A0 = (a0 >> (8 * e)) & m, B0 = (b0 >> (8 * e)) & m;
                    const u64 A1 = ((a1 >> (8 * e)) & m) << 1, B1 = (b1 >> (8 * e)) & m;
                    p[e] = (uint32_t)clmad_lo(A1, B1, clmad_lo(A0, B0, 0)) << (2 * q);
                    p[e + 4] = (uint32_t)clmad_hi(A1, B1, clmad_hi(A0, B0, 0)) << (2 * q);
                }
            } else {
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    const uint8_t x0 = (uint8_t)(a0 >> (8 * e)), y0 = (uint8_t)(b0 >> (8 * e));
                    const uint8_t x1 = (uint8_t)(a1 >> (8 * e)), y1 = (uint8_t)(b1 >> (8 * e));
                    const uint8_t p0 = s_antilog[(int)s_log[x0] + s_log[y0]] & (uint8_t)(-(int)(x0 && y0));
                    const uint8_t p1 = s_antilog[(int)s_log[x1] + s_log[y1]] & (uint8_t)(-(int)(x1 && y1));
                    p[e] = ((uint32_t)p0 ^ (uint32_t)p1 << 1) << (2 * q);
                }
            }
            // the four lanes of w add their sums: lane (w, q) keeps e in [4(q & 1), +4), then [e0, e0 + 2)
            uint32_t k4[4], k2[2];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint32_t keep = (q & 1) ? p[i + 4] : p[i], give = (q & 1) ? p[i] : p[i + 4];
                k4[i] = keep ^ __shfl_xor_sync(0xffffffffu, give, 8);
            }
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const uint32_t keep = (q >> 1) ? k4[i + 2] : k4[i], give = (q >> 1) ? k4[i] : k4[i + 2];
                k2[i] = keep ^ __shfl_xor_sync(0xffffffffu, give, 16);
            }
            const uint8_t r0 = cl ? zc_gf8_barrett(k2[0]) : reduce_zerocheck_gf8_polynomial((uint16_t)k2[0]);
            const uint8_t r1 = cl ? zc_gf8_barrett(k2[1]) : reduce_zerocheck_gf8_polynomial((uint16_t)k2[1]);
            // byte w of z's eight columns, byte k from column k
            const uint32_t zw = (((w < 4 ? cz.x : cz.y) >> (8 * (w & 3))) & 0xff) | (((w < 4 ? cz.z : cz.w) >> (8 * (w & 3))) & 0xff) << 8;
            const uint32_t zp = __shfl_xor_sync(0xffffffffu, zw, 8);
            const uint32_t z4 = (q & 1) ? zp | zw << 16 : zw | zp << 16;
            const uint32_t zo = __shfl_xor_sync(0xffffffffu, z4, 16);
            const uint32_t zlo = (q >> 1) ? zo : z4, zhi = (q >> 1) ? z4 : zo;
            const uint32_t v0 = zc_byte_bits(zlo, zhi, e0), v1 = zc_byte_bits(zlo, zhi, e0 + 1);
            chunk_ab0 = f128_add(multiply_zerocheck_value_by_generator(chunk_ab0), s_phi[r0 * 8 + w]);
            chunk_ab1 = f128_add(multiply_zerocheck_value_by_generator(chunk_ab1), s_phi[r1 * 8 + w]);
            chunk_w0 = f128_add(multiply_zerocheck_value_by_generator(chunk_w0), s_phi[v0 * 8 + w]);
            chunk_w1 = f128_add(multiply_zerocheck_value_by_generator(chunk_w1), s_phi[v1 * 8 + w]);
        }
        pab0 = f128_add(pab0, ghash_mul_karatsuba(eq_o, chunk_ab0));
        pab1 = f128_add(pab1, ghash_mul_karatsuba(eq_o, chunk_ab1));
        pw0 = f128_add(pw0, ghash_mul_karatsuba(eq_o, chunk_w0));
        pw1 = f128_add(pw1, ghash_mul_karatsuba(eq_o, chunk_w1));
    }
    const int l0 = 8 * w + e0;
    atomicXor((unsigned long long*)&raw_ab[l0].lo, pab0.lo);
    atomicXor((unsigned long long*)&raw_ab[l0].hi, pab0.hi);
    atomicXor((unsigned long long*)&raw_ab[l0 + 1].lo, pab1.lo);
    atomicXor((unsigned long long*)&raw_ab[l0 + 1].hi, pab1.hi);
    atomicXor((unsigned long long*)&raw_w[l0].lo, pw0.lo);
    atomicXor((unsigned long long*)&raw_w[l0].hi, pw0.hi);
    atomicXor((unsigned long long*)&raw_w[l0 + 1].lo, pw1.lo);
    atomicXor((unsigned long long*)&raw_w[l0 + 1].hi, pw1.hi);
}

// Round 1's outputs times C_s·C_med, c's from zc_round1_zlin's sums in place: c[λ] = scale·Σ_s φ8(M[s][λ])·w_s. 64 threads.
__global__ void zc_round1_combine(F128* __restrict__ ab, F128* __restrict__ c, const u64* __restrict__ t0,
                                  const F128* __restrict__ phi, F128 scale) {
    __shared__ F128 w[64];
    const int l = threadIdx.x;
    w[l] = c[l];
    __syncthreads();
    F128 acc{0, 0};
    for (int s = 0; s < 64; s++)
        acc = f128_add(acc, ghash_mul_karatsuba(phi[(uint8_t)(t0[(1 << (s & 7)) * 8 + ((l >> 3) ^ (s >> 3))] >> (8 * (l & 7)))], w[s]));
    ab[l] = ghash_mul_karatsuba(ab[l], scale);
    c[l] = ghash_mul_karatsuba(acc, scale);
}

// launch_zerocheck_first_round_cpu_structured's outputs from zc_round1_zlin; FC_ZC_ZLIN=0: from the table kernel
inline void launch_zc_round1(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_z, const F128* d_eq_out, long long n_out,
                             F128 scale, F128* d_ab, F128* d_c) {
    static const bool tables = [] { const char* e = std::getenv("FC_ZC_ZLIN"); return e && e[0] == '0'; }();
    if (tables) {
        launch_zerocheck_first_round_cpu_structured(d_a, d_b, d_z, d_eq_out, n_out, scale, d_ab, d_c);
        return;
    }
    constexpr int W = ZC_R1_W;
    constexpr size_t smem = zc_round1_smem();
    static const int grid = [] {
        build_zerocheck_logarithm_tables<<<1, 1>>>();
        cudaFuncSetAttribute(zc_round1_zlin<W>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        int dev = 0, sms = 1, per = 1;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, zc_round1_zlin<W>, W * 32, smem);
        return sms * (per > 0 ? per : 1);
    }();
    cudaMemset(d_ab, 0, 64 * sizeof(F128));
    cudaMemset(d_c, 0, 64 * sizeof(F128));
    zc_round1_zlin<W><<<grid, W * 32, smem>>>(d_a, d_b, d_z, d_eq_out, g_zc_t0, g_zc_phi, n_out, d_ab, d_c);
    zc_round1_combine<<<1, 64>>>(d_ab, d_c, g_zc_t0, g_zc_phi, scale);
}
