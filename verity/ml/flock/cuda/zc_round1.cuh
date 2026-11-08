// verity flock-gpu-link: zerocheck round 1 from the packed witness (launch_zc_round1). Reads the cuda-ghash TU's round-1
// tables and helpers (zerocheck_round1_cpustyle.cuh: g_zc_t0, g_zc_phi, the log tables and the table kernel).
#pragma once
#include <cstdlib>
#include "zerocheck_round1_cpustyle.cuh"
#include "zlin_transpose.cuh"

// Zerocheck round 1 (zerocheck_first_round_cpu_structured's outputs) with z folded before its extension. The table kernel's
// c output is Σ_o eq_o Σ_j γ^j φ8(Σ_k x^k ẑ_λ(o, j, k)), ẑ_λ = Σ_s z_s M[s][λ] the extension of a column's 64 bits, where
// M[8b + i][λ] = byte λ & 7 of t0[1 << i][(λ >> 3) ^ b]. φ8 embeds GF(2^8) in F128 as a ring, so that is Σ_s φ8(M[s][λ])·w_s
// with w_s = Σ_o eq_o Σ_j γ^j φ8(v_s), v_s the byte whose bit k is bit s of column k: lane l sums w_s for s = l, l + 32, and
// zc_round1_combine applies M to the 64 sums once. Only a and b go through the t0 tables.
// Lane (k, w) = (lane >> 2, lane & 3) of a warp step builds words w and w + 4 of column k's extensions of a and b: t0 holds
// words u and u ^ 4 side by side, so byte b's two words are one 16-byte read, at an offset one byte permute forms from b's
// value and the lane's offset. It interleaves the four words' bytes so that the four lane l needs (byte l & 7 of words
// l >> 3 and (l >> 3) + 4 of a and b: λ = l and l + 32) are one word of its column's row of gbuf, which lane l reads 4 bytes a
// column. a·b is the table kernel's (clmad on sm_90, the log tables elsewhere; ZC_R1_TABLES forces them). Lane l's bytes v_l
// and v_{l+32} are bit l of z's columns' halves, each rotated to its column's bit. On clmad an outer chunk's fold over j,
// Σ_j γ^j φ8(r_j) with γ = x, is Σ_i φ8(x^i)·c_i, c_i the 16-bit polynomial whose bit j is bit i of r_j (two 8x8 bit
// transposes of the chunk's 16 bytes), in place of a φ8 read and a multiplication by γ a step.
// No shared-memory conflicts: a quarter-warp's 16-byte table reads are 8 slots of one 128-byte row (copy k & 1, word pair
// w ^ (b & 3)), its gbuf stores 8 slots (odd columns' rows swap their 16-byte halves), a gbuf row read is the row, and a
// quarter-warp's eight φ8 reads lane l's copy, on bank quad l & 7. The grid is resident (zc_round1_smem bytes a block,
// ZC_R1_MINB blocks an SM) and its warps stride over the outer chunks, so each block loads the copies once; one block of 16
// warps an SM leaves the most L1 for the streamed columns.
#ifndef ZC_R1_W
#define ZC_R1_W 16
#endif
#ifndef ZC_R1_MINB
#define ZC_R1_MINB 1
#endif
template <int W>
constexpr size_t zc_round1_smem() { return 256 * 8 * sizeof(F128) + 256 * 256 + W * (1024 + 64) + 768; }

// p < 2^22 modulo the AES polynomial P (reduce_zerocheck_gf8_wide's residue): with μ = ⌊x^22 / P⌋ = 0x4696,
// q = ⌊⌊p / x^8⌋·μ / x^14⌋ is ⌊p / P⌋ (Barrett, exact over GF(2)[x]) and p - q·P's low byte the residue
__device__ __forceinline__ uint8_t zc_gf8_barrett(uint32_t p) {
    const u64 q = clmad_lo((u64)(p >> 8), 0x4696ull, 0) >> 14;
    return (uint8_t)(p ^ (uint32_t)clmad_lo(q, 0x11bull, 0));
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
    unsigned char* s_t0 = reinterpret_cast<unsigned char*>(s_phi + 256 * 8);   // entry (v, u)'s copy c at 256v + 64c + 16(u & 3) + 8(u >> 2)
    uint32_t* s_g = reinterpret_cast<uint32_t*>(s_t0 + 256 * 256);   // a warp's 8 rows of 32 words
    u64* s_z = reinterpret_cast<u64*>(s_g + W * 256);   // a warp's 8 columns of z
    uint8_t* s_log = reinterpret_cast<uint8_t*>(s_z + W * 8);
    uint8_t* s_antilog = s_log + 256;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000 && !defined(ZC_R1_TABLES)
    constexpr bool cl = true;
#else
    constexpr bool cl = false;
#endif
    for (int i = threadIdx.x; !cl && i < 256; i += blockDim.x) s_log[i] = d_zc_log[i];
    for (int i = threadIdx.x; !cl && i < 512; i += blockDim.x) s_antilog[i] = d_zc_antilog[i];
    for (int i = threadIdx.x; i < 256 * 16; i += blockDim.x) {
        const int v = i >> 4, c = (i >> 3) & 1, u = i & 7;
        *reinterpret_cast<u64*>(s_t0 + 256 * v + 64 * c + 16 * (u & 3) + 8 * (u >> 2)) = t0[v * 8 + u];
    }
    for (int i = threadIdx.x; i < 256 * 8; i += blockDim.x) s_phi[i] = phi[i >> 3];
    __syncthreads();

    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, k = lane >> 2, w = lane & 3, q = lane & 7;
    uint32_t toff[4];   // byte b's pair offset in the lane's copy: words w ^ b and (w ^ b) ^ 4
#pragma unroll
    for (int i = 0; i < 4; i++) toff[i] = (uint32_t)(64 * (k & 1) + 16 * (w ^ i));
    uint32_t rot[8];   // rotating right by rot[c] takes bit l to bit c
#pragma unroll
    for (int c = 0; c < 8; c++) rot[c] = (uint32_t)(lane - c) & 31;
    uint32_t* grow = s_g + wid * 256 + 32 * k;
    const uint32_t* gl = s_g + wid * 256 + lane;
    const uint32_t* glx = s_g + wid * 256 + (lane ^ 4);
    u64* zw = s_z + wid * 8;
    const long long stride = (long long)gridDim.x * W;
    // the lane's column k of step (o, j)
    auto at = [&](long long o, int j) { return (o * 128 + j * 8 + k) * 8; };
    long long o = (long long)blockIdx.x * W + wid;
    u64 na = 0, nb = 0, nz = 0;
    if (o < n_out) {
        na = __ldg(reinterpret_cast<const unsigned long long*>(a_packed + at(o, 15)));
        nb = __ldg(reinterpret_cast<const unsigned long long*>(b_packed + at(o, 15)));
        nz = __ldg(reinterpret_cast<const unsigned long long*>(z_packed + at(o, 15)));
    }
    F128 pab0{0, 0}, pab1{0, 0}, pw0{0, 0}, pw1{0, 0};
    for (; o < n_out; o += stride) {
        const F128 eq_o = eq_out[o];
        F128 chunk[4] = {};   // a·b's at λ = l, l + 32; w_s's at s = l, l + 32
        uint32_t rb[4][4] = {};   // on clmad: byte j & 3 of rb[·][j >> 2] the reduced byte φ8 takes at j
        for (int jj = 3; jj >= 0; jj--) {
            uint32_t cur[4] = {};
#pragma unroll
            for (int i = 3; i >= 0; i--) {
                const int j = 4 * jj + i;
                const u64 ca = na, cb = nb;
                if (w == 0) zw[k] = nz;
                const long long no = j ? o : o + stride;
                if (no < n_out) {
                    const int nj = j ? j - 1 : 15;
                    na = __ldg(reinterpret_cast<const unsigned long long*>(a_packed + at(no, nj)));
                    nb = __ldg(reinterpret_cast<const unsigned long long*>(b_packed + at(no, nj)));
                    nz = __ldg(reinterpret_cast<const unsigned long long*>(z_packed + at(no, nj)));
                }
                // words w (L) and w + 4 (H) of column k's extensions
                uint32_t aL0 = 0, aL1 = 0, aH0 = 0, aH1 = 0, bL0 = 0, bL1 = 0, bH0 = 0, bH1 = 0;
#pragma unroll
                for (int b = 0; b < 8; b++) {
                    const uint32_t sel = 0x5504u | (uint32_t)(b & 3) << 4;
                    const uint4 ta = *reinterpret_cast<const uint4*>(s_t0 + __byte_perm(b < 4 ? (uint32_t)ca : (uint32_t)(ca >> 32), toff[b & 3], sel));
                    const uint4 tb = *reinterpret_cast<const uint4*>(s_t0 + __byte_perm(b < 4 ? (uint32_t)cb : (uint32_t)(cb >> 32), toff[b & 3], sel));
                    if (b < 4) {
                        aL0 ^= ta.x; aL1 ^= ta.y; aH0 ^= ta.z; aH1 ^= ta.w;
                        bL0 ^= tb.x; bL1 ^= tb.y; bH0 ^= tb.z; bH1 ^= tb.w;
                    } else {
                        aL0 ^= ta.z; aL1 ^= ta.w; aH0 ^= ta.x; aH1 ^= ta.y;
                        bL0 ^= tb.z; bL1 ^= tb.w; bH0 ^= tb.x; bH1 ^= tb.y;
                    }
                }
                // g[e] = bytes e of words L and H of a, then of b
                const uint32_t pa0 = __byte_perm(aL0, aH0, 0x5140), pa1 = __byte_perm(aL0, aH0, 0x7362);
                const uint32_t pa2 = __byte_perm(aL1, aH1, 0x5140), pa3 = __byte_perm(aL1, aH1, 0x7362);
                const uint32_t pb0 = __byte_perm(bL0, bH0, 0x5140), pb1 = __byte_perm(bL0, bH0, 0x7362);
                const uint32_t pb2 = __byte_perm(bL1, bH1, 0x5140), pb3 = __byte_perm(bL1, bH1, 0x7362);
                const uint4 g0 = make_uint4(__byte_perm(pa0, pb0, 0x5410), __byte_perm(pa0, pb0, 0x7632),
                                            __byte_perm(pa1, pb1, 0x5410), __byte_perm(pa1, pb1, 0x7632));
                const uint4 g1 = make_uint4(__byte_perm(pa2, pb2, 0x5410), __byte_perm(pa2, pb2, 0x7632),
                                            __byte_perm(pa3, pb3, 0x5410), __byte_perm(pa3, pb3, 0x7632));
                *reinterpret_cast<uint4*>(grow + ((8 * w) ^ (4 * (k & 1)))) = g0;
                *reinterpret_cast<uint4*>(grow + ((8 * w + 4) ^ (4 * (k & 1)))) = g1;
                __syncwarp();
                uint32_t aacc0 = 0, aacc1 = 0;
                u64 pl = 0, ph = 0;
#pragma unroll
                for (int c = 0; c < 8; c++) {
                    const uint32_t x = (c & 1 ? glx : gl)[32 * c];
                    const uint32_t aH = __byte_perm(x, 0, 0x4441), bL = __byte_perm(x, 0, 0x4442), bH = __umulhi(x, 256);
                    if constexpr (cl) {
                        const u64 A = ((u64)(aH << c) << 32) | __umulhi(x << 24, 256u << c), B = ((u64)bH << 32) | bL;
                        pl = clmad_lo(A, B, pl);
                        ph = clmad_hi(A, B, ph);
                    } else {
                        const uint32_t aL = x & 0xff;
                        const uint32_t p0 = s_antilog[(int)s_log[aL] + s_log[bL]] & (uint32_t)(-(int)(aL && bL));
                        const uint32_t p1 = s_antilog[(int)s_log[aH] + s_log[bH]] & (uint32_t)(-(int)(aH && bH));
                        aacc0 ^= p0 << c;
                        aacc1 ^= p1 << c;
                    }
                }
                uint32_t r[4] = {0, 0, 0, 0};   // r0, r1, v0, v1
#pragma unroll
                for (int c = 0; c < 8; c += 2) {
                    const uint4 zz = *reinterpret_cast<const uint4*>(zw + c);
                    r[2] |= __funnelshift_r(zz.x, zz.x, rot[c]) & (1u << c);
                    r[3] |= __funnelshift_r(zz.y, zz.y, rot[c]) & (1u << c);
                    r[2] |= __funnelshift_r(zz.z, zz.z, rot[c + 1]) & (2u << c);
                    r[3] |= __funnelshift_r(zz.w, zz.w, rot[c + 1]) & (2u << c);
                }
                r[0] = cl ? zc_gf8_barrett((uint32_t)pl) : reduce_zerocheck_gf8_polynomial((uint16_t)aacc0);
                r[1] = cl ? zc_gf8_barrett((uint32_t)ph) : reduce_zerocheck_gf8_polynomial((uint16_t)aacc1);
#pragma unroll
                for (int t = 0; t < 4; t++) {
                    if constexpr (cl) cur[t] += r[t] << (8 * i);
                    else chunk[t] = f128_add(multiply_zerocheck_value_by_generator(chunk[t]), s_phi[r[t] * 8 + q]);
                }
                __syncwarp();
            }
#pragma unroll
            for (int t = 0; t < 4; t++) {
                rb[t][3] = rb[t][2]; rb[t][2] = rb[t][1]; rb[t][1] = rb[t][0]; rb[t][0] = cur[t];
            }
        }
        if constexpr (cl) {
            // Σ_j x^j φ8(r_j) = Σ_i φ8(x^i)·c_i, bit j of c_i bit i of r_j: c_2m and c_2m+1 are the halves of cw[m]
            uint32_t cw[4][4];
#pragma unroll
            for (int t = 0; t < 4; t++) {
                const u64 tl = zlin_t8((u64)rb[t][1] << 32 | rb[t][0]), th = zlin_t8((u64)rb[t][3] << 32 | rb[t][2]);
                cw[t][0] = __byte_perm((uint32_t)tl, (uint32_t)th, 0x5140);
                cw[t][1] = __byte_perm((uint32_t)tl, (uint32_t)th, 0x7362);
                cw[t][2] = __byte_perm((uint32_t)(tl >> 32), (uint32_t)(th >> 32), 0x5140);
                cw[t][3] = __byte_perm((uint32_t)(tl >> 32), (uint32_t)(th >> 32), 0x7362);
            }
            u64 s0[4] = {}, s1[4] = {}, s2[4] = {};
#pragma unroll
            for (int bit = 0; bit < 8; bit++) {
                const F128 f = s_phi[(1 << bit) * 8 + q];
#pragma unroll
                for (int t = 0; t < 4; t++) {
                    const u64 ci = bit & 1 ? __umulhi(cw[t][bit >> 1], 65536) : __byte_perm(cw[t][bit >> 1], 0, 0x4410);
                    s0[t] = clmad_lo(f.lo, ci, s0[t]);
                    s1[t] = clmad_lo(f.hi, ci, clmad_hi(f.lo, ci, s1[t]));
                    s2[t] = clmad_hi(f.hi, ci, s2[t]);
                }
            }
#pragma unroll
            for (int t = 0; t < 4; t++) chunk[t] = F128{clmad_lo(s2[t], 0x87ull, s0[t]), s1[t]};
        }
        pab0 = f128_add(pab0, ghash_mul_karatsuba(eq_o, chunk[0]));
        pab1 = f128_add(pab1, ghash_mul_karatsuba(eq_o, chunk[1]));
        pw0 = f128_add(pw0, ghash_mul_karatsuba(eq_o, chunk[2]));
        pw1 = f128_add(pw1, ghash_mul_karatsuba(eq_o, chunk[3]));
    }
    atomicXor((unsigned long long*)&raw_ab[lane].lo, pab0.lo);
    atomicXor((unsigned long long*)&raw_ab[lane].hi, pab0.hi);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].lo, pab1.lo);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].hi, pab1.hi);
    atomicXor((unsigned long long*)&raw_w[lane].lo, pw0.lo);
    atomicXor((unsigned long long*)&raw_w[lane].hi, pw0.hi);
    atomicXor((unsigned long long*)&raw_w[lane + 32].lo, pw1.lo);
    atomicXor((unsigned long long*)&raw_w[lane + 32].hi, pw1.hi);
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
    constexpr size_t smem = zc_round1_smem<W>();
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
