// Operand reuse on sm_120 `mma.sync`: what a tensor-core operand register may hold, and what it costs to reuse it.
//
//   t1_fanout    one A fragment converted once (F2FP), then fed to 8 QMMAs against 8 B tiles, no op in between
//   t1_loop      the same A fragment fed to a runtime-length loop of chained QMMAs
//   t2_f16copy   QMMA with an f16 accumulator (16x16 = two n8 tiles); its D registers are the A operand of an HMMA k16
//   t3_tf32copy  TF32 HMMA k8; its f32 D registers are the A operand of the next TF32 HMMA
//   t4_s8copy    IMMA k32 s8->s32; its s32 D registers are the A operand (four s8 digits each) of the next IMMA
//   t5_rawbits   QMMA f32 D registers fed as E4M3x4 A registers to the next QMMA (raw float bits as FP8 codes)
//   t6_bf16copy  QMMA f32 D registers fed as BF16x2 A registers to an HMMA k16 (high half = D truncated to BF16)
//   t7_split     a full-width f32 tile output minus its own TF32 truncation, the subtraction in the accumulator input:
//                an exact low digit with no CUDA arithmetic (A' needs a register permutation: four MOVs in the SASS)
//   t8_split_bf16  t7 in BF16 only (W1's formats), twice: three exact BF16 digits of a 24-bit value; t8b shows a low
//                half that is a BF16 NaN code poisons the split
//
// One warp per kernel.  Fragment layouts are PTX ISA 9.4's (g = lane>>2, t = lane&3):
//   8-bit A (k32) reg r: row g+8(r&1), cols 4t+16(r>>1)+j     8-bit B (k32) reg r: k 4t+16r+j, n g
//   f16 A (k16) reg r:   row g+8(r&1), cols 2t+8(r>>1)+{0,1}  f16 B (k16) reg r:   k 2t+8r+{0,1}, n g
//   tf32 A (k8): a0 (g,t) a1 (g+8,t) a2 (g,t+4) a3 (g+8,t+4)  tf32 B (k8): b0 k t, b1 k t+4, n g
//   f32/s32 D c_i: row g+8(i>>1), col 2t+(i&1)                f16 D reg r: row g+8r, cols 2t+{0,1}
// so the f16 D of two n8 tiles is exactly a k16 f16 A fragment.  A 32-bit D fed as A in its own register order (a_r = c_r,
// no move) is a fixed permutation of D's entries as a k8 tf32 A, and four byte digits per entry as a k32 8-bit A: A' entry
// (row, k) comes from c_r of lane (row%8, t), r = (row>=8) + 2(k>=K/2), t = k%4 (tf32) or (k%16)/4 with byte k%4 (8-bit).
// The host recomputes every result from these maps and reports exact agreement.
//
// Build: nvcc -O2 -std=c++17 -gencode=arch=compute_120a,code=sm_120a -o probe probe.cu

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));   \
            return 1;                                                                            \
        }                                                                                        \
    } while (0)

__device__ __forceinline__ void qmma_f32(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
                   "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void qmma_f16(uint32_t d[2], const uint32_t a[4], const uint32_t b[2], const uint32_t c[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]));
}

__device__ __forceinline__ void hmma_f16_k16(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
                   "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void hmma_tf32_k8(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
                   "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void imma_s8(int d[4], const uint32_t a[4], const uint32_t b[2], const int c[4]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%11,%12,%13};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]),
                   "r"(c[2]), "r"(c[3]));
}

// Four f32 -> four E4M3 codes in one register, element 0 in the low byte.
__device__ __forceinline__ uint32_t cvt_e4m3x4(float x0, float x1, float x2, float x3) {
    uint32_t r;
    asm volatile("{ .reg .b16 lo, hi;\n"
                 "  cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
                 "  cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
                 "  mov.b32 %0, {lo, hi}; }\n"
                 : "=r"(r)
                 : "f"(x0), "f"(x1), "f"(x2), "f"(x3));
    return r;
}

__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }

// A 16x32 8-bit operand fragment from a row-major tile.
__device__ __forceinline__ void load_a8(uint32_t a[4], const uint8_t* A, int g, int t) {
    for (int r = 0; r < 4; ++r) a[r] = ld32(A + (g + 8 * (r & 1)) * 32 + 4 * t + 16 * (r >> 1));
}

// A 32x8 8-bit operand fragment from an n-major (k contiguous) tile.
__device__ __forceinline__ void load_b8(uint32_t b[2], const uint8_t* Bt, int g, int t) {
    for (int r = 0; r < 2; ++r) b[r] = ld32(Bt + g * 32 + 4 * t + 16 * r);
}

template <typename T>
__device__ __forceinline__ void store_d(T* D, const T d[4], int g, int t) {
    for (int i = 0; i < 4; ++i) D[(g + 8 * (i >> 1)) * 8 + 2 * t + (i & 1)] = d[i];
}

constexpr int NB = 8;

extern "C" __global__ void __launch_bounds__(32) t1_fanout(const float* __restrict__ Af, const uint8_t* __restrict__ Bt,
                                                           float* __restrict__ D) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    uint32_t a[4];
    for (int r = 0; r < 4; ++r) {
        const float* p = Af + (g + 8 * (r & 1)) * 32 + 4 * t + 16 * (r >> 1);
        a[r] = cvt_e4m3x4(p[0], p[1], p[2], p[3]);
    }
    uint32_t b[NB][2];
#pragma unroll
    for (int n = 0; n < NB; ++n) load_b8(b[n], Bt + n * 256, g, t);
    const float z[4] = {0.f, 0.f, 0.f, 0.f};
    float d[NB][4];
#pragma unroll
    for (int n = 0; n < NB; ++n) qmma_f32(d[n], a, b[n], z);
#pragma unroll
    for (int n = 0; n < NB; ++n) store_d(D + n * 128, d[n], g, t);
}

extern "C" __global__ void __launch_bounds__(32) t1_loop(const float* __restrict__ Af, const uint8_t* __restrict__ Bt,
                                                         float* __restrict__ D, int nb) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    uint32_t a[4];
    for (int r = 0; r < 4; ++r) {
        const float* p = Af + (g + 8 * (r & 1)) * 32 + 4 * t + 16 * (r >> 1);
        a[r] = cvt_e4m3x4(p[0], p[1], p[2], p[3]);
    }
    float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int n = 0; n < nb; ++n) {
        uint32_t b[2];
        load_b8(b, Bt + n * 256, g, t);
        qmma_f32(d, a, b, d);
    }
    store_d(D, d, g, t);
}

// A 16x32 e4m3, B1 32x16 e4m3 (two n-major 8x32 tiles), B2t 8x16 f16 (n-major).  Dmid = A*B1 (16x16 f16), Dout = Dmid*B2.
extern "C" __global__ void __launch_bounds__(32) t2_f16copy(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B1t,
                                                            const __half* __restrict__ B2t, __half* __restrict__ Dmid,
                                                            float* __restrict__ Dout) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    uint32_t a[4], b0[2], b1[2];
    load_a8(a, A, g, t);
    load_b8(b0, B1t, g, t);
    load_b8(b1, B1t + 256, g, t);
    const uint32_t zc[2] = {0u, 0u};
    uint32_t d0[2], d1[2];
    qmma_f16(d0, a, b0, zc);
    qmma_f16(d1, a, b1, zc);
    const uint32_t a2[4] = {d0[0], d0[1], d1[0], d1[1]};
    uint32_t b2[2];
    for (int r = 0; r < 2; ++r) b2[r] = *reinterpret_cast<const uint32_t*>(B2t + g * 16 + 2 * t + 8 * r);
    const float z[4] = {0.f, 0.f, 0.f, 0.f};
    float d[4];
    hmma_f16_k16(d, a2, b2, z);
    store_d(Dout, d, g, t);
    for (int r = 0; r < 2; ++r) {
        *reinterpret_cast<uint32_t*>(Dmid + (g + 8 * r) * 16 + 2 * t) = d0[r];
        *reinterpret_cast<uint32_t*>(Dmid + (g + 8 * r) * 16 + 8 + 2 * t) = d1[r];
    }
}

// A 16x8 f32 (as tf32), C1 16x8 f32, B1t/B2t 8x8 f32 (n-major).  Dmid = A*B1 + C1, Dout = tf32(perm(Dmid))*B2.
extern "C" __global__ void __launch_bounds__(32) t3_tf32copy(const float* __restrict__ A, const float* __restrict__ C1,
                                                             const float* __restrict__ B1t, const float* __restrict__ B2t,
                                                             float* __restrict__ Dmid, float* __restrict__ Dout) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    uint32_t a[4] = {__float_as_uint(A[g * 8 + t]), __float_as_uint(A[(g + 8) * 8 + t]), __float_as_uint(A[g * 8 + t + 4]),
                     __float_as_uint(A[(g + 8) * 8 + t + 4])};
    uint32_t b[2] = {__float_as_uint(B1t[g * 8 + t]), __float_as_uint(B1t[g * 8 + t + 4])};
    float c[4];
    for (int i = 0; i < 4; ++i) c[i] = C1[(g + 8 * (i >> 1)) * 8 + 2 * t + (i & 1)];
    float d1[4];
    hmma_tf32_k8(d1, a, b, c);
    const uint32_t a2[4] = {__float_as_uint(d1[0]), __float_as_uint(d1[1]), __float_as_uint(d1[2]), __float_as_uint(d1[3])};
    uint32_t b2[2] = {__float_as_uint(B2t[g * 8 + t]), __float_as_uint(B2t[g * 8 + t + 4])};
    const float z[4] = {0.f, 0.f, 0.f, 0.f};
    float d[4];
    hmma_tf32_k8(d, a2, b2, z);
    store_d(Dmid, d1, g, t);
    store_d(Dout, d, g, t);
}

// A 16x32 s8, B1t/B2t 8x32 s8 (n-major).  Dmid = A*B1 (s32), Dout = bytes(perm(Dmid))*B2.
extern "C" __global__ void __launch_bounds__(32) t4_s8copy(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B1t,
                                                           const uint8_t* __restrict__ B2t, int* __restrict__ Dmid,
                                                           int* __restrict__ Dout) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    uint32_t a[4], b[2], b2[2];
    load_a8(a, A, g, t);
    load_b8(b, B1t, g, t);
    load_b8(b2, B2t, g, t);
    const int z[4] = {0, 0, 0, 0};
    int d1[4];
    imma_s8(d1, a, b, z);
    const uint32_t a2[4] = {(uint32_t)d1[0], (uint32_t)d1[1], (uint32_t)d1[2], (uint32_t)d1[3]};
    int d[4];
    imma_s8(d, a2, b2, z);
    store_d(Dmid, d1, g, t);
    store_d(Dout, d, g, t);
}

// A 16x32 e4m3, B1t/B2t 8x32 e4m3.  Dmid = A*B1 (f32), Dout = e4m3x4(bits(perm(Dmid)))*B2.
extern "C" __global__ void __launch_bounds__(32) t5_rawbits(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B1t,
                                                            const uint8_t* __restrict__ B2t, float* __restrict__ Dmid,
                                                            float* __restrict__ Dout) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    uint32_t a[4], b[2], b2[2];
    load_a8(a, A, g, t);
    load_b8(b, B1t, g, t);
    load_b8(b2, B2t, g, t);
    const float z[4] = {0.f, 0.f, 0.f, 0.f};
    float d1[4];
    qmma_f32(d1, a, b, z);
    const uint32_t a2[4] = {__float_as_uint(d1[0]), __float_as_uint(d1[1]), __float_as_uint(d1[2]), __float_as_uint(d1[3])};
    float d[4];
    qmma_f32(d, a2, b2, z);
    store_d(Dmid, d1, g, t);
    store_d(Dout, d, g, t);
}

__device__ __forceinline__ void hmma_bf16_k16(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
                   "f"(c[2]), "f"(c[3]));
}

// A 16x32 e4m3, B1t 8x32 e4m3, B2t 8x16 bf16 bits (n-major).  Dmid = A*B1 (f32), Dout = bf16x2(bits(perm(Dmid)))*B2:
// each FP32 word is two BF16 codes, its low half (mantissa tail) and its high half (the value truncated to BF16).
extern "C" __global__ void __launch_bounds__(32) t6_bf16copy(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B1t,
                                                             const uint16_t* __restrict__ B2t, float* __restrict__ Dmid,
                                                             float* __restrict__ Dout) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    uint32_t a[4], b[2], b2[2];
    load_a8(a, A, g, t);
    load_b8(b, B1t, g, t);
    for (int r = 0; r < 2; ++r) b2[r] = *reinterpret_cast<const uint32_t*>(B2t + g * 16 + 2 * t + 8 * r);
    const float z[4] = {0.f, 0.f, 0.f, 0.f};
    float d1[4];
    qmma_f32(d1, a, b, z);
    const uint32_t a2[4] = {__float_as_uint(d1[0]), __float_as_uint(d1[1]), __float_as_uint(d1[2]), __float_as_uint(d1[3])};
    float d[4];
    hmma_bf16_k16(d, a2, b2, z);
    store_d(Dmid, d1, g, t);
    store_d(Dout, d, g, t);
}

// V 16x8 full-width f32, B2t 8x8 f32 (n-major, the -1 selector).  D0 = V + 0*0 (a tile output holding V), then
// D1 = D0 + tf32(A')*B2 with A' = {c0,c2,c1,c3} of D0 (A'[r][t] = D0[r][2t], A'[r][t+4] = D0[r][2t+1]) and B2 the
// selector that sends A'[r][k] to column 2k (k < 4) or 2(k-4)+1 with weight -1: D1 = D0 - trunc_tf32(D0), the low digit,
// with the subtraction done in the accumulator input.
extern "C" __global__ void __launch_bounds__(32) t7_split(const float* __restrict__ V, const float* __restrict__ B2t,
                                                          float* __restrict__ D0out, float* __restrict__ D1out) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    float c[4];
    for (int i = 0; i < 4; ++i) c[i] = V[(g + 8 * (i >> 1)) * 8 + 2 * t + (i & 1)];
    const uint32_t za[4] = {0u, 0u, 0u, 0u}, zb[2] = {0u, 0u};
    float d0[4];
    hmma_tf32_k8(d0, za, zb, c);
    const uint32_t a2[4] = {__float_as_uint(d0[0]), __float_as_uint(d0[2]), __float_as_uint(d0[1]), __float_as_uint(d0[3])};
    uint32_t b2[2] = {__float_as_uint(B2t[g * 8 + t]), __float_as_uint(B2t[g * 8 + t + 4])};
    float d1[4];
    hmma_tf32_k8(d1, a2, b2, d0);
    store_d(D0out, d0, g, t);
    store_d(D1out, d1, g, t);
}

// t7 in BF16, W1's own formats only.  D0 = V + 0*0, then twice D_{i+1} = D_i + bf16(A'_i)*B2 with A'_i = {c0,c2,c1,c3} of
// D_i: as a k16 BF16 A, A'[r][2t+1] = hi16(D[r][2t]) and A'[r][2t+9] = hi16(D[r][2t+1]), and B2 sends those to columns
// 2t and 2t+1 with weight -1 (the low halves meet zeros).  So D1 = D0 - trunc_bf16(D0), D2 = D1 - trunc_bf16(D1): three
// BF16 digits of a 24-bit value.  A low half whose exponent field is all ones is a BF16 NaN or Inf code, and code * 0 is
// NaN, so V's mantissa bits 14..7 must not all be set (t8b sets them).
extern "C" __global__ void __launch_bounds__(32) t8_split_bf16(const float* __restrict__ V, const uint16_t* __restrict__ B2t,
                                                               float* __restrict__ D1out, float* __restrict__ D2out) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    float c[4];
    for (int i = 0; i < 4; ++i) c[i] = V[(g + 8 * (i >> 1)) * 8 + 2 * t + (i & 1)];
    const uint32_t za[4] = {0u, 0u, 0u, 0u}, zb[2] = {0u, 0u};
    uint32_t b2[2];
    for (int r = 0; r < 2; ++r) b2[r] = *reinterpret_cast<const uint32_t*>(B2t + g * 16 + 2 * t + 8 * r);
    float d0[4];
    hmma_bf16_k16(d0, za, zb, c);
    const uint32_t a1[4] = {__float_as_uint(d0[0]), __float_as_uint(d0[2]), __float_as_uint(d0[1]), __float_as_uint(d0[3])};
    float d1[4];
    hmma_bf16_k16(d1, a1, b2, d0);
    const uint32_t a2[4] = {__float_as_uint(d1[0]), __float_as_uint(d1[2]), __float_as_uint(d1[1]), __float_as_uint(d1[3])};
    float d2[4];
    hmma_bf16_k16(d2, a2, b2, d1);
    store_d(D1out, d1, g, t);
    store_d(D2out, d2, g, t);
}

// ---------------------------------------------------------------------------------------------------------------- host

static float e4m3_decode(uint8_t c) {
    const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    if (e == 15 && m == 7) return NAN;
    const float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.f + m / 8.f, e - 7);
    return s ? -v : v;
}

static uint8_t e4m3_encode_exact(float v) {
    for (int c = 0; c < 256; ++c)
        if (e4m3_decode((uint8_t)c) == v && !(c == 0x80 && v == 0.f)) return (uint8_t)c;
    fprintf(stderr, "no exact e4m3 code for %g\n", v);
    return 0;
}

// Index into a row-major 16x8 D of the entry c_r of lane (g, t), r = (row>=8) + 2(k>=K/2), t = (k % (K/2)) / E for E
// elements a register: the D entry that sits in A' entry (row, k) when D's registers are fed as A's.
static int d_index_for_a(int row, int k, int K, int E) {
    const int r = (row >= 8) + 2 * (k >= K / 2), g = row % 8, t = (k % (K / 2)) / E;
    return (g + 8 * (r >> 1)) * 8 + 2 * t + (r & 1);
}

static uint32_t rng = 0x9e3779b9u;
static int rnd(int lo, int hi) {
    rng = rng * 1664525u + 1013904223u;
    return lo + (int)((rng >> 8) % (uint32_t)(hi - lo + 1));
}

struct Result {
    const char* name;
    bool exact;
    double max_err;
    int n;
};

template <typename T>
static T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T) + 16);
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

template <typename T>
static std::vector<T> back(const T* p, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

static Result compare(const char* name, const std::vector<double>& got, const std::vector<double>& want) {
    double m = 0;
    bool exact = true;
    for (size_t i = 0; i < got.size(); ++i) {
        const double e = std::fabs(got[i] - want[i]);
        if (!(got[i] == want[i])) exact = false;
        if (!(e <= m)) m = std::isnan(e) ? INFINITY : e;
    }
    return {name, exact, m, (int)got.size()};
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    std::vector<Result> res;

    // t1: A 16x32 integers in [-16,16] (exact in E4M3), NB B tiles 8x32 integers in [-4,4].
    std::vector<float> Af(16 * 32), Avals(16 * 32);
    for (int i = 0; i < 16 * 32; ++i) Af[i] = Avals[i] = (float)rnd(-16, 16);
    std::vector<uint8_t> Bt(NB * 256);
    std::vector<float> Bv(NB * 256);
    for (int i = 0; i < NB * 256; ++i) Bt[i] = e4m3_encode_exact(Bv[i] = (float)rnd(-4, 4));
    {
        float* dA = dev(Af);
        uint8_t* dB = dev(Bt);
        float* dD;
        CK(cudaMalloc(&dD, NB * 128 * sizeof(float)));
        t1_fanout<<<1, 32>>>(dA, dB, dD);
        CK(cudaDeviceSynchronize());
        auto D = back(dD, NB * 128);
        std::vector<double> got(D.begin(), D.end()), want(NB * 128);
        for (int n = 0; n < NB; ++n)
            for (int i = 0; i < 16; ++i)
                for (int j = 0; j < 8; ++j) {
                    double s = 0;
                    for (int k = 0; k < 32; ++k) s += (double)Avals[i * 32 + k] * Bv[n * 256 + j * 32 + k];
                    want[n * 128 + i * 8 + j] = s;
                }
        res.push_back(compare("t1_fanout", got, want));

        t1_loop<<<1, 32>>>(dA, dB, dD, NB);
        CK(cudaDeviceSynchronize());
        auto L = back(dD, 128);
        std::vector<double> lg(L.begin(), L.end()), lw(128, 0.0);
        for (int n = 0; n < NB; ++n)
            for (int i = 0; i < 128; ++i) lw[i] += want[n * 128 + i];
        res.push_back(compare("t1_loop", lg, lw));
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dD);
    }

    // t2: A in [-8,8], B1 in [-2,2] (|A*B1| <= 512, exact in f16), B2 in [-3,3].
    {
        std::vector<uint8_t> A(16 * 32), B1t(2 * 256);
        std::vector<float> Av(16 * 32), B1v(2 * 256);
        for (int i = 0; i < 16 * 32; ++i) A[i] = e4m3_encode_exact(Av[i] = (float)rnd(-8, 8));
        for (int i = 0; i < 2 * 256; ++i) B1t[i] = e4m3_encode_exact(B1v[i] = (float)rnd(-2, 2));
        std::vector<__half> B2t(8 * 16);
        std::vector<float> B2v(8 * 16);
        for (int i = 0; i < 8 * 16; ++i) B2t[i] = __float2half(B2v[i] = (float)rnd(-3, 3));
        uint8_t *dA = dev(A), *dB1 = dev(B1t);
        __half* dB2 = dev(B2t);
        __half* dM;
        float* dO;
        CK(cudaMalloc(&dM, 256 * sizeof(__half)));
        CK(cudaMalloc(&dO, 128 * sizeof(float)));
        t2_f16copy<<<1, 32>>>(dA, dB1, dB2, dM, dO);
        CK(cudaDeviceSynchronize());
        auto M = back(dM, 256);
        auto O = back(dO, 128);
        std::vector<double> mg(256), mw(256), og(O.begin(), O.end()), ow(128);
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j) {
                double s = 0;
                for (int k = 0; k < 32; ++k) s += (double)Av[i * 32 + k] * B1v[(j / 8) * 256 + (j % 8) * 32 + k];
                mw[i * 16 + j] = s;
                mg[i * 16 + j] = __half2float(M[i * 16 + j]);
            }
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 8; ++j) {
                double s = 0;
                for (int k = 0; k < 16; ++k) s += mw[i * 16 + k] * B2v[j * 16 + k];
                ow[i * 8 + j] = s;
            }
        res.push_back(compare("t2_f16copy.mid", mg, mw));
        res.push_back(compare("t2_f16copy.out", og, ow));
        cudaFree(dA);
        cudaFree(dB1);
        cudaFree(dB2);
        cudaFree(dM);
        cudaFree(dO);
    }

    // t3: A in [-8,8], B1 in [-4,4], C1 = small integer + 1/3 (its dropped TF32 bits never tie, and in about half the
    // binades rounding and truncation disagree), B2 in [-3,3].
    {
        std::vector<float> A(16 * 8), C1(16 * 8), B1t(64), B2t(64);
        for (auto& x : A) x = (float)rnd(-8, 8);
        for (auto& x : C1) x = (float)rnd(-4, 4) + 1.f / 3.f;
        for (auto& x : B1t) x = (float)rnd(-4, 4);
        for (auto& x : B2t) x = (float)rnd(-3, 3);
        float *dA = dev(A), *dC = dev(C1), *dB1 = dev(B1t), *dB2 = dev(B2t), *dM, *dO;
        CK(cudaMalloc(&dM, 128 * sizeof(float)));
        CK(cudaMalloc(&dO, 128 * sizeof(float)));
        t3_tf32copy<<<1, 32>>>(dA, dC, dB1, dB2, dM, dO);
        CK(cudaDeviceSynchronize());
        auto M = back(dM, 128);
        auto O = back(dO, 128);
        std::vector<double> mg(M.begin(), M.end()), mw(128), og(O.begin(), O.end()), trunc(128), rne(128);
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 8; ++j) {
                double s = C1[i * 8 + j];
                for (int k = 0; k < 8; ++k) s += (double)A[i * 8 + k] * B1t[j * 8 + k];
                mw[i * 8 + j] = (double)(float)s;
            }
        auto tf32_trunc = [](float x) {
            uint32_t u;
            memcpy(&u, &x, 4);
            u &= 0xffffe000u;
            memcpy(&x, &u, 4);
            return x;
        };
        auto tf32_rne = [](float x) {
            uint32_t u;
            memcpy(&u, &x, 4);
            u = (u + 0xfffu + ((u >> 13) & 1u)) & 0xffffe000u;
            memcpy(&x, &u, 4);
            return x;
        };
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 8; ++j) {
                double st = 0, sr = 0;
                for (int k = 0; k < 8; ++k) {
                    const float dv = M[d_index_for_a(i, k, 8, 1)];
                    st += (double)tf32_trunc(dv) * B2t[j * 8 + k];
                    sr += (double)tf32_rne(dv) * B2t[j * 8 + k];
                }
                trunc[i * 8 + j] = st;
                rne[i * 8 + j] = sr;
            }
        res.push_back(compare("t3_tf32copy.mid", mg, mw));
        res.push_back(compare("t3_tf32copy.out_vs_truncate", og, trunc));
        res.push_back(compare("t3_tf32copy.out_vs_round", og, rne));
        res.push_back(compare("t3_tf32copy.truncate_vs_round(inexact=discriminating)", trunc, rne));
        cudaFree(dA);
        cudaFree(dC);
        cudaFree(dB1);
        cudaFree(dB2);
        cudaFree(dM);
        cudaFree(dO);
    }

    // t4: A in [-2,2], B1 in [-1,1] (|D| <= 64, signs mixed), B2 in [-3,3].
    {
        std::vector<uint8_t> A(16 * 32), B1t(256), B2t(256);
        for (auto& x : A) x = (uint8_t)(int8_t)rnd(-2, 2);
        for (auto& x : B1t) x = (uint8_t)(int8_t)rnd(-1, 1);
        for (auto& x : B2t) x = (uint8_t)(int8_t)rnd(-3, 3);
        uint8_t *dA = dev(A), *dB1 = dev(B1t), *dB2 = dev(B2t);
        int *dM, *dO;
        CK(cudaMalloc(&dM, 128 * sizeof(int)));
        CK(cudaMalloc(&dO, 128 * sizeof(int)));
        t4_s8copy<<<1, 32>>>(dA, dB1, dB2, dM, dO);
        CK(cudaDeviceSynchronize());
        auto M = back(dM, 128);
        auto O = back(dO, 128);
        std::vector<double> mg(M.begin(), M.end()), mw(128), og(O.begin(), O.end()), ow(128);
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 8; ++j) {
                long s = 0;
                for (int k = 0; k < 32; ++k) s += (long)(int8_t)A[i * 32 + k] * (int8_t)B1t[j * 32 + k];
                mw[i * 8 + j] = (double)s;
            }
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 8; ++j) {
                long s = 0;
                for (int k = 0; k < 32; ++k) {
                    const uint32_t w = (uint32_t)M[d_index_for_a(i, k, 32, 4)];
                    s += (long)(int8_t)(w >> (8 * (k % 4))) * (int8_t)B2t[j * 32 + k];
                }
                ow[i * 8 + j] = (double)s;
            }
        res.push_back(compare("t4_s8copy.mid", mg, mw));
        res.push_back(compare("t4_s8copy.out", og, ow));
        cudaFree(dA);
        cudaFree(dB1);
        cudaFree(dB2);
        cudaFree(dM);
        cudaFree(dO);
    }

    // t5: A in [-2,2], B1 in [-1,1] (|D| <= 64: no E4M3 NaN code among the float's bytes), B2 in [-1,1].
    {
        std::vector<uint8_t> A(16 * 32), B1t(256), B2t(256);
        std::vector<float> Av(16 * 32), B1v(256), B2v(256);
        for (int i = 0; i < 16 * 32; ++i) A[i] = e4m3_encode_exact(Av[i] = (float)rnd(-2, 2));
        for (int i = 0; i < 256; ++i) B1t[i] = e4m3_encode_exact(B1v[i] = (float)rnd(-1, 1));
        for (int i = 0; i < 256; ++i) B2t[i] = e4m3_encode_exact(B2v[i] = (float)rnd(-1, 1));
        uint8_t *dA = dev(A), *dB1 = dev(B1t), *dB2 = dev(B2t);
        float *dM, *dO;
        CK(cudaMalloc(&dM, 128 * sizeof(float)));
        CK(cudaMalloc(&dO, 128 * sizeof(float)));
        t5_rawbits<<<1, 32>>>(dA, dB1, dB2, dM, dO);
        CK(cudaDeviceSynchronize());
        auto M = back(dM, 128);
        auto O = back(dO, 128);
        std::vector<double> mg(M.begin(), M.end()), mw(128), og(O.begin(), O.end()), ow(128);
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 8; ++j) {
                double s = 0;
                for (int k = 0; k < 32; ++k) s += (double)Av[i * 32 + k] * B1v[j * 32 + k];
                mw[i * 8 + j] = s;
            }
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 8; ++j) {
                double s = 0;
                for (int k = 0; k < 32; ++k) {
                    uint32_t w;
                    memcpy(&w, &M[d_index_for_a(i, k, 32, 4)], 4);
                    s += (double)e4m3_decode((uint8_t)(w >> (8 * (k % 4)))) * B2v[j * 32 + k];
                }
                ow[i * 8 + j] = s;
            }
        res.push_back(compare("t5_rawbits.mid", mg, mw));
        res.push_back(compare("t5_rawbits.out", og, ow));
        cudaFree(dA);
        cudaFree(dB1);
        cudaFree(dB2);
        cudaFree(dM);
        cudaFree(dO);
    }

    // t6: A in [-4,4], B1 in [-2,2] (|D| <= 256: integers this small are exact in BF16, with a zero low half), B2 in
    // [-3,3].
    {
        std::vector<uint8_t> A(16 * 32), B1t(256);
        std::vector<float> Av(16 * 32), B1v(256), B2v(8 * 16);
        std::vector<uint16_t> B2t(8 * 16);
        for (int i = 0; i < 16 * 32; ++i) A[i] = e4m3_encode_exact(Av[i] = (float)rnd(-4, 4));
        for (int i = 0; i < 256; ++i) B1t[i] = e4m3_encode_exact(B1v[i] = (float)rnd(-2, 2));
        for (int i = 0; i < 8 * 16; ++i) {
            const float v = B2v[i] = (float)rnd(-3, 3);
            uint32_t u;
            memcpy(&u, &v, 4);
            B2t[i] = (uint16_t)(u >> 16);
        }
        uint8_t *dA = dev(A), *dB1 = dev(B1t);
        uint16_t* dB2 = dev(B2t);
        float *dM, *dO;
        CK(cudaMalloc(&dM, 128 * sizeof(float)));
        CK(cudaMalloc(&dO, 128 * sizeof(float)));
        t6_bf16copy<<<1, 32>>>(dA, dB1, dB2, dM, dO);
        CK(cudaDeviceSynchronize());
        auto M = back(dM, 128);
        auto O = back(dO, 128);
        std::vector<double> mg(M.begin(), M.end()), mw(128), og(O.begin(), O.end()), ow(128), copy(128);
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 8; ++j) {
                double s = 0;
                for (int k = 0; k < 32; ++k) s += (double)Av[i * 32 + k] * B1v[j * 32 + k];
                mw[i * 8 + j] = s;
            }
        auto bf16 = [](uint32_t h) {
            const uint32_t u = h << 16;
            float f;
            memcpy(&f, &u, 4);
            return (double)f;
        };
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 8; ++j) {
                double s = 0;
                for (int k = 0; k < 16; ++k) {
                    uint32_t w;
                    memcpy(&w, &M[d_index_for_a(i, k, 16, 2)], 4);
                    s += bf16(k % 2 ? w >> 16 : w & 0xffffu) * B2v[j * 16 + k];
                }
                ow[i * 8 + j] = s;
            }
        res.push_back(compare("t6_bf16copy.mid", mg, mw));
        res.push_back(compare("t6_bf16copy.out", og, ow));
        cudaFree(dA);
        cudaFree(dB1);
        cudaFree(dB2);
        cudaFree(dM);
        cudaFree(dO);
    }

    // t7: V with full 24-bit significands in [1, 1024); D1 must be D0 - trunc_tf32(D0) exactly, and trunc + D1 = D0.
    {
        std::vector<float> V(128), B2t(64, 0.f);
        for (auto& x : V) {
            rng = rng * 1664525u + 1013904223u;
            const uint32_t u = (uint32_t)(127 + rnd(0, 9)) << 23 | (rng & 0x7fffffu);
            memcpy(&x, &u, 4);
        }
        for (int k = 0; k < 8; ++k) B2t[(k < 4 ? 2 * k : 2 * (k - 4) + 1) * 8 + k] = -1.f;
        float *dV = dev(V), *dB2 = dev(B2t), *d0, *d1;
        CK(cudaMalloc(&d0, 128 * sizeof(float)));
        CK(cudaMalloc(&d1, 128 * sizeof(float)));
        t7_split<<<1, 32>>>(dV, dB2, d0, d1);
        CK(cudaDeviceSynchronize());
        auto D0 = back(d0, 128);
        auto D1 = back(d1, 128);
        std::vector<double> g0(D0.begin(), D0.end()), w0(V.begin(), V.end()), g1(D1.begin(), D1.end()), w1(128),
            sum(128);
        for (int i = 0; i < 128; ++i) {
            uint32_t u;
            memcpy(&u, &D0[i], 4);
            u &= 0xffffe000u;
            float hi;
            memcpy(&hi, &u, 4);
            w1[i] = (double)D0[i] - (double)hi;
            sum[i] = (double)hi + (double)D1[i];
        }
        res.push_back(compare("t7_split.d0_is_v", g0, w0));
        res.push_back(compare("t7_split.low_digit", g1, w1));
        res.push_back(compare("t7_split.hi_plus_low_is_v", sum, w0));
        cudaFree(dV);
        cudaFree(dB2);
        cudaFree(d0);
        cudaFree(d1);
    }

    // t8: V as t7 but with mantissa bits 14..7 never all set; D1 and D2 must be the low digits exactly, D2 exact in BF16,
    // and hi(D0) + hi(D1) + D2 = V.  t8b: the same with bits 14..7 all set, so every low half is a BF16 NaN code.
    for (int poison = 0; poison < 2; ++poison) {
        std::vector<float> V(128);
        for (auto& x : V) {
            rng = rng * 1664525u + 1013904223u;
            uint32_t m = rng & 0x7fffffu;
            if (poison) m |= 0x7f80u;
            else if ((m & 0x7f80u) == 0x7f80u) m &= ~0x80u;
            const uint32_t u = (uint32_t)(127 + rnd(0, 9)) << 23 | m;
            memcpy(&x, &u, 4);
        }
        std::vector<uint16_t> B2t(8 * 16, 0);
        for (int n = 0; n < 8; ++n) B2t[n * 16 + (n % 2 ? n + 8 : n + 1)] = 0xbf80u;
        float* dV = dev(V);
        uint16_t* dB2 = dev(B2t);
        float *d1, *d2;
        CK(cudaMalloc(&d1, 128 * sizeof(float)));
        CK(cudaMalloc(&d2, 128 * sizeof(float)));
        t8_split_bf16<<<1, 32>>>(dV, dB2, d1, d2);
        CK(cudaDeviceSynchronize());
        auto D1 = back(d1, 128);
        auto D2 = back(d2, 128);
        auto hi = [](float x) {
            uint32_t u;
            memcpy(&u, &x, 4);
            u &= 0xffff0000u;
            memcpy(&x, &u, 4);
            return x;
        };
        std::vector<double> g1(D1.begin(), D1.end()), w1(128), g2(D2.begin(), D2.end()), w2(128), sum(128), v(128),
            low16(128, 0.0), zero(128, 0.0), isnan1(128), all1(128, 1.0);
        for (int i = 0; i < 128; ++i) {
            const float l1 = V[i] - hi(V[i]);
            w1[i] = l1;
            w2[i] = (double)l1 - (double)hi(l1);
            sum[i] = (double)hi(V[i]) + (double)hi(D1[i]) + (double)D2[i];
            v[i] = V[i];
            uint32_t u;
            memcpy(&u, &D2[i], 4);
            low16[i] = (double)(u & 0xffffu);
            isnan1[i] = std::isnan(D1[i]) ? 1.0 : 0.0;
        }
        if (!poison) {
            res.push_back(compare("t8_split_bf16.d1_low_digit", g1, w1));
            res.push_back(compare("t8_split_bf16.d2_low_digit", g2, w2));
            res.push_back(compare("t8_split_bf16.d2_exact_in_bf16", low16, zero));
            res.push_back(compare("t8_split_bf16.three_digits_sum_to_v", sum, v));
        } else {
            res.push_back(compare("t8b_nan_low_half.d1_all_nan", isnan1, all1));
        }
        cudaFree(dV);
        cudaFree(dB2);
        cudaFree(d1);
        cudaFree(d2);
    }

    char uuid[64];
    for (int i = 0; i < 16; ++i) snprintf(uuid + 2 * i, 3, "%02x", (unsigned char)prop.uuid.bytes[i]);
    printf("{\"device\": \"%s\", \"cc\": \"%d.%d\", \"uuid\": \"%s\", \"results\": [\n", prop.name, prop.major, prop.minor,
           uuid);
    for (size_t i = 0; i < res.size(); ++i)
        printf("  {\"name\": \"%s\", \"exact\": %s, \"max_abs_err\": %.17g, \"n\": %d}%s\n", res[i].name,
               res[i].exact ? "true" : "false", res[i].max_err, res[i].n, i + 1 < res.size() ? "," : "");
    printf("]}\n");
    return 0;
}
