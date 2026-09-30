// Instruction-level W1 price microbenchmarks for sm_120 (driven by w1.py): issue rates per SM per clock of the tensor-core atoms
// and the scalar work Pearl-C prices against them.  One block of 16 warps per SM (a dynamic shared-memory request keeps a second
// block off the SM), clock64() around the loop per block, %smid recorded; every op's loop body holds exactly UNROLL x NACC of its
// instruction (w1.py's SASS gate) and its accumulators are written out, so nothing is dead.
//
//   op  kernel                 instruction                                                                     unit per instruction
//    0  qmma_e4m3_f32_rate     mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32                 4096 MAC
//    1  qmma_e4m3_f16_rate     mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16                 4096 MAC
//    2  qmma_mxf8_rate         mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X ... ue8m0  4096 MAC
//    3  omma_nvf4_rate         mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 ... ue4m3      8192 MAC
//    4  hmma_bf16_f32_rate     mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32                              2048 MAC
//    5  hmma_f16_f16_rate      mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16                                2048 MAC
//    6  fadd_rate              add.rn.f32                                                                         32 lanes
//    7  ffma_rate              fma.rn.f32                                                                         32 lanes
//    8  f2fp_rate              cvt.rn.satfinite.e4m3x2.f32 (+ one 3-input add per four converts)                64 elements
//    9  f2fp_u16_rate          the same, each result zero-extended (+ one 3-input add per two converts)            64 elements
//   10  fadd_f32x2_rate        add.rn.f32x2 (PTX 8.6; sm_120a lowers each to two FADD, sm_100a to one FADD2)       64 words
//   11  fadd_pairs_rate        two independent C-level f32 adds per chain pair (does ptxas pair them on its own?)   32 words
//   12  f2fp_e2m1_rate         cvt.rn.satfinite.e2m1x2.f32, four to a word (ptxas chains them through MERGE_C)     64 elements
//   13  f2fp_e2m1_u8_rate      the same, each result zero-extended (+ one LOP3 and half a 3-input add per convert)  64 elements
//   14  f2fp_e2m1_x2_rate      op 12 with 32 constants: twice the converts over the same per-iteration overhead     64 elements
//   15  qmma_sp_e4m3_rate      mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32  8192 logical MAC
//                              (4096 nonzero products: 2:4 along k)
//   16  cs_port_fadd_rate      the port's E4M3 cast-and-store: scalar convert with a zero partner, LOP3 + IMAD, STS.U16   32 codes
//                              (+ one FADD per code feeding it)
//   17  cs_port_nogen_rate     the same instruction mix with a live partner in place of zero, no per-code input          32 codes
//   18  cs_packed_fadd_rate    cvt.rn.satfinite.e4m3x2.f32 on two live floats, four codes to a word, STS.32 (+ FADD)      64 codes
//   19  cs_packed_nogen_rate   the same with no per-code input                                                           64 codes
//
// Build: nvcc -O2 -gencode=arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int THREADS = 512;
constexpr int NACC = 4;    // independent MMA accumulators per warp
constexpr int UNROLL = 4;  // loop-body repeats
constexpr int NCHAIN = 8;  // independent scalar chains per thread
constexpr int SUNROLL = 8;
constexpr int CHK = 16;    // chk words per thread, whatever the op writes (the host reads (blocks, THREADS, CHK))

__device__ __forceinline__ unsigned smid() {
    unsigned r;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void mma_e4m3_f32(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_e4m3_f16(uint32_t c[2], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_mxf8(float c[4], const uint32_t a[4], const uint32_t b[2], uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
}

__device__ __forceinline__ void mma_nvf4(float c[4], const uint32_t a[4], const uint32_t b[2], uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
}

// B's four registers are b0, b1 and two byte permutations of them (the row has two B words; repeating a register in the quad
// puts MOVs in the loop); e is the metadata register (row word 6).
__device__ __forceinline__ void mma_sp_e4m3(float c[4], const uint32_t a[4], const uint32_t b[2], uint32_t e) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(__byte_perm(b[0], b[1], 0x5140)),
          "r"(__byte_perm(b[0], b[1], 0x7362)), "r"(e));
}

__device__ __forceinline__ void mma_bf16_f32(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_f16_f16(uint32_t c[2], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Each lane's operand registers come from its own row of in[32][8] (words 0..5: a0..a3, b0, b1; word 6: scale A; word 7: scale
// B), so the host chooses the data (all-ones for the numeric gate, random codes for timing) and no value is warp-uniform: uniform
// operands would let ptxas move scalar work onto the uniform datapath (UF2FP, UFADD), which is not the cost being priced.
#define RATE_PROLOGUE                                                                  \
    extern __shared__ unsigned char smem_pad[];                                        \
    __shared__ long long t0;                                                           \
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;                             \
    const uint32_t* row = in + (threadIdx.x & 31) * 8;                                 \
    if (threadIdx.x == 0) smem_pad[0] = 0;                                             \
    __syncthreads();                                                                   \
    if (threadIdx.x == 0) t0 = clock64();                                              \
    __syncthreads();

#define RATE_EPILOGUE                                                                  \
    __syncthreads();                                                                   \
    if (threadIdx.x == 0) {                                                            \
        cycles[blockIdx.x] = clock64() - t0;                                           \
        sm[blockIdx.x] = (int)smid();                                                  \
    }

#define MMA_OPERANDS                                                                   \
    uint32_t a[4], b[2];                                                               \
    for (int i = 0; i < 4; i++) a[i] = row[i];                                         \
    for (int i = 0; i < 2; i++) b[i] = row[4 + i];                                     \
    const uint32_t sfa = row[6], sfb = row[7];                                         \
    (void)sfa; (void)sfb;

#define MMA_F32_RATE(KNAME, CALL)                                                                                     \
    __global__ void __launch_bounds__(THREADS) KNAME(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk,      \
                                                     long long* __restrict__ cycles, int* __restrict__ sm, int iters) { \
        RATE_PROLOGUE                                                                                                 \
        MMA_OPERANDS                                                                                                  \
        float c[NACC][4] = {};                                                                                        \
        _Pragma("unroll 1") for (int it = 0; it < iters; it++) {                                                      \
            _Pragma("unroll") for (int u = 0; u < UNROLL; u++) {                                                      \
                _Pragma("unroll") for (int j = 0; j < NACC; j++) CALL;                                                \
            }                                                                                                         \
        }                                                                                                             \
        RATE_EPILOGUE                                                                                                 \
        for (int j = 0; j < NACC; j++)                                                                                \
            for (int i = 0; i < 4; i++) chk[(size_t)tid * CHK + j * 4 + i] = __float_as_uint(c[j][i]);                 \
    }

#define MMA_F16_RATE(KNAME, CALL)                                                                                     \
    __global__ void __launch_bounds__(THREADS) KNAME(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk,      \
                                                     long long* __restrict__ cycles, int* __restrict__ sm, int iters) { \
        RATE_PROLOGUE                                                                                                 \
        MMA_OPERANDS                                                                                                  \
        uint32_t c[NACC][2] = {};                                                                                     \
        _Pragma("unroll 1") for (int it = 0; it < iters; it++) {                                                      \
            _Pragma("unroll") for (int u = 0; u < UNROLL; u++) {                                                      \
                _Pragma("unroll") for (int j = 0; j < NACC; j++) CALL;                                                \
            }                                                                                                         \
        }                                                                                                             \
        RATE_EPILOGUE                                                                                                 \
        for (int j = 0; j < NACC; j++)                                                                                \
            for (int i = 0; i < 4; i++) chk[(size_t)tid * CHK + j * 4 + i] = i < 2 ? c[j][i] : 0u;                       \
    }

MMA_F32_RATE(qmma_e4m3_f32_rate, mma_e4m3_f32(c[j], a, b))
MMA_F16_RATE(qmma_e4m3_f16_rate, mma_e4m3_f16(c[j], a, b))
MMA_F32_RATE(qmma_mxf8_rate, mma_mxf8(c[j], a, b, sfa, sfb))
MMA_F32_RATE(omma_nvf4_rate, mma_nvf4(c[j], a, b, sfa, sfb))
MMA_F32_RATE(hmma_bf16_f32_rate, mma_bf16_f32(c[j], a, b))
MMA_F16_RATE(hmma_f16_f16_rate, mma_f16_f16(c[j], a, b))
MMA_F32_RATE(qmma_sp_e4m3_rate, mma_sp_e4m3(c[j], a, b, sfa))

// Scalar rates: NCHAIN independent chains per thread; a lane's row word 0 is its chains' start, word 1 the addend y, word 2 the
// multiplier m.
#define SCALAR_RATE(KNAME, BODY)                                                                                      \
    __global__ void __launch_bounds__(THREADS) KNAME(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk,      \
                                                     long long* __restrict__ cycles, int* __restrict__ sm, int iters) { \
        RATE_PROLOGUE                                                                                                 \
        float x[NCHAIN];                                                                                              \
        for (int j = 0; j < NCHAIN; j++) x[j] = __uint_as_float(row[0]);                                              \
        const float y = __uint_as_float(row[1]), m = __uint_as_float(row[2]);                                         \
        (void)m;                                                                                                      \
        _Pragma("unroll 1") for (int it = 0; it < iters; it++) {                                                      \
            _Pragma("unroll") for (int u = 0; u < SUNROLL; u++) {                                                     \
                _Pragma("unroll") for (int j = 0; j < NCHAIN; j++) BODY;                                              \
            }                                                                                                         \
        }                                                                                                             \
        RATE_EPILOGUE                                                                                                 \
        for (int j = 0; j < NCHAIN; j++) chk[(size_t)tid * CHK + j] = __float_as_uint(x[j]);                          \
    }

SCALAR_RATE(fadd_rate, asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(x[j]) : "f"(y)))
SCALAR_RATE(ffma_rate, asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x[j]) : "f"(m), "f"(y)))

// Four E4M3 codes in one word, as an FP8 quantizer packs them: P(a, b) = code(a) << 8 | code(b) in the low half, P(c, d) in the high.
__device__ __forceinline__ uint32_t cvt_e4m3x4(float a, float b, float c, float d) {
    uint32_t r;
    asm volatile("{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, %1, %2; cvt.rn.satfinite.e4m3x2.f32 hi, %3, %4; mov.b32 %0, {lo, hi}; }"
                 : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
    return r;
}

// F2FP: each convert pairs one of the iteration's two values f, g (two integer adds per iteration, so nothing can be hoisted) with
// one of 16 per-lane constants, in both orders: 64 distinct converts per iteration, packed four to a word, and two words folded into
// a chain's word by one 3-input add (an XOR fold of both orders keeps only hi ^ lo, and f and g cancel out of it).  Row word 0 = f's
// start word (g = f + 2^21), word 1 = the constants' base word (constant k = base + k * 2^20): the constants sit one E4M3 ulp apart
// and g two from f, so the gate's check words cover 18 distinct codes per lane.
__global__ void __launch_bounds__(THREADS) f2fp_rate(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk, long long* __restrict__ cycles,
                                                     int* __restrict__ sm, int iters) {
    RATE_PROLOGUE
    uint32_t h[NCHAIN] = {};
    float xs[2 * NCHAIN];
    for (int k = 0; k < 2 * NCHAIN; k++) xs[k] = __uint_as_float(row[1] + ((uint32_t)k << 20));
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
        const float f = __uint_as_float(row[0] + (uint32_t)it);
        const float g = __uint_as_float(row[0] + (uint32_t)it + (1u << 21));
#pragma unroll
        for (int k = 0; k < 2 * NCHAIN; k++) h[k % NCHAIN] += cvt_e4m3x4(f, xs[k], xs[k], f) + cvt_e4m3x4(g, xs[k], xs[k], g);
    }
    RATE_EPILOGUE
    for (int j = 0; j < NCHAIN; j++) chk[(size_t)tid * CHK + j] = h[j];
}

__device__ __forceinline__ uint32_t cvt_e4m3x2_u32(float hi, float lo) {
    uint32_t r;
    asm volatile("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %1, %2; cvt.u32.u16 %0, t; }" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// The same 64 converts per iteration, each zero-extended to its own word and folded two at a time: a second mix of support
// instructions per F2FP, so two loops pin F2FP's own cost apart from theirs.
__global__ void __launch_bounds__(THREADS) f2fp_u16_rate(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk,
                                                         long long* __restrict__ cycles, int* __restrict__ sm, int iters) {
    RATE_PROLOGUE
    uint32_t h[NCHAIN] = {};
    float xs[2 * NCHAIN];
    for (int k = 0; k < 2 * NCHAIN; k++) xs[k] = __uint_as_float(row[1] + ((uint32_t)k << 20));
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
        const float f = __uint_as_float(row[0] + (uint32_t)it);
        const float g = __uint_as_float(row[0] + (uint32_t)it + (1u << 21));
#pragma unroll
        for (int k = 0; k < 2 * NCHAIN; k++) {
            h[k % NCHAIN] += cvt_e4m3x2_u32(f, xs[k]) + cvt_e4m3x2_u32(xs[k], f);
            h[(k + 1) % NCHAIN] += cvt_e4m3x2_u32(g, xs[k]) + cvt_e4m3x2_u32(xs[k], g);
        }
    }
    RATE_EPILOGUE
    for (int j = 0; j < NCHAIN; j++) chk[(size_t)tid * CHK + j] = h[j];
}

// Packed FP32 pairs: NCHAIN chains of two words each, a lane's row words 0 and 1 the chains' low and high starts, words 2 and 3 the
// low and high addends, so both halves are checked and a swap shows.
__global__ void __launch_bounds__(THREADS) fadd_f32x2_rate(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk,
                                                           long long* __restrict__ cycles, int* __restrict__ sm, int iters) {
    RATE_PROLOGUE
    unsigned long long x[NCHAIN];
    const unsigned long long x0 = (unsigned long long)row[1] << 32 | row[0], y = (unsigned long long)row[3] << 32 | row[2];
    for (int j = 0; j < NCHAIN; j++) x[j] = x0;
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int u = 0; u < SUNROLL; u++) {
#pragma unroll
            for (int j = 0; j < NCHAIN; j++) asm volatile("add.rn.f32x2 %0, %0, %1;" : "+l"(x[j]) : "l"(y));
        }
    }
    RATE_EPILOGUE
    for (int j = 0; j < NCHAIN; j++) {
        chk[(size_t)tid * CHK + 2 * j] = (uint32_t)x[j];
        chk[(size_t)tid * CHK + 2 * j + 1] = (uint32_t)(x[j] >> 32);
    }
}

// The same pairs as plain C adds, the code anyone writes: whatever ptxas makes of them is what they cost.  Chain j starts j ulps
// above the row's start, so no two chains are the same computation and none can be merged (the gate's start is 0, and its first add
// of 1 or 2 rounds the offset away).
__global__ void __launch_bounds__(THREADS) fadd_pairs_rate(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk,
                                                           long long* __restrict__ cycles, int* __restrict__ sm, int iters) {
    RATE_PROLOGUE
    float x[2 * NCHAIN];
    const float y0 = __uint_as_float(row[2]), y1 = __uint_as_float(row[3]);
    for (int j = 0; j < NCHAIN; j++) x[2 * j] = __uint_as_float(row[0] + j), x[2 * j + 1] = __uint_as_float(row[1] + j);
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int u = 0; u < SUNROLL; u++) {
#pragma unroll
            for (int j = 0; j < NCHAIN; j++) {
                x[2 * j] = x[2 * j] + y0;
                x[2 * j + 1] = x[2 * j + 1] + y1;
            }
        }
    }
    RATE_EPILOGUE
    for (int j = 0; j < 2 * NCHAIN; j++) chk[(size_t)tid * CHK + j] = __float_as_uint(x[j]);
}

// Four E2M1 pairs in one word, as an FP4 quantizer packs them: byte i = code(a_i) << 4 | code(b_i).
__device__ __forceinline__ uint32_t cvt_e2m1x8(float a0, float b0, float a1, float b1, float a2, float b2, float a3, float b3) {
    uint32_t r;
    asm volatile("{ .reg .b8 q0, q1, q2, q3; cvt.rn.satfinite.e2m1x2.f32 q0, %1, %2; cvt.rn.satfinite.e2m1x2.f32 q1, %3, %4;"
                 " cvt.rn.satfinite.e2m1x2.f32 q2, %5, %6; cvt.rn.satfinite.e2m1x2.f32 q3, %7, %8; mov.b32 %0, {q0, q1, q2, q3}; }"
                 : "=r"(r) : "f"(a0), "f"(b0), "f"(a1), "f"(b1), "f"(a2), "f"(b2), "f"(a3), "f"(b3));
    return r;
}

__device__ __forceinline__ uint32_t cvt_e2m1x2_u32(float hi, float lo) {
    uint32_t r;
    asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t; }" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// E2M1 casts: three loops whose F2FP, support and per-iteration counts differ, so w1.py's split solves for F2FP's own cost.  Per
// iteration each of NK per-lane constants x_k meets the iteration's two values f, g in both orders.  Row word 0 = f's start word
// (g = f + 2^21), word 1 = the constants' base word (x_k = base + k * 2^25 / NK: NK points over four binades of the grid).
template <int NK>
__device__ __forceinline__ void e2m1_consts(const uint32_t* row, float* xs) {
#pragma unroll
    for (int k = 0; k < NK; k++) xs[k] = __uint_as_float(row[1] + (uint32_t)k * ((1u << 25) / NK));
}

template <int NK>
__device__ __forceinline__ void e2m1_packed(const uint32_t* row, uint32_t* h, int iters) {
    float xs[NK];
    e2m1_consts<NK>(row, xs);
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
        const float f = __uint_as_float(row[0] + (uint32_t)it);
        const float g = __uint_as_float(row[0] + (uint32_t)it + (1u << 21));
#pragma unroll
        for (int k = 0; k < NK; k++) h[k % NCHAIN] += cvt_e2m1x8(f, xs[k], xs[k], f, g, xs[k], xs[k], g);
    }
}

__global__ void __launch_bounds__(THREADS) f2fp_e2m1_rate(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk,
                                                          long long* __restrict__ cycles, int* __restrict__ sm, int iters) {
    RATE_PROLOGUE
    uint32_t h[NCHAIN] = {};
    e2m1_packed<2 * NCHAIN>(row, h, iters);
    RATE_EPILOGUE
    for (int j = 0; j < NCHAIN; j++) chk[(size_t)tid * CHK + j] = h[j];
}

__global__ void __launch_bounds__(THREADS) f2fp_e2m1_u8_rate(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk,
                                                             long long* __restrict__ cycles, int* __restrict__ sm, int iters) {
    RATE_PROLOGUE
    uint32_t h[NCHAIN] = {};
    float xs[2 * NCHAIN];
    e2m1_consts<2 * NCHAIN>(row, xs);
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
        const float f = __uint_as_float(row[0] + (uint32_t)it);
        const float g = __uint_as_float(row[0] + (uint32_t)it + (1u << 21));
#pragma unroll
        for (int k = 0; k < 2 * NCHAIN; k++) {
            h[k % NCHAIN] += cvt_e2m1x2_u32(f, xs[k]) + cvt_e2m1x2_u32(xs[k], f);
            h[(k + 1) % NCHAIN] += cvt_e2m1x2_u32(g, xs[k]) + cvt_e2m1x2_u32(xs[k], g);
        }
    }
    RATE_EPILOGUE
    for (int j = 0; j < NCHAIN; j++) chk[(size_t)tid * CHK + j] = h[j];
}

__global__ void __launch_bounds__(THREADS) f2fp_e2m1_x2_rate(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk,
                                                             long long* __restrict__ cycles, int* __restrict__ sm, int iters) {
    RATE_PROLOGUE
    uint32_t h[NCHAIN] = {};
    e2m1_packed<4 * NCHAIN>(row, h, iters);
    RATE_EPILOGUE
    for (int j = 0; j < NCHAIN; j++) chk[(size_t)tid * CHK + j] = h[j];
}

// Cast-and-store loops (ops 16-19).  One iteration is one warp of form_s5 (pearl_c_sm120.cu at 952e3ecc) casting its 16 x 128
// tile, 64 codes per thread, and staging it in shared memory at the port's addresses: rows g and g + 8 of the warp's 16, FORM_ROW
// bytes a row, so a quad's stores hit distinct banks.  The stores are volatile, so every iteration's stores stay in the loop, and
// after timing each thread reads its own words back into chk.  Ops 16 and 18 feed the converts from 64 FADD chains (x_j += y, one
// FADD per code, as the port has one FMUL per code); ops 17 and 19 need no per-code input: each convert pairs one of the
// iteration's two values f, g (two integer adds per iteration) with a per-lane constant.  Row word 0 = the chains' base (16, 18)
// or f's start word (17, 19; g = f + 2^21); word 1 = y (16, 18) or the constants' base (17, 19).  Value j = base + j * 2^20 with
// the sign flipped on odd j: 64 values over eight binades per lane.
constexpr int FORM_ROW = 144;
constexpr int NCS = 64;

template <int N>
__device__ __forceinline__ void cs_values(uint32_t base, float* v) {
#pragma unroll
    for (int j = 0; j < N; j++) v[j] = __uint_as_float((base + ((uint32_t)j << 20)) ^ ((uint32_t)(j & 1) << 31));
}

// The port's e4m3(): one code per convert, the other half converting a zero.
__device__ __forceinline__ uint8_t e4m3_port(float f) { return (uint8_t)__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3); }

// The same convert with a live high half, written as cuda_fp8.h writes it (a plain asm): the low byte is code(lo).
__device__ __forceinline__ uint16_t cvt_e4m3x2_h(float hi, float lo) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

// Byte offset of a thread's first staged 16-bit word (port) or 32-bit word (packed) in the block's shared memory.
__device__ __forceinline__ int cs_offset(int bytes_per_quad_lane) {
    const int lane = threadIdx.x & 31;
    return (threadIdx.x >> 5) * 16 * FORM_ROW + (lane >> 2) * FORM_ROW + bytes_per_quad_lane * (lane & 3);
}

// chk word nb = row g's word | row g + 8's << 16 (port), or words 2m, 2m + 1 = row g's and row g + 8's word m (packed).
__device__ __forceinline__ void cs_port_readback(const uint8_t* s0, uint32_t* out) {
#pragma unroll
    for (int nb = 0; nb < 16; nb++)
        out[nb] = *reinterpret_cast<const volatile uint16_t*>(s0 + 8 * nb) |
                  (uint32_t)*reinterpret_cast<const volatile uint16_t*>(s0 + 8 * FORM_ROW + 8 * nb) << 16;
}

__device__ __forceinline__ void cs_packed_readback(const uint8_t* p0, uint32_t* out) {
#pragma unroll
    for (int m = 0; m < 8; m++) {
        out[2 * m] = *reinterpret_cast<const volatile uint32_t*>(p0 + 16 * m);
        out[2 * m + 1] = *reinterpret_cast<const volatile uint32_t*>(p0 + 8 * FORM_ROW + 16 * m);
    }
}

// The port's cast-and-store: per atom four scalar converts, two codes to a 16-bit word (the port's expression), one STS.U16 each.
__global__ void __launch_bounds__(THREADS) cs_port_fadd_rate(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk,
                                                             long long* __restrict__ cycles, int* __restrict__ sm, int iters) {
    RATE_PROLOGUE
    uint8_t* s0 = smem_pad + cs_offset(2);
    uint8_t* s1 = s0 + 8 * FORM_ROW;
    float x[NCS];
    cs_values<NCS>(row[0], x);
    const float y = __uint_as_float(row[1]);
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int j = 0; j < NCS; j++) x[j] = __fadd_rn(x[j], y);
#pragma unroll
        for (int nb = 0; nb < 16; nb++) {
            *reinterpret_cast<volatile uint16_t*>(s0 + 8 * nb) = e4m3_port(x[4 * nb]) | (uint16_t)e4m3_port(x[4 * nb + 1]) << 8;
            *reinterpret_cast<volatile uint16_t*>(s1 + 8 * nb) = e4m3_port(x[4 * nb + 2]) | (uint16_t)e4m3_port(x[4 * nb + 3]) << 8;
        }
    }
    RATE_EPILOGUE
    cs_port_readback(s0, chk + (size_t)tid * CHK);
}

// The port's instruction mix with no per-code input: the converts' discarded high half is f or g instead of zero.
__global__ void __launch_bounds__(THREADS) cs_port_nogen_rate(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk,
                                                              long long* __restrict__ cycles, int* __restrict__ sm, int iters) {
    RATE_PROLOGUE
    uint8_t* s0 = smem_pad + cs_offset(2);
    uint8_t* s1 = s0 + 8 * FORM_ROW;
    float xs[NCS];
    cs_values<NCS>(row[1], xs);
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
        const float f = __uint_as_float(row[0] + (uint32_t)it);
        const float g = __uint_as_float(row[0] + (uint32_t)it + (1u << 21));
#pragma unroll
        for (int nb = 0; nb < 16; nb++) {
            *reinterpret_cast<volatile uint16_t*>(s0 + 8 * nb) =
                (uint8_t)cvt_e4m3x2_h(f, xs[4 * nb]) | (uint16_t)(uint8_t)cvt_e4m3x2_h(g, xs[4 * nb + 1]) << 8;
            *reinterpret_cast<volatile uint16_t*>(s1 + 8 * nb) =
                (uint8_t)cvt_e4m3x2_h(f, xs[4 * nb + 2]) | (uint16_t)(uint8_t)cvt_e4m3x2_h(g, xs[4 * nb + 3]) << 8;
        }
    }
    RATE_EPILOGUE
    cs_port_readback(s0, chk + (size_t)tid * CHK);
}

// Packed: two live values per convert, four codes to a 32-bit word (bytes x_0..x_3 in order), one STS.32 per word.  A word holds
// four adjacent columns, where the port's lane holds two, so its columns are a permutation of the port's within 16.
__global__ void __launch_bounds__(THREADS) cs_packed_fadd_rate(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk,
                                                               long long* __restrict__ cycles, int* __restrict__ sm, int iters) {
    RATE_PROLOGUE
    uint8_t* p0 = smem_pad + cs_offset(4);
    uint8_t* p1 = p0 + 8 * FORM_ROW;
    float x[NCS];
    cs_values<NCS>(row[0], x);
    const float y = __uint_as_float(row[1]);
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int j = 0; j < NCS; j++) x[j] = __fadd_rn(x[j], y);
#pragma unroll
        for (int m = 0; m < 8; m++) {
            *reinterpret_cast<volatile uint32_t*>(p0 + 16 * m) = cvt_e4m3x4(x[8 * m + 1], x[8 * m], x[8 * m + 3], x[8 * m + 2]);
            *reinterpret_cast<volatile uint32_t*>(p1 + 16 * m) = cvt_e4m3x4(x[8 * m + 5], x[8 * m + 4], x[8 * m + 7], x[8 * m + 6]);
        }
    }
    RATE_EPILOGUE
    cs_packed_readback(p0, chk + (size_t)tid * CHK);
}

// Packed with no per-code input: word = P(f, x_2i) | P(x_2i+1, g) << 16, P(hi, lo) = code(hi) << 8 | code(lo), i = 2m + h.
__global__ void __launch_bounds__(THREADS) cs_packed_nogen_rate(const uint32_t* __restrict__ in, uint32_t* __restrict__ chk,
                                                                long long* __restrict__ cycles, int* __restrict__ sm, int iters) {
    RATE_PROLOGUE
    uint8_t* p0 = smem_pad + cs_offset(4);
    uint8_t* p1 = p0 + 8 * FORM_ROW;
    float xs[NCS / 2];
    cs_values<NCS / 2>(row[1], xs);
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
        const float f = __uint_as_float(row[0] + (uint32_t)it);
        const float g = __uint_as_float(row[0] + (uint32_t)it + (1u << 21));
#pragma unroll
        for (int m = 0; m < 8; m++) {
            *reinterpret_cast<volatile uint32_t*>(p0 + 16 * m) = cvt_e4m3x4(f, xs[4 * m], xs[4 * m + 1], g);
            *reinterpret_cast<volatile uint32_t*>(p1 + 16 * m) = cvt_e4m3x4(f, xs[4 * m + 2], xs[4 * m + 3], g);
        }
    }
    RATE_EPILOGUE
    cs_packed_readback(p0, chk + (size_t)tid * CHK);
}

typedef void (*RateFn)(const uint32_t*, uint32_t*, long long*, int*, int);
const RateFn KERNELS[] = {qmma_e4m3_f32_rate, qmma_e4m3_f16_rate, qmma_mxf8_rate, omma_nvf4_rate, hmma_bf16_f32_rate, hmma_f16_f16_rate,
                          fadd_rate, ffma_rate, f2fp_rate, f2fp_u16_rate, fadd_f32x2_rate, fadd_pairs_rate,
                          f2fp_e2m1_rate, f2fp_e2m1_u8_rate, f2fp_e2m1_x2_rate, qmma_sp_e4m3_rate,
                          cs_port_fadd_rate, cs_port_nogen_rate, cs_packed_fadd_rate, cs_packed_nogen_rate};
constexpr int NOPS = sizeof(KERNELS) / sizeof(KERNELS[0]);

}  // namespace

extern "C" {

int w1_nops(void) { return NOPS; }
int w1_threads(void) { return THREADS; }

// One launch of op over `blocks` blocks: in u32[32*8] (each lane's operand words), chk u32[blocks*THREADS*16] (accumulators),
// cycles/sm per block, the event time in ms.  smem_bytes of dynamic shared memory per block keeps one block per SM.
int w1_run(int op, int blocks, int smem_bytes, int iters, const uint32_t* in, uint32_t* chk, long long* cycles, int* sm, float* ms) {
    if (op < 0 || op >= NOPS) return (int)cudaErrorInvalidValue;
    cudaError_t err = cudaFuncSetAttribute(KERNELS[op], cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) return (int)err;
    const size_t nchk = (size_t)blocks * THREADS * CHK;
    uint32_t *dIn = nullptr, *dChk = nullptr;
    long long* dCyc = nullptr;
    int* dSm = nullptr;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    if ((err = cudaMalloc(&dIn, 32 * 8 * 4)) == cudaSuccess && (err = cudaMalloc(&dChk, nchk * 4)) == cudaSuccess &&
        (err = cudaMalloc(&dCyc, blocks * sizeof(long long))) == cudaSuccess && (err = cudaMalloc(&dSm, blocks * sizeof(int))) == cudaSuccess) {
        cudaMemcpy(dIn, in, 32 * 8 * 4, cudaMemcpyHostToDevice);
        cudaMemset(dChk, 0, nchk * 4);
        cudaEventRecord(e0);
        KERNELS[op]<<<blocks, THREADS, smem_bytes>>>(dIn, dChk, dCyc, dSm, iters);
        cudaEventRecord(e1);
        err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaEventSynchronize(e1);
        if (err == cudaSuccess) {
            cudaEventElapsedTime(ms, e0, e1);
            cudaMemcpy(chk, dChk, nchk * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(cycles, dCyc, blocks * sizeof(long long), cudaMemcpyDeviceToHost);
            cudaMemcpy(sm, dSm, blocks * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
    cudaFree(dIn);
    cudaFree(dChk);
    cudaFree(dCyc);
    cudaFree(dSm);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return (int)err;
}

int w1_device(int* sms, int* smem_per_sm, int* clock_khz) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    *sms = prop.multiProcessorCount;
    *smem_per_sm = (int)prop.sharedMemPerMultiprocessor;
    *clock_khz = 0;
    cudaDeviceGetAttribute(clock_khz, cudaDevAttrClockRate, 0);
    return 0;
}

// mma_fp8.cu's device_info, so fp8.Device.info() (and with it fp8.identity_gate) reads this library too.
int device_info(int* major, int* minor, int* sms, char* name, int name_len, unsigned char* uuid) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    *major = prop.major;
    *minor = prop.minor;
    *sms = prop.multiProcessorCount;
    int i = 0;
    for (; i < name_len - 1 && prop.name[i]; ++i) name[i] = prop.name[i];
    name[i] = 0;
    for (int j = 0; j < 16; j++) uuid[j] = (unsigned char)prop.uuid.bytes[j];
    return 0;
}

}  // extern "C"
