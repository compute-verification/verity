// Emulating the sm_120 NVFP4 atom (NVFP4_SM120: BLACKWELL_SM120_NVF4) on the RTX PRO 6000's other paths, bit for bit, and the
// hot-accumulator tweak: each route's throughput, and its words for the host to check against the pinned model.
//
// Every route computes the same thing per warp: NACC independent chains of one 16 x 8 output tile over many atoms (k = 64 each),
// chain c reading its own register-resident operand set c every iteration, so a chain of L atoms is an NVFP4 chain of K = 64 L
// over one repeated operand set the host replays.  Each chain owns a set so no per-atom emulation work is shared between chains.
// Every route keeps the mma accumulator's lane layout: word o of lane (g = lane / 4, t = lane % 4) is row g + 8 (o >> 1),
// column 2 t + (o & 1).
//
//   route        per atom per warp                                                      reproduces the atom when
//   native       1 OMMA m16n8k64 kind::mxf4nvf4 4X ue4m3                                 always (it is the atom)
//   native_hot   the same, the chain starting from C = H                                 always
//   imma_exact   8 IMMA m16n8k16 s8 (4 group dots, 4 nonzero-product counts), then the  always: the atom's align-add on the
//                five-way align-add per word in int64 on the CUDA cores                  CUDA cores
//   imma_fast    4 IMMA k16, then acc += dot * (A factor * B factor) per word and block  the chain is exact (nothing truncates)
//                in int64; one RZ conversion at the end
//   imma_floor   imma_fast's arithmetic per atom, the atom's sum floored onto H's grid   the step-floor rule (CPU: 100%)
//   bf16_fold    4 HMMA m16n8k16 bf16 per atom on E2M1 x UE4M3 folded into BF16, chained the chain is exact
//   bf16_floor   4 HMMA per atom from +0, then floor onto H's grid (two FADDs) and add   the step-floor rule
//   dp4a         CUDA cores only: 64 IDP4A per lane per atom, then imma_fast's int64 step the chain is exact
//   lut          CUDA cores only: one shared-memory lookup per product (256-entry        the chain is exact
//                table), then imma_fast's int64 step
//
// Operands are register-resident for every route (no loads in the loop), so each rate is an upper bound on a GEMM built from
// that route.  The sets repeat, so a route whose per-atom work starts from zero (every route but native, native_hot and
// bf16_fold, which accumulate into the chain) would let the compiler merge identical work across chains and iterations;
// `asm volatile("" : "+r"(x))` on the operand words before each use forbids that and emits nothing.  Raw instruction peaks for
// the same harness: peak_omma, peak_imma16, peak_hmma16, peak_dp4a.
//
// Build: nvcc -O3 -gencode=arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared -o libfp4_emulation_sm_120a.so fp4_emulation.cu

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

enum Route {
    R_NATIVE = 0, R_NATIVE_HOT, R_IMMA_EXACT, R_IMMA_FAST, R_IMMA_FLOOR, R_BF16_FOLD, R_BF16_FLOOR, R_DP4A, R_LUT, R_COUNT
};
enum Peak { P_OMMA = 0, P_IMMA16, P_HMMA16, P_DP4A, P_COUNT };

template <int R> struct Shape;
// NSETS == NACC: chain c owns set c.
template <> struct Shape<R_NATIVE> { static constexpr int NSETS = 8, NACC = 8, WORDS = 8; };
template <> struct Shape<R_NATIVE_HOT> { static constexpr int NSETS = 8, NACC = 8, WORDS = 8; };
template <> struct Shape<R_IMMA_EXACT> { static constexpr int NSETS = 2, NACC = 2, WORDS = 32; };
template <> struct Shape<R_IMMA_FAST> { static constexpr int NSETS = 4, NACC = 4, WORDS = 28; };
template <> struct Shape<R_IMMA_FLOOR> { static constexpr int NSETS = 4, NACC = 4, WORDS = 28; };
template <> struct Shape<R_BF16_FOLD> { static constexpr int NSETS = 4, NACC = 4, WORDS = 24; };
template <> struct Shape<R_BF16_FLOOR> { static constexpr int NSETS = 4, NACC = 4, WORDS = 24; };
template <> struct Shape<R_DP4A> { static constexpr int NSETS = 2, NACC = 2, WORDS = 80; };
template <> struct Shape<R_LUT> { static constexpr int NSETS = 1, NACC = 1, WORDS = 48; };

// Per-lane constants, 8 words: E0 of the 4 words (the exponent of the int64 accumulators' unit), then H of the 4 words (bits).
#define LANEC 8

__device__ __forceinline__ void keep(uint32_t& x) { asm volatile("" : "+r"(x)); }

// ---- instructions ----------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void omma(float d[4], const uint32_t* a, const uint32_t* b, uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
}

// D = A (16 x 16 s8) . B (16 x 8 s8) from C = 0: one block's exact group dots.
__device__ __forceinline__ void imma16(int d[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

__device__ __forceinline__ void imma16_acc(int d[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void hmma16(float d[4], const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ int dp4a(uint32_t a, uint32_t b, int c) {
    int d;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// ---- the atom on integers (BlockScaledAlignAdd._transducer) -----------------------------------------------------------------------

__device__ __forceinline__ long long trunc0(long long m, int e, int lsb) {
    const int d = e - lsb;
    if (d >= 0) return m << d;
    if (-d > 62) return 0;
    const long long q = (m < 0 ? -m : m) >> (-d);
    return m < 0 ? -q : q;
}

__device__ __forceinline__ uint32_t f32_rz(long long n, int lsb) {
    if (n == 0) return 0u;
    const uint32_t neg = n < 0 ? 0x80000000u : 0u;
    const unsigned long long mag = (unsigned long long)(n < 0 ? -n : n);
    const int e_val = 63 - __clzll((long long)mag) + lsb;
    const int grid = e_val >= -126 ? e_val - 23 : -149;
    const int shift = grid - lsb;
    const unsigned long long q = shift > 0 ? (shift > 63 ? 0ull : mag >> shift) : mag << (-shift);
    if (q == 0) return 0u;
    const int e = 63 - __clzll((long long)q) + grid;
    const uint32_t bits = e >= -126 ? ((uint32_t)(e + 127) << 23) | (uint32_t)(q & 0x7FFFFFull) : (uint32_t)q;
    return bits | neg;
}

// One firing on one word: c the accumulator bits; gd / nz the four group dots and nonzero-product counts; sa / sb the packed
// decoded scales of this word's row and column (half b = ma | (ea + 16) << 8 for block b).
__device__ __forceinline__ uint32_t atom_step(uint32_t c, const int gd[4], const int nz[4], uint32_t sa01, uint32_t sa23,
                                              uint32_t sb01, uint32_t sb23) {
    const int NONE = -100000;
    int anchor = NONE, xg[4];
    long long mg[4];
#pragma unroll
    for (int g = 0; g < 4; g++) {
        const uint32_t hs = ((g < 2 ? sa01 : sa23) >> (16 * (g & 1))) & 0xFFFFu;
        const uint32_t hb = ((g < 2 ? sb01 : sb23) >> (16 * (g & 1))) & 0xFFFFu;
        const int mm = (int)(hs & 0xFF) * (int)(hb & 0xFF);
        xg[g] = (int)(hs >> 8) + (int)(hb >> 8) - 34;          // (ea + 16) + (eb + 16) - 32 - 2
        mg[g] = (long long)gd[g] * mm;
        if (nz[g] > 0 && mm != 0) anchor = max(anchor, xg[g] - 27);
    }
    const int ef = (c >> 23) & 0xFF;
    const long long frac = c & 0x7FFFFF;
    long long am = ef ? (frac | 0x800000) : frac;
    if (c >> 31) am = -am;
    const int ae = ef ? ef - 150 : -149;
    if (am != 0) anchor = max(anchor, (ef ? ef : 1) - 162);
    if (anchor == NONE) return 0u;
    const int lsb = max(anchor, -174);
    long long total = trunc0(am, ae, lsb);
#pragma unroll
    for (int g = 0; g < 4; g++) total += mg[g] != 0 ? trunc0(mg[g], xg[g], lsb) : 0;
    return f32_rz(total, lsb);
}

__device__ __forceinline__ float pow2f(int e) { return __int_as_float((e + 127) << 23); }

// ---- the route kernels -------------------------------------------------------------------------------------------------------

template <int R>
__global__ void route_kernel(const uint32_t* __restrict__ frags, const uint32_t* __restrict__ lanec, long long iters,
                             uint32_t* __restrict__ out, long long* __restrict__ cycles) {
    constexpr int NSETS = Shape<R>::NSETS, NACC = Shape<R>::NACC, W = Shape<R>::WORDS;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const size_t lid = (size_t)warp * 32 + lane;
    uint32_t f[NSETS][W];
#pragma unroll
    for (int s = 0; s < NSETS; s++)
#pragma unroll
        for (int w = 0; w < W; w++) f[s][w] = frags[(lid * NSETS + s) * W + w];
    int e0[4];
    uint32_t hbits[4];
#pragma unroll
    for (int o = 0; o < 4; o++) { e0[o] = (int)lanec[lid * LANEC + o]; hbits[o] = lanec[lid * LANEC + 4 + o]; }

    __shared__ int lut[256];
    if constexpr (R == R_LUT) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            const int ha[8] = {0, 1, 2, 3, 4, 6, 8, 12};
            const int a = i >> 4, b = i & 15;
            const int va = (a & 8) ? -ha[a & 7] : ha[a & 7], vb = (b & 8) ? -ha[b & 7] : ha[b & 7];
            lut[i] = va * vb;
        }
        __syncthreads();
    }
    volatile int* vlut = lut;

    // chain state
    float fa[NACC][4];            // native, bf16 routes (and imma_exact as bits)
    long long ia[NACC][4];        // integer routes
#pragma unroll
    for (int c = 0; c < NACC; c++)
#pragma unroll
        for (int o = 0; o < 4; o++) {
            const bool hot = R == R_NATIVE_HOT || R == R_BF16_FLOOR;
            fa[c][o] = hot ? __uint_as_float(hbits[o]) : 0.f;
            if constexpr (R == R_IMMA_FLOOR) {
                const int eh = (int)((hbits[o] >> 23) & 0xFF) - 127;
                ia[c][o] = 3ll << (eh - 1 - e0[o]);             // H = 1.5 * 2^E_H at the unit 2^E0
            } else {
                ia[c][o] = 0;
            }
        }
    int gsh[4];
#pragma unroll
    for (int o = 0; o < 4; o++) gsh[o] = (int)((hbits[o] >> 23) & 0xFF) - 127 - 23 - e0[o];

    __syncthreads();
    const long long t0 = clock64();
    static_assert(NSETS == NACC, "chain c owns operand set c");
#pragma unroll 1
    for (long long it = 0; it < iters; it++) {
        {
#pragma unroll
            for (int c = 0; c < NACC; c++) {
                uint32_t* x = f[c];
                if constexpr (R != R_NATIVE && R != R_NATIVE_HOT && R != R_BF16_FOLD) {
#pragma unroll
                    for (int w = 0; w < W; w++) keep(x[w]);
                }
                if constexpr (R == R_NATIVE || R == R_NATIVE_HOT) {
                    omma(fa[c], x, x + 4, x[6], x[7]);
                } else if constexpr (R == R_IMMA_EXACT) {
                    // x: A s8 [0..8) (block b: row g, row g+8), B s8 [8..12), A nonzero [12..20), B nonzero [20..24),
                    //    SA [24..28) (row g: blocks 01, 23; row g+8: 01, 23), SB [28..32) (col 2t: 01, 23; col 2t+1: 01, 23)
                    int gd[4][4], nz[4][4];
#pragma unroll
                    for (int b = 0; b < 4; b++) {
                        imma16(gd[b], x[2 * b], x[2 * b + 1], x[8 + b]);
                        imma16(nz[b], x[12 + 2 * b], x[12 + 2 * b + 1], x[20 + b]);
                    }
#pragma unroll
                    for (int o = 0; o < 4; o++) {
                        const int r = o >> 1, q = o & 1;
                        const int g4[4] = {gd[0][o], gd[1][o], gd[2][o], gd[3][o]};
                        const int n4[4] = {nz[0][o], nz[1][o], nz[2][o], nz[3][o]};
                        const uint32_t nb = atom_step(__float_as_uint(fa[c][o]), g4, n4, x[24 + 2 * r], x[25 + 2 * r],
                                                      x[28 + 2 * q], x[29 + 2 * q]);
                        fa[c][o] = __uint_as_float(nb);
                    }
                } else if constexpr (R == R_IMMA_FAST || R == R_IMMA_FLOOR) {
                    // x: A s8 [0..8), B s8 [8..12), FA [12..20) (row g blocks 0-3, row g+8 blocks 0-3), FB [20..28)
                    long long st[4] = {0, 0, 0, 0};
#pragma unroll
                    for (int b = 0; b < 4; b++) {
                        int gd[4];
                        imma16(gd, x[2 * b], x[2 * b + 1], x[8 + b]);
#pragma unroll
                        for (int o = 0; o < 4; o++) {
                            const int ab = (int)x[12 + 4 * (o >> 1) + b] * (int)x[20 + 4 * (o & 1) + b];
                            if constexpr (R == R_IMMA_FAST) ia[c][o] += (long long)gd[o] * ab;
                            else st[o] += (long long)gd[o] * ab;
                        }
                    }
                    if constexpr (R == R_IMMA_FLOOR) {
#pragma unroll
                        for (int o = 0; o < 4; o++) ia[c][o] += (st[o] >> gsh[o]) << gsh[o];
                    }
                } else if constexpr (R == R_BF16_FOLD) {
                    // x: block b: A bf16 [6b..6b+4), B bf16 [6b+4..6b+6)
#pragma unroll
                    for (int b = 0; b < 4; b++) hmma16(fa[c], x + 6 * b, x + 6 * b + 4);
                } else if constexpr (R == R_BF16_FLOOR) {
                    float st[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int b = 0; b < 4; b++) hmma16(st, x + 6 * b, x + 6 * b + 4);
#pragma unroll
                    for (int o = 0; o < 4; o++) {
                        const float h = __uint_as_float(hbits[o]);                 // ulp(H) is the floor grid
                        fa[c][o] = __fadd_rn(fa[c][o], __fsub_rn(__fadd_rd(st[o], h), h));
                    }
                } else if constexpr (R == R_DP4A) {
                    // x: A bytes row g [0..16), row g+8 [16..32), B bytes col 2t [32..48), col 2t+1 [48..64), FA [64..72), FB [72..80)
#pragma unroll
                    for (int o = 0; o < 4; o++) {
                        const uint32_t* ar = x + 16 * (o >> 1);
                        const uint32_t* bc = x + 32 + 16 * (o & 1);
#pragma unroll
                        for (int b = 0; b < 4; b++) {
                            int gd = 0;
#pragma unroll
                            for (int w = 0; w < 4; w++) gd = dp4a(ar[4 * b + w], bc[4 * b + w], gd);
                            const int ab = (int)x[64 + 4 * (o >> 1) + b] * (int)x[72 + 4 * (o & 1) + b];
                            ia[c][o] += (long long)gd * ab;
                        }
                    }
                } else if constexpr (R == R_LUT) {
                    // x: A nibbles row g [0..8), row g+8 [8..16), B nibbles col 2t [16..24), col 2t+1 [24..32), FA [32..40), FB [40..48)
#pragma unroll
                    for (int o = 0; o < 4; o++) {
                        const uint32_t* ar = x + 8 * (o >> 1);
                        const uint32_t* bc = x + 16 + 8 * (o & 1);
#pragma unroll
                        for (int b = 0; b < 4; b++) {
                            int gd = 0;
#pragma unroll
                            for (int w = 2 * b; w < 2 * b + 2; w++) {
                                const uint32_t lo = ((ar[w] & 0x0F0F0F0Fu) << 4) | (bc[w] & 0x0F0F0F0Fu);
                                const uint32_t hi = (ar[w] & 0xF0F0F0F0u) | ((bc[w] >> 4) & 0x0F0F0F0Fu);
#pragma unroll
                                for (int i = 0; i < 4; i++) gd += vlut[(lo >> (8 * i)) & 0xFF] + vlut[(hi >> (8 * i)) & 0xFF];
                            }
                            const int ab = (int)x[32 + 4 * (o >> 1) + b] * (int)x[40 + 4 * (o & 1) + b];
                            ia[c][o] += (long long)gd * ab;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    const long long t1 = clock64();

#pragma unroll
    for (int c = 0; c < NACC; c++)
#pragma unroll
        for (int o = 0; o < 4; o++) {
            uint32_t w;
            if constexpr (R == R_IMMA_FAST || R == R_IMMA_FLOOR || R == R_DP4A || R == R_LUT) {
                w = __float_as_uint(__fmul_rz(__ll2float_rz(ia[c][o]), pow2f(e0[o])));
            } else {
                w = __float_as_uint(fa[c][o]);
            }
            out[((size_t)warp * NACC + c) * 128 + lane * 4 + o] = w;
        }
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

// ---- raw instruction peaks (NSETS = 4 operand sets through NACC = 8 chains; the fp4_attack.cu pattern) -------------------------------

template <int P>
__global__ void peak_kernel(const uint32_t* __restrict__ frags, long long iters, uint32_t* __restrict__ sink,
                            long long* __restrict__ cycles) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t* fr = frags + ((size_t)warp * 32 + lane) * 32;
    uint32_t v[4][8];
#pragma unroll
    for (int s = 0; s < 4; s++)
#pragma unroll
        for (int w = 0; w < 8; w++) v[s][w] = fr[8 * s + w];
    float df[8][4];
    int di[8][4];
#pragma unroll
    for (int c = 0; c < 8; c++)
#pragma unroll
        for (int o = 0; o < 4; o++) { df[c][o] = 0.f; di[c][o] = 0; }
    __syncthreads();
    const long long t0 = clock64();
#pragma unroll 1
    for (long long it = 0; it < iters; it++) {
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 8; c++) {
                const uint32_t* x = v[(c + j) & 3];
                if constexpr (P == P_OMMA) omma(df[c], x, x + 4, x[6], x[7]);
                else if constexpr (P == P_IMMA16) imma16_acc(di[c], x[0], x[1], x[2]);
                else if constexpr (P == P_HMMA16) hmma16(df[c], x, x + 4);
                else {
#pragma unroll
                    for (int o = 0; o < 4; o++) di[c][o] = dp4a(x[o], x[4 + o], di[c][o]);
                }
            }
    }
    __syncthreads();
    const long long t1 = clock64();
    uint32_t acc = 0;
#pragma unroll
    for (int c = 0; c < 8; c++)
#pragma unroll
        for (int o = 0; o < 4; o++) acc ^= __float_as_uint(df[c][o]) ^ (uint32_t)di[c][o];
    sink[(size_t)warp * 32 + lane] = acc;
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

typedef void (*RouteFn)(const uint32_t*, const uint32_t*, long long, uint32_t*, long long*);
typedef void (*PeakFn)(const uint32_t*, long long, uint32_t*, long long*);

RouteFn route_fn(int r) {
    switch (r) {
        case R_NATIVE: return route_kernel<R_NATIVE>;
        case R_NATIVE_HOT: return route_kernel<R_NATIVE_HOT>;
        case R_IMMA_EXACT: return route_kernel<R_IMMA_EXACT>;
        case R_IMMA_FAST: return route_kernel<R_IMMA_FAST>;
        case R_IMMA_FLOOR: return route_kernel<R_IMMA_FLOOR>;
        case R_BF16_FOLD: return route_kernel<R_BF16_FOLD>;
        case R_BF16_FLOOR: return route_kernel<R_BF16_FLOOR>;
        case R_DP4A: return route_kernel<R_DP4A>;
        case R_LUT: return route_kernel<R_LUT>;
        default: return nullptr;
    }
}

PeakFn peak_fn(int p) {
    switch (p) {
        case P_OMMA: return peak_kernel<P_OMMA>;
        case P_IMMA16: return peak_kernel<P_IMMA16>;
        case P_HMMA16: return peak_kernel<P_HMMA16>;
        case P_DP4A: return peak_kernel<P_DP4A>;
        default: return nullptr;
    }
}

template <int R> void shape_of(int* nsets, int* nacc, int* words) {
    *nsets = Shape<R>::NSETS; *nacc = Shape<R>::NACC; *words = Shape<R>::WORDS;
}

}  // namespace

extern "C" {

int emu_route_shape(int r, int* nsets, int* nacc, int* words) {
    switch (r) {
        case R_NATIVE: shape_of<R_NATIVE>(nsets, nacc, words); return 0;
        case R_NATIVE_HOT: shape_of<R_NATIVE_HOT>(nsets, nacc, words); return 0;
        case R_IMMA_EXACT: shape_of<R_IMMA_EXACT>(nsets, nacc, words); return 0;
        case R_IMMA_FAST: shape_of<R_IMMA_FAST>(nsets, nacc, words); return 0;
        case R_IMMA_FLOOR: shape_of<R_IMMA_FLOOR>(nsets, nacc, words); return 0;
        case R_BF16_FOLD: shape_of<R_BF16_FOLD>(nsets, nacc, words); return 0;
        case R_BF16_FLOOR: shape_of<R_BF16_FLOOR>(nsets, nacc, words); return 0;
        case R_DP4A: shape_of<R_DP4A>(nsets, nacc, words); return 0;
        case R_LUT: shape_of<R_LUT>(nsets, nacc, words); return 0;
        default: return -1;
    }
}

// Resident blocks per SM of route r (or peak -1 - p) at warps_per_block, so a launch can be one full wave.
int emu_occupancy(int r, int warps_per_block, int* blocks_per_sm) {
    const void* fn = r >= 0 ? (const void*)route_fn(r) : (const void*)peak_fn(-1 - r);
    if (!fn) return -1;
    return (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, fn, 32 * warps_per_block, 0);
}

int emu_device(int* sms, int* cc_major, int* cc_minor, char* uuid16) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return (int)e;
    cudaDeviceProp p;
    e = cudaGetDeviceProperties(&p, dev);
    if (e != cudaSuccess) return (int)e;
    *sms = p.multiProcessorCount;
    *cc_major = p.major;
    *cc_minor = p.minor;
    for (int i = 0; i < 16; i++) uuid16[i] = p.uuid.bytes[i];
    return 0;
}

// frags: nwarps x 32 lanes x NSETS x WORDS u32; lanec: nwarps x 32 x LANEC u32; out: nwarps x NACC x 128 u32; cycles: nblocks.
int emu_run(int r, const uint32_t* frags_h, size_t frag_words, const uint32_t* lanec_h, int nblocks, int warps_per_block,
            long long iters, uint32_t* out_h, long long* cycles_h, float* ms) {
    RouteFn fn = route_fn(r);
    if (!fn) return -1;
    int nsets, nacc, words;
    emu_route_shape(r, &nsets, &nacc, &words);
    const size_t nwarps = (size_t)nblocks * warps_per_block;
    if (frag_words != nwarps * 32 * nsets * words) return -2;
    uint32_t *f = nullptr, *l = nullptr, *o = nullptr;
    long long* cy = nullptr;
    cudaError_t e;
    if ((e = cudaMalloc(&f, frag_words * 4)) != cudaSuccess) return (int)e;
    if ((e = cudaMalloc(&l, nwarps * 32 * LANEC * 4)) != cudaSuccess) return (int)e;
    if ((e = cudaMalloc(&o, nwarps * nacc * 128 * 4)) != cudaSuccess) return (int)e;
    if ((e = cudaMalloc(&cy, nblocks * sizeof(long long))) != cudaSuccess) return (int)e;
    cudaMemcpy(f, frags_h, frag_words * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(l, lanec_h, nwarps * 32 * LANEC * 4, cudaMemcpyHostToDevice);
    cudaEvent_t s, t;
    cudaEventCreate(&s);
    cudaEventCreate(&t);
    cudaEventRecord(s);
    fn<<<nblocks, 32 * warps_per_block>>>(f, l, iters, o, cy);
    cudaEventRecord(t);
    e = cudaEventSynchronize(t);
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e == cudaSuccess) {
        cudaEventElapsedTime(ms, s, t);
        cudaMemcpy(out_h, o, nwarps * nacc * 128 * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(cycles_h, cy, nblocks * sizeof(long long), cudaMemcpyDeviceToHost);
    }
    cudaEventDestroy(s);
    cudaEventDestroy(t);
    cudaFree(f);
    cudaFree(l);
    cudaFree(o);
    cudaFree(cy);
    return (int)e;
}

// frags: nwarps x 32 x 32 u32 (4 sets x 8 words); sink: nwarps x 32.
int emu_peak(int p, const uint32_t* frags_h, int nblocks, int warps_per_block, long long iters, long long* cycles_h, float* ms) {
    PeakFn fn = peak_fn(p);
    if (!fn) return -1;
    const size_t nwarps = (size_t)nblocks * warps_per_block;
    uint32_t *f = nullptr, *o = nullptr;
    long long* cy = nullptr;
    cudaError_t e;
    if ((e = cudaMalloc(&f, nwarps * 32 * 32 * 4)) != cudaSuccess) return (int)e;
    if ((e = cudaMalloc(&o, nwarps * 32 * 4)) != cudaSuccess) return (int)e;
    if ((e = cudaMalloc(&cy, nblocks * sizeof(long long))) != cudaSuccess) return (int)e;
    cudaMemcpy(f, frags_h, nwarps * 32 * 32 * 4, cudaMemcpyHostToDevice);
    cudaEvent_t s, t;
    cudaEventCreate(&s);
    cudaEventCreate(&t);
    cudaEventRecord(s);
    fn<<<nblocks, 32 * warps_per_block>>>(f, iters, o, cy);
    cudaEventRecord(t);
    e = cudaEventSynchronize(t);
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e == cudaSuccess) {
        cudaEventElapsedTime(ms, s, t);
        cudaMemcpy(cycles_h, cy, nblocks * sizeof(long long), cudaMemcpyDeviceToHost);
    }
    cudaEventDestroy(s);
    cudaEventDestroy(t);
    cudaFree(f);
    cudaFree(o);
    cudaFree(cy);
    return (int)e;
}

}  // extern "C"
