// F2 (lane pouw rtx-pro GPU 4): one- and two-level Strassen over one NVFP4 group (K = 16) on the TF32 tensor core, against dense
// NVFP4 on sm_120a, word for word.
//
// Within one group every row of A carries one scale and every column of B one scale, so the group's scaled product is
// diag(sa) (CA CB) diag(sb): Strassen runs on the E2M1 codes as integer halves (|x| <= 12; a level's pre-additions reach 24 and
// 48, all exact in TF32), each product's sum is an integer below 2^14 (exact in the FP32 accumulator), and the recombination
// C = W M stays exact in FP32.  The integer group dots then enter the atom's align-add (nvf4_step.cuh) with the scales, as the
// atom's do.
//
// Kernels:
//   strassen_kernel<L>  per warp one tile: L = 1 is A 32 x 16 by B 16 x 16 in 7 mma.sync m16n8k8 TF32 (of 8), L = 2 is A 64 x 16 by
//                       B 16 x 32 in 49 m16n8k4 TF32 (of 64).  Each product's pre-added operands (A-side U and B-side V) arrive as
//                       fragments, resident in registers; the timed loop issues the 7^L MMAs (C = RZ) and the recombination W
//                       in FP32 FADD, and consumes every group dot (their bits summed into 4 words per lane).  The MMAs are
//                       asm volatile, which ptxas keeps in the loop; the build's SASS gate counts them there.
//   native_kernel       OMMA m16n8k64 kind::mxf4nvf4 4X ue4m3 on the same tiles (the unit, and the words' reference on the card).
//   step_kernel         the atom's align-add on the Strassen group dots of the four groups: the route's words.
//
// Build: nvcc -O2 -std=c++17 -gencode=arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared -o libf2_strassen_sm_120a.so f2_strassen.cu

#include <cuda_runtime.h>
#include <stdint.h>

#include "nvf4_step.cuh"

namespace {

#define AROW 9    // u32 per row of A or column of B: 8 words of codes (nibble j of word w = k 8w + j), 1 word of 4 UE4M3 scale bytes
#define NT 8      // native: independent tiles (chains) per warp
#define STRASSEN_TPB 128

// Strassen on 2 x 2 blocks, quadrant q = 2 i + j (11, 12, 21, 22), products M1..M7: C_q = sum_p W1[q][p] M_p.
__host__ __device__ constexpr int W1(int q, int p) {
    constexpr int w[4][7] = {{1, 0, 0, 1, -1, 0, 1}, {0, 0, 1, 0, 1, 0, 0}, {0, 1, 0, 1, 0, 0, 0}, {1, -1, 1, 0, 0, 1, 0}};
    return w[q][p];
}

// Level L's coefficient of product p in C block (I, J); level 2 is level 1 applied to level 1's blocks (p = 7 outer + inner, the
// outer quadrant from the high bit of I and J).
template <int L>
__host__ __device__ constexpr int wcoef(int I, int J, int p) {
    if constexpr (L == 1) return W1(2 * I + J, p);
    else return W1(2 * (I >> 1) + (J >> 1), p / 7) * W1(2 * (I & 1) + (J & 1), p % 7);
}

template <int L> struct Shape {
    static constexpr int NB = 1 << L, NP = L == 1 ? 7 : 49, KB = 16 >> L, AR = KB / 2, BR = KB / 4, TM = 16 * NB, TN = 8 * NB;
    static_assert(L == 1 || L == 2, "one or two levels");
};

// D = A B + 0 (ptxas reads the zero as RZ)
template <int KB>
__device__ __forceinline__ void mma_tf32(float d[4], const uint32_t* a, const uint32_t* b) {
    if constexpr (KB == 8) {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                     : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.0f));
    } else {
        static_assert(KB == 4, "m16n8k8 or m16n8k4");
        asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
                     : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(b[0]), "f"(0.0f));
    }
}

// afr: ntiles x NP x 32 lanes x AR TF32 words (the U-combined A blocks, 16 x KB, in the m16n8kKB A layout); bfr: ntiles x NP x 32 x BR
// (the V-combined B blocks, KB x 8).  Warp w runs tile w mod ntiles; out (ntiles x TM x TN, the last iteration's C as binary32
// bits) is written by the first warp of each tile; sums (warps x 32 x 4) are the bits of every group dot, summed mod 2^32.
template <int L>
__global__ void __launch_bounds__(STRASSEN_TPB) strassen_kernel(const uint32_t* __restrict__ afr, const uint32_t* __restrict__ bfr,
                                                                int ntiles, long long iters, uint32_t* __restrict__ out,
                                                                uint32_t* __restrict__ sums, long long* __restrict__ cycles) {
    using X = Shape<L>;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int tile = warp % ntiles;
    uint32_t a[X::NP][X::AR], b[X::NP][X::BR];
#pragma unroll
    for (int p = 0; p < X::NP; p++) {
#pragma unroll
        for (int r = 0; r < X::AR; r++) a[p][r] = afr[(((size_t)tile * X::NP + p) * 32 + lane) * X::AR + r];
#pragma unroll
        for (int r = 0; r < X::BR; r++) b[p][r] = bfr[(((size_t)tile * X::NP + p) * 32 + lane) * X::BR + r];
    }
    float c[X::NB * X::NB][4];
    uint32_t sum[4] = {0u, 0u, 0u, 0u};
    __syncthreads();
    const long long t0 = clock64();
#pragma unroll 1
    for (long long it = 0; it < iters; it++) {
#pragma unroll
        for (int q = 0; q < X::NB * X::NB; q++)
#pragma unroll
            for (int o = 0; o < 4; o++) c[q][o] = -0.0f;                 // -0 + m is m: the first term of each block folds away
#pragma unroll
        for (int p = 0; p < X::NP; p++) {
            float m[4];
            mma_tf32<X::KB>(m, a[p], b[p]);
#pragma unroll
            for (int q = 0; q < X::NB * X::NB; q++) {
                const int w = wcoef<L>(q / X::NB, q % X::NB, p);
#pragma unroll
                for (int o = 0; o < 4; o++) c[q][o] = w > 0 ? c[q][o] + m[o] : (w < 0 ? c[q][o] - m[o] : c[q][o]);
            }
        }
#pragma unroll
        for (int q = 0; q < X::NB * X::NB; q++)
#pragma unroll
            for (int o = 0; o < 4; o++) sum[o] += __float_as_uint(c[q][o]);
    }
    __syncthreads();
    const long long t1 = clock64();
    if (warp < ntiles) {
#pragma unroll
        for (int q = 0; q < X::NB * X::NB; q++)
#pragma unroll
            for (int o = 0; o < 4; o++) {
                const int row = 16 * (q / X::NB) + g + 8 * (o >> 1), col = 8 * (q % X::NB) + 2 * t + (o & 1);
                out[((size_t)tile * X::TM + row) * X::TN + col] = __float_as_uint(c[q][o]);
            }
    }
#pragma unroll
    for (int o = 0; o < 4; o++) sums[((size_t)warp * 32 + lane) * 4 + o] = sum[o];
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

// Native tile u of ntiles: Strassen tile s = u / per (per = TM/16 x TN/8 native tiles), whose rows are s TM.. and columns
// (s mod nct) TN..; arows: rows x AROW, brows: columns x AROW; c0 and out: ntiles_s x TM x TN (a Strassen tile's elements).
__global__ void native_kernel(const uint32_t* __restrict__ arows, const uint32_t* __restrict__ brows, const uint32_t* __restrict__ c0, int TM,
                              int TN, int nct, int ntiles, long long iters, uint32_t* __restrict__ out, long long* __restrict__ cycles) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31, g4 = lane >> 2, t4 = lane & 3;
    const int per = (TM / 16) * (TN / 8);
    uint32_t a[NT][4], b[NT][2], sfa[NT], sfb[NT];
    float d[NT][4];
    size_t el[NT];
#pragma unroll
    for (int i = 0; i < NT; i++) {
        const int u = (int)(((long long)warp * NT + i) % ntiles);
        const int s = u / per, bi = (u % per) / (TN / 8), bj = (u % per) % (TN / 8);
        const size_t r0 = (size_t)s * TM + 16 * bi, col0 = (size_t)(s % nct) * TN + 8 * bj;
        el[i] = ((size_t)s * TM + 16 * bi) * TN + 8 * bj;
#pragma unroll
        for (int r = 0; r < 4; r++) a[i][r] = arows[(r0 + g4 + 8 * (r & 1)) * AROW + t4 + 4 * (r >> 1)];
#pragma unroll
        for (int r = 0; r < 2; r++) b[i][r] = brows[(col0 + g4) * AROW + t4 + 4 * r];
        sfa[i] = arows[(r0 + 8 * (lane & 1) + g4) * AROW + 8];
        sfb[i] = brows[(col0 + g4) * AROW + 8];
#pragma unroll
        for (int o = 0; o < 4; o++) d[i][o] = __uint_as_float(c0[el[i] + (size_t)(g4 + 8 * (o >> 1)) * TN + 2 * t4 + (o & 1)]);
    }
    __syncthreads();
    const long long t0 = clock64();
#pragma unroll 1
    for (long long it = 0; it < iters; it++) {
#pragma unroll
        for (int i = 0; i < NT; i++) omma(d[i], a[i], b[i], sfa[i], sfb[i]);
    }
    __syncthreads();
    const long long t1 = clock64();
    if (out) {
#pragma unroll
        for (int i = 0; i < NT; i++) {
            if ((long long)warp * NT + i >= ntiles) continue;
#pragma unroll
            for (int o = 0; o < 4; o++) out[el[i] + (size_t)(g4 + 8 * (o >> 1)) * TN + 2 * t4 + (o & 1)] = __float_as_uint(d[i][o]);
        }
    }
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

// Element e: gdf[4 e + g] the Strassen group dot of group g (binary32 bits of an integer, quarters), mm / xg the group's scale
// terms, bit g of part[e] its participation; out[e] = the atom's align-add chained `length` times from c0[e].
__global__ void step_kernel(const uint32_t* __restrict__ gdf, const int* __restrict__ mm, const int* __restrict__ xg,
                            const uint32_t* __restrict__ part, const uint32_t* __restrict__ c0, long long n, int length,
                            uint32_t* __restrict__ out) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    int gd[4], m[4], x[4];
    bool pt[4];
#pragma unroll
    for (int g = 0; g < 4; g++) {
        gd[g] = __float2int_rz(__uint_as_float(gdf[4 * e + g]));
        m[g] = mm[4 * e + g];
        x[g] = xg[4 * e + g];
        pt[g] = (part[e] >> g) & 1u;
    }
    uint32_t c = c0[e];
    for (int l = 0; l < length; l++) c = nvf4_step(c, gd, m, x, pt);
    out[e] = c;
}

typedef void (*StrassenFn)(const uint32_t*, const uint32_t*, int, long long, uint32_t*, uint32_t*, long long*);

StrassenFn strassen_fn(int level) {
    return level == 1 ? strassen_kernel<1> : level == 2 ? strassen_kernel<2> : nullptr;
}

int upload(void** d, const void* h, size_t bytes) {
    cudaError_t e = cudaMalloc(d, bytes ? bytes : 4);
    if (e == cudaSuccess && bytes) e = cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice);
    return (int)e;
}

}  // namespace

extern "C" {

// Level's shape: products, K per block, tile rows and columns; -1 for an unsupported level.
int f2_shape(int level, int* np, int* kb, int* tm, int* tn) {
    if (level == 1) { *np = Shape<1>::NP; *kb = Shape<1>::KB; *tm = Shape<1>::TM; *tn = Shape<1>::TN; return 0; }
    if (level == 2) { *np = Shape<2>::NP; *kb = Shape<2>::KB; *tm = Shape<2>::TM; *tn = Shape<2>::TN; return 0; }
    return -1;
}

int f2_nt() { return NT; }
int f2_tpb() { return STRASSEN_TPB; }

int f2_device(int* sms, int* max_threads_per_sm, int* cc_major, int* cc_minor) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return (int)e;
    cudaDeviceProp p;
    if ((e = cudaGetDeviceProperties(&p, dev)) != cudaSuccess) return (int)e;
    *sms = p.multiProcessorCount; *max_threads_per_sm = p.maxThreadsPerMultiProcessor;
    *cc_major = p.major; *cc_minor = p.minor;
    return 0;
}

// Resident blocks per SM: kind 1 / 2 the Strassen level at STRASSEN_TPB, 0 the native kernel at tpb.
int f2_occupancy(int kind, int tpb, int* blocks_per_sm) {
    if (kind == 0) return (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, native_kernel, tpb, 0);
    StrassenFn fn = strassen_fn(kind);
    if (!fn) return -1;
    return (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, (const void*)fn, STRASSEN_TPB, 0);
}

int f2_run_strassen(int level, const uint32_t* afr, const uint32_t* bfr, int ntiles, int nblocks, long long iters, uint32_t* out_h,
                    uint32_t* sums_h, long long* cycles_h, float* ms) {
    StrassenFn fn = strassen_fn(level);
    int np, kb, tm, tn;
    if (!fn || f2_shape(level, &np, &kb, &tm, &tn)) return -1;
    const size_t warps = (size_t)nblocks * STRASSEN_TPB / 32, fa = (size_t)ntiles * np * 32 * (kb / 2), fb = (size_t)ntiles * np * 32 * (kb / 4);
    void *da = nullptr, *db = nullptr, *o = nullptr, *su = nullptr, *cy = nullptr;
    int rc = upload(&da, afr, fa * 4);
    if (!rc) rc = upload(&db, bfr, fb * 4);
    if (!rc) rc = (int)cudaMalloc(&o, (size_t)ntiles * tm * tn * 4);
    if (!rc) rc = (int)cudaMalloc(&su, warps * 32 * 4 * 4);
    if (!rc) rc = (int)cudaMalloc(&cy, nblocks * sizeof(long long));
    if (!rc) {
        cudaEvent_t s, t;
        cudaEventCreate(&s);
        cudaEventCreate(&t);
        cudaEventRecord(s);
        fn<<<nblocks, STRASSEN_TPB>>>((const uint32_t*)da, (const uint32_t*)db, ntiles, iters, (uint32_t*)o, (uint32_t*)su, (long long*)cy);
        cudaEventRecord(t);
        rc = (int)cudaEventSynchronize(t);
        if (!rc) rc = (int)cudaGetLastError();
        if (!rc) {
            cudaEventElapsedTime(ms, s, t);
            cudaMemcpy(out_h, o, (size_t)ntiles * tm * tn * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(sums_h, su, warps * 32 * 4 * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(cycles_h, cy, nblocks * sizeof(long long), cudaMemcpyDeviceToHost);
        }
        cudaEventDestroy(s);
        cudaEventDestroy(t);
    }
    cudaFree(da); cudaFree(db); cudaFree(o); cudaFree(su); cudaFree(cy);
    return rc;
}

int f2_run_native(const uint32_t* arows, long long nrows, const uint32_t* brows, long long ncols, const uint32_t* c0, int TM, int TN, int nct,
                  int ntiles_s, int ntiles, int nblocks, int tpb, long long iters, uint32_t* out_h, long long* cycles_h, float* ms) {
    const size_t nel = (size_t)ntiles_s * TM * TN;
    void *ar = nullptr, *br = nullptr, *cc = nullptr, *o = nullptr, *cy = nullptr;
    int rc = upload(&ar, arows, (size_t)nrows * AROW * 4);
    if (!rc) rc = upload(&br, brows, (size_t)ncols * AROW * 4);
    if (!rc) rc = upload(&cc, c0, nel * 4);
    if (!rc && out_h) rc = (int)cudaMalloc(&o, nel * 4);
    if (!rc) rc = (int)cudaMalloc(&cy, nblocks * sizeof(long long));
    if (!rc) {
        cudaEvent_t s, t;
        cudaEventCreate(&s);
        cudaEventCreate(&t);
        cudaEventRecord(s);
        native_kernel<<<nblocks, tpb>>>((const uint32_t*)ar, (const uint32_t*)br, (const uint32_t*)cc, TM, TN, nct, ntiles, iters, (uint32_t*)o,
                                        (long long*)cy);
        cudaEventRecord(t);
        rc = (int)cudaEventSynchronize(t);
        if (!rc) rc = (int)cudaGetLastError();
        if (!rc) {
            cudaEventElapsedTime(ms, s, t);
            if (out_h) cudaMemcpy(out_h, o, nel * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(cycles_h, cy, nblocks * sizeof(long long), cudaMemcpyDeviceToHost);
        }
        cudaEventDestroy(s);
        cudaEventDestroy(t);
    }
    cudaFree(ar); cudaFree(br); cudaFree(cc); cudaFree(o); cudaFree(cy);
    return rc;
}

int f2_run_step(const uint32_t* gdf, const int* mm, const int* xg, const uint32_t* part, const uint32_t* c0, long long n, int length,
                uint32_t* out_h) {
    void *g = nullptr, *m = nullptr, *x = nullptr, *p = nullptr, *cc = nullptr, *o = nullptr;
    int rc = upload(&g, gdf, (size_t)n * 16);
    if (!rc) rc = upload(&m, mm, (size_t)n * 16);
    if (!rc) rc = upload(&x, xg, (size_t)n * 16);
    if (!rc) rc = upload(&p, part, (size_t)n * 4);
    if (!rc) rc = upload(&cc, c0, (size_t)n * 4);
    if (!rc) rc = (int)cudaMalloc(&o, (size_t)n * 4);
    if (!rc) {
        step_kernel<<<(unsigned)((n + 255) / 256), 256>>>((const uint32_t*)g, (const int*)m, (const int*)x, (const uint32_t*)p, (const uint32_t*)cc,
                                                          n, length, (uint32_t*)o);
        rc = (int)cudaDeviceSynchronize();
        if (!rc) rc = (int)cudaGetLastError();
        if (!rc) cudaMemcpy(out_h, o, (size_t)n * 4, cudaMemcpyDeviceToHost);
    }
    cudaFree(g); cudaFree(m); cudaFree(x); cudaFree(p); cudaFree(cc); cudaFree(o);
    return rc;
}

// probe.Probe.identity's entry points (f1_rates.py's prelude and f3_lut.cu have the same three)
int device_identity(int* major, int* minor, char* name, int name_len) {
    cudaDeviceProp p;
    cudaError_t err = cudaGetDeviceProperties(&p, 0);
    if (err != cudaSuccess) return (int)err;
    *major = p.major; *minor = p.minor;
    int i = 0;
    for (; i < name_len - 1 && p.name[i]; ++i) name[i] = p.name[i];
    name[i] = 0;
    return 0;
}

int device_uuid(char* out, int out_len) {
    cudaDeviceProp p;
    cudaError_t err = cudaGetDeviceProperties(&p, 0);
    if (err != cudaSuccess) return (int)err;
    if (out_len < 41) return (int)cudaErrorInvalidValue;
    static const char* hex = "0123456789abcdef";
    const unsigned char* u = (const unsigned char*)p.uuid.bytes;
    int k = 0;
    out[k++] = 'G'; out[k++] = 'P'; out[k++] = 'U'; out[k++] = '-';
    for (int i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out[k++] = '-';
        out[k++] = hex[u[i] >> 4];
        out[k++] = hex[u[i] & 15];
    }
    out[k] = 0;
    return 0;
}

int has_mxf4(void) { return 1; }

}  // extern "C"
