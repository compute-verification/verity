// F3: LUT-GEMM over the 16-code E2M1 alphabet with tables from the weights, against dense NVFP4 on
// sm_120a, bit for bit.
//
// The weights are fixed (preprocessing/fixed-weights), so every table is built on the host.  Table width (G, C): one table per
// slice of G consecutive k of one K = 64 atom, indexed by the slice's G activation codes (16^G entries); an entry holds the exact
// partial dots sum_i num(a_i) num(w_i) (quarters, |.| <= 144 G) of C columns, two int16 per u32 word as lo + 65536 hi, so the
// 16 / G entries of one 16-element block sum exactly in int32 and unpack to the block's C group dots.  One lookup replaces G x C
// products.  G <= 2 tables live in shared memory (G = 2, C = 4: 64 KiB), G = 4 tables (4 MiB per column pair) in global memory.
//
// Arms, per thread one activation row x one column group of C columns, chained over `iters` repeats of one K = 64 atom:
//   dots    the lookups and the unpacking only; each block's group dots accumulate in int32 (the table's own price)
//   exact   the lookups, then per column the atom BlockScaledAlignAdd (verity.ml.circuits.silicon BLACKWELL_SM120_NVF4) on the CUDA cores:
//           the block scales' products, participation, and the five-way align-add truncated toward zero onto the common grid,
//           RZ to binary32 (words bit-exact against the atom for every finite accumulator)
//   native  OMMA m16n8k64 kind::mxf4nvf4 4X ue4m3 on the same rows and columns (the unit, and the words' reference on the card)
//
// What is outside the timed loop, because a GEMM amortizes it: the table addresses (activation side, reused across the N / C
// column groups), the activation scales' decode and nonzero masks (reused across columns), the weight scales' decode and masks
// (fixed weights).  What stays inside, per row x column x block, because nothing amortizes it: the scale products, the exponent
// sums and participation; a runtime zero z, loop-carried through one add per iteration, is folded into them so ptxas can't
// hoist them.  The lookups are asm volatile loads, issued every iteration.
//
// Build: nvcc -O2 -std=c++17 -gencode=arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared -o libf3_lut_sm_120a.so f3_lut.cu

#include <cuda_runtime.h>
#include <stdint.h>

#include "nvf4_step.cuh"

namespace {

enum Arm { A_DOTS = 0, A_EXACT = 1 };

#define AROW 9    // u32 per activation row: 8 words of codes (nibble j of word w = k 8w + j), 1 word of 4 UE4M3 scale bytes
#define NT 8      // native: independent tiles (chains) per warp

// ---- lookups -------------------------------------------------------------------------------------------------------------------

template <int W> struct Entry { uint32_t w[W]; };

template <int W>
__device__ __forceinline__ Entry<W> lds(uint32_t addr) {
    Entry<W> e;
    if constexpr (W == 1) {
        asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(e.w[0]) : "r"(addr));
    } else if constexpr (W == 2) {
        asm volatile("ld.volatile.shared.v2.u32 {%0,%1}, [%2];" : "=r"(e.w[0]), "=r"(e.w[1]) : "r"(addr));
    } else {
        static_assert(W == 4, "entry of 1, 2 or 4 words");
        asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(e.w[0]), "=r"(e.w[1]), "=r"(e.w[2]), "=r"(e.w[3]) : "r"(addr));
    }
    return e;
}

// A read-only global load: the caller folds the runtime zero into the address, so it is issued every iteration (a volatile
// global load would be a system-scope LDG.E.STRONG.SYS, slower than a GEMM needs).
template <int W>
__device__ __forceinline__ Entry<W> ldg(const uint32_t* __restrict__ p) {
    Entry<W> e;
#pragma unroll
    for (int i = 0; i < W; i++) e.w[i] = __ldg(p + i);
    return e;
}

// ---- the LUT kernel ------------------------------------------------------------------------------------------------------------

template <int G, int C> struct Lut {
    static constexpr int W = C / 2, NKG = 64 / G, NIDX = 1 << (4 * G), PER_BLK = 16 / G;
    static constexpr size_t TABW = (size_t)NKG * NIDX * W;     // u32 words of one column group's tables
    static constexpr bool SMEM = G <= 2;
    static constexpr size_t SMEM_BYTES = SMEM ? TABW * 4 : 0;
    static_assert(C % 2 == 0 && 16 % G == 0, "C even, G divides the 16-element block");
};

// arows: rows x AROW; tables: ng x TABW; wsc: ng x C x 4 (per column and block: mb | (eb + 10) << 4 | nonzero-code mask << 8);
// c0: rows x C accumulator words (exact); out: rows x C (exact) or rows x C x 4 (dots, int32 sums of the group dots).
template <int ARM, int G, int C>
__global__ void lut_kernel(const uint32_t* __restrict__ arows, const uint32_t* __restrict__ tables, const uint32_t* __restrict__ wsc,
                           const uint32_t* __restrict__ c0, int ng, long long iters, uint32_t zero, uint32_t* __restrict__ out,
                           long long* __restrict__ cycles) {
    using L = Lut<G, C>;
    constexpr int W = L::W;
    extern __shared__ uint32_t smem[];
    const int grp = blockIdx.x % ng;
    const uint32_t* gtab = tables + (size_t)grp * L::TABW;
    if constexpr (L::SMEM) {
        for (size_t i = threadIdx.x; i < L::TABW; i += blockDim.x) smem[i] = gtab[i];
    }
    const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t a[8];
#pragma unroll
    for (int w = 0; w < 8; w++) a[w] = arows[row * AROW + w];
    const uint32_t sa = arows[row * AROW + 8];

    // activation side (amortized over the column groups): the lookups' addresses, the scales' decode, the nonzero-code masks
    uint32_t addr[L::NKG];
    const uint32_t sbase = L::SMEM ? (uint32_t)__cvta_generic_to_shared(smem) : 0u;
#pragma unroll
    for (int kg = 0; kg < L::NKG; kg++) {
        const int bit = 4 * G * kg;
        const uint32_t idx = (a[bit >> 5] >> (bit & 31)) & (uint32_t)(L::NIDX - 1);
        addr[kg] = (uint32_t)(kg * L::NIDX + idx) * W;          // words
        if constexpr (L::SMEM) addr[kg] = sbase + 4 * addr[kg];
    }
    int ma[4], ea[4];
    uint32_t amask[4];
#pragma unroll
    for (int b = 0; b < 4; b++) {
        const uint32_t s = (sa >> (8 * b)) & 0xFF, e = (s >> 3) & 0xF, m = s & 7;
        ma[b] = e ? 8 + (int)m : (int)m;
        ea[b] = e ? (int)e - 10 : -9;
        uint32_t mk = 0;
#pragma unroll
        for (int j = 0; j < 16; j++) mk |= (((a[2 * b + (j >> 3)] >> (4 * (j & 7))) & 7u) != 0u ? 1u : 0u) << j;
        amask[b] = mk << 8;
    }
    // weight side (fixed weights)
    uint32_t wv[C][4];
#pragma unroll
    for (int c = 0; c < C; c++)
#pragma unroll
        for (int b = 0; b < 4; b++) wv[c][b] = wsc[((size_t)grp * C + c) * 4 + b];

    uint32_t acc[C];
    uint32_t dacc[C][4];                                           // mod 2^32: a long chain wraps

#pragma unroll
    for (int c = 0; c < C; c++) {
        acc[c] = ARM == A_EXACT ? c0[row * C + c] : 0u;
#pragma unroll
            for (int b = 0; b < 4; b++) dacc[c][b] = 0u;
    }
    __syncthreads();
    const long long t0 = clock64();
    uint32_t z = zero;
#pragma unroll 1
    for (long long it = 0; it < iters; it++) {
        asm volatile("add.u32 %0, %0, %1;" : "+r"(z) : "r"(zero));
        uint32_t S[4][W];
#pragma unroll
        for (int b = 0; b < 4; b++)
#pragma unroll
            for (int p = 0; p < W; p++) S[b][p] = 0u;
#pragma unroll
        for (int kg = 0; kg < L::NKG; kg++) {
            Entry<W> e;
            if constexpr (L::SMEM) e = lds<W>(addr[kg]);
            else e = ldg<W>(gtab + (addr[kg] + z));
#pragma unroll
            for (int p = 0; p < W; p++) S[kg / L::PER_BLK][p] += e.w[p];
        }
        int gd[C][4];
#pragma unroll
        for (int b = 0; b < 4; b++)
#pragma unroll
            for (int p = 0; p < W; p++) {
                const int lo = (int)(short)(S[b][p] & 0xFFFFu);
                gd[2 * p][b] = lo;
                gd[2 * p + 1][b] = ((int)S[b][p] - lo) >> 16;
            }
        if constexpr (ARM == A_EXACT) {
#pragma unroll
            for (int c = 0; c < C; c++) {
                int mm[4], xg[4];
                bool part[4];
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    const uint32_t w = wv[c][b];
                    mm[b] = (ma[b] ^ (int)z) * (int)(w & 15u);
                    xg[b] = ea[b] + (int)((w >> 4) & 15u) - 12 + (int)z;   // -2 + ea + eb, eb = field - 10
                    part[b] = ((amask[b] ^ z) & w & 0xFFFF00u) != 0u && mm[b] != 0;
                }
                acc[c] = nvf4_step(acc[c], gd[c], mm, xg, part);
            }
        } else {
#pragma unroll
            for (int c = 0; c < C; c++)
#pragma unroll
                for (int b = 0; b < 4; b++) dacc[c][b] += (uint32_t)gd[c][b];
        }
    }
    __syncthreads();
    const long long t1 = clock64();
#pragma unroll
    for (int c = 0; c < C; c++) {
        if constexpr (ARM == A_EXACT) {
            out[row * C + c] = acc[c];
        } else {
#pragma unroll
            for (int b = 0; b < 4; b++) out[(row * C + c) * 4 + b] = dacc[c][b];
        }
    }
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

// ---- the native atom on the same rows and columns -----------------------------------------------------------------------------

// Warp w runs tiles (NT w + i) mod ntiles: rows 16 tile.., the 8 columns of its rows' group (column j is group column j mod C).
// wcodes: ng x C x AROW (the weights' codes and scales); out (when set): ntiles x 16 x 8 for the first occurrence of each tile.
__global__ void native_kernel(const uint32_t* __restrict__ arows, const uint32_t* __restrict__ wcodes, const uint32_t* __restrict__ c0,
                              int C, int rows_per_group, int ng, int ntiles, long long iters, uint32_t* __restrict__ out,
                              long long* __restrict__ cycles) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31, g4 = lane >> 2, t4 = lane & 3;
    uint32_t a[NT][4], b[NT][2], sfa[NT], sfb[NT];
    float d[NT][4];
#pragma unroll
    for (int i = 0; i < NT; i++) {
        const int tile = (int)(((long long)warp * NT + i) % ntiles);
        const size_t r0 = (size_t)tile * 16;
        const int grp = (int)((r0 / rows_per_group) % ng);
        const int col = g4 % C;
        const uint32_t* wc = wcodes + ((size_t)grp * C + col) * AROW;
#pragma unroll
        for (int r = 0; r < 4; r++) a[i][r] = arows[(r0 + g4 + 8 * (r & 1)) * AROW + t4 + 4 * (r >> 1)];
#pragma unroll
        for (int r = 0; r < 2; r++) b[i][r] = wc[t4 + 4 * r];
        sfa[i] = arows[(r0 + 8 * (lane & 1) + g4) * AROW + 8];
        sfb[i] = wc[8];
#pragma unroll
        for (int o = 0; o < 4; o++) {
            const size_t rr = r0 + g4 + 8 * (o >> 1);
            d[i][o] = __uint_as_float(c0[rr * C + (2 * t4 + (o & 1)) % C]);
        }
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
            const long long tile = (long long)warp * NT + i;
            if (tile >= ntiles) continue;
#pragma unroll
            for (int o = 0; o < 4; o++)
                out[((size_t)tile * 16 + g4 + 8 * (o >> 1)) * 8 + 2 * t4 + (o & 1)] = __float_as_uint(d[i][o]);
        }
    }
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

// ---- the variants ----------------------------------------------------------------------------------------------------------------

typedef void (*LutFn)(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, int, long long, uint32_t, uint32_t*, long long*);

struct Variant { int arm, g, c; LutFn fn; size_t tabw, smem; };

#define V(ARM, G, C) {ARM, G, C, lut_kernel<ARM, G, C>, Lut<G, C>::TABW, Lut<G, C>::SMEM_BYTES}
const Variant VARIANTS[] = {
    V(A_DOTS, 1, 2), V(A_EXACT, 1, 2), V(A_DOTS, 1, 4), V(A_EXACT, 1, 4), V(A_DOTS, 1, 8), V(A_EXACT, 1, 8),
    V(A_DOTS, 2, 2), V(A_EXACT, 2, 2), V(A_DOTS, 2, 4), V(A_EXACT, 2, 4), V(A_DOTS, 4, 2), V(A_EXACT, 4, 2),
};
#undef V
constexpr int NVAR = sizeof(VARIANTS) / sizeof(VARIANTS[0]);

}  // namespace

extern "C" {

int f3_nvariants() { return NVAR; }

int f3_variant(int v, int* arm, int* g, int* c, long long* tabw, long long* smem) {
    if (v < 0 || v >= NVAR) return -1;
    *arm = VARIANTS[v].arm; *g = VARIANTS[v].g; *c = VARIANTS[v].c;
    *tabw = (long long)VARIANTS[v].tabw; *smem = (long long)VARIANTS[v].smem;
    return 0;
}

int f3_device(int* sms, int* max_threads_per_sm, int* cc_major, int* cc_minor) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return (int)e;
    cudaDeviceProp p;
    if ((e = cudaGetDeviceProperties(&p, dev)) != cudaSuccess) return (int)e;
    *sms = p.multiProcessorCount; *max_threads_per_sm = p.maxThreadsPerMultiProcessor;
    *cc_major = p.major; *cc_minor = p.minor;
    return 0;
}

// Resident blocks per SM of variant v (v = -1: native) at tpb threads, with the variant's shared memory opted in.
int f3_occupancy(int v, int tpb, int* blocks_per_sm) {
    if (v < 0) return (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, native_kernel, tpb, 0);
    if (v >= NVAR) return -1;
    const Variant& x = VARIANTS[v];
    cudaError_t e = cudaFuncSetAttribute((const void*)x.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)x.smem);
    if (e != cudaSuccess) return (int)e;
    return (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, (const void*)x.fn, tpb, x.smem);
}

static int upload(void** d, const void* h, size_t bytes) {
    cudaError_t e = cudaMalloc(d, bytes ? bytes : 4);
    if (e == cudaSuccess && bytes) e = cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice);
    return (int)e;
}

// One launch of LUT variant v: nblocks x tpb rows (arows), ng column groups; out and cycles copied back.
int f3_run_lut(int v, const uint32_t* arows, const uint32_t* tables, const uint32_t* wsc, const uint32_t* c0, int ng, int nblocks, int tpb,
               long long iters, uint32_t* out_h, long long* cycles_h, float* ms) {
    if (v < 0 || v >= NVAR) return -1;
    const Variant& x = VARIANTS[v];
    const size_t rows = (size_t)nblocks * tpb, nout = rows * x.c * (x.arm == A_EXACT ? 1 : 4);
    void *ar = nullptr, *tb = nullptr, *ws = nullptr, *cc = nullptr, *o = nullptr, *cy = nullptr;
    int rc = upload(&ar, arows, rows * AROW * 4);
    if (!rc) rc = upload(&tb, tables, (size_t)ng * x.tabw * 4);
    if (!rc) rc = upload(&ws, wsc, (size_t)ng * x.c * 4 * 4);
    if (!rc) rc = upload(&cc, c0, rows * x.c * 4);
    if (!rc) rc = (int)cudaMalloc(&o, nout * 4);
    if (!rc) rc = (int)cudaMalloc(&cy, nblocks * sizeof(long long));
    if (!rc) rc = (int)cudaFuncSetAttribute((const void*)x.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)x.smem);
    if (!rc) {
        cudaEvent_t s, t;
        cudaEventCreate(&s);
        cudaEventCreate(&t);
        cudaEventRecord(s);
        x.fn<<<nblocks, tpb, x.smem>>>((const uint32_t*)ar, (const uint32_t*)tb, (const uint32_t*)ws, (const uint32_t*)cc, ng, iters, 0u,
                                        (uint32_t*)o, (long long*)cy);
        cudaEventRecord(t);
        rc = (int)cudaEventSynchronize(t);
        if (!rc) rc = (int)cudaGetLastError();
        if (!rc) {
            cudaEventElapsedTime(ms, s, t);
            cudaMemcpy(out_h, o, nout * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(cycles_h, cy, nblocks * sizeof(long long), cudaMemcpyDeviceToHost);
        }
        cudaEventDestroy(s);
        cudaEventDestroy(t);
    }
    cudaFree(ar); cudaFree(tb); cudaFree(ws); cudaFree(cc); cudaFree(o); cudaFree(cy);
    return rc;
}

// One launch of the native atom: nblocks x tpb threads, NT tiles per warp over ntiles tiles of rows; out (ntiles x 128) if set.
int f3_run_native(const uint32_t* arows, long long nrows, const uint32_t* wcodes, const uint32_t* c0, int C, int rows_per_group, int ng,
                  int ntiles, int nblocks, int tpb, long long iters, uint32_t* out_h, long long* cycles_h, float* ms) {
    void *ar = nullptr, *wc = nullptr, *cc = nullptr, *o = nullptr, *cy = nullptr;
    int rc = upload(&ar, arows, (size_t)nrows * AROW * 4);
    if (!rc) rc = upload(&wc, wcodes, (size_t)ng * C * AROW * 4);
    if (!rc) rc = upload(&cc, c0, (size_t)nrows * C * 4);
    if (!rc && out_h) rc = (int)cudaMalloc(&o, (size_t)ntiles * 128 * 4);
    if (!rc) rc = (int)cudaMalloc(&cy, nblocks * sizeof(long long));
    if (!rc) {
        cudaEvent_t s, t;
        cudaEventCreate(&s);
        cudaEventCreate(&t);
        cudaEventRecord(s);
        native_kernel<<<nblocks, tpb>>>((const uint32_t*)ar, (const uint32_t*)wc, (const uint32_t*)cc, C, rows_per_group, ng, ntiles, iters,
                                        (uint32_t*)o, (long long*)cy);
        cudaEventRecord(t);
        rc = (int)cudaEventSynchronize(t);
        if (!rc) rc = (int)cudaGetLastError();
        if (!rc) {
            cudaEventElapsedTime(ms, s, t);
            if (out_h) cudaMemcpy(out_h, o, (size_t)ntiles * 128 * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(cycles_h, cy, nblocks * sizeof(long long), cudaMemcpyDeviceToHost);
        }
        cudaEventDestroy(s);
        cudaEventDestroy(t);
    }
    cudaFree(ar); cudaFree(wc); cudaFree(cc); cudaFree(o); cudaFree(cy);
    return rc;
}

int f3_nt() { return NT; }

// probe.Probe.identity's entry points (f1_rates.py's prelude has the same three)
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
