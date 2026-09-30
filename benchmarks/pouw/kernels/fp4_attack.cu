// Instruction-peak microbenchmarks for the FP4 attacker (GPU 7 / GPU 4), sm_120a.  Three register-resident tensor-core
// loops, each issued back to back from every warp of every SM (the mma_peak_fp4.cu pattern: NSETS operand sets rotate
// through NACC independent accumulator chains, so the loop body is NSETS x NACC MMAs with no memory traffic).  What each
// measures is the sustained issue rate of one instruction -- an upper bound on any GEMM built from it, and the price the
// design's W1 uses in place of the whitepaper peak:
//
//   kind 0  DENSE   mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.f32.e2m1.e2m1.f32.ue4m3   (the honest rate)
//   kind 1  SPARSE  mma.sp::ordered_metadata ...kind::mxf4nvf4.block_scale... m16n8k128... e2m1.e2m1... (2:4 A: the 0.5x route)
//   kind 2  INT8    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32                                       (the exact-integer route)
//
// The DENSE and SPARSE loops share the E2M1/UE4M3 fragment layout the probe pins; SPARSE adds a 2:4 metadata word and the
// second half of B.  The host asserts the outputs are finite and bit-identical across two launches; cuobjdump names the
// SASS opcode (OMMA sparse vs dense, IMMA).  Nothing here checks the arithmetic against a reference (fp4_census / tc_probe
// do that); this is a rate and a SASS gate.
//
// Build: nvcc -O2 -gencode=arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared -o libfp4_attack_sm_120a.so fp4_attack.cu
// (ptxas refuses .block_scale / .kind::mxf4nvf4 on plain sm_120, as for the probe and the peak kernel).

#include <cuda_runtime.h>
#include <stdint.h>

#define NSETS 4
#define NACC 8

// words per operand set, per lane, for each kind (host lays the fragments out the same way):
#define WORDS_DENSE 8    // a[4] b[2] sfa sfb
#define WORDS_SPARSE 12  // a[4] b[4] meta sfa sfb pad  (padded to keep the set 4-aligned)
#define WORDS_INT8 6     // a[4] b[2]

namespace {

// ---- the three instructions -----------------------------------------------------------------------------------------------

__device__ __forceinline__ void mma_dense(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
}

__device__ __forceinline__ void mma_sparse(float d[4], const uint32_t a[4], const uint32_t b[4], uint32_t meta,
                                           uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0, {%13}, {0, 0}, {%14}, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(meta), "r"(sfa), "r"(sfb));
}

__device__ __forceinline__ void mma_int8(int32_t d[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- the loops ------------------------------------------------------------------------------------------------------------

__global__ void dense_kernel(const uint32_t* __restrict__ frags, long long iters, float* __restrict__ sink) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t* f = frags + ((size_t)warp * 32 + lane) * (NSETS * WORDS_DENSE);
    uint32_t a[NSETS][4], b[NSETS][2], sfa[NSETS], sfb[NSETS];
#pragma unroll
    for (int s = 0; s < NSETS; s++) {
        a[s][0] = f[8 * s + 0]; a[s][1] = f[8 * s + 1]; a[s][2] = f[8 * s + 2]; a[s][3] = f[8 * s + 3];
        b[s][0] = f[8 * s + 4]; b[s][1] = f[8 * s + 5];
        sfa[s] = f[8 * s + 6]; sfb[s] = f[8 * s + 7];
    }
    float d[NACC][4];
#pragma unroll
    for (int c = 0; c < NACC; c++) { d[c][0] = 0.f; d[c][1] = 0.f; d[c][2] = 0.f; d[c][3] = 0.f; }
#pragma unroll 1
    for (long long i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < NSETS; j++) {
#pragma unroll
            for (int c = 0; c < NACC; c++) {
                const int s = (c + j) & (NSETS - 1);
                mma_dense(d[c], a[s], b[s], sfa[s], sfb[s]);
            }
        }
    }
    float out[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < NACC; c++) { out[0] += d[c][0]; out[1] += d[c][1]; out[2] += d[c][2]; out[3] += d[c][3]; }
    float* o = sink + ((size_t)warp * 32 + lane) * 4;
    o[0] = out[0]; o[1] = out[1]; o[2] = out[2]; o[3] = out[3];
}

__global__ void sparse_kernel(const uint32_t* __restrict__ frags, long long iters, float* __restrict__ sink) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t* f = frags + ((size_t)warp * 32 + lane) * (NSETS * WORDS_SPARSE);
    uint32_t a[NSETS][4], b[NSETS][4], meta[NSETS], sfa[NSETS], sfb[NSETS];
#pragma unroll
    for (int s = 0; s < NSETS; s++) {
        a[s][0] = f[12 * s + 0]; a[s][1] = f[12 * s + 1]; a[s][2] = f[12 * s + 2]; a[s][3] = f[12 * s + 3];
        b[s][0] = f[12 * s + 4]; b[s][1] = f[12 * s + 5]; b[s][2] = f[12 * s + 6]; b[s][3] = f[12 * s + 7];
        meta[s] = f[12 * s + 8]; sfa[s] = f[12 * s + 9]; sfb[s] = f[12 * s + 10];
    }
    float d[NACC][4];
#pragma unroll
    for (int c = 0; c < NACC; c++) { d[c][0] = 0.f; d[c][1] = 0.f; d[c][2] = 0.f; d[c][3] = 0.f; }
#pragma unroll 1
    for (long long i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < NSETS; j++) {
#pragma unroll
            for (int c = 0; c < NACC; c++) {
                const int s = (c + j) & (NSETS - 1);
                mma_sparse(d[c], a[s], b[s], meta[s], sfa[s], sfb[s]);
            }
        }
    }
    float out[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < NACC; c++) { out[0] += d[c][0]; out[1] += d[c][1]; out[2] += d[c][2]; out[3] += d[c][3]; }
    float* o = sink + ((size_t)warp * 32 + lane) * 4;
    o[0] = out[0]; o[1] = out[1]; o[2] = out[2]; o[3] = out[3];
}

__global__ void int8_kernel(const uint32_t* __restrict__ frags, long long iters, float* __restrict__ sink) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t* f = frags + ((size_t)warp * 32 + lane) * (NSETS * WORDS_INT8);
    uint32_t a[NSETS][4], b[NSETS][2];
#pragma unroll
    for (int s = 0; s < NSETS; s++) {
        a[s][0] = f[6 * s + 0]; a[s][1] = f[6 * s + 1]; a[s][2] = f[6 * s + 2]; a[s][3] = f[6 * s + 3];
        b[s][0] = f[6 * s + 4]; b[s][1] = f[6 * s + 5];
    }
    int32_t d[NACC][4];
#pragma unroll
    for (int c = 0; c < NACC; c++) { d[c][0] = 0; d[c][1] = 0; d[c][2] = 0; d[c][3] = 0; }
#pragma unroll 1
    for (long long i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < NSETS; j++) {
#pragma unroll
            for (int c = 0; c < NACC; c++) {
                const int s = (c + j) & (NSETS - 1);
                mma_int8(d[c], a[s], b[s]);
            }
        }
    }
    int32_t out[4] = {0, 0, 0, 0};
#pragma unroll
    for (int c = 0; c < NACC; c++) { out[0] += d[c][0]; out[1] += d[c][1]; out[2] += d[c][2]; out[3] += d[c][3]; }
    float* o = sink + ((size_t)warp * 32 + lane) * 4;
    o[0] = (float)out[0]; o[1] = (float)out[1]; o[2] = (float)out[2]; o[3] = (float)out[3];
}

uint32_t* g_frags = nullptr;
float* g_sink = nullptr;
int g_nwarps = 0;

}  // namespace

extern "C" {

int fp4_attack_words_per_lane(int kind) {
    switch (kind) {
        case 0: return NSETS * WORDS_DENSE;
        case 1: return NSETS * WORDS_SPARSE;
        case 2: return NSETS * WORDS_INT8;
        default: return -1;
    }
}
int fp4_attack_mmas_per_iter(void) { return NSETS * NACC; }
// flop per MMA: dense 2*16*8*64; sparse 2*16*8*128 (logical k, half the MACs real under 2:4); int8 2*16*8*32.
long long fp4_attack_flop_per_mma(int kind) {
    switch (kind) {
        case 0: return 2LL * 16 * 8 * 64;
        case 1: return 2LL * 16 * 8 * 128;
        case 2: return 2LL * 16 * 8 * 32;
        default: return -1;
    }
}

int fp4_attack_device(int* sms, int* max_threads_per_sm, int* cc_major, int* cc_minor) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return (int)e;
    cudaDeviceProp p;
    e = cudaGetDeviceProperties(&p, dev);
    if (e != cudaSuccess) return (int)e;
    *sms = p.multiProcessorCount;
    *max_threads_per_sm = p.maxThreadsPerMultiProcessor;
    *cc_major = p.major;
    *cc_minor = p.minor;
    return 0;
}

// frags_host: nwarps x 32 lanes x words_per_lane(kind) u32.
int fp4_attack_setup(const uint32_t* frags_host, int nwarps, int words_per_lane) {
    if (g_frags) { cudaFree(g_frags); g_frags = nullptr; }
    if (g_sink) { cudaFree(g_sink); g_sink = nullptr; }
    g_nwarps = nwarps;
    const size_t nbytes = (size_t)nwarps * 32 * words_per_lane * sizeof(uint32_t);
    cudaError_t e = cudaMalloc(&g_frags, nbytes);
    if (e != cudaSuccess) return (int)e;
    e = cudaMemcpy(g_frags, frags_host, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return (int)e;
    e = cudaMalloc(&g_sink, (size_t)nwarps * 32 * 4 * sizeof(float));
    if (e != cudaSuccess) return (int)e;
    return 0;
}

int fp4_attack_run(int kind, int nblocks, int warps_per_block, long long iters, float* ms_out, float* sink_host) {
    if (!g_frags || nblocks * warps_per_block != g_nwarps) return -1;
    cudaEvent_t s, t;
    cudaEventCreate(&s);
    cudaEventCreate(&t);
    cudaEventRecord(s);
    switch (kind) {
        case 0: dense_kernel<<<nblocks, 32 * warps_per_block>>>(g_frags, iters, g_sink); break;
        case 1: sparse_kernel<<<nblocks, 32 * warps_per_block>>>(g_frags, iters, g_sink); break;
        case 2: int8_kernel<<<nblocks, 32 * warps_per_block>>>(g_frags, iters, g_sink); break;
        default: return -2;
    }
    cudaEventRecord(t);
    cudaError_t e = cudaEventSynchronize(t);
    if (e != cudaSuccess) return (int)e;
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    cudaEventElapsedTime(ms_out, s, t);
    cudaEventDestroy(s);
    cudaEventDestroy(t);
    if (sink_host) {
        e = cudaMemcpy(sink_host, g_sink, (size_t)g_nwarps * 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) return (int)e;
    }
    return 0;
}

int fp4_attack_teardown(void) {
    if (g_frags) cudaFree(g_frags);
    if (g_sink) cudaFree(g_sink);
    g_frags = nullptr;
    g_sink = nullptr;
    g_nwarps = 0;
    return 0;
}

}  // extern "C"
