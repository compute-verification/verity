// Instruction-peak microbenchmark for the sm_120 block-scaled FP4 tensor-core instruction:
//
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//
// -- the instruction `benchmarks/tc_probe_fp4/mma_fp4.cu` pins (same inline PTX, same fragment layout), issued back to back
// from every warp of every SM with register-resident operands: NSETS random operand sets per lane (E2M1 codes, valid UE4M3
// scales) rotate through NACC independent accumulator chains, so the loop body is NSETS x NACC = 32 MMAs with no memory
// traffic and no dependent-issue stall.  What this measures is the sustained rate of the *instruction* (2 x 16 x 8 x 64 =
// 16,384 flop each) at the clocks the card settles to under this load -- an upper bound on any GEMM built from it, and the
// denominator `overhead.vs_native_peak` is conservative against.  Nothing here checks numerics (the probe does that): the
// host asserts the outputs are finite and bit-identical across launches, and cuobjdump names the SASS opcode.
//
// Build: nvcc -O2 -gencode=arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared -o libmma_peak_fp4_sm_120a.so mma_peak_fp4.cu
// (ptxas refuses .block_scale on plain sm_120, as for the probe).

#include <cuda_runtime.h>
#include <stdint.h>

#define NSETS 4
#define NACC 8
#define WORDS_PER_LANE (8 * NSETS)   // per operand set: a[4] b[2] sfa sfb

namespace {

__device__ __forceinline__ void mma_nvfp4_m16n8k64(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sfa,
                                                   uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
}

__global__ void nvfp4_peak_kernel(const uint32_t* __restrict__ frags, long long iters, float* __restrict__ sink) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t* f = frags + ((size_t)warp * 32 + lane) * WORDS_PER_LANE;
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
    // the iteration loop is NOT unrolled: the SASS then carries exactly NSETS x NACC MMAs, so cuobjdump's count is the
    // executed count per iteration (nvcc otherwise emits 16x/8x/4x/1x unrolled copies of the body: 29 x 32 = 928 MMAs)
#pragma unroll 1
    for (long long i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < NSETS; j++) {
#pragma unroll
            for (int c = 0; c < NACC; c++) {
                const int s = (c + j) & (NSETS - 1);        // compile-time after unrolling: no local-memory indexing
                mma_nvfp4_m16n8k64(d[c], a[s], b[s], sfa[s], sfb[s]);
            }
        }
    }
    float out[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < NACC; c++) { out[0] += d[c][0]; out[1] += d[c][1]; out[2] += d[c][2]; out[3] += d[c][3]; }
    float* o = sink + ((size_t)warp * 32 + lane) * 4;
    o[0] = out[0]; o[1] = out[1]; o[2] = out[2]; o[3] = out[3];
}

uint32_t* g_frags = nullptr;
float* g_sink = nullptr;
int g_nwarps = 0;

}  // namespace

extern "C" {

int nvfp4_peak_words_per_lane(void) { return WORDS_PER_LANE; }
int nvfp4_peak_mmas_per_iter(void) { return NSETS * NACC; }

// Device properties the host needs to size the grid: SM count and the max resident threads per SM.
int nvfp4_peak_device(int* sms, int* max_threads_per_sm, int* cc_major, int* cc_minor) {
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

// frags_host: nwarps x 32 lanes x WORDS_PER_LANE u32 (the per-lane register fragments, host order).
int nvfp4_peak_setup(const uint32_t* frags_host, int nwarps) {
    if (g_frags) { cudaFree(g_frags); g_frags = nullptr; }
    if (g_sink) { cudaFree(g_sink); g_sink = nullptr; }
    g_nwarps = nwarps;
    const size_t nbytes = (size_t)nwarps * 32 * WORDS_PER_LANE * sizeof(uint32_t);
    cudaError_t e = cudaMalloc(&g_frags, nbytes);
    if (e != cudaSuccess) return (int)e;
    e = cudaMemcpy(g_frags, frags_host, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return (int)e;
    e = cudaMalloc(&g_sink, (size_t)nwarps * 32 * 4 * sizeof(float));
    if (e != cudaSuccess) return (int)e;
    return 0;
}

// One launch of `iters` loop iterations on nblocks x warps_per_block warps (nblocks * warps_per_block == nwarps of setup);
// *ms_out = event-timed kernel milliseconds; sink_host (nwarps x 32 x 4 floats) receives the accumulator sums when not NULL.
int nvfp4_peak_run(int nblocks, int warps_per_block, long long iters, float* ms_out, float* sink_host) {
    if (!g_frags || nblocks * warps_per_block != g_nwarps) return -1;
    cudaEvent_t s, t;
    cudaEventCreate(&s);
    cudaEventCreate(&t);
    cudaEventRecord(s);
    nvfp4_peak_kernel<<<nblocks, 32 * warps_per_block>>>(g_frags, iters, g_sink);
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

int nvfp4_peak_teardown(void) {
    if (g_frags) cudaFree(g_frags);
    if (g_sink) cudaFree(g_sink);
    g_frags = nullptr;
    g_sink = nullptr;
    g_nwarps = 0;
    return 0;
}

}  // extern "C"
