// W1 price microbenchmarks on sm_120a (lane pouw rtx-pro GPU 4): the sustained issue rate and the dependent-chain latency of each
// FP4-path mma.sync row of rows_fp4.cuh, with register-resident operands (tools/native_peak/mma_peak_fp4.cu's pattern).
//
//   price_tput_kernel<KIND, MODE>  NSETS operand sets rotate through NACC independent accumulators: the `#pragma unroll 1` body is
//                                  NSETS x NACC MMAs and nothing else, issued from every warp of every SM.
//                                  MODE 0: the scale registers stay as loaded (each set has its own, so consecutive MMAs read
//                                  different scales); MODE 1: every set's sfa / sfb is rotated a byte (one PRMT each) per iteration,
//                                  so the scale values the hardware reads change as a GEMM's do.  Unscaled kinds have MODE 0 only.
//   price_latency_kernel<KIND>     one warp, one accumulator: LAT_UNROLL dependent MMAs per iteration, timed with clock64.
//
// Thread 0 of each block records its clock64 span, so the host has the SM clock of every launch (cycles / event time) and can
// state the rate per SM per clock, which does not depend on the clock.  Nothing here checks the arithmetic (probe.py does):
// the gates are finite, bit-identical outputs across two launches and the SASS count of the expected opcode per kernel.
//
// Build: nvcc -O2 -gencode=arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared -o libw1_price_sm_120a.so w1_price.cu

#include <cuda_runtime.h>
#include <stdint.h>

#include "rows_fp4.cuh"

#define NSETS 4
#define NACC 8
#define WORDS 8  // per operand set per lane: a[4] b[2] sfa sfb
#define LAT_UNROLL 8

namespace {

template <int KIND, int MODE>
__global__ void price_tput_kernel(const uint32_t* __restrict__ frags, long long iters, float* __restrict__ sink,
                                  long long* __restrict__ cycles) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t* f = frags + ((size_t)warp * 32 + lane) * (NSETS * WORDS);
    uint32_t a[NSETS][4], b[NSETS][2], sfa[NSETS], sfb[NSETS];
#pragma unroll
    for (int s = 0; s < NSETS; s++) {
        a[s][0] = f[WORDS * s + 0]; a[s][1] = f[WORDS * s + 1]; a[s][2] = f[WORDS * s + 2]; a[s][3] = f[WORDS * s + 3];
        b[s][0] = f[WORDS * s + 4]; b[s][1] = f[WORDS * s + 5];
        sfa[s] = f[WORDS * s + 6]; sfb[s] = f[WORDS * s + 7];
    }
    float d[NACC][4];
#pragma unroll
    for (int c = 0; c < NACC; c++) { d[c][0] = 0.f; d[c][1] = 0.f; d[c][2] = 0.f; d[c][3] = 0.f; }
    __syncthreads();
    const long long t0 = clock64();
#pragma unroll 1
    for (long long i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < NSETS; j++) {
#pragma unroll
            for (int c = 0; c < NACC; c++) {
                const int s = (c + j) & (NSETS - 1);
                row_mma<KIND>(d[c], a[s], b[s], d[c], sfa[s], sfb[s]);
            }
        }
        if constexpr (MODE == 1) {
#pragma unroll
            for (int s = 0; s < NSETS; s++) { sfa[s] = __byte_perm(sfa[s], 0, 0x0321); sfb[s] = __byte_perm(sfb[s], 0, 0x0321); }
        }
    }
    __syncthreads();
    const long long t1 = clock64();
    float out[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < NACC; c++) { out[0] += d[c][0]; out[1] += d[c][1]; out[2] += d[c][2]; out[3] += d[c][3]; }
    float* o = sink + ((size_t)warp * 32 + lane) * 4;
    o[0] = out[0]; o[1] = out[1]; o[2] = out[2]; o[3] = out[3];
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

template <int KIND>
__global__ void price_latency_kernel(const uint32_t* __restrict__ frags, long long iters, float* __restrict__ sink,
                                     long long* __restrict__ cycles) {
    const int lane = threadIdx.x & 31;
    const uint32_t* f = frags + (size_t)lane * (NSETS * WORDS);
    const uint32_t a[4] = {f[0], f[1], f[2], f[3]}, b[2] = {f[4], f[5]};
    const uint32_t sfa = f[6], sfb = f[7];
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    __syncwarp();
    const long long t0 = clock64();
#pragma unroll 1
    for (long long i = 0; i < iters; i++) {
#pragma unroll
        for (int u = 0; u < LAT_UNROLL; u++) row_mma<KIND>(d, a, b, d, sfa, sfb);
    }
    __syncwarp();
    const long long t1 = clock64();
    float* o = sink + (size_t)lane * 4;
    o[0] = d[0]; o[1] = d[1]; o[2] = d[2]; o[3] = d[3];
    if (lane == 0) cycles[0] = t1 - t0;
}

uint32_t* g_frags = nullptr;
float* g_sink = nullptr;
long long* g_cycles = nullptr;
int g_nwarps = 0, g_nslots = 0, g_nblocks_max = 0;
size_t g_slot_words = 0;

typedef void (*TputFn)(const uint32_t*, long long, float*, long long*);

#define TPUT(K, M) if (kind == K && mode == M) return price_tput_kernel<K, M>;
TputFn tput_fn(int kind, int mode) {
    TPUT(R_NVF4, 0) TPUT(R_NVF4, 1) TPUT(R_MXF4, 0) TPUT(R_MXF4, 1)
    TPUT(R_F8F6F4_E2M1, 0) TPUT(R_F8F6F4_E2M1_E4M3, 0) TPUT(R_F8F6F4_E4M3, 0)
    TPUT(R_MXF8F6F4_E2M1, 0) TPUT(R_MXF8F6F4_E2M1, 1) TPUT(R_MXF8F6F4_E2M1_E4M3, 0) TPUT(R_MXF8F6F4_E2M1_E4M3, 1)
    return nullptr;
}
#undef TPUT

#define LAT(K) if (kind == K) return price_latency_kernel<K>;
TputFn latency_fn(int kind) {
    LAT(R_NVF4) LAT(R_MXF4) LAT(R_F8F6F4_E2M1) LAT(R_F8F6F4_E2M1_E4M3) LAT(R_F8F6F4_E4M3) LAT(R_MXF8F6F4_E2M1)
    LAT(R_MXF8F6F4_E2M1_E4M3)
    return nullptr;
}
#undef LAT

}  // namespace

extern "C" {

int w1_words_per_lane(void) { return NSETS * WORDS; }
int w1_mmas_per_iter(void) { return NSETS * NACC; }
int w1_latency_unroll(void) { return LAT_UNROLL; }

int w1_device(int* sms, int* max_threads_per_sm, int* major, int* minor) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    *sms = prop.multiProcessorCount;
    *max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    *major = prop.major;
    *minor = prop.minor;
    return 0;
}

int device_identity(int* major, int* minor, char* name, int name_len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    *major = prop.major;
    *minor = prop.minor;
    int i = 0;
    for (; i < name_len - 1 && prop.name[i]; ++i) name[i] = prop.name[i];
    name[i] = 0;
    return 0;
}

int device_uuid(char* out, int out_len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    if (out_len < 41) return (int)cudaErrorInvalidValue;
    static const char* hex = "0123456789abcdef";
    const unsigned char* u = (const unsigned char*)prop.uuid.bytes;
    int p = 0;
    out[p++] = 'G'; out[p++] = 'P'; out[p++] = 'U'; out[p++] = '-';
    for (int i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out[p++] = '-';
        out[p++] = hex[u[i] >> 4];
        out[p++] = hex[u[i] & 15];
    }
    out[p] = 0;
    return 0;
}

int has_mxf4(void) { return 1; }

// frags: u32[nslots][nwarps * 32][NSETS][WORDS], one slot per row: the operand sets of every lane of every warp of one launch,
// uploaded once so that no allocation or copy separates two timed launches.
int w1_setup(const uint32_t* frags, int nwarps, int nslots, int nblocks_max) {
    if (g_frags) { cudaFree(g_frags); cudaFree(g_sink); cudaFree(g_cycles); g_frags = nullptr; }
    g_slot_words = (size_t)nwarps * 32 * NSETS * WORDS;
    const size_t words = g_slot_words * nslots;
    cudaError_t err = cudaMalloc(&g_frags, words * 4);
    if (err == cudaSuccess) err = cudaMalloc(&g_sink, (size_t)nwarps * 32 * 4 * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(&g_cycles, (size_t)(nblocks_max > 0 ? nblocks_max : 1) * sizeof(long long));
    if (err == cudaSuccess) err = cudaMemcpy(g_frags, frags, words * 4, cudaMemcpyHostToDevice);
    g_nwarps = nwarps;
    g_nslots = nslots;
    g_nblocks_max = nblocks_max;
    return (int)err;
}

// The resident blocks per SM of price_tput_kernel<kind, mode> at warps_per_block: a launch of more than sms x this is not one
// wave, and thread 0's clock64 span would then undercount the launch.
int w1_occupancy(int kind, int mode, int warps_per_block, int* blocks_per_sm) {
    TputFn fn = tput_fn(kind, mode);
    if (!fn) return -1;
    return (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, fn, warps_per_block * 32, 0);
}

// One event-timed launch of price_tput_kernel<kind, mode> on slot's operands over nblocks x warps_per_block warps.  sink_out (nwarps*32*4 f32) and
// cycles_out (nblocks) are copied back when non-null.  Returns 0, a CUDA error, or -1 for an unknown (kind, mode) / grid.
int w1_run(int kind, int mode, int slot, int nblocks, int warps_per_block, long long iters, float* ms, float* sink_out,
           long long* cycles_out) {
    TputFn fn = tput_fn(kind, mode);
    if (!fn || !g_frags || slot < 0 || slot >= g_nslots || nblocks * warps_per_block > g_nwarps || nblocks > g_nblocks_max) return -1;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    fn<<<nblocks, warps_per_block * 32>>>(g_frags + slot * g_slot_words, iters, g_sink, g_cycles);
    cudaEventRecord(e1);
    cudaError_t err = cudaEventSynchronize(e1);
    if (err == cudaSuccess) err = cudaGetLastError();
    cudaEventElapsedTime(ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (err == cudaSuccess && sink_out)
        err = cudaMemcpy(sink_out, g_sink, (size_t)nblocks * warps_per_block * 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess && cycles_out) err = cudaMemcpy(cycles_out, g_cycles, (size_t)nblocks * sizeof(long long), cudaMemcpyDeviceToHost);
    return (int)err;
}

// One warp's dependent chain of iters x LAT_UNROLL MMAs on operand set 0 of slot's warp 0: cycles_out[0] = its clock64 span.
int w1_latency(int kind, int slot, long long iters, float* sink_out, long long* cycles_out) {
    TputFn fn = latency_fn(kind);
    if (!fn || !g_frags || slot < 0 || slot >= g_nslots) return -1;
    fn<<<1, 32>>>(g_frags + slot * g_slot_words, iters, g_sink, g_cycles);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err == cudaSuccess && sink_out) err = cudaMemcpy(sink_out, g_sink, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess && cycles_out) err = cudaMemcpy(cycles_out, g_cycles, sizeof(long long), cudaMemcpyDeviceToHost);
    return (int)err;
}

void w1_teardown(void) {
    if (g_frags) { cudaFree(g_frags); cudaFree(g_sink); cudaFree(g_cycles); }
    g_frags = nullptr;
    g_sink = nullptr;
    g_cycles = nullptr;
}

}  // extern "C"
