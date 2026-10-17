// Is the FP32 FMA pipe free beside the integer mix of a squaring at ra-codec-theory's pipe floor?
//
// `frontier.pipe_clocks` prices the adversary's squaring with its products as IMAD.WIDE on the FMA pipe and its adds
// split between IADD3 on the ALU pipe and IMAD on the FMA pipe's spare rate. At 1,536 bits (Karatsuba depth 3) that
// is 615 IMAD.WIDE, 257 IMAD and 1,528 IADD3 a squaring, 23.9 SM clocks against 18.75 at the issue rate. This kernel
// issues those three in that ratio (12 : 5 : 30 an iteration, independent chains, full occupancy) with F FFMAs added,
// and reads each block's clock64. If the iteration's clocks don't grow with F, FFMAs ride free beside the floor.
//
//   pipemix gate                 every chain's result against the host's replay (small iteration count)
//   pipemix time W M A F ITERS   one config: {"w","m","a","f","clocks_per_iter","lanes_per_clk":{...}} as JSON
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

constexpr int CH = 4;
constexpr int THREADS = 256;

struct State {
    uint64_t w[CH];
    uint32_t m[CH], a[CH];
    float f[CH];
};

__host__ __device__ inline void init(State &s, uint32_t tid, uint32_t seed) {
    for (int c = 0; c < CH; c++) {
        uint32_t x = (tid * 0x9E3779B9u) ^ (seed + 0x85EBCA6Bu * (c + 1));
        s.w[c] = x | (uint64_t)(x * 0xC2B2AE35u) << 32;
        s.m[c] = x * 0x27D4EB2Fu;
        s.a[c] = x ^ 0x165667B1u;
        s.f[c] = (float)(x & 0xFFFF) * (1.0f / 65536.0f);
    }
}

// One iteration: W IMAD.WIDE, M IMAD, A IADD3, F FFMA, round-robin over CH chains each.
template <int W, int M, int A, int F>
__host__ __device__ inline void step(State &s, uint32_t k1, uint32_t k2, float fm, float fc) {
#pragma unroll
    for (int i = 0; i < W; i++) {
#ifdef __CUDA_ARCH__
        asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(s.w[i % CH]) : "r"(s.m[(i + 2) % CH]), "r"(k2 + i));
#else
        s.w[i % CH] += (uint64_t)s.m[(i + 2) % CH] * (k2 + i);
#endif
    }
#pragma unroll
    for (int i = 0; i < M; i++) {
#ifdef __CUDA_ARCH__
        asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(s.m[i % CH]) : "r"(s.a[(i + 3) % CH]), "r"(k1 + i));
#else
        s.m[i % CH] += s.a[(i + 3) % CH] * (k1 + i);
#endif
    }
#pragma unroll
    for (int i = 0; i < A; i++) {
#ifdef __CUDA_ARCH__
        asm volatile("add.u32 %0, %0, %1;" : "+r"(s.a[i % CH]) : "r"(s.m[(i + 1) % CH]));
#else
        s.a[i % CH] += s.m[(i + 1) % CH];
#endif
    }
#pragma unroll
    for (int i = 0; i < F; i++) {
#ifdef __CUDA_ARCH__
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(s.f[i % CH]) : "f"(fm), "f"(fc));
#else
        s.f[i % CH] = std::fmaf(s.f[i % CH], fm, fc);
#endif
    }
}

__host__ __device__ inline uint64_t digest(const State &s) {
    uint64_t h = 0;
    for (int c = 0; c < CH; c++) {
        uint32_t fb;
        memcpy(&fb, &s.f[c], 4);
        h = h * 0x100000001B3ull ^ s.w[c];
        h = h * 0x100000001B3ull ^ s.m[c];
        h = h * 0x100000001B3ull ^ s.a[c];
        h = h * 0x100000001B3ull ^ fb;
    }
    return h;
}

template <int W, int M, int A, int F>
__global__ void __launch_bounds__(THREADS) mix(uint64_t iters, uint32_t seed, uint32_t k1, uint32_t k2, float fm,
                                               float fc, uint64_t *out, long long *clocks) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    State s;
    init(s, tid, seed);
    __syncthreads();
    long long t0 = clock64();
    for (uint64_t it = 0; it < iters; it++) step<W, M, A, F>(s, k1, k2, fm, fc);
    __syncthreads();
    long long t1 = clock64();
    out[tid] = digest(s);
    if (threadIdx.x == 0) clocks[blockIdx.x] = t1 - t0;
}

template <int W, int M, int A, int F>
void host_mix(uint64_t iters, uint32_t seed, uint32_t k1, uint32_t k2, float fm, float fc, uint32_t tid,
              uint64_t *res) {
    State s;
    init(s, tid, seed);
    for (uint64_t it = 0; it < iters; it++) step<W, M, A, F>(s, k1, k2, fm, fc);
    *res = digest(s);
}

struct Result { double clocks_per_iter; int blocks; bool ok; };

template <int W, int M, int A, int F>
Result run(uint64_t iters, bool gate) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    int per_sm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, mix<W, M, A, F>, THREADS, 0));
    int blocks = per_sm * p.multiProcessorCount;
    size_t n = (size_t)blocks * THREADS;
    uint64_t *d_out;
    long long *d_clk;
    CK(cudaMalloc(&d_out, n * 8));
    CK(cudaMalloc(&d_clk, blocks * 8));
    const uint32_t seed = 12345, k1 = 0x6A09E667u, k2 = 0xBB67AE85u;
    const float fm = 0.99999988f, fc = 1.0e-3f;
    mix<W, M, A, F><<<blocks, THREADS>>>(iters, seed, k1, k2, fm, fc, d_out, d_clk);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<uint64_t> out(n);
    std::vector<long long> clk(blocks);
    CK(cudaMemcpy(out.data(), d_out, n * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(clk.data(), d_clk, blocks * 8, cudaMemcpyDeviceToHost));
    bool ok = true;
    if (gate) {
        for (size_t t = 0; t < n; t += (t < 4096 ? 1 : 997)) {
            uint64_t h;
            host_mix<W, M, A, F>(iters, seed, k1, k2, fm, fc, (uint32_t)t, &h);
            if (h != out[t]) { ok = false; fprintf(stderr, "mismatch thread %zu\n", t); break; }
        }
    }
    double mean = 0;
    for (long long c : clk) mean += (double)c;
    mean /= blocks;
    CK(cudaFree(d_out));
    CK(cudaFree(d_clk));
    return {mean / (double)iters, blocks / p.multiProcessorCount, ok};
}

template <int W, int M, int A, int F>
void report(uint64_t iters) {
    run<W, M, A, F>(iters / 8, false);
    Result r = run<W, M, A, F>(iters, false);
    double lanes = (double)r.blocks * THREADS / r.clocks_per_iter;  // threads an SM retiring one iteration a clock
    printf("{\"w\": %d, \"m\": %d, \"a\": %d, \"f\": %d, \"blocks_per_sm\": %d, \"clocks_per_iter\": %.3f, "
           "\"lanes_per_clk\": {\"imad_wide\": %.2f, \"imad\": %.2f, \"iadd\": %.2f, \"ffma\": %.2f, "
           "\"all\": %.2f}}\n",
           W, M, A, F, r.blocks, r.clocks_per_iter, W * lanes, M * lanes, A * lanes, F * lanes,
           (W + M + A + F) * lanes);
}

#define CONFIGS(X) \
    X(12, 5, 30, 0) X(12, 5, 30, 4) X(12, 5, 30, 8) X(12, 5, 30, 12) X(12, 5, 30, 16) X(12, 5, 30, 24) \
    X(0, 0, 0, 16) X(0, 0, 0, 32) X(16, 0, 0, 0) X(16, 0, 0, 16) X(0, 16, 0, 0) X(0, 16, 0, 16) \
    X(0, 0, 32, 0) X(0, 0, 32, 16) X(12, 5, 0, 0) X(0, 0, 30, 0)

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "gate")) {
        bool all = true;
#define GATE(W, M, A, F) { Result r = run<W, M, A, F>(64, true); \
        printf("{\"gate\": [%d, %d, %d, %d], \"ok\": %s}\n", W, M, A, F, r.ok ? "true" : "false"); all &= r.ok; }
        CONFIGS(GATE)
        return all ? 0 : 1;
    }
    if (argc >= 7 && !strcmp(argv[1], "time")) {
        int w = atoi(argv[2]), m = atoi(argv[3]), a = atoi(argv[4]), f = atoi(argv[5]);
        uint64_t iters = strtoull(argv[6], nullptr, 10);
#define TIME(W, M, A, F) if (w == W && m == M && a == A && f == F) { report<W, M, A, F>(iters); return 0; }
        CONFIGS(TIME)
        fprintf(stderr, "no such config\n");
        return 1;
    }
    fprintf(stderr, "usage: pipemix gate | time W M A F ITERS\n");
    return 1;
}
