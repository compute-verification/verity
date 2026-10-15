// What one CUDA-core instruction buys on sm_120 (CMP-8, Daniel's #10): the throughput of each instruction a lookup-table or
// bit-trick op could be built from, in thread instructions per SM per clock, alone and in pairs (whether two rates add,
// i.e. whether the instructions draw on separate pipes or on one budget).
//
//   ffma   FFMA                                    the slot: 128 lanes/SM/clk
//   iadd3  IADD3, lop3 LOP3 (32 one-bit lanes)     the integer ALU
//   imad   IMAD (a 32-bit MAC)
//   dp4a   IDP4A, s8 x s8 dot-4 into s32: 4 exact MACs
//   hfma2  HFMA2 f16x2, bfma2 HFMA2.BF16 bf16x2: 2 MACs (with their own rounding)
//   prmt   PRMT, a byte table of 8 entries held in two registers
//   cvt8   cvt.rn.f16x2.e4m3x2, the decode an E4M3 MAC on the CUDA cores needs
//   lds    LDS from a 16 KiB shared table at a data-dependent index (random banks); ldsc each lane in its own bank
//
// Kernel <A, B>: one wave of --blocks 8-warp blocks an SM; every thread runs 8 chains, A on chains 0-3 and B on 4-7 (A on all
// 8 when A == B), 16 steps a chain per outer round. Integer chains are coupled within their half (each step reads two other
// chains) so no step can be folded into the next; floating-point and load chains are self-chains. A pair in one instruction
// stream ("a+b") runs at the slower op's pace, since a warp issues in order. Kernel <A, B, split> ("a|b") gives warps 0-3 of
// each block op A on all 8 chains and warps 4-7 op B, and times each half by its own last warp: whether the two run beside
// each other ("a|none" is A alone on half the warps). Gates (exit 3, before any rate): two launches agree, and thread 0's
// chains equal a host recomputation for every op but hfma2, bfma2, prmt and cvt8. The job writes the SASS opcode count of
// each kernel beside the rates, which is what says one source op is one instruction.
//
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o isa_rates isa_rates.cu
//   ./isa_rates [--ms 60] [--reps 5] [--blocks 2] [--only KERNEL,KERNEL]
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            exit(2);                                                                       \
        }                                                                                  \
    } while (0)

constexpr int THREADS = 256, CH = 8, STEPS = 16, TBL = 4096, IN_WORDS = 1024;

enum Op : int { FFMA, IADD3, LOP3, IMAD, DP4A, HFMA2, BFMA2, PRMT, CVT8, LDS, LDSC, NONE };
static const char* OPS[] = {"ffma", "iadd3", "lop3", "imad", "dp4a", "hfma2", "bfma2", "prmt", "cvt8", "lds", "ldsc", "none"};
__host__ __device__ constexpr bool host_checked(int op) { return op != HFMA2 && op != BFMA2 && op != PRMT && op != CVT8; }
__host__ __device__ constexpr bool tabled(int op) { return op == LDS || op == LDSC; }

__host__ __device__ inline uint32_t dp4a_s8(uint32_t a, uint32_t b, uint32_t c) {
#ifdef __CUDA_ARCH__
    return (uint32_t)__dp4a((int)a, (int)b, (int)c);
#else
    for (int i = 0; i < 4; i++) c += (uint32_t)((int)(int8_t)(a >> 8 * i) * (int)(int8_t)(b >> 8 * i));
    return c;
#endif
}

template <int OP>
__host__ __device__ __forceinline__ void step(uint32_t (&x)[CH], float (&f)[CH], const uint32_t* tbl, int m) {
    const int b = m & ~3, m1 = b + ((m + 1) & 3), m3 = b + ((m + 3) & 3);
    if constexpr (OP == FFMA) f[m] = fmaf(f[m], 0.999f, 0.5f);
    if constexpr (OP == IADD3) x[m] = x[m] + x[m1] + x[m3];
    if constexpr (OP == LOP3) x[m] = x[m] ^ (x[m1] & ~x[m3]);
    if constexpr (OP == IMAD) x[m] = x[m] * x[m1] + x[m3];
    if constexpr (OP == DP4A) x[m] = dp4a_s8(x[m1], x[m3], x[m]);
    if constexpr (tabled(OP)) x[m] = *reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(tbl) + x[m]);
#ifdef __CUDA_ARCH__
    if constexpr (OP == HFMA2) asm("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(x[m]) : "r"(0x38003800u), "r"(0x34003400u));
    if constexpr (OP == BFMA2) asm("fma.rn.bf16x2 %0, %0, %1, %2;" : "+r"(x[m]) : "r"(0x3f003f00u), "r"(0x3e803e80u));
    if constexpr (OP == PRMT) asm("prmt.b32 %0, %1, %2, %0;" : "+r"(x[m]) : "r"(x[m1]), "r"(x[m3]));
    if constexpr (OP == CVT8) asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(x[m]) : "h"((unsigned short)x[m]));
#endif
}

__host__ __device__ inline uint32_t table_word(bool own_bank, const uint32_t* in, int i) {
    const uint32_t h = in[i % IN_WORDS] ^ (0x9e3779b9u * (uint32_t)(i + 1));
    return own_bank ? ((h & (TBL / 32 - 1)) * 32 + (uint32_t)(i & 31)) << 2 : (h & (TBL - 1)) << 2;
}

__host__ __device__ inline uint32_t start_word(int op, const uint32_t* in, int tid, int m) {
    const uint32_t w = in[(tid * CH + m) % IN_WORDS];
    if (op == LDS) return (w & (TBL - 1)) << 2;
    if (op == LDSC) return ((w & (TBL / 32 - 1)) * 32 + (uint32_t)(tid & 31)) << 2;
    return w | 1u;
}

template <int A, int B>
__host__ __device__ __forceinline__ void init(uint32_t (&x)[CH], float (&f)[CH], const uint32_t* in, int tid) {
#pragma unroll
    for (int m = 0; m < CH; m++) {
        x[m] = start_word(m < 4 ? A : B, in, tid, m);
        f[m] = (float)((tid + m) % 7);
    }
}

template <int A, int B>
__device__ __forceinline__ void run(uint32_t (&x)[CH], float (&f)[CH], const uint32_t* tbl, int outer) {
#pragma unroll 1
    for (int o = 0; o < outer; o++) {
#pragma unroll
        for (int s = 0; s < STEPS; s++) {
#pragma unroll
            for (int m = 0; m < CH; m++) {
                if (m < 4) step<A>(x, f, tbl, m);
                else step<B>(x, f, tbl, m);
            }
        }
    }
}

constexpr int WARPS = THREADS / 32, CLK = 1 + WARPS;

template <int A, int B, bool SPLIT>
__global__ void __launch_bounds__(THREADS, 4)
    rates(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, long long* __restrict__ clk, int outer) {
    constexpr bool TABLE = tabled(A) || tabled(B);
    __shared__ uint32_t tbl[TABLE ? TBL : 1];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x, warp = threadIdx.x / 32;
    const bool second = SPLIT && warp >= WARPS / 2;
    if (TABLE)
        for (int i = threadIdx.x; i < TBL; i += blockDim.x) tbl[i] = table_word(A == LDSC || B == LDSC, in, i);
    uint32_t x[CH];
    float f[CH];
    if (!SPLIT) init<A, B>(x, f, in, tid);
    else if (!second) init<A, A>(x, f, in, tid);
    else init<B, B>(x, f, in, tid);
    __syncthreads();
    const long long t0 = clock64();
    if (!SPLIT) run<A, B>(x, f, tbl, outer);
    else if (!second) run<A, A>(x, f, tbl, outer);
    else run<B, B>(x, f, tbl, outer);
    const long long t1 = clock64();
    uint32_t v = 0;
#pragma unroll
    for (int m = 0; m < CH; m++) v ^= (x[m] ^ __float_as_uint(f[m])) * (2u * m + 1u);
    out[tid] = v;
    if (tid == 0)
        for (int m = 0; m < CH; m++) {
            out[gridDim.x * blockDim.x + m] = x[m];
            out[gridDim.x * blockDim.x + CH + m] = __float_as_uint(f[m]);
        }
    if (threadIdx.x == 0) clk[CLK * blockIdx.x] = t0;
    if (threadIdx.x % 32 == 0) clk[CLK * blockIdx.x + 1 + warp] = t1;
}

template <int A, int B, bool SPLIT>
static std::vector<uint32_t> host_chains(const std::vector<uint32_t>& in, int outer) {
    constexpr int B0 = SPLIT ? A : B;
    std::vector<uint32_t> tbl(TBL);
    for (int i = 0; i < TBL; i++) tbl[i] = table_word(A == LDSC || B == LDSC, in.data(), i);
    uint32_t x[CH];
    float f[CH];
    init<A, B0>(x, f, in.data(), 0);
    for (int o = 0; o < outer; o++)
        for (int s = 0; s < STEPS; s++)
            for (int m = 0; m < CH; m++) {
                if (m < 4) step<A>(x, f, tbl.data(), m);
                else step<B0>(x, f, tbl.data(), m);
            }
    std::vector<uint32_t> r(2 * CH);
    for (int m = 0; m < CH; m++) {
        r[m] = x[m];
        memcpy(&r[CH + m], &f[m], 4);
    }
    return r;
}

struct Kernel {
    std::string name;
    int a, b;
    bool split;
    void (*fn)(const uint32_t*, uint32_t*, long long*, int);
    std::vector<uint32_t> (*chains)(const std::vector<uint32_t>&, int);
};

#define K1(A) Kernel{OPS[A], A, A, false, rates<A, A, false>, host_chains<A, A, false>}
#define K2(A, B) Kernel{std::string(OPS[A]) + "+" + OPS[B], A, B, false, rates<A, B, false>, host_chains<A, B, false>}
#define KS(A, B) Kernel{std::string(OPS[A]) + "|" + OPS[B], A, B, true, rates<A, B, true>, host_chains<A, B, true>}
static const Kernel KERNELS[] = {
    K1(FFMA), K1(IADD3), K1(LOP3), K1(IMAD), K1(DP4A), K1(HFMA2), K1(BFMA2), K1(PRMT), K1(CVT8), K1(LDS), K1(LDSC),
    K2(FFMA, IADD3), K2(FFMA, IMAD), K2(IADD3, IMAD), K2(FFMA, DP4A), K2(IADD3, DP4A), K2(FFMA, HFMA2),
    K2(IADD3, HFMA2), K2(FFMA, PRMT), K2(IADD3, PRMT), K2(FFMA, CVT8), K2(IADD3, CVT8), K2(FFMA, LDS),
    K2(DP4A, LDS), K2(FFMA, LDSC),
    KS(FFMA, NONE), KS(IADD3, NONE), KS(DP4A, NONE), KS(LDS, NONE), KS(LDSC, NONE),
    KS(FFMA, IADD3), KS(FFMA, DP4A), KS(IADD3, IMAD), KS(FFMA, LDS), KS(FFMA, LDSC), KS(DP4A, LDSC), KS(IADD3, LDSC),
};

struct Launch {
    double ms, span_a, span_b;  // the last warp of each half (the whole block unless split), in SM cycles
    std::vector<uint32_t> out;
};

static Launch launch(const Kernel& k, const uint32_t* d_in, int grid, int outer) {
    const size_t nout = (size_t)grid * THREADS + 2 * CH;
    uint32_t* d_out;
    long long* d_clk;
    CK(cudaMalloc(&d_out, nout * 4));
    CK(cudaMalloc(&d_clk, (size_t)grid * CLK * 8));
    CK(cudaMemset(d_out, 0xA5, nout * 4));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    k.fn<<<grid, THREADS>>>(d_in, d_out, d_clk, outer);
    CK(cudaEventRecord(e1));
    CK(cudaGetLastError());
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    Launch r{ms, 0, 0, std::vector<uint32_t>(nout)};
    std::vector<long long> clk((size_t)grid * CLK);
    CK(cudaMemcpy(r.out.data(), d_out, nout * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(clk.data(), d_clk, clk.size() * 8, cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid; i++)
        for (int w = 0; w < WARPS; w++) {
            const double c = (double)(clk[CLK * i + 1 + w] - clk[CLK * i]);
            double& span = k.split && w >= WARPS / 2 ? r.span_b : r.span_a;
            span = std::max(span, c);
        }
    if (!k.split) r.span_b = r.span_a;
    CK(cudaFree(d_out));
    CK(cudaFree(d_clk));
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    return r;
}

int main(int argc, char** argv) {
    double target_ms = 60;
    int reps = 5, blocks = 2;
    std::string only;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--ms")) target_ms = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--only")) only = "," + std::string(argv[i + 1]) + ",";
        else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--blocks")) blocks = atoi(argv[i + 1]);
    }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 12) {
        fprintf(stderr, "compute capability %d.%d is not sm_120\n", prop.major, prop.minor);
        return 2;
    }
    const int sms = prop.multiProcessorCount;
    std::vector<uint32_t> in(IN_WORDS);
    uint64_t z = 0x2545f4914f6cdd1dull;
    for (auto& w : in) {
        z ^= z << 13, z ^= z >> 7, z ^= z << 17;
        w = (uint32_t)z;
    }
    uint32_t* d_in;
    CK(cudaMalloc(&d_in, IN_WORDS * 4));
    CK(cudaMemcpy(d_in, in.data(), IN_WORDS * 4, cudaMemcpyHostToDevice));

    printf("{\"schema\": \"verity-pouw/saturation-isa-rates/v1\", \"device\": \"%s\", \"sms\": %d, \"chains\": %d, "
           "\"steps\": %d, \"threads_per_block\": %d, \"kernels\": [\n",
           prop.name, sms, CH, STEPS, THREADS);
    bool first = true;
    for (const Kernel& k : KERNELS) {
        if (!only.empty() && only.find("," + k.name + ",") == std::string::npos) continue;
        int bps = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k.fn, THREADS, 0));
        cudaFuncAttributes attr;
        CK(cudaFuncGetAttributes(&attr, k.fn));
        const int grid = sms * std::min(bps, blocks);
        const size_t nthreads = (size_t)grid * THREADS;
        Launch cal = launch(k, d_in, grid, 64);
        const int outer = std::max(64, (int)(64 * target_ms / std::max(cal.ms, 1e-3)));
        std::vector<Launch> runs;
        for (int r = 0; r < reps; r++) runs.push_back(launch(k, d_in, grid, outer));
        for (int r = 1; r < reps; r++)
            if (runs[r].out != runs[0].out) {
                fprintf(stderr, "GATE %s: launches disagree\n", k.name.c_str());
                return 3;
            }
        const std::vector<uint32_t> want = k.chains(in, outer);
        for (int m = 0; m < CH; m++) {
            const int op = m < 4 || k.split ? k.a : k.b;
            const uint32_t* got = &runs[0].out[nthreads];
            if (host_checked(op) && (got[m] != want[m] || got[CH + m] != want[CH + m])) {
                fprintf(stderr, "GATE %s: thread 0 chain %d is %08x/%08x, host %08x/%08x\n", k.name.c_str(), m, got[m],
                        got[CH + m], want[m], want[CH + m]);
                return 3;
            }
        }
        const bool pair = k.a != k.b, has_b = pair && k.b != NONE;
        // ops a thread of A (and of B) runs, and the threads that run them
        const double per_thread = (double)outer * STEPS * (pair && !k.split ? CH / 2 : CH);
        const double threads = k.split ? nthreads / 2.0 : (double)nthreads;
        std::vector<double> ra, rb, mhz, ms, overlap;
        for (const Launch& r : runs) {
            ra.push_back(threads * per_thread / (sms * r.span_a));
            rb.push_back(has_b ? threads * per_thread / (sms * r.span_b) : 0.0);
            mhz.push_back(std::max(r.span_a, r.span_b) / (r.ms * 1e3));
            overlap.push_back(std::min(r.span_a, r.span_b) / std::max(r.span_a, r.span_b));
            ms.push_back(r.ms);
        }
        auto med = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        printf("%s {\"kernel\": \"%s\", \"a\": \"%s\", \"b\": \"%s\", \"split\": %s, \"regs\": %d, \"blocks_per_sm\": %d, "
               "\"grid\": %d, \"outer\": %d, \"ms\": %.4f, \"sm_mhz\": %.1f, \"a_per_sm_clk\": %.2f, \"b_per_sm_clk\": %.2f, "
               "\"span_overlap\": %.3f, \"host_checked\": %s, \"gate\": \"ok\"}",
               first ? "" : ",\n", k.name.c_str(), OPS[k.a], pair ? OPS[k.b] : "", k.split ? "true" : "false",
               attr.numRegs, bps, grid, outer, med(ms), med(mhz), med(ra), med(rb), med(overlap),
               host_checked(k.a) && (k.split || host_checked(k.b)) ? "true" : "false");
        first = false;
        fflush(stdout);
    }
    printf("\n]}\n");
    CK(cudaFree(d_in));
    return 0;
}
