// Sustained MUFU exp2 rate, in results per SM clock, of the f32 instruction softmax uses against the packed ones:
//
//   f32     ex2.approx.ftz.f32      1 result a lane
//   f16x2   ex2.approx.f16x2        2 results a lane
//   bf16x2  ex2.approx.ftz.bf16x2   2 results a lane
//
// Register-resident: one block on every SM, WARPS warps a block, CHAINS independent values a thread, each step
// x <- 2^-x (a value in (0, 1) stays there). Thread 0 of each block stamps clock64 and %smid after a barrier at both
// ends, so a block's rate is its own SM's clocks. Each (instruction, WARPS) runs a short calibration and then at least
// SUSTAIN_S seconds; one step is first checked against the host's exp2. One JSON line per measurement on stdout.
//
//   nvcc -O3 -gencode arch=compute_90a,code=sm_90a -o ex2_rate ex2_rate.cu && ./ex2_rate [SUSTAIN_S]

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));           \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

constexpr int CHAINS = 16;
enum Op { F32 = 0, F16X2 = 1, BF16X2 = 2 };

// One step on a 32-bit register: 2^-x of its one f32, or of each of its two halves.
template <int OP>
__device__ __forceinline__ uint32_t step(uint32_t x) {
    uint32_t d;
    if (OP == F32) {
        asm volatile("{ .reg .f32 t; neg.f32 t, %1; ex2.approx.ftz.f32 %0, t; }" : "=r"(d) : "r"(x));
    } else if (OP == F16X2) {
        asm volatile("{ .reg .b32 t; xor.b32 t, %1, 0x80008000; ex2.approx.f16x2 %0, t; }" : "=r"(d) : "r"(x));
    } else {
        asm volatile("{ .reg .b32 t; xor.b32 t, %1, 0x80008000; ex2.approx.ftz.bf16x2 %0, t; }" : "=r"(d) : "r"(x));
    }
    return d;
}

struct Stamp {
    long long c0, c1;
    unsigned long long t0, t1;
    int smid, pad;
};

__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

template <int OP>
__global__ void rate_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, Stamp* __restrict__ st,
                            long long iters) {
    uint32_t v[CHAINS];
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) v[c] = in[c];
    __syncthreads();
    long long c0 = clock64();
    unsigned long long t0 = globaltimer();
#pragma unroll 1
    for (long long it = 0; it < iters; ++it) {
#pragma unroll
        for (int c = 0; c < CHAINS; ++c) v[c] = step<OP>(v[c]);
    }
    __syncthreads();
    long long c1 = clock64();
    unsigned long long t1 = globaltimer();
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        st[blockIdx.x] = Stamp{c0, c1, t0, t1, smid, 0};
    }
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) s ^= v[c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        for (int c = 0; c < CHAINS; ++c) out[size_t(gridDim.x) * blockDim.x + c] = v[c];
}

typedef void (*Kernel)(const uint32_t*, uint32_t*, Stamp*, long long);

static const char* NAMES[] = {"f32", "f16x2", "bf16x2"};
static const char* PTX[] = {"ex2.approx.ftz.f32", "ex2.approx.f16x2", "ex2.approx.ftz.bf16x2"};
static const int RESULTS[] = {1, 2, 2};
static const Kernel KERNELS[] = {rate_kernel<F32>, rate_kernel<F16X2>, rate_kernel<BF16X2>};

static float half_lo(uint32_t w, int op) {
    uint16_t h = uint16_t(w & 0xFFFF);
    if (op == F16X2) {
        __half x;
        memcpy(&x, &h, 2);
        return __half2float(x);
    }
    uint32_t f = uint32_t(h) << 16;
    float r;
    memcpy(&r, &f, 4);
    return r;
}

static uint32_t pack(float lo, float hi, int op) {
    if (op == F32) {
        uint32_t w;
        memcpy(&w, &lo, 4);
        return w;
    }
    uint16_t a, b;
    if (op == F16X2) {
        __half x = __float2half_rn(lo), y = __float2half_rn(hi);
        memcpy(&a, &x, 2);
        memcpy(&b, &y, 2);
    } else {
        __nv_bfloat16 x = __float2bfloat16_rn(lo), y = __float2bfloat16_rn(hi);
        memcpy(&a, &x, 2);
        memcpy(&b, &y, 2);
    }
    return uint32_t(a) | (uint32_t(b) << 16);
}

struct Result {
    double wall_s, per_clk_sm, clk_mhz, min_sm, max_sm;
    int sms;
};

static Result run(int op, int sms, int warps, long long iters, const uint32_t* in, uint32_t* out, Stamp* st) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    KERNELS[op]<<<sms, 32 * warps>>>(in, out, st, iters);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    std::vector<Stamp> h(sms);
    CK(cudaMemcpy(h.data(), st, sms * sizeof(Stamp), cudaMemcpyDeviceToHost));
    std::map<int, std::pair<long long, long long>> span;
    std::map<int, int> blocks;
    double cyc = 0, ns = 0;
    for (auto& s : h) {
        auto it = span.find(s.smid);
        if (it == span.end()) span[s.smid] = {s.c0, s.c1};
        else it->second = {std::min(it->second.first, s.c0), std::max(it->second.second, s.c1)};
        blocks[s.smid]++;
        cyc += double(s.c1 - s.c0);
        ns += double(s.t1 - s.t0);
    }
    const double per_block = double(iters) * CHAINS * warps * 32 * RESULTS[op];
    double tot_cyc = 0, mn = 1e30, mx = 0;
    for (auto& kv : span) {
        double c = double(kv.second.second - kv.second.first);
        double r = per_block * blocks[kv.first] / c;
        tot_cyc += c;
        mn = std::min(mn, r);
        mx = std::max(mx, r);
    }
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    return Result{ms / 1e3, per_block * sms / tot_cyc, cyc / ns * 1e3, mn, mx, int(span.size())};
}

int main(int argc, char** argv) {
    const double sustain_s = argc > 1 ? atof(argv[1]) : 2.5;
    int dev = 0, sms = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    sms = p.multiProcessorCount;
    printf("{\"device\": \"%s\", \"cc\": \"%d.%d\", \"sms\": %d, \"chains\": %d}\n", p.name, p.major, p.minor, sms, CHAINS);

    uint32_t* in;
    uint32_t* out;
    Stamp* st;
    CK(cudaMalloc(&in, CHAINS * 4));
    CK(cudaMalloc(&out, (size_t(sms) * 1024 + CHAINS) * 4));
    CK(cudaMalloc(&st, sms * sizeof(Stamp)));

    const int warp_counts[] = {4, 8, 16, 32};
    for (int op = 0; op < 3; ++op) {
        std::vector<float> lo(CHAINS), hi(CHAINS);
        std::vector<uint32_t> hin(CHAINS);
        for (int c = 0; c < CHAINS; ++c) {
            lo[c] = 0.03f + 0.06f * c;
            hi[c] = 0.97f - 0.05f * c;
            hin[c] = pack(lo[c], hi[c], op);
        }
        CK(cudaMemcpy(in, hin.data(), CHAINS * 4, cudaMemcpyHostToDevice));
        KERNELS[op]<<<1, 32>>>(in, out, st, 1);
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> got(CHAINS);
        CK(cudaMemcpy(got.data(), out + 32, CHAINS * 4, cudaMemcpyDeviceToHost));
        double worst = 0;
        for (int c = 0; c < CHAINS; ++c) {
            float x = op == F32 ? lo[c] : half_lo(hin[c], op);
            float want = std::exp2(-double(x));
            float g;
            if (op == F32) memcpy(&g, &got[c], 4);
            else g = half_lo(got[c], op);
            worst = std::max(worst, std::fabs(double(g) / want - 1));
        }
        printf("{\"op\": \"%s\", \"ptx\": \"%s\", \"check_max_rel\": %.3g}\n", NAMES[op], PTX[op], worst);
        fflush(stdout);
        for (int warps : warp_counts) {
            const long long cal = 4096;
            Result r0 = run(op, sms, warps, cal, in, out, st);
            long long iters = (long long)(cal * sustain_s / std::max(r0.wall_s, 1e-6)) + 1;
            Result r = run(op, sms, warps, iters, in, out, st);
            printf("{\"op\": \"%s\", \"warps_per_sm\": %d, \"iters\": %lld, \"wall_s\": %.3f, \"sms_seen\": %d, "
                   "\"results_per_instr_lane\": %d, \"results_per_clk_sm\": %.2f, \"min_sm\": %.2f, \"max_sm\": %.2f, "
                   "\"sm_clk_mhz\": %.0f, \"warp_instr_per_clk_sm\": %.4f}\n",
                   NAMES[op], warps, iters, r.wall_s, r.sms, RESULTS[op], r.per_clk_sm, r.min_sm, r.max_sm, r.clk_mhz,
                   r.per_clk_sm / (32.0 * RESULTS[op]));
            fflush(stdout);
        }
    }
    return 0;
}
