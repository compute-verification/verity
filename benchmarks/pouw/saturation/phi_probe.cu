// What a receipt step costs on sm_120 beside E4M3 mma.sync (CMP-8, Daniel's #7 (b)): the ideal-phi model's phi_{sigma,beta,t}(y, p)
// instantiated as the page's "keyed 32-bit ARX step", run over every accumulator word every PER mma rounds (commit stride
// L = 32 * PER MACs per output entry), each thread chaining its own words (one chain per accumulator tile).
//
//   alu   phi1(y, p) = rotl(t, 13) + t + k_t, t = y ^ p ^ s   all on the integer ALU: ptxas emits LOP3, LEA, IADD3 (k_t an
//                                                             immediate per word, s the readout)
//   imad  phi1(y, p) = t * (k_t | 1) + rotl(t, 13)             LOP3, SHF.L.W on the ALU, IMAD on the FMA pipe beside it
//   ROUNDS = 2 applies a second phi1 against a constant
//
// Kernels (one wave of --blocks blocks an SM, at most the occupancy's, 8 warps a block, every operand in registers):
//   mode 0  NACC independent m16n8k32 E4M3 mma.sync per round (QMMA), no phi: the tensor-core peak
//   mode 1  phi only over the same words (fixed, read from memory): what one phi step costs, in FP32 lane slots
//   mode 2  PER mma rounds then one phi step per accumulator word, the receipt at commit stride 32 * PER
//   mode 3  the same stride, staggered: PER rounds on half A, the receipt of half B (written a half-body ago), PER rounds on
//           B, the receipt of A; so no receipt waits on an mma in flight
// Gates (exit 3, before any rate is printed): every mode-2/3 kernel's accumulators equal mode 0's after the same mma rounds (so
// the phi work did not change what the tensor cores computed), two launches of every kernel agree, and mode 1's thread-0 chains
// equal a host recomputation.
//
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o phi_probe phi_probe.cu
//   ./phi_probe [--ms 60] [--reps 5] [--blocks 2] [--only KERNEL,KERNEL]
#include <cuda_runtime.h>

#include <algorithm>
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

constexpr int NACC = 8, THREADS = 256, IN_WORDS = 1024;

__host__ __device__ __forceinline__ uint32_t rotl13(uint32_t t) {
#ifdef __CUDA_ARCH__
    return __funnelshift_l(t, t, 13);
#else
    return (t << 13) | (t >> 19);
#endif
}
__host__ __device__ __forceinline__ uint32_t key(int j, int w) { return 0x9e3779b9u * (uint32_t)(j * 4 + w + 1); }

template <int VAR>
__host__ __device__ __forceinline__ uint32_t phi1(uint32_t y, uint32_t p, uint32_t s, uint32_t k) {
    const uint32_t t = y ^ p ^ s;
    return VAR == 0 ? rotl13(t) + t + k : t * (k | 1u) + rotl13(t);
}

template <int ROUNDS, int VAR>
__host__ __device__ __forceinline__ uint32_t phi(uint32_t y, uint32_t p, uint32_t s, uint32_t k) {
    y = phi1<VAR>(y, p, s, k);
    if (ROUNDS > 1) y = phi1<VAR>(y, 0x7f4a7c15u, s, k ^ 0x5851f42du);
    return y;
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__host__ __device__ inline float init_acc(uint32_t word) { return (float)(word & 0xff) * 0.25f; }

template <int ROUNDS, int VAR, int J0, int J1>
__device__ __forceinline__ void receipt(uint32_t (&y)[NACC], const float (&c)[NACC][4], uint32_t s) {
#pragma unroll
    for (int j = J0; j < J1; j++) {
#pragma unroll
        for (int w = 0; w < 4; w++) y[j] = phi<ROUNDS, VAR>(y[j], __float_as_uint(c[j][w]), s, key(j, w));
    }
}

template <int J0, int J1, int PER>
__device__ __forceinline__ void mma_rounds(float (&c)[NACC][4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#pragma unroll
    for (int r = 0; r < PER; r++) {
#pragma unroll
        for (int j = J0; j < J1; j++) mma(c[j], a, b);
    }
}

template <int MODE, int PER, int ROUNDS, int VAR>
__global__ void __launch_bounds__(THREADS, 2)
    probe(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, long long* __restrict__ clk, int outer) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t a[4], b[2], y[NACC];
    float c[NACC][4];
#pragma unroll
    for (int i = 0; i < 4; i++) a[i] = in[(tid * 7 + i) % IN_WORDS];
#pragma unroll
    for (int i = 0; i < 2; i++) b[i] = in[(tid * 5 + 4 + i) % IN_WORDS];
#pragma unroll
    for (int j = 0; j < NACC; j++) {
#pragma unroll
        for (int w = 0; w < 4; w++) c[j][w] = init_acc(in[(tid * 3 + j * 4 + w + 512) % IN_WORDS]);
        y[j] = (uint32_t)tid * NACC + j;
    }
    uint32_t s = 0;
    __syncthreads();
    const long long t0 = clock64();
#pragma unroll 1
    for (int o = 0; o < outer; o++) {
        if (MODE == 0 || MODE == 2) mma_rounds<0, NACC, PER>(c, a, b);
        if (MODE == 1 || MODE == 2) receipt<ROUNDS, VAR, 0, NACC>(y, c, ++s);
        if (MODE == 3) {
            ++s;
            mma_rounds<0, NACC / 2, PER>(c, a, b);
            receipt<ROUNDS, VAR, NACC / 2, NACC>(y, c, s);
            mma_rounds<NACC / 2, NACC, PER>(c, a, b);
            receipt<ROUNDS, VAR, 0, NACC / 2>(y, c, s);
        }
    }
    if (MODE == 3) receipt<ROUNDS, VAR, NACC / 2, NACC>(y, c, s + 1);
    const long long t1 = clock64();
    uint32_t cx = 0, yx = 0;
#pragma unroll
    for (int j = 0; j < NACC; j++) {
        yx ^= y[j] * (2u * j + 1u);
#pragma unroll
        for (int w = 0; w < 4; w++) cx ^= __float_as_uint(c[j][w]) * (uint32_t)(2 * (j * 4 + w) + 1);
    }
    out[2 * tid] = cx;
    out[2 * tid + 1] = yx;
    if (tid == 0)
        for (int j = 0; j < NACC; j++) out[2 * gridDim.x * blockDim.x + j] = y[j];
    if (threadIdx.x == 0) {
        clk[2 * blockIdx.x] = t0;
        clk[2 * blockIdx.x + 1] = t1;
    }
}

template <int ROUNDS, int VAR>
static std::vector<uint32_t> host_chains(const std::vector<uint32_t>& in, int outer) {
    std::vector<uint32_t> y(NACC);
    float c[NACC][4];
    for (int j = 0; j < NACC; j++) {
        for (int w = 0; w < 4; w++) c[j][w] = init_acc(in[(j * 4 + w + 512) % IN_WORDS]);
        y[j] = j;
    }
    for (int o = 1; o <= outer; o++)
        for (int j = 0; j < NACC; j++)
            for (int w = 0; w < 4; w++) {
                uint32_t p;
                memcpy(&p, &c[j][w], 4);
                y[j] = phi<ROUNDS, VAR>(y[j], p, (uint32_t)o, key(j, w));
            }
    return y;
}

struct Kernel {
    std::string name;
    int mode, per, rounds, var;
    void (*fn)(const uint32_t*, uint32_t*, long long*, int);
    std::vector<uint32_t> (*chains)(const std::vector<uint32_t>&, int);
};

static const char* VARS[] = {"alu", "imad"};
#define K(M, P, R, V) \
    Kernel { "mode" #M "_per" #P "_r" #R "_" + std::string(VARS[V]), M, P, R, V, probe<M, P, R, V>, host_chains<R, V> }
static const Kernel KERNELS[] = {
    K(0, 1, 1, 0),
    K(1, 1, 1, 0), K(1, 1, 1, 1), K(1, 1, 2, 0), K(1, 1, 2, 1),
    K(2, 1, 1, 0), K(2, 2, 1, 0), K(2, 4, 1, 0), K(2, 8, 1, 0),
    K(2, 1, 1, 1), K(2, 2, 1, 1), K(2, 4, 1, 1), K(2, 8, 1, 1),
    K(2, 1, 2, 0), K(2, 2, 2, 0), K(2, 4, 2, 0), K(2, 8, 2, 0),
    K(2, 1, 2, 1), K(2, 2, 2, 1), K(2, 4, 2, 1), K(2, 8, 2, 1),
    K(3, 1, 1, 0), K(3, 2, 1, 0), K(3, 4, 1, 0), K(3, 8, 1, 0),
    K(3, 1, 1, 1), K(3, 2, 1, 1), K(3, 4, 1, 1), K(3, 8, 1, 1),
    K(3, 1, 2, 0), K(3, 2, 2, 0), K(3, 4, 2, 0), K(3, 8, 2, 0),
    K(3, 1, 2, 1), K(3, 2, 2, 1), K(3, 4, 2, 1), K(3, 8, 2, 1),
    K(2, 3, 1, 0), K(2, 3, 2, 0), K(2, 3, 2, 1), K(3, 3, 1, 0), K(3, 3, 2, 0), K(3, 3, 2, 1),
};

struct Launch {
    double ms, span_cycles;
    std::vector<uint32_t> out;
};

static Launch launch(const Kernel& k, const uint32_t* d_in, int grid, int outer) {
    const size_t nthreads = (size_t)grid * THREADS, nout = 2 * nthreads + NACC;
    uint32_t* d_out;
    long long* d_clk;
    CK(cudaMalloc(&d_out, nout * 4));
    CK(cudaMalloc(&d_clk, (size_t)grid * 16));
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
    Launch r{ms, 0, std::vector<uint32_t>(nout)};
    std::vector<long long> clk((size_t)grid * 2);
    CK(cudaMemcpy(r.out.data(), d_out, nout * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(clk.data(), d_clk, clk.size() * 8, cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid; i++) r.span_cycles = std::max(r.span_cycles, (double)(clk[2 * i + 1] - clk[2 * i]));
    CK(cudaFree(d_out));
    CK(cudaFree(d_clk));
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    return r;
}

static bool same_c(const std::vector<uint32_t>& x, const std::vector<uint32_t>& z, size_t nthreads) {
    for (size_t t = 0; t < nthreads; t++)
        if (x[2 * t] != z[2 * t]) return false;
    return true;
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
        w = (uint32_t)z & 0x77777777u;  // finite E4M3 bytes (exponent <= 14), never NaN
    }
    uint32_t* d_in;
    CK(cudaMalloc(&d_in, IN_WORDS * 4));
    CK(cudaMemcpy(d_in, in.data(), IN_WORDS * 4, cudaMemcpyHostToDevice));

    printf("{\"schema\": \"verity-pouw/saturation-phi-probe/v1\", \"device\": \"%s\", \"sms\": %d, \"nacc\": %d, "
           "\"threads_per_block\": %d, \"kernels\": [\n",
           prop.name, sms, NACC, THREADS);
    bool first = true;
    std::vector<uint32_t> mode0_out;
    int mode0_rounds = 0, mode0_grid = 0;
    for (const Kernel& k : KERNELS) {
        if (!only.empty() && only.find("," + k.name + ",") == std::string::npos) continue;
        int bps = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k.fn, THREADS, 0));
        cudaFuncAttributes attr;
        CK(cudaFuncGetAttributes(&attr, k.fn));
        const int grid = sms * std::min(bps, blocks);
        const size_t nthreads = (size_t)grid * THREADS;
        Launch cal = launch(k, d_in, grid, 256);
        int outer = std::max(256, (int)(256 * target_ms / std::max(cal.ms, 1e-3)));
        if (k.mode == 0) outer = 8 * ((outer + 7) / 8);
        std::vector<Launch> runs;
        for (int r = 0; r < reps; r++) runs.push_back(launch(k, d_in, grid, outer));
        for (int r = 1; r < reps; r++)
            if (runs[r].out != runs[0].out) {
                fprintf(stderr, "GATE %s: launches disagree\n", k.name.c_str());
                return 3;
            }
        if (k.mode == 0) {
            mode0_out = runs[0].out, mode0_rounds = outer, mode0_grid = grid;
        } else if (k.mode >= 2) {
            const int rounds = outer * k.per;
            const Launch ref = rounds == mode0_rounds && grid == mode0_grid ? Launch{0, 0, mode0_out}
                                                                            : launch(KERNELS[0], d_in, grid, rounds);
            if (!same_c(ref.out, runs[0].out, nthreads)) {
                fprintf(stderr, "GATE %s: accumulators differ from mode 0 after %d mma rounds\n", k.name.c_str(), rounds);
                return 3;
            }
        } else {
            const std::vector<uint32_t> want = k.chains(in, outer);
            for (int j = 0; j < NACC; j++)
                if (runs[0].out[2 * nthreads + j] != want[j]) {
                    fprintf(stderr, "GATE %s: thread 0 chain %d is %08x, host %08x\n", k.name.c_str(), j,
                            runs[0].out[2 * nthreads + j], want[j]);
                    return 3;
                }
        }
        std::vector<double> mac_rate, phi_rate, mhz;
        const double warps = (double)nthreads / 32;
        for (const Launch& r : runs) {
            const double macs = k.mode == 1 ? 0 : warps * outer * k.per * NACC * 4096.0;
            const double phis = k.mode == 0 ? 0 : (double)nthreads * 4 * (outer * NACC + (k.mode == 3 ? NACC / 2 : 0));
            mac_rate.push_back(macs / (sms * r.span_cycles));
            phi_rate.push_back(phis / (sms * r.span_cycles));
            mhz.push_back(r.span_cycles / (r.ms * 1e3));
        }
        auto med = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        printf("%s {\"kernel\": \"%s\", \"mode\": %d, \"per\": %d, \"L\": %d, \"rounds\": %d, \"var\": \"%s\", \"regs\": %d, "
               "\"blocks_per_sm\": %d, \"grid\": %d, \"outer\": %d, \"ms\": %.4f, \"sm_mhz\": %.1f, \"macs_per_sm_clk\": %.2f, "
               "\"phi_per_sm_clk\": %.3f, \"gate\": \"ok\"}",
               first ? "" : ",\n", k.name.c_str(), k.mode, k.per, 32 * k.per, k.rounds, VARS[k.var], attr.numRegs, bps, grid,
               outer,
               med([&] { std::vector<double> v; for (auto& r : runs) v.push_back(r.ms); return v; }()), med(mhz),
               med(mac_rate), med(phi_rate));
        first = false;
        fflush(stdout);
    }
    printf("\n]}\n");
    CK(cudaFree(d_in));
    return 0;
}
