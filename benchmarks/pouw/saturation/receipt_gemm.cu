// What phi receipts cost inside a real FP8 GEMM on sm_120 (CMP-8, Daniel's #7 (b)), where phi_probe.cu measured them beside
// register-resident mma.sync only: Pearl-C's 128 x 128 mainloop as its gemm_plain runs it (../pearl_c_sm120/mainloop_sm120.cuh
// at G = 4: TMA into 3 stages of 128 bytes of depth, a producer warpgroup, 8 consumer warps of 64 x 32, every 128-deep window
// from +0 and added into the FP32 total by one FADD), with phi.cuh's step folded over the total's words right after every
// P-th window's FADD.  The total there is the running prefix at k = 128 P, 256 P, ..., K, so P sets the commit stride
// L = 128 P; each thread chains the 4 words of each of its 16 accumulator tiles (one chain a tile), and the epilogue folds its
// 16 chains into one word a tile.
//   plain             no receipt: gemm_plain's mainloop (its scales aside), bf16 out
//   alu_pP, imad_pP   phi1 alu or imad at L = 128 P;  aluR_pP  R alu rounds a word
//   ..._pPs           spread: a receipting stage's words one after each of its 64 MMAs (row block a's after atom a's), the
//                     same chains, against the unspread form's 4 words after each of the first atom's FADDs
// Gates (exit 3, before any time is printed): every kernel's FP32 totals equal plain's bit for bit and so do its bf16 outputs,
// two launches of each agree (totals and receipt words), a receipted kernel's receipt words are not all zero, a spread
// kernel's receipt words equal its unspread twin's, and 64 sampled totals are within 1e-4 of the sum of |products| of a
// double host sum (the mma's own rounding aside).  Times: --reps rounds,
// each kernel in turn within a round (so each round pairs every kernel with plain at the same clocks), bf16 out.
//
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o receipt_gemm receipt_gemm.cu
//   ./receipt_gemm [--sizes 8192,16384] [--reps 9] [--only KERNEL,KERNEL] [--profile]
//   --profile: no gates, one bf16 launch of each kernel at the first size (for ncu)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../pearl_c_sm120/mainloop_sm120.cuh"
#include "phi.cuh"

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            exit(2);                                                                       \
        }                                                                                  \
    } while (0)

constexpr int RASTER = 8;  // run.py's tile-row group
using L = ml::Mainloop<128, 128, 4, 3, false, 1, 24>;
constexpr int MI = L::MI, NI = L::NI;

template <int VAR, int ROUNDS, int P, bool SPREAD> struct Receipted : L {
    ML_DEV explicit Receipted(uint8_t* smem) : L(smem) {}
    ML_DEV static void receipt(uint32_t& y, const float (&t)[4], uint32_t w, int j) {
#pragma unroll
        for (int e = 0; e < 4; e++) y = phi<ROUNDS, VAR>(y, __float_as_uint(t[e]), w, key(j, e));
    }
    // L::mma, with each tile's receipt (RCPT) right after its FADD, among the window's first MMAs; or (RB >= 0) row block
    // RB's 16 words one after each of the atom's 16 MMAs, in the same chain order
    template <bool FROM0, bool FADD, bool RCPT, int RB = -1>
    ML_DEV static void mma_r(float (&acc)[MI][NI][4], float (&total)[MI][NI][4], uint32_t (&y)[MI][NI], uint32_t w,
                             const uint32_t (&fa)[MI][4], const uint32_t (&fb)[NI][2]) {
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int ni = 0; ni < NI; ni++) {
                if (FADD) {
#pragma unroll
                    for (int e = 0; e < 4; e++) total[mi][ni][e] = __fadd_rn(total[mi][ni][e], acc[mi][ni][e]);
                }
                if (RCPT) receipt(y[mi][ni], total[mi][ni], w, mi * NI + ni);
                if (FROM0) ml::qmma0(acc[mi][ni], fa[mi], fb[ni]);
                else ml::qmma(acc[mi][ni], fa[mi], fb[ni], acc[mi][ni]);
                if (RB >= 0) {  // after MMA q, word q % 4 of tile q / 4
                    const int q = mi * NI + ni, tj = q / 4, te = q % 4;
                    y[RB][tj] = phi<ROUNDS, VAR>(y[RB][tj], __float_as_uint(total[RB][tj][te]), w, key(RB * NI + tj, te));
                }
            }
    }
    // L::stage_body<FIRST, LAST, true>; SPREAD moves a receipting stage's receipts to row block a after atom a's MMAs
    template <bool FIRST, bool LAST, bool RCPT>
    ML_DEV void stage_r(float (&acc)[MI][NI][4], float (&total)[MI][NI][4], uint32_t (&y)[MI][NI], uint32_t kt,
                        uint32_t (&fa)[2][MI][4], uint32_t (&fb)[2][NI][2]) {
        constexpr bool S = RCPT && SPREAD;
        const uint32_t sb = stage(s), done = empty(s);
        load(fa[1], fb[1], sb, 1);
        mma_r<true, !FIRST, RCPT && !SPREAD, S ? 0 : -1>(acc, total, y, kt, fa[0], fb[0]);
        load(fa[0], fb[0], sb, 2);
        mma_r<false, false, false, S ? 1 : -1>(acc, total, y, kt, fa[1], fb[1]);
        load(fa[1], fb[1], sb, 3);
        mma_r<false, false, false, S ? 2 : -1>(acc, total, y, kt, fa[0], fb[0]);
        advance();
        if (!LAST) {
            ml::mbar_wait_warp(full(s), ph);
            load(fa[0], fb[0], stage(s), 0);
        }
        mma_r<false, false, false, S ? 3 : -1>(acc, total, y, kt, fa[1], fb[1]);
        ml::mbar_release(done);
    }
    // L::tile<1, true>: stage kt's FADD closes the prefix at k = 128 kt, receipted when P divides kt, and so is the final
    // total (k = K); K a multiple of 128 P
    ML_DEV void tile_r(int K, float (&acc)[MI][NI][4], float (&total)[MI][NI][4], uint32_t (&y)[MI][NI]) {
        const int KT = K / BK;
        uint32_t fa[2][MI][4], fb[2][NI][2];
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int ni = 0; ni < NI; ni++) {
                y[mi][ni] = 0;
#pragma unroll
                for (int e = 0; e < 4; e++) total[mi][ni][e] = 0.0f;
            }
        ml::mbar_wait_warp(full(s), ph);
        load(fa[0], fb[0], stage(s), 0);
        stage_r<true, false, false>(acc, total, y, 0, fa, fb);
#pragma unroll 1
        for (int kt = 1; kt < KT - 1; kt++) {
            if (P > 1 && kt % (P > 1 ? P : 1)) stage_r<false, false, false>(acc, total, y, kt, fa, fb);
            else stage_r<false, false, (P > 0)>(acc, total, y, kt, fa, fb);
        }
        stage_r<false, true, P == 1>(acc, total, y, KT - 1, fa, fb);
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int ni = 0; ni < NI; ni++) {
#pragma unroll
                for (int e = 0; e < 4; e++) total[mi][ni][e] = __fadd_rn(total[mi][ni][e], acc[mi][ni][e]);
                if (P > 0) receipt(y[mi][ni], total[mi][ni], KT, mi * NI + ni);
            }
    }
};

__device__ __forceinline__ void store2(float* p, float a, float b) { *reinterpret_cast<float2*>(p) = make_float2(a, b); }
__device__ __forceinline__ void store_bf2(uint16_t* p, float a, float b) {
    *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
}
__device__ __forceinline__ long long globaltimer() {
    long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// A (M x K) and B (N x K) row-major E4M3 through maps; out C (FP32 totals, fp32 != 0) or Yb (their bf16); R a word a consumer
// thread a tile, its 16 receipt chains folded; clk 4 words a block (clock64 and globaltimer at its first tile and after its last)
template <int VAR, int ROUNDS, int P, bool SPREAD = false>
__global__ void __launch_bounds__(L::NT, 1)
    rgemm(const ml::Maps* maps, float* C, uint16_t* Yb, uint32_t* R, long long* clk, int M, int N, int K, int fp32) {
    extern __shared__ __align__(1024) uint8_t smem[];
    const int tiles_m = M / 128, tiles_n = N / 128;
    Receipted<VAR, ROUNDS, P, SPREAD> loop(smem);
    loop.roles(maps, M, N, K, RASTER);
    const long long c0 = clock64(), g0 = globaltimer();
    for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
        int tm, tn;
        ml::raster(t, tiles_m, tiles_n, RASTER, tm, tn);
        float acc[MI][NI][4], total[MI][NI][4];
        uint32_t y[MI][NI];
        loop.tile_r(K, acc, total, y);
        if (K < 0) __trap();  // gemm_plain's block boundary among the chain's last MMAs (without it its epilogue spills)
        uint32_t tid;
        asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
        const int warp = tid / 32, lane = tid % 32, wm = warp / L::WARPS_N, wn = warp % L::WARPS_N;
        const int row0 = tm * 128 + wm * L::WM + (lane >> 2), col0 = tn * 128 + wn * L::WN + 2 * (lane & 3);
        uint32_t f = 0;
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int ni = 0; ni < NI; ni++) {
                const size_t at = (size_t)(row0 + 16 * mi) * N + col0 + 8 * ni, at8 = at + (size_t)8 * N;
                if (fp32) {
                    store2(C + at, total[mi][ni][0], total[mi][ni][1]);
                    store2(C + at8, total[mi][ni][2], total[mi][ni][3]);
                } else {
                    store_bf2(Yb + at, total[mi][ni][0], total[mi][ni][1]);
                    store_bf2(Yb + at8, total[mi][ni][2], total[mi][ni][3]);
                }
                f = rotl13(f) ^ y[mi][ni];
            }
        R[(size_t)t * L::NCT + tid] = f;
    }
    if (threadIdx.x == 0) {
        clk[4 * blockIdx.x] = c0;
        clk[4 * blockIdx.x + 1] = clock64();
        clk[4 * blockIdx.x + 2] = g0;
        clk[4 * blockIdx.x + 3] = globaltimer();
    }
}

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
    return x;
}
// finite E4M3 bytes of exponent field 4 ... 10 (2^-3 ... 2^3), either sign
__global__ void gen(uint8_t* p, size_t n, uint32_t salt) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t h = mix32((uint32_t)i * 0x9e3779b9u ^ salt);
        p[i] = (uint8_t)((h & 0x80u) | ((4u + (h >> 8) % 7u) << 3) | ((h >> 16) & 7u));
    }
}
__global__ void diff(const uint32_t* a, const uint32_t* b, size_t n, unsigned long long* count) {
    unsigned long long c = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) c += a[i] != b[i];
    if (c) atomicAdd(count, c);
}

using Fn = void (*)(const ml::Maps*, float*, uint16_t*, uint32_t*, long long*, int, int, int, int);
struct Kernel {
    const char* name;
    const char* var;
    int rounds, p;
    bool spread;
    Fn fn;
};
static const Kernel KERNELS[] = {
    {"plain", "none", 0, 0, false, rgemm<0, 1, 0>},
    {"alu_p1", "alu", 1, 1, false, rgemm<0, 1, 1>},       {"alu_p2", "alu", 1, 2, false, rgemm<0, 1, 2>},
    {"alu_p4", "alu", 1, 4, false, rgemm<0, 1, 4>},       {"imad_p1", "imad", 1, 1, false, rgemm<1, 1, 1>},
    {"imad_p2", "imad", 1, 2, false, rgemm<1, 1, 2>},     {"imad_p4", "imad", 1, 4, false, rgemm<1, 1, 4>},
    {"alu2_p1", "alu", 2, 1, false, rgemm<0, 2, 1>},      {"alu2_p2", "alu", 2, 2, false, rgemm<0, 2, 2>},
    {"alu3_p1", "alu", 3, 1, false, rgemm<0, 3, 1>},      {"alu4_p1", "alu", 4, 1, false, rgemm<0, 4, 1>},
    {"alu_p1s", "alu", 1, 1, true, rgemm<0, 1, 1, true>}, {"alu_p2s", "alu", 1, 2, true, rgemm<0, 1, 2, true>},
    {"imad_p1s", "imad", 1, 1, true, rgemm<1, 1, 1, true>}, {"alu2_p1s", "alu", 2, 1, true, rgemm<0, 2, 1, true>},
    {"alu2_p2s", "alu", 2, 2, true, rgemm<0, 2, 2, true>},
};

using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

// mainloop_sm120.cuh's map: UINT8, rank 2, globalDim {depth, rows}, box {128, 128}, SWIZZLE_128B, L2 promotion 128B, no fill
static CUtensorMap tmap(EncodeTiled encode, void* p, int depth, int rows) {
    CUtensorMap m;
    const cuuint64_t dims[2] = {(cuuint64_t)depth, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)depth};
    const cuuint32_t box[2] = {128, 128}, estr[2] = {1, 1};
    const CUresult r = encode(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                              CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled: %d\n", (int)r);
        exit(2);
    }
    return m;
}

static double e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
    return b & 0x80 ? -v : v;
}

static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    std::vector<int> sizes = {8192, 16384};
    int reps = 9;
    bool profile = false;
    std::string only;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--profile")) profile = true;
        else if (i + 1 < argc && !strcmp(argv[i], "--only")) only = "," + std::string(argv[++i]) + ",";
        else if (i + 1 < argc && !strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "--sizes")) {
            sizes.clear();
            for (const char* q = argv[++i]; *q; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) sizes.push_back(atoi(q));
        }
    }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 12) {
        fprintf(stderr, "compute capability %d.%d is not sm_120\n", prop.major, prop.minor);
        return 2;
    }
    const int sms = prop.multiProcessorCount;
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q));
    if (q != cudaDriverEntryPointSuccess || !fn) {
        fprintf(stderr, "no cuTensorMapEncodeTiled\n");
        return 2;
    }
    const EncodeTiled encode = reinterpret_cast<EncodeTiled>(fn);
    std::vector<const Kernel*> ks;
    for (const Kernel& k : KERNELS)
        if (k.p == 0 || only.empty() || only.find("," + std::string(k.name) + ",") != std::string::npos) ks.push_back(&k);
    std::vector<cudaFuncAttributes> attrs(ks.size());
    for (size_t j = 0; j < ks.size(); j++) {
        CK(cudaFuncSetAttribute(ks[j]->fn, cudaFuncAttributeMaxDynamicSharedMemorySize, L::SMEM));
        CK(cudaFuncGetAttributes(&attrs[j], ks[j]->fn));
    }

    printf("{\"schema\": \"verity-pouw/saturation-receipt-gemm/v1\", \"device\": \"%s\", \"sms\": %d, \"tile\": [128, 128], "
           "\"window\": 128, \"stages\": 3, \"threads_per_block\": %d, \"smem\": %d, \"results\": [\n",
           prop.name, sms, L::NT, L::SMEM);
    bool first = true;
    for (const int n : sizes) {
        if (n % 512 || n < 1024) {
            fprintf(stderr, "size %d is not a multiple of 512 (128 P for P = 4)\n", n);
            return 2;
        }
        const size_t nn = (size_t)n * n, tiles = nn / (128 * 128), rwords = tiles * L::NCT;
        const int grid = (int)std::min<size_t>(tiles, sms);
        uint8_t *A, *B;
        float *Cref, *Ct;
        uint16_t *Yref, *Yt;
        uint32_t* R;
        long long* clk;
        unsigned long long* cnt;
        ml::Maps* maps;
        CK(cudaMalloc(&A, nn));
        CK(cudaMalloc(&B, nn));
        CK(cudaMalloc(&Cref, nn * 4));
        CK(cudaMalloc(&Ct, nn * 4));
        CK(cudaMalloc(&Yref, nn * 2));
        CK(cudaMalloc(&Yt, nn * 2));
        CK(cudaMalloc(&R, rwords * 4));
        CK(cudaMalloc(&clk, (size_t)grid * 32));
        CK(cudaMalloc(&cnt, 8));
        CK(cudaMalloc(&maps, sizeof(ml::Maps)));
        gen<<<1024, 256>>>(A, nn, 0x51ed270bu);
        gen<<<1024, 256>>>(B, nn, 0x2c1b3c6du);
        CK(cudaGetLastError());
        const CUtensorMap mh[4] = {tmap(encode, A, n, n), tmap(encode, B, n, n), tmap(encode, A, n, n), tmap(encode, B, n, n)};
        static_assert(sizeof(mh) == sizeof(ml::Maps), "four 128-byte maps");
        CK(cudaMemcpy(maps, mh, sizeof(mh), cudaMemcpyHostToDevice));
        auto run = [&](const Kernel& k, float* C, uint16_t* Y, int fp32) {
            k.fn<<<grid, L::NT, L::SMEM>>>(maps, C, Y, R, clk, n, n, n, fp32);
            CK(cudaGetLastError());
        };
        auto differ = [&](const void* a, const void* b, size_t words) {
            CK(cudaMemset(cnt, 0, 8));
            diff<<<1024, 256>>>((const uint32_t*)a, (const uint32_t*)b, words, cnt);
            unsigned long long c = 0;
            CK(cudaMemcpy(&c, cnt, 8, cudaMemcpyDeviceToHost));
            return c;
        };
        auto receipts = [&]() {
            std::vector<uint32_t> h(rwords);
            CK(cudaMemcpy(h.data(), R, rwords * 4, cudaMemcpyDeviceToHost));
            return h;
        };
        auto gate = [&](const char* name, const char* what, unsigned long long bad) {
            if (bad) {
                fprintf(stderr, "GATE %s at n = %d: %s (%llu)\n", name, n, what, bad);
                exit(3);
            }
        };

        double host_err = 0;  // the sampled totals' worst |C - host| / sum |ab|
        if (!profile) {
            run(*ks[0], Cref, Yref, 1);
            run(*ks[0], nullptr, Yref, 0);
            CK(cudaDeviceSynchronize());
            for (int i = 0; i < 64; i++) {
                const size_t r = (size_t)(i * 2654435761u) % n, c = ((size_t)i * 40503u + 17) % n;
                std::vector<uint8_t> ar(n), br(n);
                float got;
                CK(cudaMemcpy(ar.data(), A + r * n, n, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(br.data(), B + c * n, n, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&got, Cref + r * n + c, 4, cudaMemcpyDeviceToHost));
                double ref = 0, mag = 0;
                for (int t = 0; t < n; t++) {
                    const double v = e4m3(ar[t]) * e4m3(br[t]);
                    ref += v, mag += std::fabs(v);
                }
                host_err = std::max(host_err, std::fabs(got - ref) / mag);
                if (!(std::fabs(got - ref) <= 1e-4 * mag)) {
                    fprintf(stderr, "GATE plain at n = %d: C[%zu][%zu] = %.6g, host %.6g (sum |ab| %.6g)\n", n, r, c, got, ref,
                            mag);
                    return 3;
                }
            }
            for (const Kernel* k : ks) {
                run(*k, Ct, Yt, 1);
                const std::vector<uint32_t> r0 = receipts();
                gate(k->name, "FP32 totals differ from plain's", differ(Cref, Ct, nn));
                CK(cudaMemset(Ct, 0xA5, nn * 4));
                run(*k, Ct, Yt, 1);
                gate(k->name, "a second launch's totals differ", differ(Cref, Ct, nn));
                gate(k->name, "a second launch's receipt words differ", receipts() != r0);
                run(*k, nullptr, Yt, 0);
                gate(k->name, "bf16 outputs differ from plain's", differ(Yref, Yt, nn / 2));
                if (k->p > 0) gate(k->name, "every receipt word is zero", std::all_of(r0.begin(), r0.end(), [](uint32_t w) { return w == 0; }));
                if (k->spread) {
                    const Kernel* twin = nullptr;
                    for (const Kernel& t : KERNELS)
                        if (!t.spread && t.p == k->p && t.rounds == k->rounds && !strcmp(t.var, k->var)) twin = &t;
                    run(*twin, Ct, Yt, 1);
                    gate(k->name, "receipt words differ from its unspread twin's", receipts() != r0);
                }
            }
        }

        const int launches = profile ? 1 : std::max(1, (int)(5.5e12 / ((double)nn * n)));  // 10 at 8,192, 1 at 16,384
        std::vector<std::vector<double>> ms(ks.size()), mhz(ks.size());
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        std::vector<long long> hc((size_t)grid * 4);
        for (int rep = 0; rep < (profile ? 1 : reps); rep++)
            for (size_t j = 0; j < ks.size(); j++) {
                CK(cudaEventRecord(e0));
                for (int l = 0; l < launches; l++) run(*ks[j], nullptr, Yt, 0);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float t = 0;
                CK(cudaEventElapsedTime(&t, e0, e1));
                ms[j].push_back(t / launches);
                CK(cudaMemcpy(hc.data(), clk, hc.size() * 8, cudaMemcpyDeviceToHost));
                std::vector<double> f;
                for (int b = 0; b < grid; b++) f.push_back(1e3 * (hc[4 * b + 1] - hc[4 * b]) / (double)(hc[4 * b + 3] - hc[4 * b + 2]));
                mhz[j].push_back(med(f));
            }
        for (size_t j = 0; j < ks.size(); j++) {
            const Kernel& k = *ks[j];
            std::vector<double> ratio, mac, phis;
            for (size_t r = 0; r < ms[j].size(); r++) {
                ratio.push_back(ms[j][r] / ms[0][r]);
                const double cycles = ms[j][r] * 1e-3 * mhz[j][r] * 1e6 * sms;
                mac.push_back((double)nn * n / cycles);
                phis.push_back(k.p ? (double)nn * (n / (128 * k.p)) * k.rounds / cycles : 0);
            }
            printf("%s {\"n\": %d, \"kernel\": \"%s\", \"var\": \"%s\", \"rounds\": %d, \"P\": %d, \"L\": %d, \"spread\": %s, "
                   "\"regs_launch\": %d, \"local_bytes\": %zu, \"grid\": %d, \"launches\": %d, \"reps\": %zu, \"ms\": %.4f, "
                   "\"ms_min\": %.4f, \"tflops\": %.1f, \"sm_mhz\": %.1f, \"macs_per_sm_clk\": %.1f, \"phi_rounds_per_sm_clk\": %.2f, "
                   "\"ms_vs_plain\": %.4f, \"host_err\": %.2e, \"gate\": \"%s\"}",
                   first ? "" : ",\n", n, k.name, k.var, k.rounds, k.p, 128 * k.p, k.spread ? "true" : "false", attrs[j].numRegs,
                   attrs[j].localSizeBytes,
                   grid, launches, ms[j].size(), med(ms[j]), *std::min_element(ms[j].begin(), ms[j].end()),
                   2.0 * nn * n / (med(ms[j]) * 1e9), med(mhz[j]), med(mac), med(phis), med(ratio), host_err,
                   profile ? "none" : "ok");
            first = false;
            fflush(stdout);
        }
        CK(cudaEventDestroy(e0));
        CK(cudaEventDestroy(e1));
        for (void* p : {(void*)A, (void*)B, (void*)Cref, (void*)Ct, (void*)Yref, (void*)Yt, (void*)R, (void*)clk, (void*)cnt,
                        (void*)maps})
            CK(cudaFree(p));
    }
    printf("\n]}\n");
    return 0;
}
