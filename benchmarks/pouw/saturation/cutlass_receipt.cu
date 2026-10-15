// What phi receipts cost inside the FP8 GEMMs vLLM actually serves on sm_120 (CMP-8 round 2): Llama-3.1-8B's linears run
// vLLM's cutlass_scaled_mm, CUTLASS's MainloopSm120TmaWarpSpecialized in two configurations (prefill: cooperative
// 128x128x128; decode at batch 32: ping-pong 32x64x128). cutlass_receipt_kernel.cu builds each, unmodified (plain), with
// phi.cuh's receipt after every P-th k-tile (L = 128 P), or as the prefix dump the gates read. This program gates them and
// times them against plain at Llama-8B's shapes.
// Gates (exit 3, before any time is printed), at one small shape per configuration:
//   - the dump kernel's prefix sums at every k-tile are within 1e-5 of the sum of |products| of a double host sum, and its
//     chain-word map covers the tile once;
//   - plain's bf16 output is bf16(sa * (sb * prefix at K)) of the dump's final prefix, bit for bit;
//   - every kernel's bf16 output equals plain's bit for bit;
//   - every receipt kernel's receipt words equal the host's phi chains over the dump's prefix sums.
// At every timed shape: every kernel's output equals plain's, two launches agree on the receipt words, and they aren't all
// zero. Times: operands rotate over sets past twice the L2; --reps rounds, each timing every kernel in turn (so each round
// pairs each kernel with plain at nearly the same clocks), with NVML's SM clock after each batch.
//
//   ./cutlass_receipt [--shapes NAME,...] [--reps 9] [--only KERNEL,...] [--profile]
//   --profile: no gates, one launch of each selected kernel at the first selected shape (for ncu)
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "cutlass_receipt.cuh"
#include "phi.cuh"

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            exit(2);                                                                       \
        }                                                                                  \
    } while (0)

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
    return x;
}
// finite E4M3 bytes of exponent field 4 ... 10 (2^-3 ... 2^3), either sign (receipt_gemm.cu's operands)
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

static double e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
    return b & 0x80 ? -v : v;
}
static uint16_t bf16_rn(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
static uint32_t phi_host(int var, int rounds, uint32_t y, uint32_t p, uint32_t s, uint32_t k) {
    auto one = [&](uint32_t y_, uint32_t p_, uint32_t k_) { return var == 0 ? phi1<0>(y_, p_, s, k_) : phi1<1>(y_, p_, s, k_); };
    y = one(y, p, k);
    for (int r = 1; r < rounds; r++) y = one(y, 0x7f4a7c15u, k ^ (0x5851f42du * (uint32_t)r));
    return y;
}
static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// NVML's SM clock of this device, by UUID (dlopen'ed, so the binary needs no NVML at link time)
struct Clock {
    using Init = int (*)();
    using ByUuid = int (*)(const char*, void**);
    using Info = int (*)(void*, int, unsigned*);
    Info info = nullptr;
    void* dev = nullptr;
    explicit Clock(const cudaDeviceProp& p) {
        void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!h) return;
        auto init = (Init)dlsym(h, "nvmlInit_v2");
        auto by = (ByUuid)dlsym(h, "nvmlDeviceGetHandleByUUID");
        info = (Info)dlsym(h, "nvmlDeviceGetClockInfo");
        if (!init || !by || !info || init() != 0) { info = nullptr; return; }
        const unsigned char* u = (const unsigned char*)p.uuid.bytes;
        char s[64];
        snprintf(s, sizeof s, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1], u[2], u[3],
                 u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
        if (by(s, &dev) != 0) info = nullptr;
    }
    double mhz() const {
        unsigned v = 0;
        return info && info(dev, 1 /* NVML_CLOCK_SM */, &v) == 0 ? (double)v : 0.0;
    }
};

struct Shape {
    const char* name;
    int m, n, k;
};
// Llama-3.1-8B's FP8 linears per layer (qkv, o, gate_up, down) at prefill 2,048 and 8,192 and decode batch 32, and a cube
static const Shape SHAPES[] = {
    {"p2048_qkv", 2048, 6144, 4096},  {"p2048_o", 2048, 4096, 4096},  {"p2048_gateup", 2048, 28672, 4096},
    {"p2048_down", 2048, 4096, 14336}, {"p8192_qkv", 8192, 6144, 4096}, {"p8192_o", 8192, 4096, 4096},
    {"p8192_gateup", 8192, 28672, 4096}, {"p8192_down", 8192, 4096, 14336}, {"d32_qkv", 32, 6144, 4096},
    {"d32_o", 32, 4096, 4096},        {"d32_gateup", 32, 28672, 4096}, {"d32_down", 32, 4096, 14336},
    {"cube8192", 8192, 8192, 8192},
};

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        if (b > a) v.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return v;
}
static bool in(const std::vector<std::string>& v, const char* s) { return std::find(v.begin(), v.end(), s) != v.end(); }

static std::vector<std::string> gates_failed;
static void gate(const char* what, const char* kernel, bool bad) {
    if (bad) {
        fprintf(stderr, "GATE FAIL %s: %s\n", kernel, what);
        gates_failed.push_back(std::string(kernel) + ": " + what);
    }
}

struct Bufs {
    uint8_t *A = nullptr, *B = nullptr;
    float *sa = nullptr, *sb = nullptr;
    void* D = nullptr;
    uint32_t* R = nullptr;
    float* dump = nullptr;
    uint32_t* map = nullptr;
};

static cr_args args_of(const Shape& s, const Bufs& b, int set, void* D) {
    return cr_args{s.m, s.n, s.k, b.A + (size_t)set * s.m * s.k, b.B + (size_t)set * s.n * s.k, b.sa, b.sb, D, b.R, b.dump, b.map};
}

static unsigned long long count_diff(const void* a, const void* b, size_t words, unsigned long long* dc) {
    CK(cudaMemset(dc, 0, 8));
    diff<<<1024, 256>>>((const uint32_t*)a, (const uint32_t*)b, words, dc);
    unsigned long long c = 0;
    CK(cudaMemcpy(&c, dc, 8, cudaMemcpyDeviceToHost));
    return c;
}

// The small-shape gates of one configuration (see the header)
static void gate_cfg(int cfg, const std::vector<const cr_kernel*>& ks, unsigned long long* dc) {
    const cr_kernel* plain = nullptr;
    const cr_kernel* dumpk = nullptr;
    for (auto* k : ks)
        if (k->cfg == cfg) {
            if (k->mode == 0) plain = k;
            if (k->mode == 2) dumpk = k;
        }
    if (!plain || !dumpk) {
        gate("no plain or dump kernel", cfg ? "cfg1" : "cfg0", true);
        return;
    }
    const Shape s = cfg == 0 ? Shape{"gate0", 256, 384, 2048} : Shape{"gate1", 32, 256, 2048};
    const int KT = s.k / 128, TM = plain->tile_m, TN = plain->tile_n, NT = plain->threads, CH = plain->chains;
    const int tiles = (s.m / TM) * (s.n / TN);
    Bufs b;
    CK(cudaMalloc(&b.A, (size_t)s.m * s.k));
    CK(cudaMalloc(&b.B, (size_t)s.n * s.k));
    gen<<<256, 256>>>(b.A, (size_t)s.m * s.k, 0x5eed0001u + cfg);
    gen<<<256, 256>>>(b.B, (size_t)s.n * s.k, 0x5eed0101u + cfg);
    std::vector<float> sa(s.m), sb(s.n);
    for (int i = 0; i < s.m; i++) sa[i] = 0.5f + (float)((i * 2654435761u) % 1000u) / 666.0f;
    for (int i = 0; i < s.n; i++) sb[i] = 0.5f + (float)((i * 40503u + 7u) % 1000u) / 666.0f;
    CK(cudaMalloc(&b.sa, s.m * 4));
    CK(cudaMalloc(&b.sb, s.n * 4));
    CK(cudaMemcpy(b.sa, sa.data(), s.m * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.sb, sb.data(), s.n * 4, cudaMemcpyHostToDevice));
    void *D0, *D1;
    const size_t dbytes = (size_t)s.m * s.n * 2;
    CK(cudaMalloc(&D0, dbytes));
    CK(cudaMalloc(&D1, dbytes));
    CK(cudaMalloc(&b.R, (size_t)tiles * NT * 4));
    CK(cudaMalloc(&b.dump, (size_t)KT * s.m * s.n * 4));
    CK(cudaMalloc(&b.map, (size_t)NT * CH * 4 * 4));
    CK(cudaMemset(b.map, 0xff, (size_t)NT * CH * 4 * 4));
    auto run = [&](const cr_kernel* k, void* D) {
        int err = 0;
        cr_impl* g = k->make(args_of(s, b, 0, D), &err);
        if (!g) {
            gate("make failed", k->name, true);
            return false;
        }
        const int rc = g->launch(nullptr);
        CK(cudaDeviceSynchronize());
        delete g;
        gate("launch failed", k->name, rc != 0);
        return rc == 0;
    };
    if (!run(plain, D0) || !run(dumpk, D1)) return;
    gate("output differs from plain's", dumpk->name, count_diff(D0, D1, dbytes / 4, dc) != 0);

    std::vector<uint8_t> A((size_t)s.m * s.k), B((size_t)s.n * s.k);
    CK(cudaMemcpy(A.data(), b.A, A.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(B.data(), b.B, B.size(), cudaMemcpyDeviceToHost));
    std::vector<float> dump((size_t)KT * s.m * s.n);
    std::vector<uint32_t> map((size_t)NT * CH * 4);
    std::vector<uint16_t> D(s.m * (size_t)s.n);
    CK(cudaMemcpy(dump.data(), b.dump, dump.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(map.data(), b.map, map.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(D.data(), D0, dbytes, cudaMemcpyDeviceToHost));
    std::vector<double> av(A.size()), bv(B.size());
    for (size_t i = 0; i < A.size(); i++) av[i] = e4m3(A[i]);
    for (size_t i = 0; i < B.size(); i++) bv[i] = e4m3(B[i]);
    double worst = 0;
    long bad_prefix = 0, bad_d = 0;
    for (int r = 0; r < s.m; r++)
        for (int c = 0; c < s.n; c++) {
            double sum = 0, mag = 0;
            for (int kt = 0; kt < KT; kt++) {
                for (int kk = kt * 128; kk < kt * 128 + 128; kk++) {
                    const double p = av[(size_t)r * s.k + kk] * bv[(size_t)c * s.k + kk];
                    sum += p;
                    mag += std::fabs(p);
                }
                const double got = dump[((size_t)kt * s.m + r) * s.n + c], e = std::fabs(got - sum) / std::max(mag, 1e-30);
                worst = std::max(worst, e);
                bad_prefix += e > 1e-5;
            }
            const float acc = dump[((size_t)(KT - 1) * s.m + r) * s.n + c];
            const float t = sb[c] * acc, u = sa[r] * t;
            bad_d += bf16_rn(u) != D[(size_t)r * s.n + c];
        }
    gate("a prefix sum is off the host's by more than 1e-5 of sum |ab|", dumpk->name, bad_prefix != 0);
    gate("plain's output isn't bf16(sa * (sb * the dump's final prefix))", plain->name, bad_d != 0);
    std::vector<int> seen(TM * TN, 0);
    for (uint32_t w : map)
        if ((w >> 16) < (uint32_t)TM && (w & 0xffff) < (uint32_t)TN) seen[(w >> 16) * TN + (w & 0xffff)]++;
    gate("the chain-word map doesn't cover the tile once", dumpk->name,
         std::any_of(seen.begin(), seen.end(), [](int v) { return v != 1; }));
    printf("    {\"gate\": \"cfg%d\", \"m\": %d, \"n\": %d, \"k\": %d, \"prefix_err_max\": %.3e, \"prefix_bad\": %ld, \"d_bad\": %ld},\n",
           cfg, s.m, s.n, s.k, worst, bad_prefix, bad_d);

    for (auto* k : ks) {
        if (k->cfg != cfg || k->mode != 1) continue;
        CK(cudaMemset(b.R, 0, (size_t)tiles * NT * 4));
        if (!run(k, D1)) continue;
        gate("output differs from plain's", k->name, count_diff(D0, D1, dbytes / 4, dc) != 0);
        std::vector<uint32_t> R((size_t)tiles * NT);
        CK(cudaMemcpy(R.data(), b.R, R.size() * 4, cudaMemcpyDeviceToHost));
        long bad = 0;
        for (int tm = 0; tm < s.m / TM; tm++)
            for (int tn = 0; tn < s.n / TN; tn++)
                for (int t = 0; t < NT; t++) {
                    uint32_t f = 0;
                    for (int ch = 0; ch < CH; ch++) {
                        uint32_t y = 0;
                        for (int kt = 0; kt < KT; kt++) {
                            if ((kt + 1) % k->p) continue;
                            for (int e = 0; e < 4; e++) {
                                const uint32_t w = map[((size_t)t * CH + ch) * 4 + e];
                                const int r = tm * TM + (int)(w >> 16), c = tn * TN + (int)(w & 0xffff);
                                y = phi_host(k->var, k->rounds, y, f2u(dump[((size_t)kt * s.m + r) * s.n + c]),
                                             (uint32_t)(kt + 1), key(ch, e));
                            }
                        }
                        f = rotl13(f) ^ y;
                    }
                    bad += R[((size_t)tm * (s.n / TN) + tn) * NT + t] != f;
                }
        gate("receipt words differ from the host's chains over the dump", k->name, bad != 0);
    }
    for (void* p : {(void*)b.A, (void*)b.B, (void*)b.sa, (void*)b.sb, D0, D1, (void*)b.R, (void*)b.dump, (void*)b.map})
        CK(cudaFree(p));
}

int main(int argc, char** argv) {
    std::vector<std::string> shapes_sel, only;
    int reps = 9;
    bool profile = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--shapes" && i + 1 < argc) shapes_sel = split(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "--only" && i + 1 < argc) only = split(argv[++i]);
        else if (a == "--profile") profile = true;
        else {
            fprintf(stderr, "usage: %s [--shapes NAME,...] [--reps N] [--only KERNEL,...] [--profile]\n", argv[0]);
            return 2;
        }
    }
    std::vector<const cr_kernel*> ks = cr_registry();
    std::sort(ks.begin(), ks.end(), [](const cr_kernel* a, const cr_kernel* b) {
        return std::make_tuple(a->cfg, a->mode != 0, std::string(a->name)) < std::make_tuple(b->cfg, b->mode != 0, std::string(b->name));
    });
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    Clock clock(prop);
    unsigned long long* dc;
    CK(cudaMalloc(&dc, 8));

    printf("{\"schema\": \"verity-pouw/saturation-cutlass-receipt/v1\", \"device\": \"%s\", \"sms\": %d, \"l2_bytes\": %d,\n",
           prop.name, prop.multiProcessorCount, prop.l2CacheSize);
    printf("  \"kernels\": [\n");
    for (size_t i = 0; i < ks.size(); i++) {
        auto* k = ks[i];
        printf("    {\"name\": \"%s\", \"cfg\": %d, \"var\": %d, \"rounds\": %d, \"p\": %d, \"L\": %d, \"mode\": %d, \"tile\": [%d, %d], "
               "\"threads\": %d, \"chains\": %d, \"stages\": %d, \"smem_bytes\": %d}%s\n",
               k->name, k->cfg, k->var, k->rounds, k->p, 128 * k->p, k->mode, k->tile_m, k->tile_n, k->threads, k->chains,
               k->stages, k->smem_bytes, i + 1 < ks.size() ? "," : "");
    }
    printf("  ],\n  \"gates\": [\n");
    if (!profile) {
        gate_cfg(0, ks, dc);
        gate_cfg(1, ks, dc);
    }
    printf("    {}\n  ],\n");
    fflush(stdout);
    if (!gates_failed.empty()) {
        printf("  \"failed\": %zu\n}\n", gates_failed.size());
        return 3;
    }

    const size_t L2 = (size_t)prop.l2CacheSize;
    printf("  \"rows\": [\n");
    bool first_row = true;
    for (const Shape& s : SHAPES) {
        if (!shapes_sel.empty() && !in(shapes_sel, s.name)) continue;
        const int cfg = s.m >= 128 ? 0 : 1;
        std::vector<const cr_kernel*> sk;
        for (auto* k : ks)
            if (k->cfg == cfg && k->mode != 2 && (k->mode == 0 || only.empty() || in(only, k->name))) sk.push_back(k);
        const int TM = sk[0]->tile_m, TN = sk[0]->tile_n;
        const int tiles = (s.m / TM) * (s.n / TN), NT = sk[0]->threads;
        const size_t ab = (size_t)s.m * s.k + (size_t)s.n * s.k;
        const int sets = profile ? 1 : (int)std::min<size_t>(16, std::max<size_t>(1, (2 * L2 + ab - 1) / ab));
        Bufs b;
        CK(cudaMalloc(&b.A, (size_t)sets * s.m * s.k));
        CK(cudaMalloc(&b.B, (size_t)sets * s.n * s.k));
        gen<<<1024, 256>>>(b.A, (size_t)sets * s.m * s.k, 0xa0000000u + s.m);
        gen<<<1024, 256>>>(b.B, (size_t)sets * s.n * s.k, 0xb0000000u + s.n + s.k);
        std::vector<float> sa(s.m, 1.0f), sb(s.n, 1.0f);
        CK(cudaMalloc(&b.sa, s.m * 4));
        CK(cudaMalloc(&b.sb, s.n * 4));
        CK(cudaMemcpy(b.sa, sa.data(), s.m * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.sb, sb.data(), s.n * 4, cudaMemcpyHostToDevice));
        const size_t dbytes = (size_t)s.m * s.n * 2;
        void *D, *D0;
        CK(cudaMalloc(&D, dbytes));
        CK(cudaMalloc(&D0, dbytes));
        CK(cudaMalloc(&b.R, (size_t)tiles * NT * 4));
        std::vector<std::vector<cr_impl*>> impl(sk.size());
        for (size_t j = 0; j < sk.size(); j++)
            for (int t = 0; t < sets; t++) {
                int err = 0;
                cr_impl* g = sk[j]->make(args_of(s, b, t, D), &err);
                if (!g) {
                    fprintf(stderr, "%s at %s: make error %d\n", sk[j]->name, s.name, err);
                    return 2;
                }
                impl[j].push_back(g);
            }
        auto launch = [&](size_t j, int t) {
            if (impl[j][t % sets]->launch(nullptr) != 0) {
                fprintf(stderr, "%s at %s: launch failed\n", sk[j]->name, s.name);
                exit(2);
            }
        };
        if (profile) {
            for (size_t j = 0; j < sk.size(); j++) launch(j, 0);
            CK(cudaDeviceSynchronize());
            printf("    {}\n  ]\n}\n");
            return 0;
        }
        launch(0, 0);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(D0, D, dbytes, cudaMemcpyDeviceToDevice));
        std::vector<uint32_t> r0((size_t)tiles * NT), r1(r0.size());
        for (size_t j = 1; j < sk.size(); j++) {
            CK(cudaMemset(b.R, 0, r0.size() * 4));
            launch(j, 0);
            CK(cudaDeviceSynchronize());
            gate("output differs from plain's at a timed shape", sk[j]->name, count_diff(D0, D, dbytes / 4, dc) != 0);
            CK(cudaMemcpy(r0.data(), b.R, r0.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemset(b.R, 0, r0.size() * 4));
            launch(j, 0);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(r1.data(), b.R, r1.size() * 4, cudaMemcpyDeviceToHost));
            gate("a second launch's receipt words differ", sk[j]->name, r0 != r1);
            gate("every receipt word is zero", sk[j]->name, std::all_of(r0.begin(), r0.end(), [](uint32_t w) { return w == 0; }));
        }
        if (!gates_failed.empty()) {
            printf("    {}\n  ],\n  \"failed\": %zu\n}\n", gates_failed.size());
            return 3;
        }
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        auto batch = [&](size_t j, int launches) {
            CK(cudaEventRecord(e0));
            for (int t = 0; t < launches; t++) launch(j, t);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            return (double)ms / launches;
        };
        const double pilot = batch(0, sets);
        const int launches = std::max(sets, (int)std::ceil(25.0 / std::max(pilot, 1e-3) / sets) * sets);
        for (size_t j = 0; j < sk.size(); j++) batch(j, launches);
        std::vector<std::vector<double>> ms(sk.size()), mhz(sk.size()), wall(sk.size()), cyc(sk.size());
        for (int r = 0; r < reps; r++)
            for (size_t j = 0; j < sk.size(); j++) {
                ms[j].push_back(batch(j, launches));
                mhz[j].push_back(clock.mhz());
            }
        for (int r = 0; r < reps; r++)
            for (size_t j = 0; j < sk.size(); j++) {
                wall[j].push_back(ms[j][r] / ms[0][r]);
                cyc[j].push_back(ms[j][r] * mhz[j][r] / std::max(ms[0][r] * mhz[0][r], 1e-30));
            }
        for (size_t j = 0; j < sk.size(); j++) {
            auto* k = sk[j];
            const double t = med(ms[j]), f = med(mhz[j]);
            const double words = k->mode == 1 ? (double)s.m * s.n * (s.k / (128.0 * k->p)) : 0.0;
            printf("%s    {\"shape\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"kernel\": \"%s\", \"cfg\": %d, \"var\": %d, "
                   "\"rounds\": %d, \"p\": %d, \"L\": %d, \"sets\": %d, \"launches\": %d, \"reps\": %d, \"ms\": %.5f, "
                   "\"ms_min\": %.5f, \"tflops\": %.1f, \"sm_mhz\": %.0f, \"macs_per_sm_clk\": %.1f, \"receipt_words\": %.0f, "
                   "\"phi1_per_sm_clk\": %.3f, \"wall_vs_plain\": %.4f, \"cycles_vs_plain\": %.4f}",
                   first_row ? "" : ",\n", s.name, s.m, s.n, s.k, k->name, k->cfg, k->var, k->rounds, k->p, 128 * k->p, sets,
                   launches, reps, t, *std::min_element(ms[j].begin(), ms[j].end()), 2.0 * s.m * s.n * (double)s.k / (t * 1e9), f,
                   f > 0 ? (double)s.m * s.n * s.k / (t * 1e-3 * f * 1e6 * prop.multiProcessorCount) : 0.0, words,
                   f > 0 ? words * k->rounds / (t * 1e-3 * f * 1e6 * prop.multiProcessorCount) : 0.0, med(wall[j]),
                   med(cyc[j]));
            first_row = false;
            fflush(stdout);
        }
        for (auto& v : impl)
            for (auto* g : v) delete g;
        for (void* p : {(void*)b.A, (void*)b.B, (void*)b.sa, (void*)b.sb, D, D0, (void*)b.R}) CK(cudaFree(p));
        CK(cudaEventDestroy(e0));
        CK(cudaEventDestroy(e1));
    }
    printf("\n  ]\n}\n");
    return 0;
}
