// cuda/sha512.cuh's hm96_finish_leaves, for test_hm96_leaves.py: hm96_leaves finishes leaves [base, n) of n digests at blocks
// of tpb threads, as launch_merkle512 (256) and flock_cuda_hm96_leaves (128) launch it, leaving the digests below base.
// Built with -DHM96_MAIN it is also `hm96_leaves LOG_N REPS`: one JSON line with the mean time a launch over REPS of 2^LOG_N
// leaves at 256 threads, from a fixed draw of digests, salts and tables, and a 64-bit FNV-1a of the leaves, which a build
// against another sha512.cuh prints alike when its leaves agree.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../cuda/sha512.cuh"

extern "C" int hm96_leaves(const uint8_t* x, const uint8_t* salts, long long n, long long base, int tpb, const uint8_t* prefixes,
                           const uint64_t* tables, uint8_t* out) {
    if (n < 1 || base < 0 || base >= n || tpb < 4 || tpb > HM96_TPB || tpb % 4) return 1;
    const size_t tb = (size_t)192 * 256 * 8 * sizeof(uint64_t);
    uint8_t *d_x = nullptr, *d_s = nullptr, *d_pre = nullptr;
    uint64_t *d_t = nullptr, *d_mid = nullptr;
    cudaError_t e = cudaMalloc(&d_x, (size_t)n * 64);
    if (e == cudaSuccess) e = cudaMalloc(&d_s, (size_t)(n - base) * 192);
    if (e == cudaSuccess) e = cudaMalloc(&d_pre, 256);
    if (e == cudaSuccess) e = cudaMalloc(&d_t, tb);
    if (e == cudaSuccess) e = cudaMalloc(&d_mid, 16 * sizeof(uint64_t));
    if (e == cudaSuccess) e = cudaMemcpy(d_x, x, (size_t)n * 64, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(d_s, salts, (size_t)(n - base) * 192, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(d_pre, prefixes, 256, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(d_t, tables, tb, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) {
        hm96_midstates<<<1, 1>>>(d_pre, d_mid);
        const long long m = n - base;
        hm96_finish_leaves<<<(unsigned)((m + tpb - 1) / tpb), tpb>>>(d_x, (const uint32_t*)d_s, base, n, d_t, d_mid);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess) e = cudaMemcpy(out, d_x, (size_t)n * 64, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_s); cudaFree(d_pre); cudaFree(d_t); cudaFree(d_mid);
    return e == cudaSuccess ? 0 : 2;
}

#ifdef HM96_MAIN
#include <vector>

static uint64_t hm_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static std::vector<uint64_t> draw(size_t words, uint64_t seed) {
    std::vector<uint64_t> v(words);
    for (size_t i = 0; i < words; i++) v[i] = hm_mix(seed ^ hm_mix(i));
    return v;
}

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: hm96_leaves LOG_N REPS\n"); return 2; }
    const long long n = 1LL << atoi(argv[1]);
    const int reps = atoi(argv[2]);
    const auto x = draw((size_t)n * 8, 1), s = draw((size_t)n * 24, 2), t = draw((size_t)192 * 256 * 8, 3), pre = draw(32, 4);
    uint8_t *d_x = nullptr, *d_tree = nullptr, *d_s = nullptr, *d_pre = nullptr;
    uint64_t *d_t = nullptr, *d_mid = nullptr;
    cudaMalloc(&d_x, (size_t)n * 64); cudaMalloc(&d_tree, (size_t)n * 64); cudaMalloc(&d_s, (size_t)n * 192);
    cudaMalloc(&d_pre, 256); cudaMalloc(&d_t, t.size() * 8); cudaMalloc(&d_mid, 128);
    cudaMemcpy(d_x, x.data(), (size_t)n * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, s.data(), (size_t)n * 192, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pre, pre.data(), 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, t.data(), t.size() * 8, cudaMemcpyHostToDevice);
    hm96_midstates<<<1, 1>>>(d_pre, d_mid);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float total = 0;
    for (int r = 0; r <= reps; r++) {
        cudaMemcpy(d_tree, d_x, (size_t)n * 64, cudaMemcpyDeviceToDevice);
        cudaEventRecord(a);
        hm96_finish_leaves<<<(unsigned)((n + 255) / 256), 256>>>(d_tree, (const uint32_t*)d_s, 0, n, d_t, d_mid);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        if (r) total += ms;
    }
    const cudaError_t e = cudaGetLastError();
    std::vector<uint8_t> out((size_t)n * 64);
    cudaMemcpy(out.data(), d_tree, out.size(), cudaMemcpyDeviceToHost);
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint8_t c : out) h = (h ^ c) * 0x100000001b3ull;
    printf("{\"log_n\": %d, \"reps\": %d, \"ms\": %.4f, \"fnv\": \"%016llx\", \"error\": \"%s\"}\n", atoi(argv[1]), reps,
           total / reps, (unsigned long long)h, cudaGetErrorString(e));
    return e == cudaSuccess ? 0 : 1;
}
#endif
