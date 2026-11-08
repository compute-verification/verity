// hm96_finish_leaves (verity/ml/flock/cuda/sha512.cuh) timed: `hm96_leaves LOG_N REPS` prints one JSON line with the mean time
// a launch over REPS launches of 2^LOG_N leaves at 256 threads a block, from a fixed draw of digests, salts and tables, and a
// 64-bit FNV-1a of the leaves. Built against another tree's sha512.cuh (its cuda/ directory first on the include path), it
// prints the same FNV when that kernel's leaves are the same. From the tree's root, on a leased GPU:
//
//   nvcc -O3 -std=c++17 -arch=sm_90 -I verity/ml/flock/cuda -o hm96_leaves benchmarks/flock_kernels/hm96_leaves.cu
//   ./hm96_leaves 23 20
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#include "sha512.cuh"

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
