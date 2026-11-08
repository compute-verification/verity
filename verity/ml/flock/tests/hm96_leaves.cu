// cuda/sha512.cuh's launch_hm96_finish, for test_hm96_leaves.py: hm96_leaves finishes leaves [base, n) of n digests, hashing
// at blocks of tpb threads as launch_merkle512 (256) and flock_cuda_hm96_leaves (128) do, leaving the digests below base, and
// refuses a tpb hm96_tpb_fits refuses. benchmarks/flock_kernels/hm96_leaves.cu times it.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../cuda/sha512.cuh"

extern "C" int hm96_leaves(const uint8_t* x, const uint8_t* salts, long long n, long long base, int tpb, const uint8_t* prefixes,
                           const uint64_t* tables, uint8_t* out) {
    if (n < 1 || base < 0 || base >= n || !hm96_tpb_fits(tpb)) return 1;
    std::vector<uint32_t> frag(2 * HM96_FRAGS);
    hm96_fragments(tables, frag.data());
    uint8_t *d_x = nullptr, *d_s = nullptr, *d_pre = nullptr;
    uint32_t* d_f = nullptr;
    uint64_t* d_mid = nullptr;
    cudaError_t e = cudaMalloc(&d_x, (size_t)n * 64);
    if (e == cudaSuccess) e = cudaMalloc(&d_s, (size_t)(n - base) * 192);
    if (e == cudaSuccess) e = cudaMalloc(&d_pre, 256);
    if (e == cudaSuccess) e = cudaMalloc(&d_f, frag.size() * sizeof(uint32_t));
    if (e == cudaSuccess) e = cudaMalloc(&d_mid, 16 * sizeof(uint64_t));
    if (e == cudaSuccess) e = cudaMemcpy(d_x, x, (size_t)n * 64, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(d_s, salts, (size_t)(n - base) * 192, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(d_pre, prefixes, 256, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(d_f, frag.data(), frag.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (e == cudaSuccess) {
        hm96_midstates<<<1, 1>>>(d_pre, d_mid);
        launch_hm96_finish(d_x, (const uint32_t*)d_s, base, n, tpb, d_f, d_mid);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess) e = cudaMemcpy(out, d_x, (size_t)n * 64, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_s); cudaFree(d_pre); cudaFree(d_f); cudaFree(d_mid);
    return e == cudaSuccess ? 0 : 2;
}
