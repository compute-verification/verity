// cuda/zc_round1.cuh against the table kernel it replaces (cuda-ghash/zerocheck_round1_cpustyle.cuh), for test_zc_round1.py.
// zr_check draws a packed witness of n_out outer chunks (128 columns of a, b and z each), eq_out, the extension's columns and
// the scale from seed, runs launch_zerocheck_first_round_cpu_structured and launch_zc_round1 on them, and counts the words
// that differ: bad[0] of the 64 a·b sums, bad[1] of the 64 c sums.
// Built with -DZR_MAIN it is also `zc_round1 N_OUT REPS`: one JSON line with each path's mean time over REPS and the words
// that differ.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "phi8_table.cuh"
#include "../cuda/zc_round1.cuh"

__host__ __device__ inline uint64_t zr_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

__global__ void zr_fill(uint64_t* p, long long n, uint64_t seed) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        p[i] = zr_mix(seed ^ zr_mix((uint64_t)i));
}

static uint8_t zr_gf8_mul(uint8_t x, uint8_t y) {
    uint8_t p = 0;
    for (int i = 0; i < 8; i++) {
        if ((y >> i) & 1) p ^= x;
        x = (uint8_t)((x << 1) ^ ((x >> 7) * 0x1b));
    }
    return p;
}

struct ZrCase {
    long long n_out = 0;
    uint8_t *a = nullptr, *b = nullptr, *z = nullptr;
    F128 *eq = nullptr, *ab = nullptr, *c = nullptr;
    F128 scale{};
};

// GF(2^8)'s products and φ8 as the prover uploads them, and the extension's 64 columns drawn from seed
static cudaError_t zr_setup(ZrCase& c, long long n_out, uint64_t seed) {
    c.n_out = n_out;
    cudaError_t e = cudaSuccess;
    auto get = [&](auto** p, size_t bytes) { if (e == cudaSuccess) e = cudaMalloc((void**)p, bytes); };
    get(&c.a, n_out * 1024); get(&c.b, n_out * 1024); get(&c.z, n_out * 1024);
    get(&c.eq, n_out * sizeof(F128)); get(&c.ab, 64 * sizeof(F128)); get(&c.c, 64 * sizeof(F128));
    if (e != cudaSuccess) return e;
    zr_fill<<<1024, 256>>>((uint64_t*)c.a, n_out * 128, seed ^ 1);
    zr_fill<<<1024, 256>>>((uint64_t*)c.b, n_out * 128, seed ^ 2);
    zr_fill<<<1024, 256>>>((uint64_t*)c.z, n_out * 128, seed ^ 3);
    zr_fill<<<1024, 256>>>((uint64_t*)c.eq, n_out * 2, seed ^ 4);
    static uint8_t f8mul[256 * 256], mcol[64 * 64];
    for (int x = 0; x < 256; x++)
        for (int y = 0; y < 256; y++) f8mul[x * 256 + y] = zr_gf8_mul((uint8_t)x, (uint8_t)y);
    for (int i = 0; i < 64 * 64; i++) mcol[i] = (uint8_t)zr_mix(seed ^ 5 ^ zr_mix((uint64_t)i));
    c.scale = F128{zr_mix(seed ^ 6), zr_mix(seed ^ 7)};
    e = upload_zerocheck_first_round_tables(mcol, f8mul, PHI_8_TABLE);
    return e == cudaSuccess ? cudaDeviceSynchronize() : e;
}

static void zr_free(ZrCase& c) {
    for (void* p : {(void*)c.a, (void*)c.b, (void*)c.z, (void*)c.eq, (void*)c.ab, (void*)c.c}) cudaFree(p);
}

static void ref_round1(ZrCase& c) {
    launch_zerocheck_first_round_cpu_structured(c.a, c.b, c.z, c.eq, c.n_out, c.scale, c.ab, c.c);
}
static void zlin_round1(ZrCase& c) { launch_zc_round1(c.a, c.b, c.z, c.eq, c.n_out, c.scale, c.ab, c.c); }

// both paths' 128 words (a·b's, then c's) and the words that differ
static cudaError_t zr_compare(ZrCase& c, long long* bad) {
    F128 want[128], got[128];
    ref_round1(c);
    cudaError_t e = cudaMemcpy(want, c.ab, 64 * sizeof(F128), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(want + 64, c.c, 64 * sizeof(F128), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return e;
    zlin_round1(c);
    e = cudaMemcpy(got, c.ab, 64 * sizeof(F128), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(got + 64, c.c, 64 * sizeof(F128), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return e;
    bad[0] = bad[1] = 0;
    for (int i = 0; i < 128; i++) bad[i / 64] += want[i].lo != got[i].lo || want[i].hi != got[i].hi;
    return cudaGetLastError();
}

extern "C" int zr_check(long long n_out, uint64_t seed, long long* bad) {
    ZrCase c;
    cudaError_t e = zr_setup(c, n_out, seed);
    if (e == cudaSuccess) e = zr_compare(c, bad);
    zr_free(c);
    return (int)e;
}

#ifdef ZR_MAIN
template <class F>
static double mean_ms(F f, int reps) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    f();
    cudaEventRecord(t0);
    for (int i = 0; i < reps; i++) f();
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / reps;
}

int main(int argc, char** argv) {
    const long long n_out = argc > 1 ? atoll(argv[1]) : 1LL << 22;
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    ZrCase c;
    cudaError_t e = zr_setup(c, n_out, 0x5a17);
    if (e != cudaSuccess) { fprintf(stderr, "setup: %s\n", cudaGetErrorString(e)); return 1; }
    long long bad[2];
    e = zr_compare(c, bad);
    if (e != cudaSuccess) { fprintf(stderr, "compare: %s\n", cudaGetErrorString(e)); return 1; }
    printf("{\"n_out\": %lld, \"reps\": %d, \"tables_ms\": %.3f, \"zlin_ms\": %.3f, \"bad\": [%lld, %lld]}\n", n_out, reps,
           mean_ms([&] { ref_round1(c); }, reps), mean_ms([&] { zlin_round1(c); }, reps), bad[0], bad[1]);
    e = cudaDeviceSynchronize();
    zr_free(c);
    return e == cudaSuccess ? 0 : 1;
}
#endif
