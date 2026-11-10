// cuda/ring_switch.cuh's tensor-core fold and combine against its split ones, for test_ring_switch.py. rs_combine_check draws eq
// halves for 2^d rows split at k and both claims' weights from seed, runs chunk_ring_combine_basis_split and
// chunk_ring_combine_basis_bmma, and counts the rows that differ in bad[0]; rs_fold_check draws a witness of 2^d rows and eq
// halves, folds it with chunk_ring_fold_rows_split and chunk_ring_fold_rows_bmma, each through an XOR of its partials, and counts
// the 128 sums that differ in bad[0].
// Built with -DRS_MAIN it is also `ring_switch D REPS`: a JSON line per check at small sizes, then one with each kernel's mean time
// over REPS at 2^D rows split at D / 2, launched as the prover launches them, and the rows and sums that differ.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../cuda/ring_switch.cuh"

__host__ __device__ inline uint64_t rs_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

__global__ void rs_fill(uint64_t* p, long long n, uint64_t seed) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        p[i] = rs_mix(seed ^ rs_mix((uint64_t)i));
}

__global__ void rs_diff(const F128* a, const F128* b, long long n, unsigned long long* bad) {
    unsigned long long k = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        k += a[i].lo != b[i].lo || a[i].hi != b[i].hi;
    if (k) atomicAdd(bad, k);
}

#define RS_CK(x) do { if ((x) != cudaSuccess) return 1; } while (0)

struct RsCombine {
    int d = 0, k = 0;
    F128 *lo_ab = nullptr, *hi_ab = nullptr, *lo_c = nullptr, *hi_c = nullptr, *w = nullptr, *split = nullptr, *bmma = nullptr;
    unsigned long long* bad = nullptr;
    int alloc(int d_, int k_, uint64_t seed) {
        d = d_; k = k_;
        const long long n = 1LL << d, nlo = 1LL << k, nhi = 1LL << (d - k);
        RS_CK(cudaMalloc(&lo_ab, nlo * 16)); RS_CK(cudaMalloc(&hi_ab, nhi * 16));
        RS_CK(cudaMalloc(&lo_c, nlo * 16)); RS_CK(cudaMalloc(&hi_c, nhi * 16));
        RS_CK(cudaMalloc(&w, 256 * 16)); RS_CK(cudaMalloc(&split, n * 16)); RS_CK(cudaMalloc(&bmma, n * 16));
        RS_CK(cudaMalloc(&bad, sizeof(unsigned long long)));
        int s = 0;
        for (auto [p, words] : {std::pair<F128*, long long>{lo_ab, nlo}, {hi_ab, nhi}, {lo_c, nlo}, {hi_c, nhi}, {w, 256}})
            rs_fill<<<256, 256>>>((uint64_t*)p, 2 * words, seed ^ (0x1000ull * ++s));
        RS_CK(cudaMemset(split, 0xA5, n * 16));
        RS_CK(cudaMemset(bmma, 0x5A, n * 16));
        return cudaDeviceSynchronize() != cudaSuccess;
    }
    void run_split() {
        const long long n = 1LL << d;
        chunk_ring_combine_basis_split<<<(unsigned)std::min<long long>((n + 255) / 256, 4096), 256>>>(lo_ab, hi_ab, lo_c, hi_c, k, w,
                                                                                                       w + 128, n, split);
    }
    void run_bmma() {
        const long long nb = 1LL << (d - k);
        chunk_ring_combine_basis_bmma<<<(unsigned)std::min<long long>(nb, 4096), 256>>>(lo_ab, hi_ab, lo_c, hi_c, k, w, w + 128, nb,
                                                                                         bmma);
    }
    long long diff() {
        cudaMemset(bad, 0, sizeof(unsigned long long));
        rs_diff<<<1024, 256>>>(split, bmma, 1LL << d, bad);
        unsigned long long h = ~0ull;
        if (cudaMemcpy(&h, bad, sizeof h, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
        return (long long)h;
    }
    ~RsCombine() {
        for (void* p : {(void*)lo_ab, (void*)hi_ab, (void*)lo_c, (void*)hi_c, (void*)w, (void*)split, (void*)bmma, (void*)bad})
            if (p) cudaFree(p);
    }
};

__global__ void rs_reduce(const F128* partials, int chunks, F128* out) {
    const int t = threadIdx.x;
    F128 x{0, 0};
    for (int c = 0; c < chunks; c++) { x.lo ^= partials[(long long)t * chunks + c].lo; x.hi ^= partials[(long long)t * chunks + c].hi; }
    out[t] = x;
}

struct RsFold {
    static constexpr int CHUNKS = 1024;   // the prover's ring_switch_chunks
    int d = 0, k = 0, bmma_chunks = 0;
    F128 *witness = nullptr, *lo = nullptr, *hi = nullptr, *partials = nullptr, *sums = nullptr;
    int alloc(int d_, int k_, uint64_t seed) {
        d = d_; k = k_;
        const long long n = 1LL << d, nlo = 1LL << k, nhi = 1LL << (d - k);
        bmma_chunks = ring_fold_bmma_blocks(k, nhi, CHUNKS);
        RS_CK(cudaMalloc(&witness, n * 16)); RS_CK(cudaMalloc(&lo, nlo * 16)); RS_CK(cudaMalloc(&hi, nhi * 16));
        RS_CK(cudaMalloc(&partials, 128LL * CHUNKS * 16)); RS_CK(cudaMalloc(&sums, 2 * 128 * 16));
        int s = 0;
        for (auto [p, words] : {std::pair<F128*, long long>{witness, n}, {lo, nlo}, {hi, nhi}})
            rs_fill<<<1024, 256>>>((uint64_t*)p, 2 * words, seed ^ (0x2000ull * ++s));
        return cudaDeviceSynchronize() != cudaSuccess;
    }
    void run_split() {
        chunk_ring_fold_rows_split<<<CHUNKS, 256>>>(witness, lo, hi, k, 1LL << d, CHUNKS, partials);
        rs_reduce<<<1, 128>>>(partials, CHUNKS, sums);
    }
    void run_bmma() {
        chunk_ring_fold_rows_bmma<<<bmma_chunks, 256>>>(witness, lo, hi, k, 1LL << (d - k), partials);
        rs_reduce<<<1, 128>>>(partials, bmma_chunks, sums + 128);
    }
    long long diff() {
        F128 h[256];
        if (cudaMemcpy(h, sums, sizeof h, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
        long long bad = 0;
        for (int t = 0; t < 128; t++) bad += h[t].lo != h[128 + t].lo || h[t].hi != h[128 + t].hi;
        return bad;
    }
    ~RsFold() {
        for (void* p : {(void*)witness, (void*)lo, (void*)hi, (void*)partials, (void*)sums})
            if (p) cudaFree(p);
    }
};

extern "C" int rs_fold_check(int d, int k, uint64_t seed, long long* bad) {
    if (k < 8 || k > d) return 2;
    RsFold f;
    if (f.alloc(d, k, seed)) return 1;
    f.run_split();
    f.run_bmma();
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    bad[0] = f.diff();
    return bad[0] < 0;
}

extern "C" int rs_combine_check(int d, int k, uint64_t seed, long long* bad) {
    if (k < 4 || k > d) return 2;
    RsCombine c;
    if (c.alloc(d, k, seed)) return 1;
    c.run_split();
    c.run_bmma();
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    bad[0] = c.diff();
    return bad[0] < 0;
}

#ifdef RS_MAIN
template <typename F>
static double rs_time(int reps, F&& f) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    f();
    cudaEventRecord(a);
    for (int r = 0; r < reps; r++) f();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

int main(int argc, char** argv) {
    const int d = argc > 1 ? atoi(argv[1]) : 28, reps = argc > 2 ? atoi(argv[2]) : 5, k = d / 2;
    for (auto [cd, ck] : {std::pair<int, int>{8, 4}, {9, 4}, {12, 4}, {12, 6}, {12, 10}, {16, 8}, {20, 10}, {21, 10}})
        for (uint64_t seed : {1, 2}) {
            long long bad[1] = {-1};
            const int rc = rs_combine_check(cd, ck, seed, bad);
            printf("{\"check\": \"combine\", \"d\": %d, \"k\": %d, \"seed\": %llu, \"rc\": %d, \"rows_differ\": %lld}\n", cd, ck,
                   (unsigned long long)seed, rc, bad[0]);
        }
    for (auto [cd, ck] : {std::pair<int, int>{8, 8}, {12, 8}, {16, 8}, {16, 11}, {16, 12}, {20, 10}, {22, 11}, {24, 12}})
        for (uint64_t seed : {1, 2}) {
            long long bad[1] = {-1};
            const int rc = rs_fold_check(cd, ck, seed, bad);
            printf("{\"check\": \"fold\", \"d\": %d, \"k\": %d, \"seed\": %llu, \"rc\": %d, \"sums_differ\": %lld}\n", cd, ck,
                   (unsigned long long)seed, rc, bad[0]);
        }
    double fold_split = 0, fold_bmma = 0, split = 0, bmma = 0;
    long long fold_bad = -1, rows_bad = -1;
    {
        RsFold f;
        if (f.alloc(d, k, 7)) { printf("{\"error\": \"alloc\"}\n"); return 1; }
        fold_split = rs_time(reps, [&] { f.run_split(); });
        fold_bmma = rs_time(reps, [&] { f.run_bmma(); });
        cudaDeviceSynchronize();
        fold_bad = f.diff();
    }
    {
        RsCombine c;
        if (c.alloc(d, k, 7)) { printf("{\"error\": \"alloc\"}\n"); return 1; }
        split = rs_time(reps, [&] { c.run_split(); });
        bmma = rs_time(reps, [&] { c.run_bmma(); });
        cudaDeviceSynchronize();
        rows_bad = c.diff();
    }
    const cudaError_t e = cudaDeviceSynchronize();
    printf("{\"d\": %d, \"k\": %d, \"reps\": %d, \"fold_split_ms\": %.3f, \"fold_bmma_ms\": %.3f, \"fold_sums_differ\": %lld, "
           "\"combine_split_ms\": %.3f, \"combine_bmma_ms\": %.3f, \"combine_rows_differ\": %lld, \"cuda\": \"%s\"}\n",
           d, k, reps, fold_split, fold_bmma, fold_bad, split, bmma, rows_bad, cudaGetErrorString(e));
    return 0;
}
#endif
