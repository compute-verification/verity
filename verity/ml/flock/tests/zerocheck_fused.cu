// cuda/zerocheck_fused.cuh against round 2's fold and the tail's first lookahead (cuda-ghash/zerocheck_round2.cuh and
// zerocheck_tail.cuh), for test_zerocheck_fused.py. zf_check draws a packed witness of 2^log_rows rows for a and b, a
// GF(2)-linear fold table, split-eq tables and coins from seed, runs both paths with the fused kernels reading the table as
// `table` says (fused_round2), and counts the words that differ: bad[0] of round 2's 8 messages, bad[1] of the first
// lookahead's 8 and bad[2] of its folded a and b.
// Built with -DZF_MAIN it is also `zerocheck_fused LOG_ROWS REPS`: one JSON line with the mean time over REPS of round 2 and
// of the first lookahead, on round 2's tables and on the packed witness under each `table`, and the words that differ.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../cuda/zerocheck_fused.cuh"

__host__ __device__ inline uint64_t zf_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

__global__ void zf_fill(uint64_t* p, long long n, uint64_t seed) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        p[i] = zf_mix(seed ^ zf_mix((uint64_t)i));
}

__global__ void zf_diff(const F128* a, const F128* b, long long n, unsigned long long* bad) {
    unsigned long long k = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        k += a[i].lo != b[i].lo || a[i].hi != b[i].hi;
    if (k) atomicAdd(bad, k);
}

static void fill(void* p, long long words64, uint64_t seed) { zf_fill<<<1024, 256>>>((uint64_t*)p, words64, seed); }

static F128 host_word(uint64_t seed, int i) { return F128{zf_mix(seed ^ (2 * i)), zf_mix(seed ^ (2 * i + 1))}; }

struct ZfCase {
    long long n_out = 0;
    int lobits = 0;
    uint8_t *a = nullptr, *b = nullptr;
    F128 *ft = nullptr, *eqlo = nullptr, *eqhi = nullptr, *part = nullptr, *out = nullptr;
    F128 *am = nullptr, *bm = nullptr, *ref_a = nullptr, *ref_b = nullptr, *got_a = nullptr, *got_b = nullptr;
    unsigned long long* bad = nullptr;
    F128 r1{}, r2{}, s1{}, s2{}, s3{};
};

// a and b as the prover's packed witness (8 bytes a row), the fold table as prove_chunk builds it from 64 weights, and the
// split-eq tables at the prover's sizes for m = log_rows + 6
static cudaError_t zf_setup(ZfCase& c, int log_rows, uint64_t seed) {
    c.n_out = 1LL << log_rows;
    const int dfull = log_rows - 1;
    c.lobits = dfull > 7 ? dfull - 7 : 0;
    const long long q = c.n_out / 4;
    cudaError_t e = cudaSuccess;
    auto get = [&](auto** p, size_t bytes) { if (e == cudaSuccess) e = cudaMalloc((void**)p, bytes); };
    get(&c.a, c.n_out * 8); get(&c.b, c.n_out * 8);
    get(&c.ft, ZF_WORDS * sizeof(F128));
    get(&c.eqlo, (1LL << c.lobits) * sizeof(F128)); get(&c.eqhi, (1LL << (dfull - c.lobits)) * sizeof(F128));
    get(&c.part, 8 * ZT_MAX_BLOCKS * sizeof(F128)); get(&c.out, 16 * sizeof(F128));
    get(&c.am, c.n_out * sizeof(F128)); get(&c.bm, c.n_out * sizeof(F128));
    get(&c.ref_a, q * sizeof(F128)); get(&c.ref_b, q * sizeof(F128));
    get(&c.got_a, q * sizeof(F128)); get(&c.got_b, q * sizeof(F128));
    get(&c.bad, sizeof(unsigned long long));
    if (e != cudaSuccess) return e;
    fill(c.a, c.n_out, seed ^ 1); fill(c.b, c.n_out, seed ^ 2);
    fill(c.eqlo, 2LL << c.lobits, seed ^ 3); fill(c.eqhi, 2LL << (dfull - c.lobits), seed ^ 4);
    F128 ws[64];
    for (int i = 0; i < 64; i++) ws[i] = host_word(seed ^ 5, i);
    static F128 ft[ZF_WORDS];
    for (int j = 0; j < 8; j++)
        for (int v = 0; v < 256; v++) {
            F128 acc{0, 0};
            for (int bb = 0; bb < 8; bb++)
                if ((v >> bb) & 1) { acc.lo ^= ws[8 * j + bb].lo; acc.hi ^= ws[8 * j + bb].hi; }
            ft[j * 256 + v] = acc;
        }
    e = cudaMemcpy(c.ft, ft, sizeof(ft), cudaMemcpyHostToDevice);
    c.r1 = host_word(seed ^ 6, 0); c.r2 = host_word(seed ^ 6, 1);
    c.s1 = host_word(seed ^ 6, 2); c.s2 = host_word(seed ^ 6, 3); c.s3 = host_word(seed ^ 6, 4);
    return e == cudaSuccess ? cudaDeviceSynchronize() : e;
}

static void zf_free(ZfCase& c) {
    for (void* p : {(void*)c.a, (void*)c.b, (void*)c.ft, (void*)c.eqlo, (void*)c.eqhi, (void*)c.part, (void*)c.out, (void*)c.am,
                    (void*)c.bm, (void*)c.ref_a, (void*)c.ref_b, (void*)c.got_a, (void*)c.got_b, (void*)c.bad})
        cudaFree(p);
}

static void ref_round2(ZfCase& c) {
    launch_zerocheck_second_round_fold_with_lookahead(c.a, c.b, c.ft, c.eqlo, c.eqhi, c.lobits, c.n_out, c.am, c.bm, c.s1, c.part,
                                                      c.out);
}
static void ref_lookahead(ZfCase& c) {
    launch_zerocheck_tail_lookahead(c.am, c.bm, c.ref_a, c.ref_b, c.eqlo, c.eqhi, 2, c.lobits, c.n_out / 16, c.r1, c.r2, c.s2, c.s3,
                                    c.part, c.out + 8);
}
// table: 0 and 1 the modes of those numbers, 4 to 7 mode 2 in chunks of that many bits
template <int T, int C>
static void fused(ZfCase& c) {
    launch_zerocheck_second_round_message<T, C>(c.a, c.b, c.ft, c.eqlo, c.eqhi, c.lobits, c.n_out, c.s1, c.part, c.out);
}
template <int T, int C>
static void fused_tail(ZfCase& c) {
    launch_zerocheck_tail_lookahead_packed<T, C>(c.a, c.b, c.ft, c.got_a, c.got_b, c.eqlo, c.eqhi, 2, c.lobits, c.n_out / 16, c.r1,
                                                 c.r2, c.s2, c.s3, c.part, c.out + 8);
}
static void fused_round2(ZfCase& c, int table) {
    switch (table) {
        case 0: return fused<0, 4>(c);
        case 1: return fused<1, 4>(c);
        case 4: return fused<2, 4>(c);
        case 5: return fused<2, 5>(c);
        case 6: return fused<2, 6>(c);
        default: return fused<2, 7>(c);
    }
}
static void fused_lookahead(ZfCase& c, int table) {
    switch (table) {
        case 0: return fused_tail<0, 4>(c);
        case 1: return fused_tail<1, 4>(c);
        case 4: return fused_tail<2, 4>(c);
        case 5: return fused_tail<2, 5>(c);
        case 6: return fused_tail<2, 6>(c);
        default: return fused_tail<2, 7>(c);
    }
}
// both paths' messages (16 words: round 2's, then the lookahead's) and the words that differ
static cudaError_t zf_compare(ZfCase& c, int table, long long* bad) {
    F128 want[16], got[16];
    ref_round2(c); ref_lookahead(c);
    cudaError_t e = cudaMemcpy(want, c.out, sizeof(want), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemset(c.out, 0, sizeof(got));
    if (e != cudaSuccess) return e;
    fused_round2(c, table); fused_lookahead(c, table);
    e = cudaMemcpy(got, c.out, sizeof(got), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemset(c.bad, 0, sizeof(unsigned long long));
    if (e != cudaSuccess) return e;
    bad[0] = bad[1] = 0;
    for (int i = 0; i < 16; i++) bad[i / 8] += want[i].lo != got[i].lo || want[i].hi != got[i].hi;
    zf_diff<<<1024, 256>>>(c.ref_a, c.got_a, c.n_out / 4, c.bad);
    zf_diff<<<1024, 256>>>(c.ref_b, c.got_b, c.n_out / 4, c.bad);
    unsigned long long k = 0;
    e = cudaMemcpy(&k, c.bad, sizeof(k), cudaMemcpyDeviceToHost);
    bad[2] = (long long)k;
    return e;
}

extern "C" int zf_check(int log_rows, int table, uint64_t seed, long long* bad) {
    ZfCase c;
    cudaError_t e = zf_setup(c, log_rows, seed);
    if (e == cudaSuccess) e = zf_compare(c, table, bad);
    if (e == cudaSuccess) e = cudaGetLastError();
    zf_free(c);
    return (int)e;
}

#ifdef ZF_MAIN
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
    const int log_rows = argc > 1 ? atoi(argv[1]) : 29, reps = argc > 2 ? atoi(argv[2]) : 5;
    ZfCase c;
    cudaError_t e = zf_setup(c, log_rows, 0x5a17);
    if (e != cudaSuccess) { fprintf(stderr, "setup: %s\n", cudaGetErrorString(e)); return 1; }
    printf("{\"log_rows\": %d, \"reps\": %d, \"tables\": {\"round2_ms\": %.3f, \"lookahead_ms\": %.3f}", log_rows, reps,
           mean_ms([&] { ref_round2(c); }, reps), mean_ms([&] { ref_lookahead(c); }, reps));
    for (int t : {0, 1, 4, 5, 6, 7}) {
        long long bad[3];
        e = zf_compare(c, t, bad);
        if (e != cudaSuccess) { fprintf(stderr, "compare: %s\n", cudaGetErrorString(e)); return 1; }
        printf(", \"packed%d\": {\"round2_ms\": %.3f, \"lookahead_ms\": %.3f, \"bad\": [%lld, %lld, %lld]}", t,
               mean_ms([&] { fused_round2(c, t); }, reps), mean_ms([&] { fused_lookahead(c, t); }, reps), bad[0], bad[1], bad[2]);
    }
    printf("}\n");
    e = cudaDeviceSynchronize();
    zf_free(c);
    return e == cudaSuccess ? 0 : 1;
}
#endif
