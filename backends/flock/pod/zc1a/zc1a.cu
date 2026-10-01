// 1a beside C-Flock's zerocheck round 1 (zerocheck_first_round_cpu_structured<14>, cuda-ghash/zerocheck_round1_cpustyle.cuh):
// the same kernel with its GF(2^8) products on the ALU, four to a 32-bit word (SWAR, xtime reduction by 0x1b), in place of the
// log and antilog lookups in shared memory (1a-prmt and 1a-imad: two ways to form the byte masks), and a third variant, 1a-clmad,
// whose products are four to a carry-less multiply (sm_120's clmad, as f128.cuh's GHASH uses), reduced once per sum. A prototype:
// it includes C-Flock's headers read-only and is never linked into the prover. It checks the SWAR products against the tables
// (all 65,536 pairs, in each byte lane), the 22-bit reduction on every input, each variant's per-j table index against the
// current kernel's (every pair at each of the eight positions), then all four kernels' 64 + 64 outputs byte for byte on the
// same inputs at several m, and times them at one m. One JSON object per line, each prefixed "ZC1A ".
// usage: zc1a [--m 13,20,28,32,35] [--time-m 35] [--iters 10] | zc1a --ncu M (one launch of each kernel at m = M, for ncu)
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "zerocheck_round1_cpustyle.cuh"
#include "phi8_table.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

// each byte to 0xff if its top bit is set, else 0x00. MODE 0: prmt's sign replication; MODE 1: a shift, a mask and a multiply
template <int MODE>
__device__ __forceinline__ uint32_t zc1a_top_mask(uint32_t x) {
    if (MODE == 0) {
        uint32_t r;
        asm("prmt.b32 %0, %1, %2, 0xBA98;" : "=r"(r) : "r"(x), "r"(0u));
        return r;
    }
    return ((x >> 7) & 0x01010101u) * 0xffu;
}

// four GF(2^8) products mod x^8 + x^4 + x^3 + x + 1, byte by byte: b's bit i (moved to its byte's top) selects a·x^i
template <int MODE>
__device__ __forceinline__ uint32_t zc1a_mul4(uint32_t a, uint32_t b) {
    uint32_t p = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        p ^= a & zc1a_top_mask<MODE>(b << (7 - i));
        if (i < 7) a = ((a << 1) & 0xfefefefeu) ^ (zc1a_top_mask<MODE>(a) & 0x1b1b1b1bu);
    }
    return p;
}

// byte sel (0..7) of a 64-bit word, in byte 0
__device__ __forceinline__ uint32_t zc1a_byte(u64 w, uint32_t sel) {
    return __byte_perm((uint32_t)w, (uint32_t)(w >> 32), sel);
}

// byte 0 of each of four words, in that order
__device__ __forceinline__ uint32_t zc1a_pack(uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3) {
    return __byte_perm(__byte_perm(e0, e1, 0x0040), __byte_perm(e2, e3, 0x0040), 0x5410);
}

// XOR over k of p_k << k, for p_0..p_3 the bytes of lo and p_4..p_7 those of hi (the kernel's aacc)
__device__ __forceinline__ uint16_t zc1a_shift_sum(uint32_t lo, uint32_t hi) {
    return (uint16_t)((lo & 0xffu) ^ ((lo >> 7) & 0x1feu) ^ ((lo >> 14) & 0x3fcu) ^ ((lo >> 21) & 0x7f8u)
                      ^ ((hi << 4) & 0xff0u) ^ ((hi >> 3) & 0x1fe0u) ^ ((hi >> 10) & 0x3fc0u) ^ ((hi >> 17) & 0x7f80u));
}

// sum over k < 4 of x_k·y_k·x^k as polynomials over GF(2), unreduced (18 bits), from one 64 x 64 carry-less product: x_k at
// x^(18k) and y_k at x^(51 - 17k) put each x_k·y_k (degree <= 14) at x^(51 + k), in bits 51..68, and every x_k·y_l with k != l
// at x^(51 + k + 17(k - l)), entirely below bit 51 or above bit 68. Bytes only: x_k, y_k < 256
__device__ __forceinline__ uint32_t zc1a_clmad4(const uint32_t* x, const uint32_t* y) {
    const u64 a = (u64)x[0] | (u64)x[1] << 18 | (u64)x[2] << 36 | (u64)x[3] << 54;
    const u64 b = (u64)y[0] << 51 | (u64)y[1] << 34 | (u64)y[2] << 17 | (u64)y[3];
    const u64 lo = clmul_lo(a, b), hi = clmul_hi(a, b);
    return (uint32_t)((lo >> 51) | (hi << 13)) & 0x3ffffu;
}

// p (< 2^22) mod x^8 + x^4 + x^3 + x + 1: x^8 = 0x1b folds the bits from 8 up, 22 -> 18 -> 14 -> 10 -> 8 bits
__device__ __forceinline__ uint32_t zc1a_reduce22(uint32_t p) {
#pragma unroll
    for (int i = 0; i < 4; i++) p = (uint32_t)clmad_lo((u64)(p >> 8), 0x1bull, (u64)(p & 0xffu));
    return p;
}

// one lane's s_phi index at one j: (sum over k of x_k·y_k·x^k) mod x^8 + x^4 + x^3 + x + 1, from byte 0 of each x_k and y_k
// (MODE 2: the other bytes zero). The current kernel's is reduce(XOR_k antilog[log x_k + log y_k] << k), the same residue
template <int MODE>
__device__ __forceinline__ uint32_t zc1a_index(const uint32_t* x, const uint32_t* y) {
    if (MODE == 2) return zc1a_reduce22(zc1a_clmad4(x, y) ^ (zc1a_clmad4(x + 4, y + 4) << 4));
    const uint32_t lo = zc1a_mul4<MODE>(zc1a_pack(x[0], x[1], x[2], x[3]), zc1a_pack(y[0], y[1], y[2], y[3]));
    const uint32_t hi = zc1a_mul4<MODE>(zc1a_pack(x[4], x[5], x[6], x[7]), zc1a_pack(y[4], y[5], y[6], y[7]));
    return reduce_zerocheck_gf8_polynomial(zc1a_shift_sum(lo, hi));
}

// zerocheck_first_round_cpu_structured with only its k loop changed: its 16 table products per lane per j become four SWAR
// words (MODE 0, 1) or four carry-less products (MODE 2), and its two reductions of the products' sums zc1a_index's. The bounds
// keep the two 448-thread blocks per SM that the original's launcher documents (72 registers).
template <int W, int MODE>
__global__ void __launch_bounds__(W * 32, 2)
zc1a_first_round(const uint8_t* __restrict__ a_packed, const uint8_t* __restrict__ b_packed,
                 const uint8_t* __restrict__ c_packed, const F128* __restrict__ eq_out, const u64* __restrict__ t0,
                 const uint8_t* __restrict__ f8mul, const F128* __restrict__ phi, long long n_out, long long G,
                 F128* __restrict__ raw_ab, F128* __restrict__ raw_c) {
    __shared__ u64 s_t0[256 * 8];
    __shared__ F128 s_phi[256];
    __shared__ u64 s_scol[W * 24];
    __shared__ u64 s_wbuf[W * 192];
    for (int i = threadIdx.x; i < 256 * 8; i += blockDim.x) s_t0[i] = t0[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_phi[i] = phi[i];
    __syncthreads();

    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u64* scol = s_scol + wid * 24;
    u64* wbuf = s_wbuf + wid * 192;
    long long warp_id = (long long)blockIdx.x * W + wid;
    long long o0 = warp_id * G, o1 = o0 + G;
    if (o1 > n_out) o1 = n_out;
    if (o0 >= o1) return;

    int wlo = lane >> 3, whi = wlo + 4, sh = (lane & 7) * 8;
    const uint32_t sel = lane & 7;
    F128 pab0{0, 0}, pab1{0, 0}, pc0{0, 0}, pc1{0, 0};

    for (long long o = o0; o < o1; o++) {
        F128 eq_o = eq_out[o];
        F128 chunk_ab0{0, 0}, chunk_ab1{0, 0}, chunk_c0{0, 0}, chunk_c1{0, 0};
        for (int j = 15; j >= 0; j--) {
            long long col0 = o * 128 + j * 8;
            long long base = col0 * 8;
            if (lane < 24) {
                const uint8_t* Wt = (lane < 8) ? a_packed : (lane < 16) ? b_packed : c_packed;
                int cl = lane & 7;
                scol[lane] = *(const u64*)(Wt + base + cl * 8);
            }
            __syncwarp();
#pragma unroll
            for (int r = 0; r < 6; r++) {
                int bld = lane + 32 * r;
                int cl = bld / 24, win = bld - cl * 24, wt = win >> 3, wcol = win & 7;
                u64 src = scol[wt * 8 + cl];
                u64 word = 0;
#pragma unroll
                for (int b = 0; b < 8; b++)
                    word ^= s_t0[(int)((src >> (8 * b)) & 0xff) * 8 + (wcol ^ b)];
                wbuf[cl * 24 + win] = word;
            }
            __syncwarp();
            uint32_t eaL[8], eaH[8], ebL[8], ebH[8];
            uint16_t cacc0 = 0, cacc1 = 0;
            auto ex = [&](u64 w) -> uint32_t { return MODE == 2 ? (uint32_t)(w >> sh) & 0xffu : zc1a_byte(w, sel); };
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const u64* wc = wbuf + k * 24;
                eaL[k] = ex(wc[wlo]);     eaH[k] = ex(wc[whi]);
                ebL[k] = ex(wc[8 + wlo]); ebH[k] = ex(wc[8 + whi]);
                uint8_t cL = (uint8_t)(wc[16 + wlo] >> sh), cH = (uint8_t)(wc[16 + whi] >> sh);
                cacc0 ^= (uint16_t)cL << k;
                cacc1 ^= (uint16_t)cH << k;
            }
            chunk_ab0 = f128_add(multiply_zerocheck_value_by_generator(chunk_ab0), s_phi[zc1a_index<MODE>(eaL, ebL)]);
            chunk_ab1 = f128_add(multiply_zerocheck_value_by_generator(chunk_ab1), s_phi[zc1a_index<MODE>(eaH, ebH)]);
            chunk_c0  = f128_add(multiply_zerocheck_value_by_generator(chunk_c0),  s_phi[reduce_zerocheck_gf8_polynomial(cacc0)]);
            chunk_c1  = f128_add(multiply_zerocheck_value_by_generator(chunk_c1),  s_phi[reduce_zerocheck_gf8_polynomial(cacc1)]);
            __syncwarp();
        }
        pab0 = f128_add(pab0, ghash_mul_karatsuba(eq_o, chunk_ab0));
        pab1 = f128_add(pab1, ghash_mul_karatsuba(eq_o, chunk_ab1));
        pc0  = f128_add(pc0,  ghash_mul_karatsuba(eq_o, chunk_c0));
        pc1  = f128_add(pc1,  ghash_mul_karatsuba(eq_o, chunk_c1));
    }
    atomicXor((unsigned long long*)&raw_ab[lane].lo, pab0.lo);
    atomicXor((unsigned long long*)&raw_ab[lane].hi, pab0.hi);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].lo, pab1.lo);
    atomicXor((unsigned long long*)&raw_ab[lane + 32].hi, pab1.hi);
    atomicXor((unsigned long long*)&raw_c[lane].lo, pc0.lo);
    atomicXor((unsigned long long*)&raw_c[lane].hi, pc0.hi);
    atomicXor((unsigned long long*)&raw_c[lane + 32].lo, pc1.lo);
    atomicXor((unsigned long long*)&raw_c[lane + 32].hi, pc1.hi);
}

// launch_zerocheck_first_round_cpu_structured with the variant kernel: the same grid, memsets and scale
template <int MODE>
static void zc1a_launch(const uint8_t* d_a, const uint8_t* d_b, const uint8_t* d_c, const F128* d_eq_out, long long n_out,
                        F128 scale, F128* d_round1_ab, F128* d_round1_c) {
    constexpr int W = 14;
    constexpr long long G = ZC_OUTER_CHUNKS_PER_WARP;
    long long warps = (n_out + G - 1) / G;
    int blocks = (int)((warps + W - 1) / W);
    cudaMemset(d_round1_ab, 0, 64 * sizeof(F128));
    cudaMemset(d_round1_c, 0, 64 * sizeof(F128));
    zc1a_first_round<W, MODE><<<blocks, W * 32>>>(d_a, d_b, d_c, d_eq_out, g_zc_t0, g_zc_f8mul, g_zc_phi, n_out, G,
                                                  d_round1_ab, d_round1_c);
    scale_zerocheck_first_round<<<1, 64>>>(d_round1_ab, d_round1_c, scale);
}

__device__ __forceinline__ uint8_t zc1a_table_product(uint32_t x, uint32_t y) {
    return d_zc_antilog[(int)d_zc_log[x] + d_zc_log[y]] & (uint8_t)(-(int)(x && y));
}

// every product: the tables' (antilog[log x + log y], 0 if x or y is 0) against the field's (f8mul) and against zc1a_mul4, in
// four byte lanes holding four different pairs, so a carry from one lane into the next shows. 65,536 threads
template <int MODE>
__global__ void zc1a_check_products(const uint8_t* __restrict__ f8mul, unsigned* bad) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 65536) return;
    const uint32_t a = (uint32_t)t >> 8, b = (uint32_t)t & 0xff;
    const uint32_t xs[4] = {a, b, a ^ 0x5a, a}, ys[4] = {b, a, b, b ^ 0xff};
    uint32_t x4 = 0, y4 = 0;
    for (int q = 0; q < 4; q++) { x4 |= xs[q] << (8 * q); y4 |= ys[q] << (8 * q); }
    const uint32_t p4 = zc1a_mul4<MODE>(x4, y4);
    for (int q = 0; q < 4; q++) {
        const uint8_t tab = zc1a_table_product(xs[q], ys[q]);
        if (tab != f8mul[xs[q] * 256 + ys[q]]) atomicAdd(&bad[0], 1u);
        if (((p4 >> (8 * q)) & 0xff) != tab) atomicAdd(&bad[1], 1u);
    }
}

// zc1a_reduce22 against bit-by-bit long division, on every p < 2^22
__global__ void zc1a_check_reduce22(unsigned* bad) {
    for (uint32_t p = blockIdx.x * blockDim.x + threadIdx.x; p < (1u << 22); p += gridDim.x * blockDim.x) {
        uint32_t r = p;
        for (int d = 21; d >= 8; d--) if ((r >> d) & 1) r ^= 0x11bu << (d - 8);
        if (zc1a_reduce22(p) != r) atomicAdd(bad, 1u);
    }
}

// zc1a_index against the current kernel's (its products' shifted sum, then reduce_zerocheck_gf8_polynomial), with each of the
// eight positions k taking every pair (x, y) once, through eight different bijections of (a, b). 65,536 threads
template <int MODE>
__global__ void zc1a_check_index(unsigned* bad) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 65536) return;
    const uint32_t a = (uint32_t)t >> 8, b = (uint32_t)t & 0xff;
    const uint32_t x[8] = {a, b, a ^ 0x5a, a, b ^ 0x3c, (a + 17) & 0xff, b, a ^ 0xff};
    const uint32_t y[8] = {b, a, b, b ^ 0xff, a, b ^ 0xa5, (a * 3 + 1) & 0xff, (b + 101) & 0xff};
    uint16_t acc = 0;
    for (int k = 0; k < 8; k++) acc ^= (uint16_t)zc1a_table_product(x[k], y[k]) << k;
    if (zc1a_index<MODE>(x, y) != reduce_zerocheck_gf8_polynomial(acc)) atomicAdd(bad, 1u);
}

__host__ __device__ inline u64 zc1a_mix(u64 x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// the packed columns of a satisfied witness: a and b pseudo-random (sparse: each bit set with probability about 1/8), c = a & b
__global__ void zc1a_fill(u64* a, u64* b, u64* c, size_t n, u64 seed, int sparse) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        u64 x = zc1a_mix(seed + 2 * i), y = zc1a_mix(seed + 2 * i + 1);
        if (sparse) { x &= zc1a_mix(x) & zc1a_mix(~x); y &= zc1a_mix(y) & zc1a_mix(~y); }
        a[i] = x; b[i] = y; c[i] = x & y;
    }
}

__global__ void zc1a_fill_eq(F128* e, long long n, u64 seed) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        e[i] = F128{zc1a_mix(seed ^ (u64)(3 * i)), zc1a_mix(seed ^ (u64)(3 * i + 1))};
}

static uint8_t gf8_mul_host(uint8_t x, uint8_t y) {
    uint8_t p = 0;
    for (int i = 0; i < 8; i++) {
        if (y & 1) p ^= x;
        y >>= 1;
        x = (uint8_t)((x << 1) ^ ((x & 0x80) ? 0x1b : 0));
    }
    return p;
}

struct Bufs { uint8_t *a, *b, *c; F128 *eq, *ab, *cc; };
constexpr int NK = 4;
static const char* KNAME[NK] = {"current", "1a-prmt", "1a-imad", "1a-clmad"};

static void run_kernel(int which, const Bufs& d, long long n_out, F128 scale) {
    if (which == 0) launch_zerocheck_first_round_cpu_structured(d.a, d.b, d.c, d.eq, n_out, scale, d.ab, d.cc);
    else if (which == 1) zc1a_launch<0>(d.a, d.b, d.c, d.eq, n_out, scale, d.ab, d.cc);
    else if (which == 2) zc1a_launch<1>(d.a, d.b, d.c, d.eq, n_out, scale, d.ab, d.cc);
    else zc1a_launch<2>(d.a, d.b, d.c, d.eq, n_out, scale, d.ab, d.cc);
}

static void fill(const Bufs& d, int m, u64 seed, int sparse) {
    const size_t words = ((size_t)1 << (m - 3)) / 8;
    const long long n_out = 1LL << (m - 13);
    zc1a_fill<<<4096, 256>>>((u64*)d.a, (u64*)d.b, (u64*)d.c, words, seed * 0x100000001b3ull, sparse);
    zc1a_fill_eq<<<1024, 256>>>(d.eq, n_out, seed * 0x2545f4914f6cdd1dull);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
}

static std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) { v.push_back(atoi(p)); while (*p && *p != ',') p++; if (*p) p++; }
    return v;
}

int main(int argc, char** argv) {
    std::vector<int> ms = {13, 20, 28, 32, 35};
    int time_m = 35, iters = 10, ncu_m = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if (k == "--m") ms = parse_list(argv[i + 1]);
        else if (k == "--time-m") time_m = atoi(argv[i + 1]);
        else if (k == "--iters") iters = atoi(argv[i + 1]);
        else if (k == "--ncu") ncu_m = atoi(argv[i + 1]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    // C-Flock's own bench's word-build map (bench_zc_cpustyle.cu): the products' check doesn't depend on it
    std::vector<uint8_t> mcol(64 * 64), f8mul((size_t)256 * 256);
    for (size_t i = 0; i < mcol.size(); i++) mcol[i] = (uint8_t)(i * 7 + 1);
    for (int x = 0; x < 256; x++) for (int y = 0; y < 256; y++) f8mul[x * 256 + y] = gf8_mul_host((uint8_t)x, (uint8_t)y);
    CK(upload_zerocheck_first_round_tables(mcol.data(), f8mul.data(), PHI_8_TABLE));
    build_zerocheck_logarithm_tables<<<1, 1>>>();
    CK(cudaDeviceSynchronize());

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("ZC1A {\"device\": \"%s\", \"sm\": %d, \"sm_count\": %d}\n", prop.name, prop.major * 10 + prop.minor, prop.multiProcessorCount);
    const void* fns[NK] = {(const void*)zerocheck_first_round_cpu_structured<14>, (const void*)zc1a_first_round<14, 0>,
                           (const void*)zc1a_first_round<14, 1>, (const void*)zc1a_first_round<14, 2>};
    for (int w = 0; w < NK; w++) {
        cudaFuncAttributes fa;
        CK(cudaFuncGetAttributes(&fa, fns[w]));
        int per_sm = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fns[w], 14 * 32, 0));
        printf("ZC1A {\"kernel\": \"%s\", \"registers\": %d, \"local_bytes\": %zu, \"static_shared_bytes\": %zu, \"blocks_per_sm\": %d}\n",
               KNAME[w], fa.numRegs, fa.localSizeBytes, fa.sharedSizeBytes, per_sm);
    }

    bool ok = true;
    unsigned* d_bad;
    CK(cudaMalloc(&d_bad, 2 * sizeof(unsigned)));
    uint8_t* d_f8mul;
    CK(cudaMalloc(&d_f8mul, f8mul.size()));
    CK(cudaMemcpy(d_f8mul, f8mul.data(), f8mul.size(), cudaMemcpyHostToDevice));
    unsigned bad[2];
    for (int mode = 0; mode < 2; mode++) {
        CK(cudaMemset(d_bad, 0, 2 * sizeof(unsigned)));
        if (mode == 0) zc1a_check_products<0><<<256, 256>>>(d_f8mul, d_bad);
        else zc1a_check_products<1><<<256, 256>>>(d_f8mul, d_bad);
        CK(cudaMemcpy(bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost));
        printf("ZC1A {\"check\": \"products\", \"kernel\": \"%s\", \"pairs\": 262144, \"tables_vs_field_bad\": %u, \"swar_vs_tables_bad\": %u}\n",
               KNAME[1 + mode], bad[0], bad[1]);
        ok = ok && bad[0] == 0 && bad[1] == 0;
    }
    CK(cudaMemset(d_bad, 0, 2 * sizeof(unsigned)));
    zc1a_check_reduce22<<<1024, 256>>>(d_bad);
    CK(cudaMemcpy(bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost));
    printf("ZC1A {\"check\": \"reduce22\", \"inputs\": 4194304, \"bad\": %u}\n", bad[0]);
    ok = ok && bad[0] == 0;
    for (int mode = 0; mode < 3; mode++) {
        CK(cudaMemset(d_bad, 0, 2 * sizeof(unsigned)));
        if (mode == 0) zc1a_check_index<0><<<256, 256>>>(d_bad);
        else if (mode == 1) zc1a_check_index<1><<<256, 256>>>(d_bad);
        else zc1a_check_index<2><<<256, 256>>>(d_bad);
        CK(cudaMemcpy(bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost));
        printf("ZC1A {\"check\": \"index\", \"kernel\": \"%s\", \"cases\": 65536, \"pairs_per_position\": 65536, \"bad\": %u}\n",
               KNAME[1 + mode], bad[0]);
        ok = ok && bad[0] == 0;
    }
    CK(cudaGetLastError());

    int max_m = ncu_m ? ncu_m : std::max(time_m, *std::max_element(ms.begin(), ms.end()));
    const size_t packed = (size_t)1 << (max_m - 3);
    Bufs d;
    CK(cudaMalloc(&d.a, packed)); CK(cudaMalloc(&d.b, packed)); CK(cudaMalloc(&d.c, packed));
    CK(cudaMalloc(&d.eq, (size_t)(1LL << (max_m - 13)) * sizeof(F128)));
    CK(cudaMalloc(&d.ab, 64 * sizeof(F128))); CK(cudaMalloc(&d.cc, 64 * sizeof(F128)));
    const F128 scale{zc1a_mix(77), zc1a_mix(78)};

    if (ncu_m) {
        fill(d, ncu_m, 1, 0);
        for (int w = 0; w < NK; w++) run_kernel(w, d, 1LL << (ncu_m - 13), scale);
        CK(cudaDeviceSynchronize());
        printf("ZC1A {\"ncu_m\": %d}\n", ncu_m);
        return ok ? 0 : 1;
    }

    std::vector<F128> out[NK];
    for (int sparse = 0; sparse < 2; sparse++) {
        for (u64 seed = 1; seed <= 2; seed++) {
            for (int m : ms) {
                const long long n_out = 1LL << (m - 13);
                fill(d, m, seed, sparse);
                for (int w = 0; w < NK; w++) {
                    CK(cudaMemset(d.ab, 0xa5, 64 * sizeof(F128)));
                    CK(cudaMemset(d.cc, 0xa5, 64 * sizeof(F128)));
                    run_kernel(w, d, n_out, scale);
                    CK(cudaGetLastError());
                    out[w].assign(128, F128{0, 0});
                    CK(cudaMemcpy(out[w].data(), d.ab, 64 * sizeof(F128), cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(out[w].data() + 64, d.cc, 64 * sizeof(F128), cudaMemcpyDeviceToHost));
                }
                int nonzero = 0;
                for (const F128& x : out[0]) nonzero += (x.lo || x.hi);
                std::string same;
                for (int w = 1; w < NK; w++) {
                    const bool s1 = memcmp(out[0].data(), out[w].data(), 128 * sizeof(F128)) == 0;
                    same += std::string(", \"") + KNAME[w] + "_identical\": " + (s1 ? "true" : "false");
                    ok = ok && s1;
                }
                printf("ZC1A {\"check\": \"outputs\", \"m\": %d, \"inputs\": \"%s\", \"seed\": %llu, \"bytes\": %zu, \"current_nonzero\": %d%s, "
                       "\"current_ab0\": \"%016llx%016llx\"}\n",
                       m, sparse ? "sparse" : "uniform", (unsigned long long)seed, 128 * sizeof(F128), nonzero, same.c_str(),
                       out[0][0].hi, out[0][0].lo);
            }
        }
    }

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const long long n_out = 1LL << (time_m - 13);
    for (int sparse = 0; sparse < 2; sparse++) {
        fill(d, time_m, 3, sparse);
        for (int w = 0; w < NK; w++) run_kernel(w, d, n_out, scale);
        CK(cudaDeviceSynchronize());
        std::vector<float> ms_[NK];
        for (int it = 0; it < iters; it++) {
            for (int w = 0; w < NK; w++) {
                CK(cudaEventRecord(e0));
                run_kernel(w, d, n_out, scale);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float t = 0;
                CK(cudaEventElapsedTime(&t, e0, e1));
                ms_[w].push_back(t);
            }
        }
        for (int w = 0; w < NK; w++) {
            std::vector<float> s = ms_[w];
            std::sort(s.begin(), s.end());
            std::string all;
            for (float t : ms_[w]) all += (all.empty() ? "" : ", ") + std::to_string(t);
            printf("ZC1A {\"time\": \"launch\", \"kernel\": \"%s\", \"m\": %d, \"inputs\": \"%s\", \"iters\": %d, \"median_ms\": %.4f, "
                   "\"min_ms\": %.4f, \"max_ms\": %.4f, \"ms\": [%s]}\n",
                   KNAME[w], time_m, sparse ? "sparse" : "uniform", iters, s[s.size() / 2], s.front(), s.back(), all.c_str());
        }
    }
    printf("ZC1A {\"all_identical\": %s}\n", ok ? "true" : "false");
    return ok ? 0 : 1;
}
