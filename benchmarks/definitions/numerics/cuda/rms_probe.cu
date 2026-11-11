// Characterise the approximate scalar instructions the batch-invariant Triton ``_rms_norm_kernel`` uses
// (read off its PTX / SASS on sm_89):
//   * ``math.sqrt``  -> PTX ``sqrt.approx.ftz.f32``  -> SASS ``MUFU.SQRT``
//   * ``arith.divf`` -> PTX ``div.full.f32``         -> SASS: range-scale, ``MUFU.RCP``, ``FMUL``/``FFMA`` (no refinement)
// Both are deterministic functions of their 32-bit inputs.  This probe (1) dumps the MUFU.SQRT mantissa tables
// and (2) verifies, on ALL 2^32 input bit patterns, that the CPU-side models used by ``rms_triton_model.cpp``
// reproduce the hardware bit for bit:
//     sqrt.approx.ftz.f32(x) == T_sqrt[parity(exp(x))][mantissa(x)] * 2^floor(exp(x)/2)      (+ ftz / special rules)
//     div.full.f32(1, x)     == scale-aware  1 * MUFU.RCP(x)   (SASS sequence of the Triton kernel; MUFU.RCP table
//                               is the one measured by mufu_probe.cu and reused from the attention model)
//     div.full.f32(x, n)     == x * MUFU.RCP(n)                (n = 1536, the profile's hidden size; runtime arg)
//
// Modes:
//   table  <sqrt_out.bin>                         : writes sqrt(1.m) then sqrt(2*1.m) for m in [0, 2^23), u32 LE (16 MB each)
//   verify <sqrt_table.bin> <mufu_table.bin> <n>  : exhaustive check of the three models; JSON summary on stdout
//
// Build: nvcc -O2 -arch=sm_89 -o rms_probe rms_probe.cu

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__device__ __forceinline__ float sqrt_approx_ftz(float x) {
    float y;
    asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float rcp_approx_ftz(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float div_full(float a, float b) {
    float y;
    asm volatile("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float rsqrt_approx_ftz(float x) {
    float y;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float rsqrt_approx(float x) {  // non-ftz: what rsqrtf() compiles to without --use_fast_math
    float y;
    asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__global__ void k_table(uint32_t* out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (1u << 23)) return;
    out[i] = __float_as_uint(sqrt_approx_ftz(__uint_as_float(0x3F800000u | i)));                // [1, 2)
    out[(1u << 23) + i] = __float_as_uint(sqrt_approx_ftz(__uint_as_float(0x40000000u | i)));  // [2, 4)
}

__global__ void k_rsq_table(uint32_t* out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (1u << 23)) return;
    out[i] = __float_as_uint(rsqrt_approx_ftz(__uint_as_float(0x3F800000u | i)));                // rsqrt(1.m)   in (0.707, 1]
    out[(1u << 23) + i] = __float_as_uint(rsqrt_approx_ftz(__uint_as_float(0x40000000u | i)));  // rsqrt(2*1.m) in (0.5, 0.707]
}

// MUFU.RSQ: rsqrt(1.m * 2^k) = T[parity(k)][m] * 2^(-(k - parity)/2); ftz on subnormal input (-> +inf), +0 -> +inf,
// -0 -> -inf, negative -> NaN, +inf -> +0.
__device__ __forceinline__ uint32_t model_rsq(uint32_t xb, const uint32_t* T) {
    uint32_t s = xb & 0x80000000u, ax = xb & 0x7FFFFFFFu;
    int e = (ax >> 23) & 0xFF;
    if (e == 0xFF) { if (ax & 0x7FFFFF) return 0x7FFFFFFFu; return s ? 0x7FFFFFFFu : 0u; }
    if (e == 0) return s | 0x7F800000u;
    if (s) return 0x7FFFFFFFu;
    int k = e - 127, par = k & 1;
    uint32_t t = T[((uint32_t)par << 23) | (ax & 0x7FFFFF)];
    int re = (int)((t >> 23) & 0xFF) - (k - par) / 2;
    return ((uint32_t)re << 23) | (t & 0x7FFFFF);
}
// rsqrt.approx.f32 (non-ftz) as ptxas emits it: |x| < 2^-126 (or NaN) -> x *= 2^24, MUFU.RSQ, result *= 2^12
__device__ __forceinline__ uint32_t model_rsqrt_nonftz(uint32_t xb, const uint32_t* T) {
    float x = __uint_as_float(xb);
    bool p0 = !(fabsf(x) < 1.1754943508222875e-38f);
    if (!p0) x = x * 16777216.0f;
    float r = __uint_as_float(model_rsq(__float_as_uint(x), T));
    if (!p0) r = r * 4096.0f;
    return __float_as_uint(r);
}

__global__ void k_rsq_verify(uint32_t base, const uint32_t* T, unsigned long long* counts, uint32_t* first, int nfirst) {
    uint32_t i = base + blockIdx.x * blockDim.x + threadIdx.x;
    float x = __uint_as_float(i);
    uint32_t hw = __float_as_uint(rsqrt_approx_ftz(x)), md = model_rsq(i, T);
    bool hn = ((hw & 0x7F800000u) == 0x7F800000u) && (hw & 0x7FFFFFu), mn = ((md & 0x7F800000u) == 0x7F800000u) && (md & 0x7FFFFFu);
    if (!(hw == md || (hn && mn))) {
        unsigned long long k = atomicAdd(&counts[0], 1ull);
        if (k < (unsigned long long)nfirst) { first[3 * k] = i; first[3 * k + 1] = hw; first[3 * k + 2] = md; }
    }
    uint32_t hw2 = __float_as_uint(rsqrt_approx(x)), md2 = model_rsqrt_nonftz(i, T);
    hn = ((hw2 & 0x7F800000u) == 0x7F800000u) && (hw2 & 0x7FFFFFu); mn = ((md2 & 0x7F800000u) == 0x7F800000u) && (md2 & 0x7FFFFFu);
    if (!(hw2 == md2 || (hn && mn))) {
        unsigned long long k = atomicAdd(&counts[1], 1ull);
        if (k < (unsigned long long)nfirst) { first[3 * nfirst + 3 * k] = i; first[3 * nfirst + 3 * k + 1] = hw2; first[3 * nfirst + 3 * k + 2] = md2; }
    }
}

// ---- models (mirrored in rms_triton_model.cpp) --------------------------------------------------------------

__device__ __forceinline__ uint32_t model_sqrt(uint32_t xb, const uint32_t* T) {
    uint32_t s = xb & 0x80000000u;
    uint32_t ax = xb & 0x7FFFFFFFu;
    int e = (ax >> 23) & 0xFF;
    if (e == 0xFF) { if (ax & 0x7FFFFF) return 0x7FFFFFFFu; return s ? 0x7FFFFFFFu : 0x7F800000u; }  // sqrt(-inf)=NaN
    if (e == 0) return s;                       // ftz: +-0 and subnormals -> +-0
    if (s) return 0x7FFFFFFFu;                  // negative normal -> NaN
    int k = e - 127;                            // unbiased exponent
    int par = k & 1;                            // parity (works for negative k: -1 & 1 == 1)
    uint32_t t = T[(par << 23) | (ax & 0x7FFFFF)];  // sqrt(1.m) or sqrt(2*1.m); exponent field 127 or 127
    int half = (k - par) / 2;                   // floor(k/2)
    int re = (int)((t >> 23) & 0xFF) + half;
    return ((uint32_t)re << 23) | (t & 0x7FFFFF);
}

__device__ __forceinline__ uint32_t model_rcp(uint32_t xb, const uint32_t* R) {
    uint32_t s = xb & 0x80000000u;
    uint32_t ax = xb & 0x7FFFFFFFu;
    int e = (ax >> 23) & 0xFF;
    if (e == 0xFF) { if (ax & 0x7FFFFF) return 0x7FFFFFFFu; return s; }
    if (e == 0) return s | 0x7F800000u;
    uint32_t r = R[ax & 0x7FFFFF];
    int re = (int)((r >> 23) & 0xFF) - (e - 127);
    if (re <= 0) return s;
    if (re >= 0xFF) return s | 0x7F800000u;
    return s | ((uint32_t)re << 23) | (r & 0x7FFFFF);
}

// div.full.f32(a, b) as ptxas emits it in the Triton kernel (sm_89):
//   P0 = |b| >  2^126  (8.507059e37)          -> a *= 0.25, b *= 0.25
//   P1 = |b| >= 2^-126 (1.1754944e-38) or NaN  ; if !P1 -> a *= 2^24, b *= 2^24
//   r = MUFU.RCP(b) ; result = a * r  (FMUL, round-to-nearest, no ftz)
__device__ __forceinline__ float model_div_full(float a, float b, const uint32_t* R) {
    float ab = fabsf(b);
    bool p0 = ab > 8.507059173023462e37f;
    bool p1 = !(ab < 1.1754943508222875e-38f);  // GEU: >= or unordered
    if (p0) { a = a * 0.25f; b = b * 0.25f; }
    if (!p1) { a = a * 16777216.0f; b = b * 16777216.0f; }
    float r = __uint_as_float(model_rcp(__float_as_uint(b), R));
    return __fmul_rn(a, r);
}

__global__ void k_verify(uint32_t base, const uint32_t* T, const uint32_t* R, float n_cols,
                         unsigned long long* counts, uint32_t* first, int nfirst) {
    uint32_t i = base + blockIdx.x * blockDim.x + threadIdx.x;
    float x = __uint_as_float(i);
    bool nan_in = ((i & 0x7F800000u) == 0x7F800000u) && (i & 0x7FFFFFu);
    // (a) sqrt
    uint32_t hw = __float_as_uint(sqrt_approx_ftz(x));
    uint32_t md = model_sqrt(i, T);
    bool hw_nan = ((hw & 0x7F800000u) == 0x7F800000u) && (hw & 0x7FFFFFu);
    bool md_nan = ((md & 0x7F800000u) == 0x7F800000u) && (md & 0x7FFFFFu);
    if (!(hw == md || (hw_nan && md_nan))) {
        unsigned long long k = atomicAdd(&counts[0], 1ull);
        if (k < (unsigned long long)nfirst) { first[3 * k] = i; first[3 * k + 1] = hw; first[3 * k + 2] = md; }
    }
    if (i == 0x7FC00000u) { first[3 * nfirst] = hw; }
    if (i == 0xBF800000u) { first[3 * nfirst + 1] = hw; }  // sqrt(-1)
    // (b) div.full(1, x)
    uint32_t hw_d = __float_as_uint(div_full(1.0f, x));
    uint32_t md_d = __float_as_uint(model_div_full(1.0f, x, R));
    hw_nan = ((hw_d & 0x7F800000u) == 0x7F800000u) && (hw_d & 0x7FFFFFu);
    md_nan = ((md_d & 0x7F800000u) == 0x7F800000u) && (md_d & 0x7FFFFFu);
    if (!(hw_d == md_d || (hw_nan && md_nan))) {
        unsigned long long k = atomicAdd(&counts[1], 1ull);
        if (k < (unsigned long long)nfirst) { first[3 * nfirst + 4 + 3 * k] = i; first[3 * nfirst + 5 + 3 * k] = hw_d; first[3 * nfirst + 6 + 3 * k] = md_d; }
    }
    // (c) div.full(x, n_cols)
    uint32_t hw_c = __float_as_uint(div_full(x, n_cols));
    uint32_t md_c = __float_as_uint(model_div_full(x, n_cols, R));
    hw_nan = ((hw_c & 0x7F800000u) == 0x7F800000u) && (hw_c & 0x7FFFFFu);
    md_nan = ((md_c & 0x7F800000u) == 0x7F800000u) && (md_c & 0x7FFFFFu);
    if (!(hw_c == md_c || (hw_nan && md_nan))) {
        unsigned long long k = atomicAdd(&counts[2], 1ull);
        if (k < (unsigned long long)nfirst) { first[6 * nfirst + 8 + 3 * k] = i; first[6 * nfirst + 9 + 3 * k] = hw_c; first[6 * nfirst + 10 + 3 * k] = md_c; }
    }
    (void)nan_in;
}

#define CK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } } while (0)

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: rms_probe table <out.bin> | verify <sqrt_table.bin> <mufu_table.bin> <n_cols>\n"); return 2; }
    const size_t N = 1u << 23;
    uint32_t* d_sqrt; CK(cudaMalloc(&d_sqrt, 2 * N * 4));
    std::vector<uint32_t> h_sqrt(2 * N);
    if (!strcmp(argv[1], "table")) {
        k_table<<<(N + 255) / 256, 256>>>(d_sqrt);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h_sqrt.data(), d_sqrt, 2 * N * 4, cudaMemcpyDeviceToHost));
        FILE* f = fopen(argv[2], "wb");
        fwrite(h_sqrt.data(), 4, 2 * N, f); fclose(f);
        printf("wrote %s: sqrt(1.m) table then sqrt(2*1.m) table (2 x 2^23 x u32)\n", argv[2]);
        return 0;
    }
    if (!strcmp(argv[1], "rsq_table")) {
        k_rsq_table<<<(N + 255) / 256, 256>>>(d_sqrt);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h_sqrt.data(), d_sqrt, 2 * N * 4, cudaMemcpyDeviceToHost));
        FILE* f = fopen(argv[2], "wb");
        fwrite(h_sqrt.data(), 4, 2 * N, f); fclose(f);
        printf("wrote %s: rsqrt(1.m) table then rsqrt(2*1.m) table (2 x 2^23 x u32)\n", argv[2]);
        return 0;
    }
    if (!strcmp(argv[1], "rsq_verify")) {
        FILE* f = fopen(argv[2], "rb");
        if (!f || fread(h_sqrt.data(), 4, 2 * N, f) != 2 * N) { fprintf(stderr, "cannot read %s\n", argv[2]); return 2; }
        fclose(f);
        CK(cudaMemcpy(d_sqrt, h_sqrt.data(), 2 * N * 4, cudaMemcpyHostToDevice));
        unsigned long long* d_counts; CK(cudaMalloc(&d_counts, 2 * 8)); CK(cudaMemset(d_counts, 0, 2 * 8));
        const int nfirst = 16;
        uint32_t* d_first; CK(cudaMalloc(&d_first, 6 * nfirst * 4)); CK(cudaMemset(d_first, 0, 6 * nfirst * 4));
        const uint32_t chunk = 1u << 26;
        for (uint64_t base = 0; base < (1ull << 32); base += chunk)
            k_rsq_verify<<<chunk / 256, 256>>>((uint32_t)base, d_sqrt, d_counts, d_first, nfirst);
        CK(cudaDeviceSynchronize());
        unsigned long long counts[2]; CK(cudaMemcpy(counts, d_counts, 2 * 8, cudaMemcpyDeviceToHost));
        std::vector<uint32_t> first(6 * nfirst); CK(cudaMemcpy(first.data(), d_first, 6 * nfirst * 4, cudaMemcpyDeviceToHost));
        printf("{\"inputs\": 4294967296, \"rsqrt_ftz_mismatches\": %llu, \"rsqrt_nonftz_mismatches\": %llu,\n", counts[0], counts[1]);
        const char* names[2] = {"first_rsqrt_ftz", "first_rsqrt_nonftz"};
        for (int j = 0; j < 2; ++j) {
            printf(" \"%s\": [", names[j]);
            for (int k = 0; k < nfirst && (unsigned long long)k < counts[j]; ++k)
                printf("%s[\"%08x\", \"%08x\", \"%08x\"]", k ? ", " : "", first[3 * nfirst * j + 3 * k], first[3 * nfirst * j + 3 * k + 1], first[3 * nfirst * j + 3 * k + 2]);
            printf("]%s\n", j < 1 ? "," : "}");
        }
        return 0;
    }
    if (!strcmp(argv[1], "verify")) {
        if (argc < 5) { fprintf(stderr, "verify needs <sqrt_table.bin> <mufu_table.bin> <n_cols>\n"); return 2; }
        FILE* f = fopen(argv[2], "rb");
        if (!f || fread(h_sqrt.data(), 4, 2 * N, f) != 2 * N) { fprintf(stderr, "cannot read %s\n", argv[2]); return 2; }
        fclose(f);
        // mufu table file: ex2 table (2^23) then rcp table (2^23)
        std::vector<uint32_t> h_rcp(N);
        f = fopen(argv[3], "rb");
        if (!f || fseek(f, (long)(N * 4), SEEK_SET) != 0 || fread(h_rcp.data(), 4, N, f) != N) { fprintf(stderr, "cannot read %s\n", argv[3]); return 2; }
        fclose(f);
        float n_cols = (float)atof(argv[4]);
        uint32_t* d_rcp; CK(cudaMalloc(&d_rcp, N * 4));
        CK(cudaMemcpy(d_sqrt, h_sqrt.data(), 2 * N * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rcp, h_rcp.data(), N * 4, cudaMemcpyHostToDevice));
        unsigned long long* d_counts; CK(cudaMalloc(&d_counts, 3 * 8)); CK(cudaMemset(d_counts, 0, 3 * 8));
        const int nfirst = 16;
        const size_t nf = 9 * nfirst + 16;
        uint32_t* d_first; CK(cudaMalloc(&d_first, nf * 4)); CK(cudaMemset(d_first, 0, nf * 4));
        const uint32_t chunk = 1u << 26;
        for (uint64_t base = 0; base < (1ull << 32); base += chunk)
            k_verify<<<chunk / 256, 256>>>((uint32_t)base, d_sqrt, d_rcp, n_cols, d_counts, d_first, nfirst);
        CK(cudaDeviceSynchronize());
        unsigned long long counts[3]; CK(cudaMemcpy(counts, d_counts, 3 * 8, cudaMemcpyDeviceToHost));
        std::vector<uint32_t> first(nf); CK(cudaMemcpy(first.data(), d_first, nf * 4, cudaMemcpyDeviceToHost));
        printf("{\"inputs\": 4294967296, \"n_cols\": %g, \"sqrt_mismatches\": %llu, \"div_full_1_x_mismatches\": %llu, \"div_full_x_n_mismatches\": %llu,\n",
               n_cols, counts[0], counts[1], counts[2]);
        printf(" \"sqrt_nan_out\": \"%08x\", \"sqrt_neg1_out\": \"%08x\",\n", first[3 * nfirst], first[3 * nfirst + 1]);
        const char* names[3] = {"first_sqrt", "first_div_full_1_x", "first_div_full_x_n"};
        size_t offs[3] = {0, (size_t)3 * nfirst + 4, (size_t)6 * nfirst + 8};
        for (int j = 0; j < 3; ++j) {
            printf(" \"%s\": [", names[j]);
            for (int k = 0; k < nfirst && (unsigned long long)k < counts[j]; ++k)
                printf("%s[\"%08x\", \"%08x\", \"%08x\"]", k ? ", " : "", first[offs[j] + 3 * k], first[offs[j] + 3 * k + 1], first[offs[j] + 3 * k + 2]);
            printf("]%s\n", j < 2 ? "," : "}");
        }
        return 0;
    }
    fprintf(stderr, "unknown mode\n");
    return 2;
}
