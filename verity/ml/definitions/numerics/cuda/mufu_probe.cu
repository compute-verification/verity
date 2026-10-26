// Characterise the two approximate MUFU functions FlashAttention-2 (built with --use_fast_math)
// uses in its forward kernel: ``exp2f`` -> ``ex2.approx.ftz.f32`` (MUFU.EX2) and ``1.f / x`` ->
// ``rcp.approx.ftz.f32`` (MUFU.RCP).  Both are deterministic functions of the 32-bit input.
//
// Two modes:
//   table <out.bin>   : writes ex2.approx.ftz.f32(i / 2^23) for i in [0, 2^23) (FP32 bits, u32 LE), then
//                       rcp.approx.ftz.f32(1 + i / 2^23) for i in [0, 2^23).  16 MB each.
//   verify            : checks the structural hypotheses on ALL 2^32 input bit patterns, on the GPU:
//                       ex2(x) == ldexp(T_ex2[floor(frac(x) * 2^23)], floor(x)) with ftz / overflow rules,
//                       rcp(x) == ldexp(T_rcp[mantissa(x)], -exponent(x)) with ftz / inf rules,
//                       and prints the mismatch count and the first few mismatching inputs per function.
//
// Build: nvcc -O2 -arch=sm_89 -o mufu_probe mufu_probe.cu      (any sm_80+ card; the MUFU is what is measured)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

__device__ __forceinline__ float ex2_approx_ftz(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float rcp_approx_ftz(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__global__ void k_table(uint32_t* ex2_out, uint32_t* rcp_out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (1u << 23)) return;
    // x = i / 2^23 exactly (i < 2^23 so representable)
    float x = (float)i * (1.0f / 8388608.0f);
    ex2_out[i] = __float_as_uint(ex2_approx_ftz(x));
    float m = __uint_as_float(0x3F800000u | i);  // 1.m in [1, 2)
    rcp_out[i] = __float_as_uint(rcp_approx_ftz(m));
}

// --- the CPU-side model, mirrored on the device -------------------------------------------------------

// FRAC_MODE: how the hardware quantises the fraction of x to the 23-bit table index.
//   0 = truncate; 1 = round to nearest (ties away from zero in magnitude of the fraction);
//   2 = round to nearest, ties to even.  A rounded fraction that reaches 1.0 carries into floor(x).
#ifndef FRAC_MODE
#define FRAC_MODE 0
#endif

__device__ __forceinline__ uint32_t model_ex2(uint32_t xb, const uint32_t* T) {
    uint32_t ax = xb & 0x7FFFFFFFu;
    int e = (ax >> 23) & 0xFF;
    if (e == 0xFF) {  // inf / nan
        if (ax & 0x7FFFFF) return 0x7FFFFFFFu;  // canonical NaN (checked separately; value irrelevant)
        return (xb >> 31) ? 0u : 0x7F800000u;   // ex2(-inf) = +0, ex2(+inf) = +inf
    }
    if (e == 0) return 0x3F800000u;  // ftz: subnormal / zero input -> 2^0 = 1
#if FRAC_MODE == 3 || FRAC_MODE == 4
    // H3: the hardware forms the 23-bit fixed-point value of x by truncating |x| to 23 fractional bits and,
    // for negative x, inverting the bits (ones' complement, no +1).  So fixed(x) = floor(|x|*2^23) for x>0 and
    // fixed(x) = -floor(|x|*2^23) - 1 for x<0; then n = fixed >> 23 (floor), j = fixed & (2^23-1).
    // H4 (FRAC_MODE 4): as H3, except that when the truncated fraction bits of |x| are all zero (so the
    // inverted fraction would be all ones) the +1 carry IS applied: fixed(x) = -floor(|x|*2^23) exactly.
    // Equivalently: negative x -> n = -q - (r != 0), j = (r == 0) ? 0 : 2^23 - 1 - r, with
    // K = floor(|x| 2^23) = q 2^23 + r.
    {
        int unb = e - 127;
        if (unb >= 30) return (xb >> 31) ? 0u : 0x7F800000u;  // |x| >= 2^30: overflow / underflow(ftz)
        uint64_t mant = (uint64_t)((ax & 0x7FFFFF) | 0x800000);
        uint64_t K = (unb >= 0) ? (mant << unb) : (-unb >= 24 ? 0 : (mant >> (-unb)));  // floor(|x| * 2^23); mant < 2^24
        long long fixed;
        if (!(xb >> 31)) fixed = (long long)K;
        else if (FRAC_MODE == 4 && (K & 0x7FFFFF) == 0) fixed = -(long long)K;
        else fixed = -(long long)K - 1;
        long long n = fixed >> 23;  // arithmetic shift = floor
        uint32_t j = (uint32_t)(fixed & 0x7FFFFF);
        uint32_t t = T[j];
        long long re = (long long)((t >> 23) & 0xFF) + n;
        if (re >= 255) return 0x7F800000u;
        if (re <= 0) return 0u;  // ftz (subnormal result flushed to zero)
        return ((uint32_t)re << 23) | (t & 0x7FFFFF);
    }
#endif
    // exact value x = (-1)^s * 1.m * 2^(e-127).  Split into floor(x) and the quantised fraction.
    // Use 64-bit fixed point with 40 fractional bits when |x| < 2^24, else x is an integer.
    int unb = e - 127;
    uint64_t mant = (uint64_t)((ax & 0x7FFFFF) | 0x800000);  // 24 bits, value = mant * 2^(unb-23)
    long long n;      // floor(x)
    uint32_t f23;     // quantised frac(x) * 2^23
    if (unb >= 23) {
        // integer (possibly huge): result overflows for x >= 128, underflows/ftz for x <= -127 (handled below)
        if (unb >= 31) { n = (xb >> 31) ? -(1LL << 40) : (1LL << 40); f23 = 0; }
        else {
            long long v = (long long)(mant << (unb - 23));
            n = (xb >> 31) ? -v : v; f23 = 0;
        }
    } else {
        // shift so that we have 40 fractional bits: value = mant * 2^(unb-23) = (mant << (40 + unb - 23)) / 2^40
        int sh = 40 + unb - 23;  // in [-125, 39]
        uint64_t fixed;          // |x| * 2^40 (exact when sh >= 0; truncated when sh < 0 -- then |x| < 2^-17)
        bool inexact = false;
        if (sh >= 0) fixed = mant << sh;
        else { fixed = mant >> (-sh); inexact = (mant & ((1ull << (-sh)) - 1)) != 0; }
        uint64_t ip = fixed >> 40;
        uint64_t fp = fixed & ((1ull << 40) - 1);
        uint64_t fr;  // fraction * 2^40 (exact or, when inexact, the truncated value; the true value is in (fr, fr+1))
        if (xb >> 31) {  // negative: floor(-v) = -ceil(v)
            if (fp == 0 && !inexact) { n = -(long long)ip; fr = 0; }
            else { n = -(long long)ip - 1; fr = (1ull << 40) - fp; if (inexact) fr -= 1; }
        } else { n = (long long)ip; fr = fp; }
        // quantise fr (40 fractional bits) to 23 bits
        uint64_t q;
        if (FRAC_MODE == 0) q = fr >> 17;
        else {
            uint64_t low = fr & ((1ull << 17) - 1);
            uint64_t half = 1ull << 16;
            q = fr >> 17;
            bool tail = inexact;  // the true fraction has a nonzero tail beyond the 40 bits
            if (low > half || (low == half && (tail || FRAC_MODE == 1 || (q & 1)))) q += 1;
        }
        if (q >= (1ull << 23)) { q -= (1ull << 23); n += 1; }
        f23 = (uint32_t)q;
    }
    uint32_t t = T[f23];  // in [1, 2): exponent field 127
    // ldexp(t, n)
    long long te = (long long)((t >> 23) & 0xFF) + n;
    if (te >= 0xFF) return 0x7F800000u;
    if (te <= 0) return 0u;  // ftz: subnormal result flushed to +0
    return (uint32_t)(te << 23) | (t & 0x7FFFFF);
}

__device__ __forceinline__ uint32_t model_rcp(uint32_t xb, const uint32_t* R) {
    uint32_t s = xb & 0x80000000u;
    uint32_t ax = xb & 0x7FFFFFFFu;
    int e = (ax >> 23) & 0xFF;
    if (e == 0xFF) { if (ax & 0x7FFFFF) return 0x7FFFFFFFu; return s; }  // 1/inf = +-0
    if (e == 0) return s | 0x7F800000u;  // ftz: 1/(+-0 or subnormal) = +-inf
    uint32_t r = R[ax & 0x7FFFFF];       // rcp(1.m) in (0.5, 1]
    int re = (int)((r >> 23) & 0xFF) - (e - 127);
    if (re <= 0) return s;               // ftz: subnormal result -> +-0
    if (re >= 0xFF) return s | 0x7F800000u;
    return s | ((uint32_t)re << 23) | (r & 0x7FFFFF);
}

__global__ void k_verify(uint32_t base, const uint32_t* T, const uint32_t* R, unsigned long long* counts,
                         uint32_t* first_ex2, uint32_t* first_rcp, int nfirst) {
    uint32_t i = base + blockIdx.x * blockDim.x + threadIdx.x;
    float x = __uint_as_float(i);
    uint32_t hw_e = __float_as_uint(ex2_approx_ftz(x));
    uint32_t md_e = model_ex2(i, T);
    bool nan_in = ((i & 0x7F800000u) == 0x7F800000u) && (i & 0x7FFFFFu);
    if (hw_e != md_e && !nan_in) {
        unsigned long long k = atomicAdd(&counts[0], 1ull);
        if (k < (unsigned long long)nfirst) { first_ex2[2 * k] = i; first_ex2[2 * k + 1] = hw_e; }
        // histogram by (sign, exponent field): counts[2 + sign*256 + e]
        atomicAdd(&counts[2 + (i >> 31) * 256 + ((i >> 23) & 0xFF)], 1ull);
    }
    uint32_t hw_r = __float_as_uint(rcp_approx_ftz(x));
    uint32_t md_r = model_rcp(i, R);
    if (hw_r != md_r && !nan_in) {
        unsigned long long k = atomicAdd(&counts[1], 1ull);
        if (k < (unsigned long long)nfirst) { first_rcp[2 * k] = i; first_rcp[2 * k + 1] = hw_r; }
    }
    if (nan_in) {
        // record the canonical NaN outputs once
        if (i == 0x7FC00000u) { first_ex2[2 * nfirst] = hw_e; first_rcp[2 * nfirst] = hw_r; }
    }
}

__global__ void k_probe(const uint32_t* in, uint32_t* out, int n, const uint32_t* T) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    uint32_t i = in[k];
    out[2 * k] = __float_as_uint(ex2_approx_ftz(__uint_as_float(i)));
    out[2 * k + 1] = model_ex2(i, T);
}

#define CK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } } while (0)

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: mufu_probe table <out.bin> | verify <table.bin>\n"); return 2; }
    const size_t N = 1u << 23;
    uint32_t *d_ex2, *d_rcp;
    CK(cudaMalloc(&d_ex2, N * 4)); CK(cudaMalloc(&d_rcp, N * 4));
    std::vector<uint32_t> h_ex2(N), h_rcp(N);
    if (!strcmp(argv[1], "table")) {
        k_table<<<(N + 255) / 256, 256>>>(d_ex2, d_rcp);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h_ex2.data(), d_ex2, N * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_rcp.data(), d_rcp, N * 4, cudaMemcpyDeviceToHost));
        FILE* f = fopen(argv[2], "wb");
        fwrite(h_ex2.data(), 4, N, f); fwrite(h_rcp.data(), 4, N, f); fclose(f);
        printf("wrote %s: ex2 table (2^23 x u32) then rcp table (2^23 x u32)\n", argv[2]);
        return 0;
    }
    if (!strcmp(argv[1], "verify")) {
        FILE* f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
        if (fread(h_ex2.data(), 4, N, f) != N || fread(h_rcp.data(), 4, N, f) != N) { fprintf(stderr, "short read\n"); return 2; }
        fclose(f);
        CK(cudaMemcpy(d_ex2, h_ex2.data(), N * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rcp, h_rcp.data(), N * 4, cudaMemcpyHostToDevice));
        const size_t ncounts = 2 + 512;
        unsigned long long* d_counts; CK(cudaMalloc(&d_counts, ncounts * 8)); CK(cudaMemset(d_counts, 0, ncounts * 8));
        const int nfirst = 32;
        uint32_t *d_fe, *d_fr; CK(cudaMalloc(&d_fe, (2 * nfirst + 2) * 4)); CK(cudaMalloc(&d_fr, (2 * nfirst + 2) * 4));
        CK(cudaMemset(d_fe, 0, (2 * nfirst + 2) * 4)); CK(cudaMemset(d_fr, 0, (2 * nfirst + 2) * 4));
        const uint32_t chunk = 1u << 26;
        for (uint64_t base = 0; base < (1ull << 32); base += chunk) {
            k_verify<<<chunk / 256, 256>>>((uint32_t)base, d_ex2, d_rcp, d_counts, d_fe, d_fr, nfirst);
        }
        CK(cudaDeviceSynchronize());
        std::vector<unsigned long long> counts(ncounts); CK(cudaMemcpy(counts.data(), d_counts, ncounts * 8, cudaMemcpyDeviceToHost));
        std::vector<uint32_t> fe(2 * nfirst + 2), fr(2 * nfirst + 2);
        CK(cudaMemcpy(fe.data(), d_fe, fe.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(fr.data(), d_fr, fr.size() * 4, cudaMemcpyDeviceToHost));
        printf("{\"inputs\": 4294967296, \"frac_mode\": %d, \"ex2_mismatches\": %llu, \"rcp_mismatches\": %llu, \"nan_out_ex2\": \"%08x\", \"nan_out_rcp\": \"%08x\",\n",
               (int)FRAC_MODE, counts[0], counts[1], fe[2 * nfirst], fr[2 * nfirst]);
        printf(" \"ex2_mismatch_hist\": {");
        bool firsth = true;
        for (int s = 0; s < 2; ++s) for (int e = 0; e < 256; ++e) {
            unsigned long long c = counts[2 + s * 256 + e];
            if (c) { printf("%s\"%c%d\": %llu", firsth ? "" : ", ", s ? '-' : '+', e - 127, c); firsth = false; }
        }
        printf("},\n");
        printf(" \"first_ex2\": [");
        for (int k = 0; k < nfirst && (unsigned long long)k < counts[0]; ++k) printf("%s[\"%08x\", \"%08x\"]", k ? ", " : "", fe[2 * k], fe[2 * k + 1]);
        printf("],\n \"first_rcp\": [");
        for (int k = 0; k < nfirst && (unsigned long long)k < counts[1]; ++k) printf("%s[\"%08x\", \"%08x\"]", k ? ", " : "", fr[2 * k], fr[2 * k + 1]);
        printf("]}\n");
        return 0;
    }
    if (!strcmp(argv[1], "probe")) {
        // probe <table.bin> <hex>...: print hardware and model outputs for specific inputs
        FILE* f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
        if (fread(h_ex2.data(), 4, N, f) != N || fread(h_rcp.data(), 4, N, f) != N) { fprintf(stderr, "short read\n"); return 2; }
        fclose(f);
        CK(cudaMemcpy(d_ex2, h_ex2.data(), N * 4, cudaMemcpyHostToDevice));
        int n = argc - 3;
        std::vector<uint32_t> in(n);
        for (int k = 0; k < n; ++k) in[k] = (uint32_t)strtoul(argv[3 + k], nullptr, 16);
        uint32_t *d_in, *d_out; CK(cudaMalloc(&d_in, n * 4)); CK(cudaMalloc(&d_out, n * 8));
        CK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
        k_probe<<<(n + 255) / 256, 256>>>(d_in, d_out, n, d_ex2);
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> out(2 * n); CK(cudaMemcpy(out.data(), d_out, n * 8, cudaMemcpyDeviceToHost));
        for (int k = 0; k < n; ++k) {
            float x = 0; memcpy(&x, &in[k], 4);
            printf("%08x (%.9g) hw=%08x model=%08x %s\n", in[k], x, out[2 * k], out[2 * k + 1], out[2 * k] == out[2 * k + 1] ? "" : "MISMATCH");
        }
        return 0;
    }
    if (!strcmp(argv[1], "dump")) {
        // dump <table.bin> <hex_lo> <hex_hi> <out.bin>: write hw ex2 for every input in [lo, hi) as u32 (for offline analysis)
        FILE* f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
        if (fread(h_ex2.data(), 4, N, f) != N || fread(h_rcp.data(), 4, N, f) != N) { fprintf(stderr, "short read\n"); return 2; }
        fclose(f);
        CK(cudaMemcpy(d_ex2, h_ex2.data(), N * 4, cudaMemcpyHostToDevice));
        uint32_t lo = (uint32_t)strtoul(argv[3], nullptr, 16), hi = (uint32_t)strtoul(argv[4], nullptr, 16);
        int n = (int)(hi - lo);
        std::vector<uint32_t> in(n);
        for (int k = 0; k < n; ++k) in[k] = lo + k;
        uint32_t *d_in, *d_out; CK(cudaMalloc(&d_in, (size_t)n * 4)); CK(cudaMalloc(&d_out, (size_t)n * 8));
        CK(cudaMemcpy(d_in, in.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
        k_probe<<<(n + 255) / 256, 256>>>(d_in, d_out, n, d_ex2);
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> out(2 * (size_t)n); CK(cudaMemcpy(out.data(), d_out, (size_t)n * 8, cudaMemcpyDeviceToHost));
        FILE* g = fopen(argv[5], "wb");
        fwrite(out.data(), 4, out.size(), g); fclose(g);
        printf("wrote %d (hw, model) pairs to %s\n", n, argv[5]);
        return 0;
    }
    fprintf(stderr, "unknown mode\n");
    return 2;
}
