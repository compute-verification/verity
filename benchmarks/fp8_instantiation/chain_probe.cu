// Long-K chains of `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (SASS QMMA.16832.F32.E4M3.E4M3) on sm_120.
//
// One warp per 16 x 8 output tile chains one accumulator over all of K (mode "u", unpromoted), or restarts it at +0 every
// G atoms and promotes it into an FP32 total with one __fadd_rn (mode "p", Pearl-C's sm120 chain at G = 4).  Operands are
// raw E4M3 bytes, A as 16 x K rows and Bt as 8 x K rows (k contiguous), so word D[w][i][j] is the device's own function of
// row i of A and row j of Bt.
//
//   chain_probe gpu   OUTDIR [--ops FILE]...  generate every family, run both modes on the GPU, write OUTDIR/<family>.<mode>.bin
//   chain_probe check OUTDIR                  regenerate the integer families on the CPU and compare (no GPU)
//   chain_probe predict                       the closed forms alone, per family (no GPU, no device words)
//
// The integer families sit on the grid |x| <= 16 (every integer there is an E4M3 value).  `check` compares each device word
// with the exact int64 sum and with three closed forms of the chain on the integer grid: per-atom RZ to 24 bits (what
// GroupSum's 26-bit adder computes when every term is an integer and |C| < 2^26), per-atom RNE, and a per-product FP32
// RNE chain (what an FMA chain on the CUDA cores would give).  `--ops FILE` adds one family from a file (u32 M, N, K, then
// M x K bytes of A and N x K bytes of Bt, M % 16 == 0, N % 8 == 0); the i-th file's words go to OUTDIR/ops<i>.<mode>.bin
// (M x N, row-major) and are compared in Python.
//
// Build: nvcc -O2 -std=c++17 -arch=sm_120a -o chain_probe chain_probe.cu   (no fast-math, no FTZ)
// CPU only (check, predict): g++ -O2 -std=c++17 -DNO_CUDA_KERNEL -x c++ -o chain_probe_cpu chain_probe.cu

#ifndef NO_CUDA_KERNEL
#include <cuda_runtime.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#ifndef NO_CUDA_KERNEL
__device__ __forceinline__ void mma_e4m3(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// One warp per tile; A tile = 16 rows of K bytes, Bt tile = 8 rows of K bytes.  g > 0: promote every g atoms.
__global__ void chain_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt, uint32_t* __restrict__ D,
                             int ntiles, int K, int g) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= ntiles) return;
    const int lane = threadIdx.x & 31, r = lane >> 2, t = lane & 3;
    const uint8_t* a = A + (size_t)warp * 16 * K;
    const uint8_t* b = Bt + (size_t)warp * 8 * K;
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, tot[4] = {0.f, 0.f, 0.f, 0.f};
    const int steps = K / 32;
    for (int s = 0; s < steps; ++s) {
        const uint8_t* as = a + s * 32;
        const uint8_t* bs = b + s * 32;
        uint32_t af[4], bf[2];
        af[0] = *reinterpret_cast<const uint32_t*>(as + (size_t)r * K + t * 4);
        af[1] = *reinterpret_cast<const uint32_t*>(as + (size_t)(r + 8) * K + t * 4);
        af[2] = *reinterpret_cast<const uint32_t*>(as + (size_t)r * K + t * 4 + 16);
        af[3] = *reinterpret_cast<const uint32_t*>(as + (size_t)(r + 8) * K + t * 4 + 16);
        bf[0] = *reinterpret_cast<const uint32_t*>(bs + (size_t)r * K + t * 4);
        bf[1] = *reinterpret_cast<const uint32_t*>(bs + (size_t)r * K + t * 4 + 16);
        float d[4];
        mma_e4m3(d, af, bf, acc);
        for (int i = 0; i < 4; ++i) acc[i] = d[i];
        if (g > 0 && ((s + 1) % g == 0 || s + 1 == steps)) {
            for (int i = 0; i < 4; ++i) {
                tot[i] = __fadd_rn(tot[i], acc[i]);
                acc[i] = 0.f;
            }
        }
    }
    const float* out = g > 0 ? tot : acc;
    uint32_t* d = D + (size_t)warp * 128;
    d[r * 8 + 2 * t] = __float_as_uint(out[0]);
    d[r * 8 + 2 * t + 1] = __float_as_uint(out[1]);
    d[(r + 8) * 8 + 2 * t] = __float_as_uint(out[2]);
    d[(r + 8) * 8 + 2 * t + 1] = __float_as_uint(out[3]);
}
#endif

// ---- host: E4M3 encoding of grid values, deterministic families ------------------------------------------------------

static uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// The E4M3 byte of an integer |v| <= 16 (exact: 1..16 all have at most 4 significant bits).
static uint8_t e4m3_of_int(int v) {
    if (v == 0) return 0;
    uint8_t sign = v < 0 ? 0x80 : 0;
    int m = v < 0 ? -v : v;
    if (m > 16) { fprintf(stderr, "grid value %d outside |x| <= 16\n", v); exit(2); }
    int e = 31 - __builtin_clz((unsigned)m);       // floor(log2 m)
    int frac = ((m << 3) >> e) - 8;                // (m / 2^e - 1) * 8, exact for m <= 16
    if ((((frac + 8) << e) >> 3) != m) { fprintf(stderr, "not exact: %d\n", v); exit(2); }
    return sign | (uint8_t)(((e + 7) << 3) | frac);
}

// One integer family: tiles of A (16 x K) and Bt (8 x K) integer values.
struct Family {
    std::string name;
    int ntiles, K;
    std::vector<int8_t> a, b;  // ntiles*16*K, ntiles*8*K
};

static void fill_tile(Family& f, int w, uint64_t seed) {
    const int K = f.K;
    int8_t* a = f.a.data() + (size_t)w * 16 * K;
    int8_t* b = f.b.data() + (size_t)w * 8 * K;
    uint64_t s = seed * 1000003ull + (uint64_t)w;
    auto rnd = [&](int lo, int hi) { return lo + (int)(splitmix(s) % (uint64_t)(hi - lo + 1)); };
    const std::string& n = f.name;
    for (int i = 0; i < 16; ++i)
        for (int k = 0; k < K; ++k) {
            int x;
            if (n == "int_random") x = rnd(-16, 16);
            else if (n == "int_max" || n == "int_max_neg" || n == "int_halfway" || n == "int_beyond") x = 16;
            else if (n == "int_alt") x = k < K / 2 ? 16 : (k % 32 == 0 ? 0 : -15);
            else if (n == "int_signs") x = (splitmix(s) & 1) ? -16 : 16;
            else if (n == "int_near_top") x = (k == 0 ? 15 : 16) * ((i & 1) ? -1 : 1);
            else if (n == "int_large_small") x = k < K - 2048 ? 16 : rnd(-15, 15);
            else if (n == "int_mixed_atoms") x = (k % 32 == 0) ? 16 : rnd(-3, 3);
            else { fprintf(stderr, "unknown family %s\n", n.c_str()); exit(2); }
            a[(size_t)i * K + k] = (int8_t)x;
        }
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < K; ++k) {
            int y;
            if (n == "int_random") y = rnd(-16, 16);
            else if (n == "int_max") y = 16;
            else if (n == "int_max_neg") y = -16;
            else if (n == "int_signs") y = 16;
            else if (n == "int_alt") y = k < K / 2 ? 16 : 15;                 // climb to 2^23, then odd atom sums of -6975
            else if (n == "int_halfway") y = k < K / 2 ? 16 : -16;           // climb to 2^23, cancel back to 0
            else if (n == "int_near_top") y = (k == 0 ? 15 : 16);             // 225 then 256s: odd prefixes up to 2^24 - 31
            else if (n == "int_large_small") y = k < K - 2048 ? 16 : rnd(1, 15);
            else if (n == "int_mixed_atoms") y = (k % 32 == 0) ? 16 : rnd(-3, 3);
            else if (n == "int_beyond") {
                // 65536 products of 256 reach 2^24 exactly; the next atom adds a small odd total (column j picks it), the
                // last atom nothing: exact 2^24 + odd, where RZ, RNE and a per-product RNE chain all differ.
                if (k < 65536) y = 16;
                else if (k < 65536 + 32) y = (k - 65536) < (2 * j + 1) ? 1 : 0;
                else y = 0;
            } else { fprintf(stderr, "unknown family %s\n", n.c_str()); exit(2); }
            b[(size_t)j * K + k] = (int8_t)y;
        }
    if (n == "int_beyond") {
        // A's rows: x = 16 on the first 65536, then 1 (so the tail products are y), with every odd row negated.
        for (int i = 0; i < 16; ++i)
            for (int k = 0; k < K; ++k) a[(size_t)i * K + k] = (int8_t)((k < 65536 ? 16 : 1) * ((i & 1) ? -1 : 1));
    }
}

static std::vector<Family> families() {
    struct Spec { const char* name; int ntiles, K; };
    const Spec specs[] = {
        {"int_random", 64, 65536},   {"int_max", 2, 65536},          {"int_max_neg", 2, 65536},
        {"int_alt", 2, 65536},       {"int_signs", 16, 65536},       {"int_halfway", 2, 65536},
        {"int_near_top", 4, 65536},  {"int_large_small", 16, 65536}, {"int_mixed_atoms", 16, 65536},
        {"int_beyond", 2, 65536 + 64},
    };
    std::vector<Family> out;
    for (const Spec& sp : specs) {
        Family f{sp.name, sp.ntiles, sp.K, {}, {}};
        f.a.resize((size_t)f.ntiles * 16 * f.K);
        f.b.resize((size_t)f.ntiles * 8 * f.K);
        for (int w = 0; w < f.ntiles; ++w) fill_tile(f, w, 20261015ull + out.size());
        out.push_back(std::move(f));
    }
    return out;
}

static void write_file(const std::string& path, const void* p, size_t n) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp || fwrite(p, 1, n, fp) != n) { fprintf(stderr, "write %s failed\n", path.c_str()); exit(3); }
    fclose(fp);
}

static std::vector<uint8_t> read_file(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) { fprintf(stderr, "read %s failed\n", path.c_str()); exit(3); }
    std::vector<uint8_t> buf;
    uint8_t tmp[1 << 16];
    size_t n;
    while ((n = fread(tmp, 1, sizeof tmp, fp)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    fclose(fp);
    return buf;
}

// ---- host: closed forms on the integer grid ---------------------------------------------------------------------------

static int bitlen(uint64_t v) { return v ? 64 - __builtin_clzll(v) : 0; }

static int64_t round24(int64_t v, bool nearest_even) {
    uint64_t m = v < 0 ? (uint64_t)(-v) : (uint64_t)v;
    int sh = bitlen(m) - 24;
    if (sh <= 0) return v;
    uint64_t q = m >> sh, rem = m & ((1ull << sh) - 1), half = 1ull << (sh - 1);
    if (nearest_even && (rem > half || (rem == half && (q & 1)))) q += 1;
    int64_t r = (int64_t)(q << sh);
    return v < 0 ? -r : r;
}

struct Pred { int64_t exact, rz, rne_step, rne_seq, rz_promoted, max_prefix; bool window_ok; };

static Pred predict(const int8_t* x, const int8_t* y, int K, int g) {
    Pred p{0, 0, 0, 0, 0, 0, true};
    int64_t local = 0, tot = 0;
    for (int s = 0; s < K / 32; ++s) {
        int64_t atom = 0;
        for (int k = s * 32; k < s * 32 + 32; ++k) {
            int64_t pr = (int64_t)x[k] * y[k];
            atom += pr;
            p.rne_seq = round24(p.rne_seq + pr, true);
        }
        if ((p.rz < 0 ? -p.rz : p.rz) >= (1ll << 26)) p.window_ok = false;  // the closed form needs |C| < 2^26
        p.exact += atom;
        int64_t ap = p.exact < 0 ? -p.exact : p.exact;
        if (ap > p.max_prefix) p.max_prefix = ap;
        p.rz = round24(p.rz + atom, false);
        p.rne_step = round24(p.rne_step + atom, true);
        local = round24(local + atom, false);
        if ((s + 1) % g == 0 || s + 1 == K / 32) {
            tot = round24(tot + local, true);
            local = 0;
        }
    }
    p.rz_promoted = tot;
    return p;
}

static int64_t word_int(uint32_t w, bool* ok) {
    float f;
    memcpy(&f, &w, 4);
    double d = (double)f;
    *ok = (d == (double)(int64_t)d);
    return (int64_t)d;
}

static int check(const std::string& outdir) {
    std::vector<Family> fams = families();
    printf("{\"check\": [\n");
    bool first = true;
    int bad_families = 0;
    for (const Family& f : fams) {
        for (int mode = 0; mode < 2; ++mode) {
            const char* tag = mode ? "p" : "u";
            std::vector<uint8_t> raw = read_file(outdir + "/" + f.name + "." + tag + ".bin");
            const uint32_t* D = reinterpret_cast<const uint32_t*>(raw.data());
            long n = 0, vs_exact = 0, vs_rz = 0, vs_rne_step = 0, vs_rne_seq = 0, vs_rz_promoted = 0, nonint = 0, outside = 0;
            long neg_zero = 0;
            int64_t max_prefix = 0;
            char example[512] = "";
            for (int w = 0; w < f.ntiles; ++w)
                for (int i = 0; i < 16; ++i)
                    for (int j = 0; j < 8; ++j) {
                        const int8_t* x = f.a.data() + ((size_t)w * 16 + i) * f.K;
                        const int8_t* y = f.b.data() + ((size_t)w * 8 + j) * f.K;
                        Pred p = predict(x, y, f.K, 4);
                        uint32_t word = D[(size_t)w * 128 + i * 8 + j];
                        if (word == 0x80000000u) ++neg_zero;
                        bool ok;
                        int64_t v = word_int(word, &ok);
                        ++n;
                        if (!ok) { ++nonint; continue; }
                        if (!p.window_ok) ++outside;
                        if (p.max_prefix > max_prefix) max_prefix = p.max_prefix;
                        int64_t model = mode ? p.rz_promoted : p.rz;
                        if (v != p.exact) ++vs_exact;
                        if (v != model) {
                            ++vs_rz;
                            if (!example[0])
                                snprintf(example, sizeof example, "tile %d (%d,%d): device %lld exact %lld model %lld", w, i, j,
                                         (long long)v, (long long)p.exact, (long long)model);
                        }
                        if (v != p.rne_step) ++vs_rne_step;
                        if (v != p.rne_seq) ++vs_rne_seq;
                        if (v != p.rz_promoted) ++vs_rz_promoted;
                    }
            if (vs_rz || nonint || outside) ++bad_families;
            printf("%s  {\"family\": \"%s\", \"mode\": \"%s\", \"K\": %d, \"elements\": %ld, \"max_abs_prefix\": %lld, "
                   "\"mismatch_vs_exact\": %ld, \"mismatch_vs_closed_form\": %ld, \"mismatch_vs_rne_per_atom\": %ld, "
                   "\"mismatch_vs_rne_per_product\": %ld, \"mismatch_vs_promoted_closed_form\": %ld, \"non_integer_words\": %ld, "
                   "\"outside_window\": %ld, \"negative_zero_words\": %ld, \"first_closed_form_mismatch\": \"%s\"}",
                   first ? "" : ",\n", f.name.c_str(), tag, f.K, n, (long long)max_prefix, vs_exact, vs_rz, vs_rne_step,
                   vs_rne_seq, vs_rz_promoted, nonint, outside, neg_zero, example);
            first = false;
        }
    }
    printf("\n], \"families_with_closed_form_mismatches\": %d}\n", bad_families);
    return bad_families ? 1 : 0;
}

static int predict_all() {
    printf("{\"predict\": [\n");
    bool first = true;
    for (const Family& f : families()) {
        long n = 0, rz_inexact = 0, promoted_inexact = 0, rz_vs_rne = 0, rz_vs_seq = 0, outside = 0;
        int64_t max_prefix = 0;
        for (int w = 0; w < f.ntiles; ++w)
            for (int i = 0; i < 16; ++i)
                for (int j = 0; j < 8; ++j) {
                    Pred p = predict(f.a.data() + ((size_t)w * 16 + i) * f.K, f.b.data() + ((size_t)w * 8 + j) * f.K, f.K, 4);
                    ++n;
                    rz_inexact += p.rz != p.exact;
                    promoted_inexact += p.rz_promoted != p.exact;
                    rz_vs_rne += p.rz != p.rne_step;
                    rz_vs_seq += p.rz != p.rne_seq;
                    outside += !p.window_ok;
                    if (p.max_prefix > max_prefix) max_prefix = p.max_prefix;
                }
        printf("%s  {\"family\": \"%s\", \"K\": %d, \"elements\": %ld, \"max_abs_prefix\": %lld, \"closed_form_inexact\": %ld, "
               "\"promoted_inexact\": %ld, \"rz_differs_from_rne_per_atom\": %ld, \"rz_differs_from_rne_per_product\": %ld, "
               "\"outside_window\": %ld}",
               first ? "" : ",\n", f.name.c_str(), f.K, n, (long long)max_prefix, rz_inexact, promoted_inexact, rz_vs_rne,
               rz_vs_seq, outside);
        first = false;
    }
    printf("\n]}\n");
    return 0;
}

#ifndef NO_CUDA_KERNEL
static void run_gpu(const uint8_t* A, const uint8_t* Bt, uint32_t* D, int ntiles, int K, int g) {
    uint8_t *dA, *dB;
    uint32_t* dD;
    size_t na = (size_t)ntiles * 16 * K, nb = (size_t)ntiles * 8 * K, nd = (size_t)ntiles * 128 * 4;
    if (cudaMalloc(&dA, na) || cudaMalloc(&dB, nb) || cudaMalloc(&dD, nd)) { fprintf(stderr, "cudaMalloc\n"); exit(4); }
    cudaMemcpy(dA, A, na, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, Bt, nb, cudaMemcpyHostToDevice);
    cudaMemset(dD, 0xA5, nd);  // poisoned: a word the kernel does not write shows up as a mismatch
    const int warps = 4;
    chain_kernel<<<(ntiles + warps - 1) / warps, 32 * warps>>>(dA, dB, dD, ntiles, K, g);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "kernel: %s\n", cudaGetErrorString(err)); exit(4); }
    cudaMemcpy(D, dD, nd, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);
}

static int gpu(const std::string& outdir, const std::vector<std::string>& ops_files) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device %s cc %d.%d\n", prop.name, prop.major, prop.minor);
    for (const Family& f : families()) {
        std::vector<uint8_t> A(f.a.size()), B(f.b.size());
        for (size_t i = 0; i < A.size(); ++i) A[i] = e4m3_of_int(f.a[i]);
        for (size_t i = 0; i < B.size(); ++i) B[i] = e4m3_of_int(f.b[i]);
        std::vector<uint32_t> D((size_t)f.ntiles * 128);
        for (int mode = 0; mode < 2; ++mode) {
            run_gpu(A.data(), B.data(), D.data(), f.ntiles, f.K, mode ? 4 : 0);
            write_file(outdir + "/" + f.name + (mode ? ".p.bin" : ".u.bin"), D.data(), D.size() * 4);
        }
        printf("family %s: %d tiles, K = %d\n", f.name.c_str(), f.ntiles, f.K);
    }
    for (size_t oi = 0; oi < ops_files.size(); ++oi) {
        const char* ops = ops_files[oi].c_str();
        std::vector<uint8_t> raw = read_file(ops);
        uint32_t M, N, K;
        memcpy(&M, raw.data(), 4); memcpy(&N, raw.data() + 4, 4); memcpy(&K, raw.data() + 8, 4);
        const uint8_t* a = raw.data() + 12;
        const uint8_t* b = a + (size_t)M * K;
        if (M % 16 || N % 8 || K % 128 || raw.size() != 12 + (size_t)(M + N) * K) { fprintf(stderr, "bad ops file\n"); exit(2); }
        // every (16-row block of A, 8-row block of Bt) pair is one tile
        int tm = M / 16, tn = N / 8, ntiles = tm * tn;
        std::vector<uint8_t> A((size_t)ntiles * 16 * K), B((size_t)ntiles * 8 * K);
        for (int bi = 0; bi < tm; ++bi)
            for (int bj = 0; bj < tn; ++bj) {
                int w = bi * tn + bj;
                memcpy(A.data() + (size_t)w * 16 * K, a + (size_t)bi * 16 * K, (size_t)16 * K);
                memcpy(B.data() + (size_t)w * 8 * K, b + (size_t)bj * 8 * K, (size_t)8 * K);
            }
        std::vector<uint32_t> D((size_t)ntiles * 128), Out((size_t)M * N);
        for (int mode = 0; mode < 2; ++mode) {
            run_gpu(A.data(), B.data(), D.data(), ntiles, K, mode ? 4 : 0);
            for (int bi = 0; bi < tm; ++bi)
                for (int bj = 0; bj < tn; ++bj)
                    for (int i = 0; i < 16; ++i)
                        for (int j = 0; j < 8; ++j)
                            Out[(size_t)(bi * 16 + i) * N + bj * 8 + j] = D[(size_t)(bi * tn + bj) * 128 + i * 8 + j];
            write_file(outdir + "/ops" + std::to_string(oi) + (mode ? ".p.bin" : ".u.bin"), Out.data(), Out.size() * 4);
        }
        printf("ops file %s: M %u N %u K %u\n", ops, M, N, K);
    }
    return 0;
}
#endif

int main(int argc, char** argv) {
    if (argc == 2 && !strcmp(argv[1], "predict")) return predict_all();
    if (argc < 3) { fprintf(stderr, "usage: chain_probe gpu|check OUTDIR [--ops FILE]... | predict\n"); return 2; }
    std::string cmd = argv[1], outdir = argv[2];
    std::vector<std::string> ops;
    for (int i = 3; i + 1 < argc; i += 2)
        if (!strcmp(argv[i], "--ops")) ops.push_back(argv[i + 1]);
    if (cmd == "check") return check(outdir);
#ifndef NO_CUDA_KERNEL
    if (cmd == "gpu") return gpu(outdir, ops);
#endif
    fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
