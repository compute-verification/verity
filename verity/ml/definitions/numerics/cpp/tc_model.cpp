// CPU twin of veritor.core.silicon / verity.ml.definitions.numerics.mma (integer-exact tensor-core
// semantics) plus a parametrised full-GEMM model: per-output-element chains of
// ``tc_dot`` steps over K-tiles, optional split-K with an FP32 reduction of the
// partial chains, and the bias / output-rounding epilogue.  OpenMP over outputs.
//
// Derived from gpu/tensor-core-semantics/ref_chain.cpp (same group_sum, same Term).
//
// Build: g++ -O2 -std=c++17 -fopenmp -fPIC -shared -o libtc_model.so tc_model.cpp

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct Term {
    bool neg;
    int exp;
    int64_t mag;  // value = (-1)^neg * mag * 2^(exp-23); mag == 0 is zero
};

inline Term bf16_product(uint32_t a, uint32_t b) {
    int ea = (a >> 7) & 0xFF, eb = (b >> 7) & 0xFF;
    int64_t sa = (a & 0x7F) | (ea ? 0x80 : 0), sb = (b & 0x7F) | (eb ? 0x80 : 0);
    int64_t mag = (sa * sb) << 9;
    if (!mag) return Term{false, 0, 0};
    return Term{((a ^ b) >> 15 & 1) != 0, (ea ? ea : 1) + (eb ? eb : 1) - 254, mag};
}

inline Term f16_product(uint32_t a, uint32_t b) {
    int ea = (a >> 10) & 0x1F, eb = (b >> 10) & 0x1F;
    int64_t sa = (a & 0x3FF) | (ea ? 0x400 : 0), sb = (b & 0x3FF) | (eb ? 0x400 : 0);
    int64_t mag = (sa * sb) << 3;
    if (!mag) return Term{false, 0, 0};
    return Term{((a ^ b) >> 15 & 1) != 0, (ea ? ea : 1) + (eb ? eb : 1) - 30, mag};
}

inline Term fp32_term(uint32_t bits) {
    int e = (bits >> 23) & 0xFF;
    int64_t m = bits & 0x7FFFFF;
    bool neg = (bits >> 31) != 0;
    if (e == 0) return m ? Term{neg, -126, m} : Term{false, 0, 0};
    return Term{neg, e - 127, (int64_t(1) << 23) | m};
}

inline uint32_t pack_fp32(const Term& t) {
    if (!t.mag) return 0;
    if (t.exp > 127) return (uint32_t(t.neg) << 31) | 0x7F800000u;
    int e = t.exp + 127;
    if (!(t.mag & (int64_t(1) << 23))) e -= 1;
    return (uint32_t(t.neg) << 31) | (uint32_t(e) << 23) | (uint32_t(t.mag) & 0x7FFFFF);
}

inline int bit_length(int64_t x) { return x ? 64 - __builtin_clzll((unsigned long long)x) : 0; }

inline Term group_sum(const Term* terms, int n, int width, int zero_exp) {
    const int rescale = width - 24;
    int max_exp = zero_exp;
    for (int i = 0; i < n; ++i)
        if (terms[i].mag && terms[i].exp > max_exp) max_exp = terms[i].exp;
    int64_t total = 0;
    for (int i = 0; i < n; ++i) {
        if (!terms[i].mag) continue;
        int64_t scaled = rescale >= 0 ? terms[i].mag << rescale : terms[i].mag >> (-rescale);
        int shift = max_exp - terms[i].exp;
        int64_t aligned = shift >= 63 ? 0 : scaled >> shift;
        total += terms[i].neg ? -aligned : aligned;
    }
    if (!total) return Term{false, 0, 0};
    bool neg = total < 0;
    int64_t mag = neg ? -total : total;
    int bl = bit_length(mag);
    int exp = max_exp + bl - width;
    if (bl > width) mag >>= (bl - width);
    else mag <<= (width - bl);
    if (exp < -126) {
        int s = -126 - exp;
        mag = s >= 63 ? 0 : mag >> s;
        exp = -126;
    }
    mag = rescale >= 0 ? mag >> rescale : mag << (-rescale);
    if (!mag) return Term{false, 0, 0};
    return Term{neg, exp, mag};
}

struct Params {
    const int* groups;
    int ngroups;
    int width;
    int zero_exp;
    int ks;  // instruction K = sum(groups)
};

// One instruction: acc <- tc_dot(acc, a[0..ks), b[0..ks)).  Returns true when saturated.
template <bool BF16>
inline bool tc_step(Term& acc, const uint16_t* a, const uint16_t* b, const Params& P) {
    Term terms[65];
    int off = 0;
    for (int gi = 0; gi < P.ngroups; ++gi) {
        terms[0] = acc;
        for (int p = 0; p < P.groups[gi]; ++p)
            terms[1 + p] = BF16 ? bf16_product(a[off + p], b[off + p]) : f16_product(a[off + p], b[off + p]);
        acc = group_sum(terms, 1 + P.groups[gi], P.width, P.zero_exp);
        off += P.groups[gi];
        if (acc.mag && acc.exp > 127) return true;
    }
    return false;
}

// Chain over K-tiles [t0, t1) of a row/column pair in tile order given by `order` (or ascending).
template <bool BF16>
inline uint32_t chain(uint32_t acc_bits, const uint16_t* a, const uint16_t* b, int t0, int t1, const int* order,
                      const Params& P) {
    Term acc = fp32_term(acc_bits);
    for (int t = t0; t < t1; ++t) {
        int tile = order ? order[t] : t;
        int off = tile * P.ks;
        if (tc_step<BF16>(acc, a + off, b + off, P)) break;
    }
    return pack_fp32(acc);
}

inline float f32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline uint32_t u32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// The NaN word a GPU kernel writes when it converts an FP32 NaN to BF16: cvt.rn.bf16.f32 / F2F
// produce the canonical 0x7FFF (measured on every NaN output of cuBLAS's sliced/split-K kernels
// on the A100, results/.../raw/w4b/overflow_diagnosis.json); torch's CPU/c10 conversion gives 0x7FC0.
inline uint16_t f32_to_bf16_rne(uint32_t bits, uint16_t nan_word) {
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu)) return nan_word;
    uint32_t lsb = (bits >> 16) & 1u;
    return (uint16_t)((bits + 0x7FFFu + lsb) >> 16);
}

inline float bf16_f32(uint16_t w) { return f32(uint32_t(w) << 16); }

// Split-K reduction of `n` FP32 partials.  mode 0: sequential ascending ((p0+p1)+p2)...;
// mode 1: pairwise tree (p0+p1)+(p2+p3)...; mode 2: sequential descending.  Modes 3/4/5 are the
// same three orders applied to partials first rounded to BF16 (RNE) and widened again -- what
// cuBLASLt's splitKreduce_kernel does when the workspace partials are stored as __nv_bfloat16
// (torch's default allow_bf16_reduced_precision_reduction=True).
inline float reduce_partials_f32(const float* p, int n, int mode) {
    if (n == 1) return p[0];
    if (mode == 1) {
        std::vector<float> v(p, p + n);
        while (v.size() > 1) {
            std::vector<float> w;
            for (size_t i = 0; i + 1 < v.size(); i += 2) w.push_back(v[i] + v[i + 1]);
            if (v.size() & 1) w.push_back(v.back());
            v.swap(w);
        }
        return v[0];
    }
    if (mode == 2) {
        float s = p[n - 1];
        for (int i = n - 2; i >= 0; --i) s += p[i];
        return s;
    }
    float s = p[0];
    for (int i = 1; i < n; ++i) s += p[i];
    return s;
}

// mode 6: BF16 partials accumulated sequentially with a BF16 rounding after every add (a reduce
// kernel that keeps its running sum in the BF16 output).
// mode 7/8: serial split-K through the BF16 output (CUTLASS semaphore-ordered split-K, seen as a
// Memset + one GEMM kernel and no reduce kernel): split 0 writes bf16(p0); every later split reads
// the bf16 output back, adds its own FP32 partial and writes bf16 again --
// s = bf16(p0); s = bf16(f32(s) + p_i).  7 = ascending split order, 8 = descending.
inline float reduce_partials(const float* p, int n, int mode, uint16_t nan_word) {
    if (mode < 3) return reduce_partials_f32(p, n, mode);
    if (mode == 7 || mode == 8) {
        int first = mode == 7 ? 0 : n - 1, step = mode == 7 ? 1 : -1;
        float s = bf16_f32(f32_to_bf16_rne(u32(p[first]), nan_word));
        for (int k = 1, i = first + step; k < n; ++k, i += step) s = bf16_f32(f32_to_bf16_rne(u32(s + p[i]), nan_word));
        return s;
    }
    float q[64];
    for (int i = 0; i < n; ++i) q[i] = bf16_f32(f32_to_bf16_rne(u32(p[i]), nan_word));
    if (mode == 6) {
        float s = q[0];
        for (int i = 1; i < n; ++i) s = bf16_f32(f32_to_bf16_rne(u32(s + q[i]), nan_word));
        return s;
    }
    return reduce_partials_f32(q, n, mode - 3);
}

}  // namespace

extern "C" {

// Per-record tc_dot on n records: acc[n], a[n*ks], b[n*ks] -> d[n].  dtype 0 = bf16, 1 = f16.
void tc_records(int dtype, const uint32_t* acc, const uint16_t* a, const uint16_t* b, uint32_t* d, long n,
                const int* groups, int ngroups, int width, int zero_exp) {
    int ks = 0;
    for (int i = 0; i < ngroups; ++i) ks += groups[i];
    Params P{groups, ngroups, width, zero_exp, ks};
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        Term acc_t = fp32_term(acc[i]);
        if (dtype == 0)
            tc_step<true>(acc_t, a + i * ks, b + i * ks, P);
        else
            tc_step<false>(acc_t, a + i * ks, b + i * ks, P);
        d[i] = pack_fp32(acc_t);
    }
}

// Full GEMM model.  A: M x K words (row-major), Bt: N x K words (row-major, i.e. torch Linear
// weight [out, in]), bias: N words or NULL, acc_init: M x N f32 bits or NULL.
//   nsplit          : number of K splits (>= 1); split s covers tiles [bounds[s], bounds[s+1]) where
//                     bounds has nsplit+1 entries in tile units (K/ks tiles total); NULL = equal splits.
//   reduce_mode     : how the split partials are summed in FP32 (see reduce_partials).
//   order           : optional permutation of the K tiles (length K/ks) applied before splitting.
//   bias_mode       : 0 none; 1 fp32 add of the bias after the (reduced) chain, then RNE->bf16;
//                     2 bias as the initial accumulator of the first split's chain (term of group 1);
//                     3 chain -> RNE->bf16, then bf16 + bias in fp32 -> RNE->bf16 (torch-style separate add);
//                     4 bias added in fp32 to split partial 0 before the split reduction.
//   out_f32         : M x N f32 bits of the value before output rounding (after bias for modes 1/2/4;
//                     for mode 3 the pre-bias chain value) -- may be NULL.
//   out_bf16        : M x N bf16 words (RNE of the fp32 result) -- may be NULL.
//   nan_word        : the BF16 word written for an FP32 NaN (0x7FFF = GPU cvt.rn.bf16.f32 canonical NaN;
//                     0x7FC0 = torch CPU/c10 conversion).  Applies to every FP32 -> BF16 rounding step.
void gemm_model(int dtype, const uint16_t* A, const uint16_t* Bt, const uint16_t* bias, const uint32_t* acc_init,
                int M, int N, int K, const int* groups, int ngroups, int width, int zero_exp, int nsplit,
                const int* bounds, int reduce_mode, const int* order, int bias_mode, uint32_t* out_f32,
                uint16_t* out_bf16, int nan_word_i) {
    const uint16_t nan_word = (uint16_t)nan_word_i;
    int ks = 0;
    for (int i = 0; i < ngroups; ++i) ks += groups[i];
    Params P{groups, ngroups, width, zero_exp, ks};
    const int ntiles = K / ks;
    std::vector<int> bnd(nsplit + 1);
    if (bounds) {
        for (int s = 0; s <= nsplit; ++s) bnd[s] = bounds[s];
    } else {
        for (int s = 0; s <= nsplit; ++s) bnd[s] = (int)((long)ntiles * s / nsplit);
    }
#pragma omp parallel for schedule(dynamic, 64)
    for (long idx = 0; idx < (long)M * N; ++idx) {
        int i = idx / N, j = idx % N;
        const uint16_t* a = A + (size_t)i * K;
        const uint16_t* b = Bt + (size_t)j * K;
        uint32_t acc0 = acc_init ? acc_init[idx] : 0u;
        if (bias_mode == 2 && bias) acc0 = u32(bf16_f32(bias[j]));
        float partials[64];
        int np = nsplit > 64 ? 64 : nsplit;
        for (int s = 0; s < np; ++s) {
            uint32_t a0 = (s == 0) ? acc0 : 0u;
            uint32_t r = dtype == 0 ? chain<true>(a0, a, b, bnd[s], bnd[s + 1], order, P)
                                    : chain<false>(a0, a, b, bnd[s], bnd[s + 1], order, P);
            partials[s] = f32(r);
        }
        if (bias_mode == 4 && bias) partials[0] = partials[0] + bf16_f32(bias[j]);
        float v = reduce_partials(partials, np, reduce_mode, nan_word);
        uint16_t o;
        if (bias_mode == 1 && bias) {
            v = v + bf16_f32(bias[j]);
            o = f32_to_bf16_rne(u32(v), nan_word);
        } else if (bias_mode == 3 && bias) {
            uint16_t r = f32_to_bf16_rne(u32(v), nan_word);
            float w = bf16_f32(r) + bf16_f32(bias[j]);
            o = f32_to_bf16_rne(u32(w), nan_word);
        } else {
            o = f32_to_bf16_rne(u32(v), nan_word);
        }
        if (out_f32) out_f32[idx] = u32(v);
        if (out_bf16) out_bf16[idx] = o;
    }
}

}  // extern "C"
