// Independent CPU model of vLLM's batch-invariant Triton ``_rms_norm_kernel`` (W11), as compiled for sm_89:
// BLOCK_SIZE = 1024, HAS_WEIGHT = True, num_warps = 4, one CTA (128 threads, 8 columns per thread) per row.
// Every rounding point below was read off the kernel's PTX and SASS:
//
//   per thread (columns 8t .. 8t+7 of each 1024-column chunk, masked as a whole when 8t+off >= n_cols):
//       acc = x1*x1 ; acc = fma(x0,x0,acc) ; acc = fma(x2,x2,acc) ; ... ; acc = fma(x7,x7,acc)     (FMUL/FFMA, RN, no ftz)
//   warp:   butterfly xor 16, 8, 4, 2, 1 (SHFL.BFLY + FADD) -> every lane holds the same warp sum
//   CTA:    lanes 0..3 of warp 0 read the 4 warp sums, butterfly xor 2, 1 -> (w0 + w2) + (w1 + w3)
//   chunks: sum = sum + chunk_sum (FADD; sum starts at +0)
//   scalar: mean_eps = FFMA(MUFU.RCP(float(n_cols)), sum, eps)     -- div.full.f32 + add.f32 fused by ptxas
//           rms      = MUFU.SQRT(mean_eps)                          -- sqrt.approx.ftz.f32
//           inv      = MUFU.RCP(rms) * 1.0f                         -- div.full.f32(1, rms); both with range scaling
//   output: out_j = bf16_rn( (inv * x_j) * w_j )                    -- two FMULs, F2FP.BF16.F32.PACK_AB
//
// MUFU.RCP and MUFU.SQRT are the hardware tables measured on the RTX 4090 (cuda/mufu_probe.cu, cuda/rms_probe.cu;
// both verified on all 2^32 inputs).  The ``variant`` bit mask deliberately breaks single dataflow choices so
// the check harness can show that each alternative mismatches the production kernel (0 = the model).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

inline float bf16_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// F2FP.BF16.F32 (round to nearest even; NaN -> canonical 0x7FFF; subnormals kept)
inline uint16_t f32_bf16_rn(float f) {
    uint32_t u = f2u(f);
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu)) return 0x7FFF;
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

inline uint32_t mufu_rcp(uint32_t xb, const uint32_t* R) {
    uint32_t s = xb & 0x80000000u, ax = xb & 0x7FFFFFFFu;
    int e = (ax >> 23) & 0xFF;
    if (e == 0xFF) { if (ax & 0x7FFFFF) return 0x7FFFFFFFu; return s; }
    if (e == 0) return s | 0x7F800000u;
    uint32_t r = R[ax & 0x7FFFFF];
    int re = (int)((r >> 23) & 0xFF) - (e - 127);
    if (re <= 0) return s;
    if (re >= 0xFF) return s | 0x7F800000u;
    return s | ((uint32_t)re << 23) | (r & 0x7FFFFF);
}

inline uint32_t mufu_sqrt(uint32_t xb, const uint32_t* T) {
    uint32_t s = xb & 0x80000000u, ax = xb & 0x7FFFFFFFu;
    int e = (ax >> 23) & 0xFF;
    if (e == 0xFF) { if (ax & 0x7FFFFF) return 0x7FFFFFFFu; return s ? 0x7FFFFFFFu : 0x7F800000u; }
    if (e == 0) return s;                 // ftz: +-0 / subnormal -> +-0
    if (s) return 0x7FFFFFFFu;            // negative -> NaN
    int k = e - 127, par = k & 1;
    uint32_t t = T[((uint32_t)par << 23) | (ax & 0x7FFFFF)];
    int re = (int)((t >> 23) & 0xFF) + (k - par) / 2;
    return ((uint32_t)re << 23) | (t & 0x7FFFFF);
}

// div.full.f32(a, b) as ptxas emits it: range scaling, MUFU.RCP, then a multiply (returned separately so the
// caller can fuse it into an FFMA, as the kernel's first division is).
struct DivFull { float a; float r; };
inline DivFull div_full_parts(float a, float b, const uint32_t* R) {
    float ab = std::fabs(b);
    bool p0 = ab > 8.507059173023462e37f;
    bool p1 = !(ab < 1.1754943508222875e-38f);
    if (p0) { a = a * 0.25f; b = b * 0.25f; }
    if (!p1) { a = a * 16777216.0f; b = b * 16777216.0f; }
    return {a, u2f(mufu_rcp(f2u(b), R))};
}

enum Variant {
    V_NO_EPS_FUSION = 1,      // mean = RN(rcp(n) * sum); mean_eps = RN(mean + eps)
    V_SEQUENTIAL_SUM = 2,     // sum = sequential fp32 accumulation of RN(x^2)
    V_IEEE_SQRT = 4,          // sqrtf
    V_IEEE_RCP = 8,           // 1.0f / x and sum / n correctly rounded (mean_eps = RN(sum / n + eps))
    V_THREAD_ORDER_X0 = 16,   // acc = x0^2 ; fma x1 .. x7
    V_FP64_SUM = 32,          // sum in double, rounded once
    V_OUT_XW_FIRST = 64,      // out = RN(RN(x * w) * inv)
    V_UNFUSED_SQUARES = 128,  // acc = RN(acc + RN(x^2))
};

}  // namespace

extern "C" void rms_triton_rows(const uint16_t* x, const uint16_t* w, long rows, int n_cols, float eps,
                                const uint32_t* rcp_tab, const uint32_t* sqrt_tab, uint16_t* out, float* dbg,
                                float* pre, int variant) {
    const int BLOCK = 1024, THREADS = 128, PER = 8;
#pragma omp parallel for schedule(dynamic, 16)
    for (long r = 0; r < rows; ++r) {
        const uint16_t* xr = x + r * n_cols;
        uint16_t* orow = out + r * n_cols;
        float sum = 0.0f;
        if (variant & V_FP64_SUM) {
            double d = 0.0;
            for (int j = 0; j < n_cols; ++j) { double v = bf16_f32(xr[j]); d += v * v; }
            sum = (float)d;
        } else if (variant & V_SEQUENTIAL_SUM) {
            for (int j = 0; j < n_cols; ++j) { float v = bf16_f32(xr[j]); sum = sum + v * v; }
        } else {
            for (int off = 0; off < n_cols; off += BLOCK) {
                float lane[THREADS];
                for (int t = 0; t < THREADS; ++t) {
                    int c0 = off + PER * t;
                    if (c0 >= n_cols) { lane[t] = 0.0f; continue; }
                    float v[PER];
                    for (int j = 0; j < PER; ++j) v[j] = (c0 + j < n_cols) ? bf16_f32(xr[c0 + j]) : 0.0f;
                    float acc;
                    if (variant & V_UNFUSED_SQUARES) {
                        acc = v[1] * v[1];
                        acc = acc + v[0] * v[0];
                        for (int j = 2; j < PER; ++j) acc = acc + v[j] * v[j];
                    } else if (variant & V_THREAD_ORDER_X0) {
                        acc = v[0] * v[0];
                        for (int j = 1; j < PER; ++j) acc = std::fmaf(v[j], v[j], acc);
                    } else {
                        acc = v[1] * v[1];
                        acc = std::fmaf(v[0], v[0], acc);
                        for (int j = 2; j < PER; ++j) acc = std::fmaf(v[j], v[j], acc);
                    }
                    lane[t] = acc;
                }
                float wsum[4];
                for (int wp = 0; wp < 4; ++wp) {
                    float v[32], nv[32];
                    for (int i = 0; i < 32; ++i) v[i] = lane[32 * wp + i];
                    for (int s = 16; s >= 1; s >>= 1) {
                        for (int i = 0; i < 32; ++i) nv[i] = v[i] + v[i ^ s];
                        for (int i = 0; i < 32; ++i) v[i] = nv[i];
                    }
                    wsum[wp] = v[0];
                }
                float chunk = (wsum[0] + wsum[2]) + (wsum[1] + wsum[3]);
                sum = sum + chunk;
            }
        }
        float mean_eps, rms, inv;
        if (variant & V_IEEE_RCP) {
            float mean = sum / (float)n_cols;
            mean_eps = mean + eps;
        } else {
            DivFull d = div_full_parts(sum, (float)n_cols, rcp_tab);
            if (variant & V_NO_EPS_FUSION) { float mean = d.a * d.r; mean_eps = mean + eps; }
            else mean_eps = std::fmaf(d.r, d.a, eps);
        }
        if (variant & V_IEEE_SQRT) rms = std::sqrt(mean_eps);
        else rms = u2f(mufu_sqrt(f2u(mean_eps), sqrt_tab));
        if (variant & V_IEEE_RCP) inv = 1.0f / rms;
        else { DivFull d = div_full_parts(1.0f, rms, rcp_tab); inv = d.a * d.r; }
        if (dbg) { dbg[4 * r] = sum; dbg[4 * r + 1] = mean_eps; dbg[4 * r + 2] = rms; dbg[4 * r + 3] = inv; }
        for (int j = 0; j < n_cols; ++j) {
            float xv = bf16_f32(xr[j]), wv = bf16_f32(w[j]);
            float o = (variant & V_OUT_XW_FIRST) ? (xv * wv) * inv : (inv * xv) * wv;
            if (pre) pre[r * n_cols + j] = o;
            orow[j] = f32_bf16_rn(o);
        }
    }
}

// ---- vllm::rms_norm_kernel<c10::BFloat16, VEC_SIZE=8, NUM_DIMS=2, HasWeight=true> (torch.ops._C.rms_norm) ----------
// Read off the sm_89 SASS of _C_stable_libtorch.abi3.so:
//   launch:  grid = rows, block = min(n_cols / 8, 1024) threads; thread t owns 8-vectors i = t, t + block, ...
//   thread:  variance = 0 ; FFMA(x_j, x_j, variance) for j = 0..7 of each vector, vectors in order (no ftz)
//   warp:    cub WarpReduce: SHFL.DOWN 1, 2, 4, 8, 16 with predicated FADD (lane i adds lane i+off when it is valid)
//   block:   cub BlockReduce<1024> WARP_REDUCTIONS: agg = w0 ; agg = agg + w_k for k = 1 .. nwarps-1 (sequential)
//   scalar:  mean = agg / float(n_cols)  (IEEE div.rn: MUFU.RCP + 4 FFMA + FCHK slow path)  ; arg = mean + eps (FADD)
//            s = rsqrt.approx.f32(arg): |arg| < 2^-126 -> arg *= 2^24 ; MUFU.RSQ ; s *= 2^12
//   output:  out_j = bf16_rn( (s * x_j) * w_j )   (two FMULs, F2FP.BF16.F32.PACK_AB)

namespace {
inline uint32_t mufu_rsq(uint32_t xb, const uint32_t* T) {
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
inline float rsqrt_nonftz(float x, const uint32_t* T) {
    bool p0 = !(std::fabs(x) < 1.1754943508222875e-38f);
    if (!p0) x = x * 16777216.0f;
    float r = u2f(mufu_rsq(f2u(x), T));
    if (!p0) r = r * 4096.0f;
    return r;
}
enum CudaVariant {
    C_BUTTERFLY_WARP = 1,     // Triton-style xor butterfly instead of cub's shfl_down tree
    C_SEQUENTIAL_SUM = 2,     // sequential fp32 sum of squares
    C_IEEE_RSQRT = 4,         // 1.0f / sqrtf(arg)
    C_DIV_FULL = 8,           // mean via MUFU.RCP * agg (div.full) instead of IEEE division
    C_FP64_SUM = 32,          // sum in double
    C_OUT_XW_FIRST = 64,      // out = RN(RN(x * w) * s)
    C_WARP_TREE_REVERSED = 128,  // warp aggregates summed from the last warp
};
}  // namespace

extern "C" void rms_cuda_rows(const uint16_t* x, const uint16_t* w, long rows, int n_cols, float eps,
                              const uint32_t* rcp_tab, const uint32_t* rsq_tab, uint16_t* out, float* dbg, float* pre,
                              int variant) {
    const int VEC = 8;
    if (n_cols % VEC != 0) return;  // other VEC_SIZE instantiations: not modelled (caller gates)
    const int nvec = n_cols / VEC;
    const int block = nvec < 1024 ? nvec : 1024;
    const int nwarps = (block + 31) / 32;
#pragma omp parallel for schedule(dynamic, 16)
    for (long r = 0; r < rows; ++r) {
        const uint16_t* xr = x + r * n_cols;
        uint16_t* orow = out + r * n_cols;
        float agg = 0.0f;
        if (variant & C_FP64_SUM) {
            double d = 0.0;
            for (int j = 0; j < n_cols; ++j) { double v = bf16_f32(xr[j]); d += v * v; }
            agg = (float)d;
        } else if (variant & C_SEQUENTIAL_SUM) {
            for (int j = 0; j < n_cols; ++j) { float v = bf16_f32(xr[j]); agg = std::fmaf(v, v, agg); }
        } else {
            std::vector<float> lane(block, 0.0f);
            for (int t = 0; t < block; ++t) {
                float acc = 0.0f;
                for (int i = t; i < nvec; i += block)
                    for (int j = 0; j < VEC; ++j) { float v = bf16_f32(xr[i * VEC + j]); acc = std::fmaf(v, v, acc); }
                lane[t] = acc;
            }
            std::vector<float> wagg(nwarps);
            for (int wp = 0; wp < nwarps; ++wp) {
                int valid = block - 32 * wp; if (valid > 32) valid = 32;
                float v[32], nv[32];
                for (int i = 0; i < 32; ++i) v[i] = (i < valid) ? lane[32 * wp + i] : 0.0f;
                if (variant & C_BUTTERFLY_WARP) {
                    for (int s = 16; s >= 1; s >>= 1) {
                        for (int i = 0; i < 32; ++i) nv[i] = v[i] + v[i ^ s];
                        for (int i = 0; i < 32; ++i) v[i] = nv[i];
                    }
                } else {
                    for (int off = 1; off <= 16; off <<= 1) {
                        for (int i = 0; i < 32; ++i) nv[i] = (i + off < valid) ? v[i] + v[i + off] : v[i];
                        for (int i = 0; i < 32; ++i) v[i] = nv[i];
                    }
                }
                wagg[wp] = v[0];
            }
            if (variant & C_WARP_TREE_REVERSED) { agg = wagg[nwarps - 1]; for (int k = nwarps - 2; k >= 0; --k) agg = agg + wagg[k]; }
            else { agg = wagg[0]; for (int k = 1; k < nwarps; ++k) agg = agg + wagg[k]; }
        }
        float mean;
        if (variant & C_DIV_FULL) { DivFull d = div_full_parts(agg, (float)n_cols, rcp_tab); mean = d.a * d.r; }
        else mean = agg / (float)n_cols;
        float arg = mean + eps;
        float s = (variant & C_IEEE_RSQRT) ? 1.0f / std::sqrt(arg) : rsqrt_nonftz(arg, rsq_tab);
        if (dbg) { dbg[4 * r] = agg; dbg[4 * r + 1] = arg; dbg[4 * r + 2] = mean; dbg[4 * r + 3] = s; }
        for (int j = 0; j < n_cols; ++j) {
            float xv = bf16_f32(xr[j]), wv = bf16_f32(w[j]);
            float o = (variant & C_OUT_XW_FIRST) ? (xv * wv) * s : (s * xv) * wv;
            if (pre) pre[r * n_cols + j] = o;
            orow[j] = f32_bf16_rn(o);
        }
    }
}

extern "C" void rms_mufu_rsq_words(const uint32_t* x, uint32_t* y, long n, const uint32_t* T) {
    for (long i = 0; i < n; ++i) y[i] = mufu_rsq(x[i], T);
}
// rsqrtf as compiled without fast-math (rsqrt.approx.f32): the MUFU.RSQ with the subnormal range scaling
extern "C" void rms_rsqrt_words(const uint32_t* x, uint32_t* y, long n, const uint32_t* T) {
    for (long i = 0; i < n; ++i) y[i] = f2u(rsqrt_nonftz(u2f(x[i]), T));
}

// Element-wise table lookups (for tests / probes).
extern "C" void rms_mufu_sqrt_words(const uint32_t* x, uint32_t* y, long n, const uint32_t* T) {
    for (long i = 0; i < n; ++i) y[i] = mufu_sqrt(x[i], T);
}
extern "C" void rms_div_full_words(const uint32_t* a, const uint32_t* b, uint32_t* y, long n, const uint32_t* R) {
    for (long i = 0; i < n; ++i) { DivFull d = div_full_parts(u2f(a[i]), u2f(b[i]), R); y[i] = f2u(d.a * d.r); }
}
