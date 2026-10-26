// CPU model of the FlashAttention-2 forward kernel vLLM runs on Ada/Ampere for BF16 hdim-128
// (``flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, bf16>, ...,
// Is_even_MN=false, Is_even_K=true, Split=false>`` from vllm-flash-attn @ 506341a, built with
// --use_fast_math), one output row (query row x head) at a time, bit-exact by construction from:
//
//   * the Ampere ``mma.sync.m16n8k16`` BF16/FP32 model (tc_model.cpp, = veritor.core.silicon) for
//     both tensor-core GEMMs (S = Q K^T over 8 k-steps of 16; O += P V over 4 k-steps of 16 per
//     64-key block), accumulators kept as FP32 words between instructions;
//   * the kernel's online softmax (softmax.h): blocks visited from the LAST 64-key block down to
//     block 0; per block: row max (exact), rescale of the running sums and of O by
//     ex2((m_prev - m_new) * scale_log2) (FADD, FMUL, MUFU.EX2), P = ex2(fma(S, scale_log2,
//     -m_new * scale_log2)) (FFMA + MUFU.EX2), per-lane FP32 partial row sums in the lane's element
//     order (lane q of the quad owns columns 8*nj + 2q + j, j fastest) where, as COMPILED, the rescale of
//     the running partial sum is contracted into the first add of the block: l = FFMA(scores_scale, l,
//     p[first col]) (nvcc fmad contraction across `row_sum *= scores_scale; reduce_sum(...)`, verified in
//     the production SASS), P rounded to BF16 (RNE) before the PV MMA;
//   * the epilogue (normalize_softmax_lse): quad allreduce (l0 + l2) + (l1 + l3), inv = 1/sum via
//     MUFU.RCP (1.0 when sum == 0 or NaN), O *= inv (FMUL), RNE to BF16;
//   * every FP32 op with FTZ (--use_fast_math: .ftz on FADD/FMUL/FFMA; MUFU.EX2/RCP are the
//     .approx.ftz variants, reproduced from the measured 2^23-entry tables + reduction rules in
//     mufu_probe.cu).
//
// Build: g++ -O2 -std=c++17 -fopenmp -fPIC -shared -ffp-contract=off -o libfa2_model.so fa2_model.cpp

#include "tc_model.cpp"  // Term, bf16_product, group_sum, tc_step, f32_to_bf16_rne, ... (anonymous namespace)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

inline float ftz(float x) {
    uint32_t u = u32(x);
    if ((u & 0x7F800000u) == 0) return f32(u & 0x80000000u);  // subnormal (or zero) -> signed zero
    return x;
}
inline float fmul_ftz(float a, float b) { return ftz(ftz(a) * ftz(b)); }
inline float fadd_ftz(float a, float b) { return ftz(ftz(a) + ftz(b)); }
inline float fsub_ftz(float a, float b) { return ftz(ftz(a) - ftz(b)); }
inline float ffma_ftz(float a, float b, float c) { return ftz(std::fmaf(ftz(a), ftz(b), ftz(c))); }

// ex2.approx.ftz.f32 (MUFU.EX2): T[j] = hw(j / 2^23) for j in [0, 2^23); reduction rule measured
// exhaustively over the 2^32 inputs on sm_89 (mufu_probe.cu FRAC_MODE 4):  K = floor(|x| 2^23),
// q = K >> 23, r = K & (2^23-1); x >= 0: n = q, j = r; x < 0: r == 0 ? (n = -q, j = 0)
// : (n = -(q+1), j = 2^23-1-r) [ones' complement of the truncated magnitude, carry only when the
// fraction bits are zero].  Result = T[j] * 2^n with overflow -> +inf and subnormal -> +0.
inline uint32_t mufu_ex2_bits(uint32_t xb, const uint32_t* T) {
    uint32_t ax = xb & 0x7FFFFFFFu;
    int e = (ax >> 23) & 0xFF;
    if (e == 0xFF) {
        if (ax & 0x7FFFFF) return 0x7FFFFFFFu;         // NaN -> canonical NaN
        return (xb >> 31) ? 0u : 0x7F800000u;            // ex2(-inf) = +0, ex2(+inf) = +inf
    }
    if (e == 0) return 0x3F800000u;                      // ftz: +-0 / subnormal -> 1.0
    int unb = e - 127;
    if (unb >= 30) return (xb >> 31) ? 0u : 0x7F800000u;
    uint64_t mant = (uint64_t)((ax & 0x7FFFFF) | 0x800000);
    // mant < 2^24, so every right shift of 24 or more is 0 (|x| < 2^-23 reads T[0] = 1.0 at n = 0); a shift count >= 64
    // is undefined in C++ (x86-64 and arm64 take it mod 64), while the device the rule was verified on clamps it
    uint64_t K = (unb >= 0) ? (mant << unb) : (-unb >= 24 ? 0 : (mant >> (-unb)));
    long long fixed;
    if (!(xb >> 31)) fixed = (long long)K;
    else if ((K & 0x7FFFFF) == 0) fixed = -(long long)K;
    else fixed = -(long long)K - 1;
    long long n = fixed >> 23;
    uint32_t j = (uint32_t)(fixed & 0x7FFFFF);
    uint32_t t = T[j];
    long long re = (long long)((t >> 23) & 0xFF) + n;
    if (re >= 255) return 0x7F800000u;
    if (re <= 0) return 0u;
    return ((uint32_t)re << 23) | (t & 0x7FFFFF);
}
inline float mufu_ex2(float x, const uint32_t* T) { return f32(mufu_ex2_bits(u32(x), T)); }

// rcp.approx.ftz.f32 (MUFU.RCP): R[m] = hw(1.m) for the 2^23 mantissas; exponent handled exactly
// (measured exhaustively, 0 mismatches over 2^32 inputs on sm_89).
inline uint32_t mufu_rcp_bits(uint32_t xb, const uint32_t* R) {
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
inline float mufu_rcp(float x, const uint32_t* R) { return f32(mufu_rcp_bits(u32(x), R)); }

const int AMPERE_GROUPS[2] = {8, 8};
const Params AMPERE_BF16{AMPERE_GROUPS, 2, 25, -132, 16};
// Hopper wgmma.mma_async m64nNk16 bf16->f32 (veritor.core.silicon_total HOPPER_BF16_WGMMA_K16, pinned on an H100 by
// gpu/tc-hopper): ONE group of 16 products + accumulator, 26-bit adder, exponent floor -133.  Used by fa3_rows.
const int HOPPER_GROUPS[1] = {16};
const Params HOPPER_BF16_WGMMA{HOPPER_GROUPS, 1, 26, -133, 16};

struct Fa2Cfg {
    int D;            // head dim (128)
    int kBlockN;      // 64 (FA2) / 128 (FA3, hdim 128 causal)
    float scale_log2; // params.scale_softmax_log2 = float(float(softmax_scale) * M_LOG2E)
    const uint32_t* ex2_tab;
    const uint32_t* rcp_tab;
    int variant;      // dataflow-hypothesis switches (0 = the kernel as read from the source; see fa2_rows)
    const Params* tc; // the tensor-core step: AMPERE_BF16 (mma.sync, FA2) or HOPPER_BF16_WGMMA (wgmma, FA3)
    float inv_degenerate; // inv_sum when sum == 0 or NaN: FA2 `? 1.f` (softmax.h normalize_softmax_lse), FA3 `? 0.f` (hopper/softmax.h finalize)
};

// One output row.  q: D words.  K/V: key rows for this row's KV head, row `col` at
// K + col * kv_stride (words).  col_limit: number of visible keys (causal: pos + 1); seqlen_k: the
// kernel's actual_seqlen_k (block range); n_block_max as the kernel computes it for this row's m_block.
// dbg (optional, per row): dbg_s[n_block_max*kBlockN] S words, dbg_p[...] P (fp32 words before the
// BF16 cast), dbg_m[n_block_max] running max after each block (in processing order), dbg_l[4] final
// lane sums, dbg_o_pre[D] O before normalisation.
void fa2_row(const Fa2Cfg& C, const uint16_t* q, const uint16_t* K, const uint16_t* V, long kv_stride,
             int col_limit, int n_block_max, uint16_t* out, uint32_t* out_f32,
             uint32_t* dbg_s, uint32_t* dbg_p, uint32_t* dbg_m, uint32_t* dbg_l, uint32_t* dbg_o_pre, uint8_t* flag) {
    const int D = C.D, BN = C.kBlockN, ksteps = D / 16;
    // flag bit 0: a non-finite intermediate (S overflow / inf / NaN in S, P, the rescale factor, the row sums or
    // O) occurred.  The shared MMA model saturates instead of following IEEE inf/NaN rules and the kernel's
    // inf/NaN paths were not measured, so such a row is reported `unresolved`, never `exact`.
    uint8_t nf = 0;
    auto chk = [&nf](float x) { if (std::isinf(x) || std::isnan(x)) nf |= 1; };
    float m = -INFINITY;
    float l[4] = {0.f, 0.f, 0.f, 0.f};
    std::vector<uint32_t> o_(D, 0u);
    std::vector<float> s_(BN), p_(BN);
    std::vector<uint16_t> pb_(BN);
    uint32_t* o = o_.data();
    float* s = s_.data();
    float* p = p_.data();
    uint16_t* pb = pb_.data();
    uint16_t vcol[16];
    float l_pending_sc[4] = {1.f, 1.f, 1.f, 1.f};
    bool first = true;
    for (int step = 0; step < n_block_max; ++step) {
        int nb = (C.variant & 64) ? step : (n_block_max - 1 - step);
        // S = Q K^T for this block (8 mma k-steps of 16 over the head dim, accumulator starts at 0)
        float bmax = -INFINITY;
        for (int c = 0; c < BN; ++c) {
            int col = nb * BN + c;
            if (col < col_limit) {
                Term acc{false, 0, 0};
                const uint16_t* krow = K + (long)col * kv_stride;
                for (int ii = 0; ii < ksteps; ++ii) {
                    int i = (C.variant & 4) ? (ksteps - 1 - ii) : ii;
                    if (tc_step<true>(acc, q + 16 * i, krow + 16 * i, *C.tc)) break;
                }
                s[c] = f32(pack_fp32(acc));
                chk(s[c]);
            } else {
                s[c] = -INFINITY;
            }
            bmax = std::max(bmax, s[c]);  // MaxOp<float> = max(x, y); no NaNs expected
            if (dbg_s) dbg_s[nb * BN + c] = u32(s[c]);
        }
        float m_new = first ? bmax : std::max(m, bmax);
        if (!first) {
            float cur = (m_new == -INFINITY) ? 0.f : m_new;
            float sc = mufu_ex2(fmul_ftz(fsub_ftz(m, cur), C.scale_log2), C.ex2_tab);
            if (m != -INFINITY) chk(sc);  // (-inf - cur) -> ex2(-inf) = 0 is the kernel's own first-block path
            // The compiled kernel contracts `row_sum *= scores_scale; row_sum += p[first col]` into ONE FFMA per
            // lane (nvcc fmad contraction across the inlined statements; confirmed in the production SASS:
            // `FFMA.FTZ R118, R100, R171, R118` = scores_scale * l_old + p_first).  So the rescaled partial sum
            // is never rounded on its own.  Variant bit 128 restores the source-literal FMUL-then-FADD form.
            if (C.variant & 128) { for (int i = 0; i < 4; ++i) l[i] = fmul_ftz(l[i], sc); }
            else { for (int i = 0; i < 4; ++i) l_pending_sc[i] = sc; }
            for (int d = 0; d < D; ++d) o[d] = u32(fmul_ftz(f32(o[d]), sc));
        }
        float max_scaled = (m_new == -INFINITY) ? 0.f : fmul_ftz(m_new, C.scale_log2);
        for (int c = 0; c < BN; ++c) {
            if (C.variant & 32) p[c] = mufu_ex2(fsub_ftz(fmul_ftz(s[c], C.scale_log2), max_scaled), C.ex2_tab);
            else p[c] = mufu_ex2(ffma_ftz(s[c], C.scale_log2, -max_scaled), C.ex2_tab);
            if (dbg_p) dbg_p[nb * BN + c] = u32(p[c]);
            chk(p[c]);
        }
        for (int c = 0; c < BN; ++c) pb[c] = f32_to_bf16_rne(u32(p[c]), 0x7FFF);
        // per-lane partial sums: lane qd owns columns 8*nj + 2*qd + j (j fastest), sequential FADD
        if (C.variant & 8) {
            for (int c = 0; c < BN; ++c) l[0] = fadd_ftz(l[0], (C.variant & 16) ? bf16_f32(pb[c]) : p[c]);
        } else if (C.variant & 1) {
            for (int qd = 0; qd < 4; ++qd)
                for (int j = 0; j < 2; ++j)
                    for (int nj = 0; nj < BN / 8; ++nj) l[qd] = fadd_ftz(l[qd], (C.variant & 16) ? bf16_f32(pb[8 * nj + 2 * qd + j]) : p[8 * nj + 2 * qd + j]);
        } else {
            for (int qd = 0; qd < 4; ++qd)
                for (int nj = 0; nj < BN / 8; ++nj)
                    for (int j = 0; j < 2; ++j) {
                        float pv = (C.variant & 16) ? bf16_f32(pb[8 * nj + 2 * qd + j]) : p[8 * nj + 2 * qd + j];
                        if (!(C.variant & 128) && nj == 0 && j == 0 && !first) l[qd] = ffma_ftz(l[qd], l_pending_sc[qd], pv);
                        else l[qd] = fadd_ftz(l[qd], pv);
                    }
        }
        // O += P V  (4 mma k-steps of 16 keys; V rows beyond seqlen_k are zero-filled by the kernel,
        // rows in [col_limit, seqlen_k) are real data multiplied by P = 0)
        for (int d = 0; d < D; ++d) {
            Term acc = fp32_term(o[d]);
            for (int ii = 0; ii < BN / 16; ++ii) {
                int i = (C.variant & 2) ? (BN / 16 - 1 - ii) : ii;
                for (int t = 0; t < 16; ++t) {
                    int col = nb * BN + 16 * i + t;
                    vcol[t] = (col < col_limit) ? V[(long)col * kv_stride + d] : (uint16_t)0;
                }
                if (tc_step<true>(acc, pb + 16 * i, vcol, *C.tc)) break;
            }
            o[d] = pack_fp32(acc);
            chk(f32(o[d]));
        }
        m = m_new;
        if (dbg_m) dbg_m[nb] = u32(m);
        first = false;
    }
    float sum = fadd_ftz(fadd_ftz(l[0], l[2]), fadd_ftz(l[1], l[3]));
    chk(sum);
    if (flag) *flag = nf;
    float inv = (sum == 0.f || sum != sum) ? C.inv_degenerate : mufu_rcp(sum, C.rcp_tab);
    if (dbg_l) for (int i = 0; i < 4; ++i) dbg_l[i] = u32(l[i]);
    for (int d = 0; d < D; ++d) {
        if (dbg_o_pre) dbg_o_pre[d] = o[d];
        float v = fmul_ftz(f32(o[d]), inv);
        if (out_f32) out_f32[d] = u32(v);
        out[d] = f32_to_bf16_rne(u32(v), 0x7FFF);
    }
}

}  // namespace

extern "C" {

// Batch of output rows.  Q: [nrows, D] words (one head's query per row).  K, V: [Lk, KVH, D] words
// (the fixture's canonical KV rows); row i reads KV head kvh[i].  col_limit[i] visible keys,
// n_block_max[i] the kernel's block count for that row's m_block.  Outputs [nrows, D].
// dbg_* may be NULL; when given they are [nrows, max_blocks*kBlockN] / [nrows, max_blocks] / [nrows, 4] / [nrows, D].
void fa2_rows(const uint16_t* Q, const uint16_t* K, const uint16_t* V, int KVH, int D, const int* kvh,
              const int* col_limit, const int* n_block_max, int nrows, int kBlockN, float scale_log2,
              const uint32_t* ex2_tab, const uint32_t* rcp_tab, uint16_t* out, uint32_t* out_f32,
              uint32_t* dbg_s, uint32_t* dbg_p, uint32_t* dbg_m, uint32_t* dbg_l, uint32_t* dbg_o_pre, int max_blocks,
              int variant, uint8_t* flags) {
    // variant bits (hypothesis tests; 0 = kernel as read from the source): 1 lane-sum order nj-fastest;
    // 2 PV k-steps reversed; 4 QK k-steps reversed; 8 single sequential row sum (no lanes); 16 row sum over
    // BF16-rounded P; 32 FMUL+FSUB instead of FFMA for the exp2 argument; 64 blocks ascending; 128 source-literal
    // row-sum rescale (FMUL then FADD) instead of the compiled FFMA contraction.  Every bit was measured to be
    // WRONG against the production kernel on the natural fixture (see results/<id>/dataflow_variants.json).
    Fa2Cfg C{D, kBlockN, scale_log2, ex2_tab, rcp_tab, variant, &AMPERE_BF16, 1.f};
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < nrows; ++i) {
        const uint16_t* Kh = K + (long)kvh[i] * D;
        const uint16_t* Vh = V + (long)kvh[i] * D;
        fa2_row(C, Q + (long)i * D, Kh, Vh, (long)KVH * D, col_limit[i], n_block_max[i], out + (long)i * D,
                out_f32 ? out_f32 + (long)i * D : nullptr,
                dbg_s ? dbg_s + (long)i * max_blocks * kBlockN : nullptr,
                dbg_p ? dbg_p + (long)i * max_blocks * kBlockN : nullptr,
                dbg_m ? dbg_m + (long)i * max_blocks : nullptr,
                dbg_l ? dbg_l + (long)i * 4 : nullptr,
                dbg_o_pre ? dbg_o_pre + (long)i * D : nullptr, flags ? flags + i : nullptr);
    }
}

// FlashAttention-3 (hopper/, sm_90: `flash::FlashAttnFwdSm90<CollectiveMainloopFwdSm90<2, ClusterShape 1x1x1,
// TileShape (128|64)x128x128, hdim 128, bf16, Is_causal, ..., MmaPV_is_RS, IntraWGOverlap, PackGQA, Split=false>>`
// as vLLM launches it on an H100, VLLM_BATCH_INVARIANT num_splits=1): the SAME per-row recurrence as fa2_row with
//   * kBlockN = 128 keys per online-softmax block (tile_size_fwd_sm90, hdim 128 causal), blocks last -> first;
//   * every tensor-core step a wgmma m64nNk16 (HOPPER_BF16_WGMMA: one group of 16 + acc, 26 bits, floor -133) --
//     S = Q K^T as 8 ascending k16 steps from a zero accumulator (scale-d = 0 == acc + 0, measured), O += P V as 8
//     ascending k16 steps per block, the same per-thread accumulator ownership (cols 8*nj + 2*(lane%4) + j);
//   * rescale_o then the next P V gemm (IntraWGOverlap, RescaleOBeforeGemm = false for hdim 128: the rescale of
//     block n closes fwd_step(n) and the gemm P(n) V(n) opens fwd_step(n-1)) -- rescale-then-add, as FA2;
//   * finalize: inv_sum = (sum == 0 || sum != sum) ? 0.f : 1.f / sum (FA2: ? 1.f), scores_scale = inv_sum * 1.f.
// kBlockN is checked against the sm90 bf16 forward tiles vLLM reaches rather than left free, so the entry point denotes one
// kernel family: 128 (hdim 128 causal / paged non-TMA; hdim 64 in the 192 x 128 tile) and 192 (hdim 64 under one MMA
// warpgroup, seqlen_q * qheads_per_kvhead <= 64: the 64 x 192 tile) [R13 hopper].  D: 64 or 128 (ksteps = D / 16).
// hdim 256 (GLM-5.3's dense MHA prefill, qk 192 + 64 = v 256) runs the 128 x 80 tile: kBlockN 80 (5 P.V k16 steps a block,
// 16 QK k16 steps) and RescaleOBeforeGemm = true, which moves block n's rescale of O from the end of fwd_step(n) to just before
// P(n+1) V(n+1)'s gemm in fwd_step(n-1) (and the tail): the same O = O * scale + P V per element, so the same recurrence.
void fa3_rows(const uint16_t* Q, const uint16_t* K, const uint16_t* V, int KVH, int D, const int* kvh,
              const int* col_limit, const int* n_block_max, int nrows, int kBlockN, float scale_log2,
              const uint32_t* ex2_tab, const uint32_t* rcp_tab, uint16_t* out, uint32_t* out_f32,
              uint32_t* dbg_s, uint32_t* dbg_p, uint32_t* dbg_m, uint32_t* dbg_l, uint32_t* dbg_o_pre, int max_blocks,
              int variant, uint8_t* flags) {
    bool const hd128 = (kBlockN == 128 || kBlockN == 192) && (D == 64 || D == 128);
    bool const hd256 = kBlockN == 80 && D == 256;
    if (!hd128 && !hd256) {  // flag bit 1: refused
        if (flags) for (int i = 0; i < nrows; ++i) flags[i] = 2;
        return;
    }
    Fa2Cfg C{D, kBlockN, scale_log2, ex2_tab, rcp_tab, variant, &HOPPER_BF16_WGMMA, 0.f};
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < nrows; ++i) {
        const uint16_t* Kh = K + (long)kvh[i] * D;
        const uint16_t* Vh = V + (long)kvh[i] * D;
        fa2_row(C, Q + (long)i * D, Kh, Vh, (long)KVH * D, col_limit[i], n_block_max[i], out + (long)i * D,
                out_f32 ? out_f32 + (long)i * D : nullptr,
                dbg_s ? dbg_s + (long)i * max_blocks * kBlockN : nullptr,
                dbg_p ? dbg_p + (long)i * max_blocks * kBlockN : nullptr,
                dbg_m ? dbg_m + (long)i * max_blocks : nullptr,
                dbg_l ? dbg_l + (long)i * 4 : nullptr,
                dbg_o_pre ? dbg_o_pre + (long)i * D : nullptr, flags ? flags + i : nullptr);
    }
}

// Elementwise helpers exposed for tests / stage checks.
void mufu_ex2_words(const uint32_t* x, uint32_t* y, long n, const uint32_t* T) {
    for (long i = 0; i < n; ++i) y[i] = mufu_ex2_bits(x[i], T);
}
void mufu_rcp_words(const uint32_t* x, uint32_t* y, long n, const uint32_t* R) {
    for (long i = 0; i < n; ++i) y[i] = mufu_rcp_bits(x[i], R);
}

}  // extern "C"
