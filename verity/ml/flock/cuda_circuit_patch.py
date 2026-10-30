#!/usr/bin/env python3
"""Install the circuit prover (verity/ml/flock/cuda/prove_circuit.cuh) into a Flock b684b12 checkout (lane flock-circuit).

Run from the flock checkout after verity/ml/flock/cuda_chunk_patch.py, which installs prove_chunk.cuh and the live hook.
flock_cuda_prove_circuit is flock_cuda_prove_chunk's body with three parts replaced (each anchor must occur once): the shape
check (fc_bad_shape), the witness (fc_witness: device compressions and units, or the host's witness) and the lincheck fold
(fc_fold: every slot range of every type). Everything else (commit, zerocheck, lincheck rounds, ring switch, Ligerito, the
live coins) is the chunk prover's, so no second copy of it is kept in the repository. Its buffer lifetimes change so that
rep 1 can prove from rep 0's witness and level-0 commitment (rep_reuse, FcKept), off under M1.

M1 (lane flock-zk) adds Ligerito level 0's hiding behind FlockCircuitParams::zk_on, as a handful of anchored insertions (each
anchor must occur once): level 0's commitment (fc_commit_l0_zk; M0's commit is the other branch), lincheck's folded comb, the
lane phase's hiding step in the F256 ladder (verity/ml/flock/cuda/ligerito_zk.cuh, called from run_ligerito_f256) and the zk
section of the proof stream (fc_write_zk).
"""
import pathlib
import shutil

here = pathlib.Path(__file__).resolve().parent
src = (here / "cuda" / "prove_chunk.cuh").read_text()
start = src.index("int flock_cuda_prove_chunk(")
end = src.rindex("}  // extern \"C\"")
body = src[start:end].rstrip()
assert body.endswith("}")


def put(s: str, a: str, new: str) -> str:
    assert s.count(a) == 1, a
    return s.replace(a, new, 1)


def cut(s: str, a: str, b: str, new: str, keep_b: bool = True) -> str:
    i = s.index(a)
    assert s.count(a) == 1, a
    j = s.index(b, i)
    return s[:i] + new + (s[j:] if keep_b else s[j + len(b):])


body = body.replace(
    "int flock_cuda_prove_chunk(const FlockCudaProveParams* P, const FlockChunkParams* C, uint8_t** out, size_t* out_len) {",
    "int flock_cuda_prove_circuit(const FlockCudaProveParams* P, const FlockCircuitParams* N, uint8_t** out, size_t* out_len) {\n"
    "    const FlockChunkParams* C = &N->c;", 1)
assert "flock_cuda_prove_circuit" in body
body = cut(body, "    const long long n_comp = C->mode == 1", "    const int n_blocks_log", "    const bool pure = true;\n")
body = cut(body, "    if (pure ? ((C->host_z", "{ printf(\"FFI chunk: bad shape\\n\"); return 201; }",
           "    if (fc_bad_shape(P, N)) { printf(\"FFI circuit: bad shape\\n\"); return 201; }", keep_b=False)
body = cut(body, "    if (pure && C->host_z) {", "    CKI(phase(0));",
           "    std::vector<F128> fc_zk_comb(64);\n    CKI(fc_witness(P, N, m, k_log, len, df, d_a, d_b, d_zlin));\n")
body = cut(body, "        CKI(chunk_quirky_eq_device(d_eq_inner, zc_z, xab_inner, k_skip));", "        if (C->n_delta_a) chunk_delta_scatter",
           "        CKI(chunk_quirky_eq_device(d_eq_inner, zc_z, xab_inner, k_skip));\n        CK(cudaGetLastError());\n"
           "        CKI(fc_fold(P, N, d_eq_inner, alpha, k_log, d_comb));\n")
# M1: level 0's commitment with its hiding, lincheck's folded comb, the ladder's zk argument, the zk section
body = put(body, "    std::vector<MHash> l0_cap;\n    {\n        int k_code = (log_n - P->initial_k) + P->log_inv_rates[0];",
           "    std::vector<MHash> l0_cap;\n    F128 *d_zk_pads = nullptr, *d_zk_extra = nullptr;\n    if (N->zk_on) {\n"
           "        CKI(fc_commit_l0_zk(P, N, log_n, df, &d_zk_pads, &d_zk_extra, &d_cw0, &d_tree0, &l0_bl, &l0_ni, l0_cap));\n"
           "    } else {\n        int k_code = (log_n - P->initial_k) + P->log_inv_rates[0];")
body = put(body, "        CK(cudaMemcpy(lc_zpart.data(), cZ, 64 * sizeof(F128), cudaMemcpyDeviceToHost));\n",
           "        CK(cudaMemcpy(lc_zpart.data(), cZ, 64 * sizeof(F128), cudaMemcpyDeviceToHost));\n"
           "        if (N->zk_on) CK(cudaMemcpy(fc_zk_comb.data(), cC, 64 * sizeof(F128), cudaMemcpyDeviceToHost));\n")
body = put(body, "    LigF256Proof lig;\n",
           "    LigF256Proof lig;\n    LigZk fc_zk;\n"
           "    if (N->zk_on) { fc_zk.rep = N->zk_rep; fc_zk.t_pad = N->zk_t_pad; fc_zk.tamper = N->zk_tamper; fc_zk.lanes = 1 << P->initial_k;\n"
           "                    fc_zk.d_pads = d_zk_pads; fc_zk.d_extra = d_zk_extra; }\n")
body = put(body, "d_cw0, d_tree0, l0_bl, l0_ni, arena, lig);", "d_cw0, d_tree0, l0_bl, l0_ni, arena, lig, N->zk_on ? &fc_zk : nullptr);")
body = put(body, "    ffi_free(d_cw0); ffi_free(d_tree0);\n",
           "    ffi_free(d_cw0); ffi_free(d_tree0);\n    if (d_zk_pads) ffi_free(d_zk_pads);\n    if (d_zk_extra) ffi_free(d_zk_extra);\n")
body = put(body, "    CKI(phase(6));\n", "    if (N->zk_on) fc_write_zk(W, fc_zk, fc_zk_comb);\n    CKI(phase(6));\n")
# rep 1 from rep 0's witness and level-0 commitment (FlockCircuitParams.rep_reuse), not under M1 (its level 0 and lane phase
# are its own): keeping holds z, a and b to the end of rep 0 and gives its Ligerito a copy of z (the folds overwrite it), three
# more witness-sized arena buffers; rep 1 transposes z for the lincheck again
W3 = "fc_keep_extra(packed_witness_bytes)"
body = put(body, "    CK(device_arena.begin(arena_bytes));\n",
           "    const bool take = N->rep_reuse == 2 && !N->zk_on && g_fc_kept.live && g_fc_kept.tag == N->reuse_tag;\n"
           "    if (take && (g_fc_kept.m != m || g_fc_kept.k_log != k_log)) { printf(\"FFI circuit: the kept witness is another shape's\\n\"); return 207; }\n"
           "    if (!take) fc_kept_drop();\n"
           f"    const bool keep = !take && N->rep_reuse == 1 && !N->zk_on && fc_can_keep(arena_bytes, {W3}, FC_KEEP_MARGIN);\n"
           "    if (!take) {\n"
           f"        const size_t want = arena_bytes + (keep ? {W3} : 0);\n"
           "        CK(device_arena.begin(want));\n"
           "        g_fc_arena = std::max(g_fc_arena, want);\n"
           "    }\n")
alloc = ("    CK(ffi_malloc(&df, len * sizeof(F128)));\n    CK(ffi_malloc(&d_a, len * sizeof(F128)));\n"
         "    CK(ffi_malloc(&d_b, len * sizeof(F128)));\n    CK(ffi_malloc(&d_zlin, (size_t)len * 16));\n")
wit = "    std::vector<F128> fc_zk_comb(64);\n    CKI(fc_witness(P, N, m, k_log, len, df, d_a, d_b, d_zlin));\n"
body = put(body, alloc + wit,
           "    std::vector<F128> fc_zk_comb(64);\n"
           "    if (take) {\n        df = g_fc_kept.df; d_a = g_fc_kept.d_a; d_b = g_fc_kept.d_b;\n"
           "        CK(ffi_malloc(&d_zlin, (size_t)len * 16));\n"
           "        fc_zlin_transpose(df, k_log, (1LL << (m - k_log)) / 8, d_zlin);\n        CK(cudaGetLastError());\n"
           "    } else {\n" + "".join("    " + x for x in (alloc + wit.split("\n", 1)[1]).splitlines(True)) + "    }\n")
body = put(body, "    if (N->zk_on) {\n        CKI(fc_commit_l0_zk(",
           "    if (take) {\n"
           "        d_cw0 = g_fc_kept.d_cw0; d_tree0 = g_fc_kept.d_tree0; l0_bl = g_fc_kept.l0_bl; l0_ni = g_fc_kept.l0_ni;\n"
           "        l0_cap = g_fc_kept.l0_cap;\n        fc_rejoin_level0(d_tree0);\n"
           "    } else if (N->zk_on) {\n        CKI(fc_commit_l0_zk(")
body = put(body, "    ffi_free(d_a); ffi_free(d_b);\n", "    if (!keep) { ffi_free(d_a); ffi_free(d_b); }\n")
body = put(body, "    LigF256Proof lig;\n",
           "    F128* d_lz = df;\n    if (keep) {\n        CK(ffi_malloc(&d_lz, len * sizeof(F128)));\n"
           "        CK(cudaMemcpy(d_lz, df, len * sizeof(F128), cudaMemcpyDeviceToDevice));\n    }\n    LigF256Proof lig;\n")
body = put(body, "first_u0, first_u2, df, dcb,\n", "first_u0, first_u2, d_lz, dcb,\n")
body = put(body, "    ffi_free(d_cw0); ffi_free(d_tree0);\n    if (d_zk_pads) ffi_free(d_zk_pads);\n    if (d_zk_extra) ffi_free(d_zk_extra);\n"
                 "    ffi_free(df); ffi_free(dcb);\n",
           "    if (keep) ffi_free(d_lz);\n    else { ffi_free(d_cw0); ffi_free(d_tree0); ffi_free(df); }\n"
           "    if (d_zk_pads) ffi_free(d_zk_pads);\n    if (d_zk_extra) ffi_free(d_zk_extra);\n    ffi_free(dcb);\n")
body = put(body, "    CK(device_arena.finish());\n    return 0;\n}",
           "    if (keep) {\n        // the arena stays open for rep 1\n"
           "        g_fc_kept = FcKept{true, N->reuse_tag, m, k_log, df, d_a, d_b, d_cw0, d_tree0, l0_bl, l0_ni, l0_cap};\n"
           "        return 0;\n    }\n    if (take) g_fc_kept = FcKept{};\n    CK(device_arena.finish());\n    return 0;\n}")
for gone in ("C->comp_slots", "C->units", "C->unit_log", "C->unit_pos0", "C->mode", "C->u_"):
    assert gone not in body, f"{gone} left in the circuit prover"

g = pathlib.Path("cuda-ghash")
text = (here / "cuda" / "prove_circuit.cuh").read_text() + "\nextern \"C\" {\n\n" + body + "\n\n}  // extern \"C\"\n"
(g / "prove_circuit.cuh").write_text(text)
# M1: the F256 ladder's hiding step (ligerito_zk.cuh) and its five call points in run_ligerito_f256
(g / "ligerito_zk.cuh").write_text((here / "cuda" / "ligerito_zk.cuh").read_text())
lg = g / "ligerito_f256.cuh"
t = lg.read_text()
if '#include "ligerito_zk.cuh"' not in t:
    t = put(t, "// Run the complete F256 ladder.", '#include "ligerito_zk.cuh"\n\n// Run the complete F256 ladder.')
    t = put(t, "                             const LigAlloc& A, LigF256Proof& out) {",
            "                             const LigAlloc& A, LigF256Proof& out, LigZk* zk = nullptr) {")
    i = t.index("out.initial_cap.data(),")
    assert t.count("out.initial_cap.data(),") == 1
    j = t.index("\n", i) + 1
    t = t[:j] + "    if (zk) lf_zk_open(ch, *zk, target);\n" + t[j:]
    t = put(t, "            target = f128_add_hd(target, f128_mul_hd(beta, y));\n",
            "            if (zk) zk->ood.push_back({beta, y});\n            target = f128_add_hd(target, f128_mul_hd(beta, y));\n")
    t = put(t, "    for (int j = 0; j < initial_k; j++) {\n        F256Ext r256 = lf_frch256(ch.sample_f256());\n",
            "    for (int j = 0; j < initial_k; j++) {\n        F256Ext r256 = lf_frch256(ch.sample_f256());\n"
            "        if (zk) zk->challenges.push_back(r256);\n")
    t = put(t, "            // Code switch: the F256 f-array IS the split F128 table; the\n",
            "            if (zk) {\n                int zrc = lf_zk_lane_phase(ch, *zk, (F256Ext*)regions[cur_f], (const F256Ext*)regions[cur_b], n256,\n"
            "                                           d_p256_0, d_p256_2, d_out256, A);\n                if (zrc) return zrc;\n            }\n"
            "            // Code switch: the F256 f-array IS the split F128 table; the\n")
    t = put(t, "        rc = query_phase(0, out.initial_open, alpha, queries);\n        if (rc) return rc;\n",
            "        rc = query_phase(0, out.initial_open, alpha, queries);\n        if (rc) return rc;\n"
            "        if (zk) zk->queries0.assign(queries.begin(), queries.end());\n")
    lg.write_text(t)
ffi = g / "prove_ffi.cu"
s = ffi.read_text()
assert '#include "prove_chunk.cuh"' in s, "run verity/ml/flock/cuda_chunk_patch.py first"
if '#include "prove_circuit.cuh"' not in s:
    s = s.replace('#include "prove_chunk.cuh"', '#include "prove_chunk.cuh"\n#include "prove_circuit.cuh"', 1)
    ffi.write_text(s)
print("prove_circuit.cuh installed:", len(text.splitlines()), "lines")

# Ligerito level 0's OOD samples: the eq table of each sample as its halves, eq[x] = lo[x mod 2^k] hi[x >> k] (the same elements as
# build_eq_device's 2^log_n table), which the basis message and the glue form as they read it: no 2^log_n table is built or read
# back (FC_LIG_EQ_FULL=1: the full table)
ig = g / "introduce_glue.cuh"
t = ig.read_text()
if "igl_eq_split" not in t:
    t += '''
// basis_message_evaluation_partial and combine_basis_polynomials with B an eq table's halves (build_eq_device over the first k
// challenges and the rest): B[x] = lo[x mod 2^k] hi[x >> k], the same elements as the full table's
__device__ __forceinline__ F128 igl_eq_split(const F128* __restrict__ lo, const F128* __restrict__ hi, int k, long long x) {
    return ghash_mul_karatsuba(lo[x & ((1LL << k) - 1)], hi[x >> k]);
}

__global__ void basis_message_evaluation_partial_split(const F128* __restrict__ F, const F128* __restrict__ lo,
                                                       const F128* __restrict__ hi, int k, long long half, F128* p0, F128* p2,
                                                       F128* podd) {
    __shared__ F128 s0[IGL_TPB], s2[IGL_TPB], sodd[IGL_TPB];
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    F128 e0{0, 0}, e2{0, 0}, eodd{0, 0};
    for (long long j = t; j < half; j += stride) {
        F128 f0 = F[2 * j], f1 = F[2 * j + 1];
        F128 b0 = igl_eq_split(lo, hi, k, 2 * j), b1 = igl_eq_split(lo, hi, k, 2 * j + 1);
        e0 = f128_add(e0, ghash_mul_karatsuba(f0, b0));
        e2 = f128_add(e2, ghash_mul_karatsuba(f128_add(f0, f1), f128_add(b0, b1)));
        eodd = f128_add(eodd, ghash_mul_karatsuba(f1, b1));
    }
    int x = threadIdx.x;
    s0[x] = e0; s2[x] = e2; sodd[x] = eodd;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s0[x] = f128_add(s0[x], s0[x + s]); s2[x] = f128_add(s2[x], s2[x + s]); sodd[x] = f128_add(sodd[x], sodd[x + s]); }
        __syncthreads();
    }
    if (x == 0) { p0[blockIdx.x] = s0[0]; p2[blockIdx.x] = s2[0]; podd[blockIdx.x] = sodd[0]; }
}

__global__ void combine_basis_polynomials_split(F128* __restrict__ cb, const F128* __restrict__ lo, const F128* __restrict__ hi, int k,
                                                F128 beta, long long n) {
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    cb[j] = f128_add(cb[j], ghash_mul_karatsuba(beta, igl_eq_split(lo, hi, k, j)));
}

inline void launch_basis_message_evaluation_split(const F128* dF, const F128* lo, const F128* hi, int k, long long half, F128* d_p0,
                                                  F128* d_p2, F128* d_podd, F128* d_u0, F128* d_u2, F128* d_hnew) {
    int blocks = igl_blocks(half);
    basis_message_evaluation_partial_split<<<blocks, IGL_TPB>>>(dF, lo, hi, k, half, d_p0, d_p2, d_podd);
    combine_basis_message_evaluation<<<1, IGL_TPB>>>(d_p0, d_p2, d_podd, blocks, d_u0, d_u2, d_hnew);
}

inline void launch_glue_split(F128* d_cb, const F128* lo, const F128* hi, int k, F128 beta, long long n, int tpb = IGL_TPB) {
    long long blocks = (n + tpb - 1) / tpb;
    combine_basis_polynomials_split<<<(unsigned)blocks, tpb>>>(d_cb, lo, hi, k, beta, n);
}

static inline bool igl_eq_full() {
    static const bool v = [] { const char* e = std::getenv("FC_LIG_EQ_FULL"); return e && e[0] == '1'; }();
    return v;
}
'''
    ig.write_text(t)
lg = g / "ligerito_f256.cuh"
t = lg.read_text()
if "launch_glue_split" not in t:
    t = put(t, """        F128* d_eq;
        LFCK(A.alloc((void**)&d_eq, (size_t)len * sizeof(F128)));
        for (int o = 0; o < C.ood_samples[0]; o++) {""",
            """        F128 *d_eq = nullptr, *d_eq_lo = nullptr, *d_eq_hi = nullptr;
        const int eq_k = log_n / 2;
        if (igl_eq_full()) {
            LFCK(A.alloc((void**)&d_eq, (size_t)len * sizeof(F128)));
        } else {
            LFCK(A.alloc((void**)&d_eq_lo, ((size_t)1 << eq_k) * sizeof(F128)));
            LFCK(A.alloc((void**)&d_eq_hi, ((size_t)1 << (log_n - eq_k)) * sizeof(F128)));
        }
        for (int o = 0; o < C.ood_samples[0]; o++) {""")
    t = put(t, """            build_eq_device(d_eq, zf.data(), log_n);
            launch_basis_message_evaluation(d_f, d_eq, len / 2, d_p0, d_p2, d_podd, d_u0, d_u2, d_hnew);""",
            """            if (d_eq) {
                build_eq_device(d_eq, zf.data(), log_n);
                launch_basis_message_evaluation(d_f, d_eq, len / 2, d_p0, d_p2, d_podd, d_u0, d_u2, d_hnew);
            } else {
                build_eq_device(d_eq_lo, zf.data(), eq_k);
                build_eq_device(d_eq_hi, zf.data() + eq_k, log_n - eq_k);
                launch_basis_message_evaluation_split(d_f, d_eq_lo, d_eq_hi, eq_k, len / 2, d_p0, d_p2, d_podd, d_u0, d_u2, d_hnew);
            }""")
    t = put(t, """            launch_glue(d_b, d_eq, beta, len);
            LFCK(cudaGetLastError());
        }
        LFCK(A.release(d_eq));""",
            """            if (d_eq) launch_glue(d_b, d_eq, beta, len);
            else launch_glue_split(d_b, d_eq_lo, d_eq_hi, eq_k, beta, len);
            LFCK(cudaGetLastError());
        }
        for (F128* q : {d_eq, d_eq_hi, d_eq_lo})
            if (q) LFCK(A.release(q));""")
    lg.write_text(t)
