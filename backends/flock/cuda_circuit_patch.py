#!/usr/bin/env python3
"""Install the circuit prover (backends/flock/cuda/prove_circuit.cuh) into a Flock b684b12 checkout (lane flock-circuit).

Run from the flock checkout after backends/flock/cuda_chunk_patch.py, which installs prove_chunk.cuh and the live hook.
flock_cuda_prove_circuit is flock_cuda_prove_chunk's body with three parts replaced (each anchor must occur once): the shape
check (fc_bad_shape), the witness (fc_witness: device compressions and units, or the host's witness) and the lincheck fold
(fc_fold: every slot range of every type). Everything else (commit, zerocheck, lincheck rounds, ring switch, Ligerito, the
live coins) is the chunk prover's, so no second copy of it is kept in the repository. Its buffer lifetimes change so that
rep 1 can prove from rep 0's witness and level-0 commitment (rep_reuse, FcKept).
"""
import pathlib
import shutil

here = pathlib.Path(__file__).resolve().parent
src = (here / "cuda" / "prove_chunk.cuh").read_text()
start = src.index("int flock_cuda_prove_chunk(")
end = src.rindex("}  // extern \"C\"")
body = src[start:end].rstrip()
assert body.endswith("}")


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
           "    CKI(fc_witness(P, N, m, k_log, len, df, d_a, d_b, d_zlin));\n")
body = cut(body, "        build_quirky_eq_device(d_eq_inner, zc_z, xab_inner, k_skip);", "        if (C->n_delta_a) chunk_delta_scatter",
           "        build_quirky_eq_device(d_eq_inner, zc_z, xab_inner, k_skip);\n        CK(cudaGetLastError());\n"
           "        CKI(fc_fold(P, N, d_eq_inner, alpha, k_log, d_comb));\n")


def once(s: str, a: str, new: str) -> str:
    assert s.count(a) == 1, a
    return s.replace(a, new)


# rep 1 from rep 0's witness and level-0 commitment (FlockCircuitParams.rep_reuse): keeping holds z, a, b and the lincheck's z
# to the end of rep 0 and gives its Ligerito a copy of z (the folds overwrite it), four more witness-sized arena buffers
W4 = "4 * packed_witness_bytes"
body = once(body, "    CK(device_arena.begin(arena_bytes));\n",
            "    const bool take = N->rep_reuse == 2 && g_fc_kept.live && g_fc_kept.tag == N->reuse_tag;\n"
            "    if (take && (g_fc_kept.m != m || g_fc_kept.k_log != k_log)) { printf(\"FFI circuit: the kept witness is another shape's\\n\"); return 207; }\n"
            "    if (!take) fc_kept_drop();\n"
            f"    const bool keep = !take && N->rep_reuse == 1 && fc_can_keep(arena_bytes, {W4}, FC_KEEP_MARGIN);\n"
            "    if (!take) {\n"
            f"        const size_t want = arena_bytes + (keep ? {W4} : 0);\n"
            "        CK(device_arena.begin(want));\n"
            "        g_fc_arena = std::max(g_fc_arena, want);\n"
            "    }\n")
alloc = ("    CK(ffi_malloc(&df, len * sizeof(F128)));\n    CK(ffi_malloc(&d_a, len * sizeof(F128)));\n"
         "    CK(ffi_malloc(&d_b, len * sizeof(F128)));\n    CK(ffi_malloc(&d_zlin, (size_t)len * 16));\n"
         "    CKI(fc_witness(P, N, m, k_log, len, df, d_a, d_b, d_zlin));\n")
body = once(body, alloc, "    if (take) {\n        df = g_fc_kept.df; d_a = g_fc_kept.d_a; d_b = g_fc_kept.d_b; d_zlin = g_fc_kept.d_zlin;\n"
            "    } else {\n" + "".join("    " + x for x in alloc.splitlines(True)) + "    }\n")
body = once(body, "    std::vector<MHash> l0_cap;\n    {\n",
            "    std::vector<MHash> l0_cap;\n    if (take) {\n"
            "        d_cw0 = g_fc_kept.d_cw0; d_tree0 = g_fc_kept.d_tree0; l0_bl = g_fc_kept.l0_bl; l0_ni = g_fc_kept.l0_ni;\n"
            "        l0_cap = g_fc_kept.l0_cap;\n        fc_rejoin_level0(d_tree0);\n    } else {\n")
body = once(body, "    ffi_free(d_a); ffi_free(d_b);\n", "    if (!keep) { ffi_free(d_a); ffi_free(d_b); }\n")
body = once(body, "    ffi_free(d_zlin);\n", "    if (!keep) ffi_free(d_zlin);\n")
body = once(body, "    LigF256Proof lig;\n",
            "    F128* d_lz = df;\n    if (keep) {\n        CK(ffi_malloc(&d_lz, len * sizeof(F128)));\n"
            "        CK(cudaMemcpy(d_lz, df, len * sizeof(F128), cudaMemcpyDeviceToDevice));\n    }\n    LigF256Proof lig;\n")
body = once(body, "first_u0, first_u2, df, dcb,\n", "first_u0, first_u2, d_lz, dcb,\n")
body = once(body, "    ffi_free(d_cw0); ffi_free(d_tree0);\n    ffi_free(df); ffi_free(dcb);\n",
            "    if (keep) ffi_free(d_lz);\n    else { ffi_free(d_cw0); ffi_free(d_tree0); ffi_free(df); }\n    ffi_free(dcb);\n")
body = once(body, "    CK(device_arena.finish());\n    return 0;\n}",
            "    if (keep) {\n        // the arena stays open for rep 1\n"
            "        g_fc_kept = FcKept{true, N->reuse_tag, m, k_log, df, d_a, d_b, d_zlin, d_cw0, d_tree0, l0_bl, l0_ni, l0_cap};\n"
            "        return 0;\n    }\n    if (take) g_fc_kept = FcKept{};\n    CK(device_arena.finish());\n    return 0;\n}")
for gone in ("C->comp_slots", "C->units", "C->unit_log", "C->unit_pos0", "C->mode", "C->u_"):
    assert gone not in body, f"{gone} left in the circuit prover"

g = pathlib.Path("cuda-ghash")
text = (here / "cuda" / "prove_circuit.cuh").read_text() + "\nextern \"C\" {\n\n" + body + "\n\n}  // extern \"C\"\n"
(g / "prove_circuit.cuh").write_text(text)
ffi = g / "prove_ffi.cu"
s = ffi.read_text()
assert '#include "prove_chunk.cuh"' in s, "run backends/flock/cuda_chunk_patch.py first"
if '#include "prove_circuit.cuh"' not in s:
    s = s.replace('#include "prove_chunk.cuh"', '#include "prove_chunk.cuh"\n#include "prove_circuit.cuh"', 1)
    ffi.write_text(s)
print("prove_circuit.cuh installed:", len(text.splitlines()), "lines")
