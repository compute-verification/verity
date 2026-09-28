#!/usr/bin/env python3
"""Install the circuit prover (backends/flock/cuda/prove_circuit.cuh) into a Flock b684b12 checkout (lane flock-circuit).

Run from the flock checkout after backends/flock/cuda_chunk_patch.py, which installs prove_chunk.cuh and the live hook.
flock_cuda_prove_circuit is flock_cuda_prove_chunk's body with three parts replaced (each anchor must occur once): the shape
check (fc_bad_shape), the witness (fc_witness: device compressions and units, or the host's witness) and the lincheck fold
(fc_fold: every slot range of every type). Everything else (commit, zerocheck, lincheck rounds, ring switch, Ligerito, the
live coins) is the chunk prover's, so no second copy of it is kept in the repository.
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
