#!/usr/bin/env python3
"""SHA-512 Merkle trees with hm96-sha512/v1 leaves on the device for verity/flock-circuit: install verity/protocols/flock/prover/cuda/sha512.cuh into a Flock b684b12
checkout, with the GPU's one SHA-512 (verity/definitions/kernels/sha512.cuh) beside it as kernels_sha512.cuh, and widen every Merkle
node from 32 to 64 bytes (FLOCK_HB): the tree kernels, the node gathers and cap copies, the proof writer, and the tree
allocations and cap observations of the chunk and circuit provers and the Ligerito levels, whose salts the device copies
beside level 0's folds.

Run from the flock checkout after verity/protocols/flock/prover/cuda_circuit_patch.py, together with verity/protocols/flock/prover/flock-sha512-b684b12.patch
(the Rust side: 64-byte digests, HashKind::Sha512); flock-live's feature `sha512` selects it. Every edit must match exactly
once (or the stated count), so a changed upstream fails here rather than committing with mixed widths.
"""
import pathlib
import re
import shutil

g = pathlib.Path("cuda-ghash")
if "FLOCK_HB" in (g / "merkle_open.hpp").read_text():
    raise SystemExit("cuda_sha512_patch: already applied")
# The GPU's one SHA-512 and the header that includes it, each by its path from the repository's root, and that header's
# include line, whatever its way there: this runs under a bare python3, which can't import verity for the paths.
KERNELS_SHA = re.compile(r'#include "\.\./[\w./-]*/sha512\.cuh"\n')
shutil.copy(pathlib.Path(__file__).resolve().parents[4] / "verity/definitions/kernels/sha512.cuh", g / "kernels_sha512.cuh")
s = (pathlib.Path(__file__).resolve().parents[4] / "verity/protocols/flock/prover/cuda/sha512.cuh").read_text()
found = KERNELS_SHA.findall(s)
assert len(found) == 1, f"cuda/sha512.cuh: {KERNELS_SHA.pattern!r} occurs {len(found)} times, expected 1"
(g / "sha512.cuh").write_text(s.replace(found[0], '#include "kernels_sha512.cuh"\n'))

GUARD = "#ifndef FLOCK_HB\n#define FLOCK_HB 64   // Merkle node bytes: SHA-512 (verity's cuda_sha512_patch.py)\n#endif\n"


def edit(name: str, pairs: list, count: int = 1) -> None:
    p = g / name
    s = p.read_text()
    for a, b, *n in pairs:
        want = n[0] if n else count
        assert s.count(a) == want, f"{name}: {a!r} occurs {s.count(a)} times, expected {want}"
        s = s.replace(a, b)
    p.write_text(s)


edit("merkle.cuh", [
    ('#include "sha256.cuh"\n', '#include "sha256.cuh"\n#include "sha512.cuh"\n' + GUARD),
    ("typedef uint8_t Hash[32];", "typedef uint8_t Hash[FLOCK_HB];"),
    ("int tpb = 256, int kway = 1) {\n",
     "int tpb = 256, int kway = 1) {\n    if (FLOCK_HB == 64) { launch_merkle512(d_data, d_tree, num_leaves, leaf_size, tpb); return; }\n"),
])
edit("merkle_open.hpp", [
    ("#include <algorithm>\n", "#include <algorithm>\n" + GUARD),
    ("struct MHash { uint8_t b[32]; };", "struct MHash { uint8_t b[FLOCK_HB]; };"),
    ("memcmp(a.b, c.b, 32)", "memcmp(a.b, c.b, FLOCK_HB)"),
])
p = g / "merkle_open_device.cuh"
s = p.read_text()
n_mul, n_loop = len(re.findall(r"\* 32\b", s)), s.count("j < 32;")
assert (n_mul, n_loop) == (10, 1), (n_mul, n_loop)
p.write_text(re.sub(r"\* 32\b", "* FLOCK_HB", s).replace("j < 32;", "j < FLOCK_HB;"))
edit("prove_ffi.cu", [
    ("    void hash(const uint8_t* h) { buf.insert(buf.end(), h, h + 32); }", "    void hash(const uint8_t* h) { buf.insert(buf.end(), h, h + FLOCK_HB); }"),
    ("(size_t)(2 * l0_bl - 1) * 32)", "(size_t)(2 * l0_bl - 1) * FLOCK_HB)"),
    ("l0_cap.size() * 32", "l0_cap.size() * FLOCK_HB"),
])
SALTS = ("        W.hashes(lo.path);\n",
         "        W.hashes(lo.path);\n"
         "        if (g_hm96.tree_cb) { W.u64(lo.salts.size() / 192); W.buf.insert(W.buf.end(), lo.salts.begin(), lo.salts.end()); }\n")
for f in ("prove_chunk.cuh", "prove_circuit.cuh"):
    edit(f, [
        ("(size_t)(2 * l0_bl - 1) * 32)", "(size_t)(2 * l0_bl - 1) * FLOCK_HB)"),
        ("l0_cap.size() * 32", "l0_cap.size() * FLOCK_HB", 2),
        SALTS,
    ])
edit("prove_circuit.cuh", [
    ("uint8_t** out, size_t* out_len) {\n    const FlockChunkParams* C = &N->c;\n",
     "uint8_t** out, size_t* out_len) {\n    const FlockChunkParams* C = &N->c;\n"
     "    Hm96Guard hm96_guard(N->hm96_tree, N->hm96_ctx, N->hm96_prefixes, N->hm96_tables, N->hm96_level0_leaves);\n"),
])
edit("ligerito_f256.cuh", [
    ("out.initial_cap.size() * 32", "out.initial_cap.size() * FLOCK_HB"),
    ("(size_t)(2 * bl - 1) * 32)", "(size_t)(2 * bl - 1) * FLOCK_HB)"),
    ("cap.data(), cap.size() * 32", "cap.data(), cap.size() * FLOCK_HB"),
    ("    std::vector<MHash> path;   // capped per-query siblings, sample order\n",
     "    std::vector<MHash> path;   // capped per-query siblings, sample order\n"
     "    std::vector<uint8_t> salts;   // hm96-sha512: each opened row's 192-byte salt, in sample order\n"),
    ("                                               caps[open_level]);\n",
     "                                               caps[open_level]);\n"
     "        open.salts = hm96_opened_salts(d_prev_tree, q_out);\n"),
    # the recursive levels' salts copied to the device beside level 0's folds (hm96_copy_ahead) from the first fold's coin, which
    # under the ZK server comes after its Lean finish drew them; fc_arena_bytes counts them, each held until its tree is built
    ("    // ---- initial_k folds + code switch ----\n",
     "    std::vector<uint8_t*> d_salts(r + 1, nullptr);\n"
     "    auto copy_salts_ahead = [&]() -> int {\n"
     "        if (!g_hm96.tree_cb || !g_hm96.copy) return 0;\n"
     "        std::vector<long long> leaves;\n"
     "        for (int lvl = 1, dim = n1 + 1; lvl <= r; lvl++) {\n"
     "            const int log_cols = dim - C.recursive_ks[lvl - 1];\n"
     "            leaves.push_back(1LL << (log_cols + C.log_inv_rates[lvl]));\n"
     "            LFCK(A.alloc((void**)&d_salts[lvl], (size_t)leaves.back() * 192));\n"
     "            dim = log_cols + 1;\n"
     "        }\n"
     "        if (hm96_copy_ahead(leaves, std::vector<uint8_t*>(d_salts.begin() + 1, d_salts.end()))) return 0;\n"
     "        for (int lvl = 1; lvl <= r; lvl++) LFCK(A.release(d_salts[lvl]));\n"
     "        std::fill(d_salts.begin(), d_salts.end(), nullptr);\n"
     "        return 0;\n"
     "    };\n"
     "    // ---- initial_k folds + code switch ----\n"),
    ("        if (j == 0) {\n            long long half = len / 2;\n",
     "        if (j == 0) {\n            long long half = len / 2;\n"
     "            if (int rc = copy_salts_ahead()) return rc;\n"),
    ("        launch_merkle((const uint8_t*)d_cw_out, d_tree_out, bl, num_ntts * 16);\n",
     "        launch_merkle((const uint8_t*)d_cw_out, d_tree_out, bl, num_ntts * 16);\n"
     "        if (d_salts[level]) {\n"
     "            LFCK(cudaMemsetAsync(d_salts[level], 0, (size_t)bl * 192));\n"
     "            LFCK(A.release(d_salts[level]));\n"
     "            d_salts[level] = nullptr;\n"
     "        }\n"),
])
print("cuda_sha512_patch: SHA-512 Merkle nodes (64 bytes) installed")
