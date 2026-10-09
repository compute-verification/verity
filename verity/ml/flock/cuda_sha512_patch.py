#!/usr/bin/env python3
"""SHA-512 Merkle trees with hm96-sha512/v1 leaves on the device for verity/flock-circuit: install verity/ml/flock/cuda/sha512.cuh into a Flock b684b12
checkout and widen every Merkle node from 32 to 64 bytes (FLOCK_HB): the tree kernels, the node gathers and cap copies, the
proof writer, and the tree allocations and cap observations of the chunk and circuit provers and the Ligerito levels.

Run from the flock checkout after verity/ml/flock/cuda_circuit_patch.py, together with verity/ml/flock/flock-sha512-b684b12.patch
(the Rust side: 64-byte digests, HashKind::Sha512); flock-live's feature `sha512` selects it. Every edit must match exactly
once (or the stated count), so a changed upstream fails here rather than committing with mixed widths.
"""
import pathlib
import re
import shutil

here = pathlib.Path(__file__).resolve().parent
g = pathlib.Path("cuda-ghash")
if "FLOCK_HB" in (g / "merkle_open.hpp").read_text():
    raise SystemExit("cuda_sha512_patch: already applied")
shutil.copy(here / "cuda" / "sha512.cuh", g / "sha512.cuh")

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
])
print("cuda_sha512_patch: SHA-512 Merkle nodes (64 bytes) installed")
