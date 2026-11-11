#!/usr/bin/env python3
"""Install the chunk-statement prover (verity/protocols/flock/prover/cuda/prove_chunk.cuh) into a Flock b684b12 checkout.

The NTT's shared-memory chunks (ntt_f128.cuh) run on ntt_tile.cuh's kernel, up to 8 layers a pass.

Run from the flock checkout after `git apply` of flock-link-b684b12.patch and flock-gpu-link-b684b12.patch and after
copying verity/protocols/flock/prover/live to crates/flock-live. It applies the CUDA half of cuda_live_patch.py (the live hook in
challenger.hpp, pow_grind.cuh and prove_ffi.cu; its gpu_roundtrip.rs half needs flock-128's harness and is skipped),
copies prove_chunk.cuh next to prove_ffi.cu and includes it at the end, has the device arena take small blocks from its top,
moves zerocheck round 1's GF(2^8) products from its shared-memory log tables onto clmad on sm_90
(zerocheck_round1_cpustyle.cuh), and points build.rs at SM (default 90) and CUDA (default /usr/local/cuda-13.3).
"""
import os
import pathlib
import runpy
import shutil
import sys

root = pathlib.Path(__file__).resolve().parents[4]
try:
    runpy.run_path(str(root / "verity/protocols/flock/prover/cuda_live_patch.py"))
except AssertionError as e:
    if "g128_patch" not in str(e):
        raise
    print("cuda_live_patch.py: CUDA half applied, harness half skipped")

assert "live_cb" in pathlib.Path("cuda-ghash/challenger.hpp").read_text(), "live hook missing"
for f in ("async_copies.cuh", "prove_chunk.cuh", "pure_sha256.cuh", "ring_switch.cuh", "zc_round1.cuh", "zerocheck_fused.cuh",
          "zlin_transpose.cuh"):
    shutil.copy(root / "verity/protocols/flock/prover/cuda" / f, f"cuda-ghash/{f}")
pf = pathlib.Path("cuda-ghash/prove_ffi.cu")
s = pf.read_text()
if '#include "prove_chunk.cuh"' not in s:
    s = s.rstrip("\n") + '\n\n#include "prove_chunk.cuh"\n'
    print("prove_ffi.cu includes prove_chunk.cuh")
# the device arena takes blocks under 64 MiB from the top of the highest free block that holds them and larger ones first fit
# from the bottom, so the small buffers one phase leaves live don't split the space the next phase's large ones need
first_fit = """        for (auto block = free_.begin(); block != free_.end(); block++) {
            if (block->second < aligned) continue;
            size_t offset = block->first;
            size_t remaining = block->second - aligned;
            free_.erase(block);
            if (remaining) free_.emplace(offset + aligned, remaining);
            *ptr = base_ + offset;
            live_.emplace(*ptr, Block{offset, aligned});
            return cudaSuccess;
        }
        return cudaErrorMemoryAllocation;
"""
if first_fit in s:
    s = s.replace(first_fit, """        const bool top = aligned < ((size_t)64 << 20);
        auto block = free_.end();
        for (auto b = free_.begin(); b != free_.end(); b++) {
            if (b->second < aligned) continue;
            block = b;
            if (!top) break;
        }
        if (block == free_.end()) return cudaErrorMemoryAllocation;
        const size_t start = block->first, remaining = block->second - aligned;
        const size_t offset = top ? start + remaining : start;
        free_.erase(block);
        if (remaining) free_.emplace(top ? start : start + aligned, remaining);
        *ptr = base_ + offset;
        live_.emplace(*ptr, Block{offset, aligned});
        return cudaSuccess;
""", 1)
    print("prove_ffi.cu: the arena's small blocks from the top")
assert "const bool top = aligned < ((size_t)64 << 20);" in s, "DeviceArena::allocate changed upstream"
# the device arena zeroes, when a prove call finishes, every byte the call allocated (up to its high-water mark since begin()),
# so no value outlives the call in device memory; the next call, or the regrow that frees it, finds zeros
if "    size_t high_ = 0;\n" not in s:
    for a, b in (("        free_.clear();\n        free_.emplace(0, capacity_);\n",
                  "        free_.clear();\n        free_.emplace(0, capacity_);\n        high_ = 0;\n"),
                 ("        *ptr = base_ + offset;\n",
                  "        *ptr = base_ + offset;\n        if (offset + aligned > high_) high_ = offset + aligned;\n"),
                 ("        active_ = false;\n        return live_.empty() ? cudaSuccess : cudaErrorInvalidValue;\n",
                  "        active_ = false;\n        if (!live_.empty()) return cudaErrorInvalidValue;\n"
                  "        const size_t used = high_;\n        high_ = 0;\n        if (!used) return cudaSuccess;\n"
                  "        cudaError_t err = cudaMemsetAsync(base_, 0, used);\n"
                  "        return err != cudaSuccess ? err : cudaStreamSynchronize(0);\n"),
                 ("    bool active_ = false;\n", "    bool active_ = false;\n    size_t high_ = 0;\n")):
        assert s.count(a) == 1, f"prove_ffi.cu's DeviceArena changed: {a!r}"
        s = s.replace(a, b)
    print("prove_ffi.cu: the device arena zeroes what each prove call used")
pf.write_text(s)

# Round 1 folds each lane's eight k's as Σ_k x^k·(a_k·b_k) in GF(2^8) (AES polynomial), reduced once. Under clmad the eight
# products stay unreduced: one carry-less product of (aL·x^k + aH·x^(32+k)) and (bL + bH·x^32) puts aL·bL·x^k in its low
# half's bits [0, 22), aH·bH·x^k in its high half's and the cross terms in the low half's bits [32, 54); reducing the 22-bit
# sum gives the same residue the tables' sum of reduced products does, so the kernel's output is the same bytes. FORM: 0 the
# device's form (clmad on sm_90 only: on sm_89 and sm_120 clmad is far slower than a shared load), 1 the tables, 2 clmad.
# MINB: the launch bound's blocks per SM. The launcher's W = 14 needs two blocks of 448 threads per SM, so at most 72
# registers a thread, which sm_120's build meets by itself and sm_90's (78-80) only under the bound.
zc = pathlib.Path("cuda-ghash/zerocheck_round1_cpustyle.cuh")
s = zc.read_text()
if "reduce_zerocheck_gf8_wide" not in s:
    for a, new in [
        ("// GHASH multiply-by-γ (γ=0x02): left shift by 1 bit, reduce by 0x87.\n",
         "// GF(2^8) reduce of a <22-bit poly (four folds of x^8 = x^4+x^3+x+1: below 2^18, 2^14, 2^10, 2^8).\n"
         "__device__ __forceinline__ uint8_t reduce_zerocheck_gf8_wide(uint32_t p) {\n"
         "#pragma unroll\n"
         "    for (int i = 0; i < 4; i++) { uint32_t h = p >> 8; p = (p & 0xff) ^ h ^ (h << 1) ^ (h << 3) ^ (h << 4); }\n"
         "    return (uint8_t)p;\n"
         "}\n\n"
         "// GHASH multiply-by-γ (γ=0x02): left shift by 1 bit, reduce by 0x87.\n"),
        ("template <int W>\n__global__ void zerocheck_first_round_cpu_structured(",
         "template <int W, int FORM = 0, int MINB = 2>\n"
         "__global__ void __launch_bounds__(W * 32, MINB) zerocheck_first_round_cpu_structured("),
        ("    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_log[i] = d_zc_log[i];\n"
         "    for (int i = threadIdx.x; i < 512; i += blockDim.x) s_antilog[i] = d_zc_antilog[i];\n",
         "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000\n"
         "    constexpr bool cl = FORM != 1;\n"
         "#else\n"
         "    constexpr bool cl = FORM == 2;\n"
         "#endif\n"
         "    for (int i = threadIdx.x; !cl && i < 256; i += blockDim.x) s_log[i] = d_zc_log[i];\n"
         "    for (int i = threadIdx.x; !cl && i < 512; i += blockDim.x) s_antilog[i] = d_zc_antilog[i];\n"),
        ("            uint16_t aacc0 = 0, aacc1 = 0, cacc0 = 0, cacc1 = 0;\n",
         "            uint16_t aacc0 = 0, aacc1 = 0, cacc0 = 0, cacc1 = 0;\n            u64 pl = 0, ph = 0;\n"),
        ("                // A·B = antilog[log A + log B], 0-masked; 768 B tables in shared.\n"
         "                uint8_t p0 = s_antilog[(int)s_log[aL] + s_log[bL]] & (uint8_t)(-(int)(aL && bL));\n"
         "                uint8_t p1 = s_antilog[(int)s_log[aH] + s_log[bH]] & (uint8_t)(-(int)(aH && bH));\n"
         "                aacc0 ^= (uint16_t)p0 << k;\n"
         "                aacc1 ^= (uint16_t)p1 << k;\n",
         "                if constexpr (cl) {\n"
         "                    const u64 A = ((u64)((uint32_t)aH << k) << 32) | ((uint32_t)aL << k), B = ((u64)bH << 32) | bL;\n"
         "                    pl = clmad_lo(A, B, pl);\n"
         "                    ph = clmad_hi(A, B, ph);\n"
         "                } else {\n"
         "                    // A·B = antilog[log A + log B], 0-masked; 768 B tables in shared.\n"
         "                    uint8_t p0 = s_antilog[(int)s_log[aL] + s_log[bL]] & (uint8_t)(-(int)(aL && bL));\n"
         "                    uint8_t p1 = s_antilog[(int)s_log[aH] + s_log[bH]] & (uint8_t)(-(int)(aH && bH));\n"
         "                    aacc0 ^= (uint16_t)p0 << k;\n"
         "                    aacc1 ^= (uint16_t)p1 << k;\n"
         "                }\n"),
        ("            chunk_ab0 = f128_add(multiply_zerocheck_value_by_generator(chunk_ab0), s_phi[reduce_zerocheck_gf8_polynomial(aacc0)]);\n"
         "            chunk_ab1 = f128_add(multiply_zerocheck_value_by_generator(chunk_ab1), s_phi[reduce_zerocheck_gf8_polynomial(aacc1)]);\n",
         "            const uint8_t r0 = cl ? reduce_zerocheck_gf8_wide((uint32_t)pl) : reduce_zerocheck_gf8_polynomial(aacc0);\n"
         "            const uint8_t r1 = cl ? reduce_zerocheck_gf8_wide((uint32_t)ph) : reduce_zerocheck_gf8_polynomial(aacc1);\n"
         "            chunk_ab0 = f128_add(multiply_zerocheck_value_by_generator(chunk_ab0), s_phi[r0]);\n"
         "            chunk_ab1 = f128_add(multiply_zerocheck_value_by_generator(chunk_ab1), s_phi[r1]);\n"),
    ]:
        assert s.count(a) == 1, f"{zc}: {a!r} occurs {s.count(a)} times, expected 1"
        s = s.replace(a, new)
    zc.write_text(s)
    print("zerocheck_round1_cpustyle.cuh: round 1's products on clmad (sm_90)")

# The NTT's shared-memory chunks run ntt_tile.cuh's kernel (FC_NTT_REGS=0: the tile kernel), up to 8 layers a pass
# (FC_NTT_KMAX lowers it); the register kernels' chunks (4, 2 and 1 layers) and ntt_can_fuse_source are unchanged.
shutil.copy(root / "verity/protocols/flock/prover/cuda/ntt_tile.cuh", "cuda-ghash/ntt_tile.cuh")
nt = pathlib.Path("cuda-ghash/ntt_f128.cuh")
s = nt.read_text()
if '#include "ntt_tile.cuh"' not in s:
    for a, new in [
        ("// Launch one K-layer smem-fused chunk at `layer`, lane-tiling so per-block smem",
         '#include "ntt_tile.cuh"\n\n// Launch one K-layer smem-fused chunk at `layer`, lane-tiling so per-block smem'),
        ("    cudaFuncSetAttribute(additive_ntt_shared_memory_tile<K>,",
         "    if (ntt_tile_regs_on()) {\n"
         "        launch_ntt_tile_regs<K>(d_data, d_tw, layer, log_d, num_ntts, log_lb, smem, src ? src : d_data, smask);\n"
         "        return;\n"
         "    }\n"
         "    cudaFuncSetAttribute(additive_ntt_shared_memory_tile<K>,"),
        ("template <int MaxFusedLayers = 7, size_t SharedMemoryBytes = 32 * 1024>",
         "template <int MaxFusedLayers = 8, size_t SharedMemoryBytes = 32 * 1024>"),
        ("    static_assert(MaxFusedLayers >= 1 && MaxFusedLayers <= 7);",
         "    static_assert(MaxFusedLayers >= 1 && MaxFusedLayers <= 8);"),
        ("    int npass = (total + MaxFusedLayers - 1) / MaxFusedLayers;",
         "    const int kmax = ntt_tile_kmax(MaxFusedLayers);\n    int npass = (total + kmax - 1) / kmax;"),
        ("            case 7: launch_ntt_shared_memory_chunk<7, SharedMemoryBytes>",
         "            case 8: launch_ntt_shared_memory_chunk<8, SharedMemoryBytes>(d_data, d_tw, layer, log_d, num_ntts, tpb, src, smask); break;\n"
         "            case 7: launch_ntt_shared_memory_chunk<7, SharedMemoryBytes>"),
    ]:
        assert s.count(a) == 1, f"{nt}: {a!r} occurs {s.count(a)} times, expected 1"
        s = s.replace(a, new)
    nt.write_text(s)
    print("ntt_f128.cuh: shared-memory chunks on ntt_tile.cuh's kernel, up to 8 layers a pass")


def edit(path: pathlib.Path, pairs: list) -> None:
    s = path.read_text()
    for a, new, n in pairs:
        assert s.count(a) == n, f"{path}: {a!r} occurs {s.count(a)} times, expected {n}"
        s = s.replace(a, new)
    path.write_text(s)


# The host waits on the device only for a value it reads (async_copies.cuh): the eq tables' and Ligerito's uploads are queued
# behind the kernels before them, a transcript read's values come back with one wait, an opening's rows and paths come back
# with the next one, and a tree's cap read is its only wait. The same copies in the same stream order, so the same bytes.
eq = pathlib.Path("cuda-ghash/induce_sumcheck.cuh")
if '#include "async_copies.cuh"' not in eq.read_text():
    edit(eq, [
        ('#include "f128.cuh"\n', '#include "f128.cuh"\n#include "async_copies.cuh"\n', 1),
        ("    if (d <= 0) { F128 one{1ull, 0ull}; (void)cudaMemcpy(d_out, &one, sizeof(F128), cudaMemcpyHostToDevice); return; }\n",
         "    if (d <= 0) { F128 one{1ull, 0ull}; (void)hc_put(d_out, &one, sizeof(F128)); return; }\n", 1),
        ("        cudaMemcpyToSymbol(g_eq_chal, challenges, (size_t)d * sizeof(F128));\n",
         "        (void)hc_put_symbol(g_eq_chal, challenges, (size_t)d * sizeof(F128));\n", 1),
        ("    F128 one{1ull, 0ull};\n    (void)cudaMemcpy(d_out, &one, sizeof(F128), cudaMemcpyHostToDevice);\n",
         "    F128 one{1ull, 0ull};\n    (void)hc_put(d_out, &one, sizeof(F128));\n", 1),
    ])
    mo = pathlib.Path("cuda-ghash/merkle_open_device.cuh")
    edit(mo, [
        ('#include "merkle_open.hpp"   // MHash, merkle_multi_proof_indices\n',
         '#include "merkle_open.hpp"   // MHash, merkle_multi_proof_indices\n#include "async_copies.cuh"\n', 1),
    ])
    mo.write_text(mo.read_text() + """
// merkle_capped_paths_device's nodes into `out`, queued: `out` holds them after the next hc_wait. The scratch grows and stays;
// a call reuses it behind the previous call's read in stream order.
inline void merkle_capped_paths_queue(std::vector<MHash>& out, const uint8_t* d_tree, size_t num_leaves,
                                      const std::vector<size_t>& queries, uint32_t cap_depth) {
    const std::vector<size_t> idxs = merkle_capped_path_indices(num_leaves, queries, cap_depth);
    const int n = (int)idxs.size();
    out.assign(n, MHash{});
    if (n == 0) return;
    static unsigned long long* d_idx = nullptr;
    static uint8_t* d_out = nullptr;
    static int cap = 0;
    if (n > cap) {
        if (d_idx) { MOD_CK(cudaFree(d_idx)); MOD_CK(cudaFree(d_out)); }
        cap = n + (n >> 1);
        MOD_CK(cudaMalloc(&d_idx, (size_t)cap * sizeof(unsigned long long)));
        MOD_CK(cudaMalloc(&d_out, (size_t)cap * sizeof(MHash)));
    }
    const std::vector<unsigned long long> h_idx(idxs.begin(), idxs.end());
    MOD_CK(hc_put(d_idx, h_idx.data(), (size_t)n * sizeof(unsigned long long)));
    const int tpb = 128;
    gather_tree_nodes<<<(n + tpb - 1) / tpb, tpb>>>(d_tree, d_idx, n, d_out);
    MOD_CK(cudaGetLastError());
    MOD_CK(hc_get(out.data(), d_out, (size_t)n * sizeof(MHash)));
}
""")
    lg = pathlib.Path("cuda-ghash/ligerito_f256.cuh")
    read3 = ("            LFCK(cudaMemcpy(&{a}, d_u0, sizeof(F128), cudaMemcpyDeviceToHost));\n"
             "            LFCK(cudaMemcpy(&{b}, d_u2, sizeof(F128), cudaMemcpyDeviceToHost));\n"
             "            LFCK(cudaMemcpy(&y, d_hnew, sizeof(F128), cudaMemcpyDeviceToHost));\n")
    wait3 = ("            LFCK(hc_get(&{a}, d_u0, sizeof(F128))); LFCK(hc_get(&{b}, d_u2, sizeof(F128)));\n"
             "            LFCK(hc_get(&y, d_hnew, sizeof(F128))); LFCK(hc_wait());\n")
    edit(lg, [
        ('#include "merkle_open_device.cuh"\n', '#include "merkle_open_device.cuh"\n#include "async_copies.cuh"\n', 1),
        ("                             const LigAlloc& A, LigF256Proof& out) {\n",
         "                             const LigAlloc& A, LigF256Proof& out) {\n    HcDrop hc_guard;\n", 1),
        ("        cudaMemcpy(h, d_out256, sizeof(h), cudaMemcpyDeviceToHost);\n",
         "        (void)hc_get(h, d_out256, sizeof(h));\n        (void)hc_wait();\n", 2),
        (read3.format(a="ou0", b="ou2"), wait3.format(a="ou0", b="ou2"), 1),
        (read3.format(a="mu0", b="mu2"), wait3.format(a="mu0", b="mu2"), 1),
        ("        launch_merkle((const uint8_t*)d_cw_out, d_tree_out, bl, num_ntts * 16);\n        LFCK(cudaDeviceSynchronize());\n",
         "        launch_merkle((const uint8_t*)d_cw_out, d_tree_out, bl, num_ntts * 16);\n", 1),
        ("            LFCK(cudaMemcpy(d_ind_q, qull.data(), qull.size() * sizeof(unsigned long long),\n"
         "                            cudaMemcpyHostToDevice));\n",
         "            LFCK(hc_put(d_ind_q, qull.data(), qull.size() * sizeof(unsigned long long)));\n", 1),
        ("            LFCK(cudaMemcpy(open.rows_flat.data(), d_rows, open.rows_flat.size() * sizeof(F128),\n"
         "                            cudaMemcpyDeviceToHost));\n",
         "            LFCK(hc_get(open.rows_flat.data(), d_rows, open.rows_flat.size() * sizeof(F128)));\n", 1),
        ("        open.path = merkle_capped_paths_device(d_prev_tree, (size_t)prev_bl, q_out,\n",
         "        merkle_capped_paths_queue(open.path, d_prev_tree, (size_t)prev_bl, q_out,\n", 1),
        ("        LFCK(cudaMemcpy(d_ind_q, qull.data(), qull.size() * sizeof(unsigned long long),\n"
         "                        cudaMemcpyHostToDevice));\n"
         "        LFCK(cudaMemcpy(d_ind_alpha, wcomb.data(), wcomb.size() * sizeof(F128),\n"
         "                        cudaMemcpyHostToDevice));\n",
         "        LFCK(hc_put(d_ind_q, qull.data(), qull.size() * sizeof(unsigned long long)));\n"
         "        LFCK(hc_put(d_ind_alpha, wcomb.data(), wcomb.size() * sizeof(F128)));\n", 1),
        ("        LFCK(cudaMemcpy(&m0, d_u0, sizeof(F128), cudaMemcpyDeviceToHost));\n"
         "        LFCK(cudaMemcpy(&m2, d_u2, sizeof(F128), cudaMemcpyDeviceToHost));\n",
         "        LFCK(hc_get(&m0, d_u0, sizeof(F128))); LFCK(hc_get(&m2, d_u2, sizeof(F128))); LFCK(hc_wait());\n", 1),
        ("            LFCK(cudaMemcpy(out.yr.data(), regions[cur_f], out.yr.size() * sizeof(F128),\n"
         "                            cudaMemcpyDeviceToHost));\n",
         "            LFCK(hc_get(out.yr.data(), regions[cur_f], out.yr.size() * sizeof(F128))); LFCK(hc_wait());\n", 1),
        ("    LFCK(A.release(d_hnew));\n    return 0;\n}\n", "    LFCK(A.release(d_hnew));\n    LFCK(hc_wait());\n    return 0;\n}\n", 1),
    ])
    print("induce_sumcheck.cuh, merkle_open_device.cuh, ligerito_f256.cuh: the host waits only for the values it reads")

sm = os.environ.get("SM", "90")
cuda = os.environ.get("CUDA", "/usr/local/cuda-13.3")
b = pathlib.Path("crates/flock-cuda-ffi/build.rs")
s = b.read_text()
s = s.replace("arch=compute_120,code=sm_120", f"arch=compute_{sm},code=sm_{sm}")
s = s.replace('"/usr/local/cuda/lib64"', f'"{cuda}/lib64"').replace('"/usr/local/cuda/lib64/stubs"', f'"{cuda}/lib64/stubs"')
b.write_text(s)
print(f"build.rs: sm_{sm}, {cuda}")
sys.exit(0)
