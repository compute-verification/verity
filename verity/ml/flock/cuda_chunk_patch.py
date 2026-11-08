#!/usr/bin/env python3
"""Install the chunk-statement prover (verity/ml/flock/cuda/prove_chunk.cuh) into a Flock b684b12 checkout.

Run from the flock checkout after `git apply` of flock-link-b684b12.patch and flock-gpu-link-b684b12.patch and after
copying verity/ml/flock/live to crates/flock-live. It applies the CUDA half of cuda_live_patch.py (the live hook in
challenger.hpp, pow_grind.cuh and prove_ffi.cu; its gpu_roundtrip.rs half needs flock-128's harness and is skipped),
copies prove_chunk.cuh next to prove_ffi.cu and includes it at the end, moves zerocheck round 1's GF(2^8) products from its
shared-memory log tables onto clmad on sm_90 (zerocheck_round1_cpustyle.cuh), and points build.rs at SM (default 90) and
CUDA (default /usr/local/cuda-13.3).
"""
import os
import pathlib
import runpy
import shutil
import sys

here = pathlib.Path(__file__).resolve().parent
try:
    runpy.run_path(str(here / "cuda_live_patch.py"))
except AssertionError as e:
    if "g128_patch" not in str(e):
        raise
    print("cuda_live_patch.py: CUDA half applied, harness half skipped")

assert "live_cb" in pathlib.Path("cuda-ghash/challenger.hpp").read_text(), "live hook missing"
for f in ("prove_chunk.cuh", "pure_sha256.cuh", "zerocheck_fused.cuh"):
    shutil.copy(here / "cuda" / f, f"cuda-ghash/{f}")
pf = pathlib.Path("cuda-ghash/prove_ffi.cu")
s = pf.read_text()
if '#include "prove_chunk.cuh"' not in s:
    pf.write_text(s.rstrip("\n") + '\n\n#include "prove_chunk.cuh"\n')
    print("prove_ffi.cu includes prove_chunk.cuh")
# the device arena zeroes, when a prove call finishes, every byte the call allocated (up to its high-water mark since begin()),
# so no value outlives the call in device memory; the next call, or the regrow that frees it, finds zeros
s = pf.read_text()
if "    size_t high_ = 0;\n" not in s:
    for a, b in (("        free_.clear();\n        free_.emplace(0, capacity_);\n",
                  "        free_.clear();\n        free_.emplace(0, capacity_);\n        high_ = 0;\n"),
                 ("            *ptr = base_ + offset;\n",
                  "            *ptr = base_ + offset;\n            if (offset + aligned > high_) high_ = offset + aligned;\n"),
                 ("        active_ = false;\n        return live_.empty() ? cudaSuccess : cudaErrorInvalidValue;\n",
                  "        active_ = false;\n        if (!live_.empty()) return cudaErrorInvalidValue;\n"
                  "        const size_t used = high_;\n        high_ = 0;\n        if (!used) return cudaSuccess;\n"
                  "        cudaError_t err = cudaMemsetAsync(base_, 0, used);\n"
                  "        return err != cudaSuccess ? err : cudaStreamSynchronize(0);\n"),
                 ("    bool active_ = false;\n", "    bool active_ = false;\n    size_t high_ = 0;\n")):
        assert s.count(a) == 1, f"prove_ffi.cu's DeviceArena changed: {a!r}"
        s = s.replace(a, b)
    pf.write_text(s)
    print("prove_ffi.cu: the device arena zeroes what each prove call used")

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

sm = os.environ.get("SM", "90")
cuda = os.environ.get("CUDA", "/usr/local/cuda-13.3")
b = pathlib.Path("crates/flock-cuda-ffi/build.rs")
s = b.read_text()
s = s.replace("arch=compute_120,code=sm_120", f"arch=compute_{sm},code=sm_{sm}")
s = s.replace('"/usr/local/cuda/lib64"', f'"{cuda}/lib64"').replace('"/usr/local/cuda/lib64/stubs"', f'"{cuda}/lib64/stubs"')
b.write_text(s)
print(f"build.rs: sm_{sm}, {cuda}")
sys.exit(0)
