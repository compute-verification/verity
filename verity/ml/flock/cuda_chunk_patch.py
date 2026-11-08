#!/usr/bin/env python3
"""Install the chunk-statement prover (verity/ml/flock/cuda/prove_chunk.cuh) into a Flock b684b12 checkout.

Run from the flock checkout after `git apply` of flock-link-b684b12.patch and flock-gpu-link-b684b12.patch and after
copying verity/ml/flock/live to crates/flock-live. It applies the CUDA half of cuda_live_patch.py (the live hook in
challenger.hpp, pow_grind.cuh and prove_ffi.cu; its gpu_roundtrip.rs half needs flock-128's harness and is skipped),
copies prove_chunk.cuh next to prove_ffi.cu and includes it at the end, has the device arena take small blocks from its top,
and points build.rs at SM (default 90) and CUDA (default /usr/local/cuda-13.3).
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
pf.write_text(s)

sm = os.environ.get("SM", "90")
cuda = os.environ.get("CUDA", "/usr/local/cuda-13.3")
b = pathlib.Path("crates/flock-cuda-ffi/build.rs")
s = b.read_text()
s = s.replace("arch=compute_120,code=sm_120", f"arch=compute_{sm},code=sm_{sm}")
s = s.replace('"/usr/local/cuda/lib64"', f'"{cuda}/lib64"').replace('"/usr/local/cuda/lib64/stubs"', f'"{cuda}/lib64/stubs"')
b.write_text(s)
print(f"build.rs: sm_{sm}, {cuda}")
sys.exit(0)
