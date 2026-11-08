#!/usr/bin/env python3
"""Install the chunk-statement prover (verity/ml/flock/cuda/prove_chunk.cuh) into a Flock b684b12 checkout.

Run from the flock checkout after `git apply` of flock-link-b684b12.patch and flock-gpu-link-b684b12.patch and after
copying verity/ml/flock/live to crates/flock-live. It applies the CUDA half of cuda_live_patch.py (the live hook in
challenger.hpp, pow_grind.cuh and prove_ffi.cu; its gpu_roundtrip.rs half needs flock-128's harness and is skipped),
copies prove_chunk.cuh next to prove_ffi.cu and includes it at the end, and points build.rs at SM (default 90) and
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
for f in ("prove_chunk.cuh", "pure_sha256.cuh"):
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

sm = os.environ.get("SM", "90")
cuda = os.environ.get("CUDA", "/usr/local/cuda-13.3")
b = pathlib.Path("crates/flock-cuda-ffi/build.rs")
s = b.read_text()
s = s.replace("arch=compute_120,code=sm_120", f"arch=compute_{sm},code=sm_{sm}")
s = s.replace('"/usr/local/cuda/lib64"', f'"{cuda}/lib64"').replace('"/usr/local/cuda/lib64/stubs"', f'"{cuda}/lib64/stubs"')
b.write_text(s)
print(f"build.rs: sm_{sm}, {cuda}")
sys.exit(0)
