# The plain NVFP4 and FP8 GEMMs: design

Each file and who reads it, beside [the README](README.md).

| File | What it is |
| --- | --- |
| `mainloop_nvf4.cuh` | the TMA-fed, warp-specialized mainloop, NVFP4 and FP8 |
| `nvf4_plain.cu` | the GEMMs on it as entries of the GPU harness's baseline registry; `../harness/native/build.sh` builds it into the harness's library |

## Who reads it

The harness reads `nvf4_plain.cu`: its build compiles it, with the mainloop, into the harness's library, and
`../../../tests/benchmarks/pouw/test_harness.py` reads its text for its registry entries. Nothing here commits, hashes or checks a proof.
