# Pearl-C's BLAKE3 hashing (frozen)

Pearl-C hashed under four formats until 2026-10-06: -h1 (BLAKE3 messages, leaves and frame-b3 trees), -h2 (frame-b3s
trees), -h3 (-h2's trees, each row of A seeding its own E_A line) and -h3s (-h3 entirely through SHA-512). Daniel's ruling
of that day (`.agents/skills/friction/SKILL.md`, "Rulings") standardized on SHA-512: -h3s became the only -h3, and the
BLAKE3 code moved here. It is kept, not maintained: nothing imports it, no suite runs it, and nothing here is expected to
run against the current tree. Pearl's own FP8 scheme (`pearl-fp8-v4`, `schemes/pearl_kw.py`) keeps its BLAKE3 and is not
here.

| path | what it was |
|---|---|
| `protocol/` | `serving.py`, `schemes/pearl_c.py`, `schemes/pearl_c4.py`, `tests/pearl_c_vectors.py` and `tests/vectors/pearl_c.json` as they stood at d9820913f: the `h1`, `h2` and BLAKE3 `h3` `HASH_FORMATS` entries, `TREE_HASH`'s frame-b3 and frame-b3s paths, `digest_keys`, and the registered `pearl-c-h100-v1-h1` / `-h2` with their vectors |
| `pearl_c/` | the H100 -h1 bench (`pearl_c.cu`, `run.py`, `run.sh`, `ship.sh`, `build.sh`, `bench.py`, `bench.json`, `vendor.py`, the FTZ gate, `acc.h`, and its CPU half `twin.py`), the -h1 ship, stand-in and vectors (`h1_*`, `verify_h1.py`), the BLAKE3 kernels (`hash.cuh`, `hash.cu`, `hash_sm120.cuh`, `hash_h2.cuh`), `decode_served.py`, `h2_rows_stats_probe.py`, and the copies of `fixture.py`, `simt_twin.py` and `hash_host.cpp` that built them |
| `pearl_c_sm120/` | the sm_120 pipeline before the collapse (`pearl_c_sm120.cu` with the BLAKE3 kernels and the fused `gemm_pearlc_b`, `run.py` with -h1, -h2, -h3 and -h3s, its `fixture.py`, `sass_gate.py` and `build.sh`), -h2's rows form `s` (`h2_rows_stats.cuh`), and the panel's -h1 arm and its replay (`pearlc_arm.py`, `verify.py`) |
| `exhaustion/` | the salt-to-deadline window of `pearl-c-sm120-v1-h1` / `-h2` on node 2's cards, and its audit |
| `pouw_pearl_c_host.py`, `pouw_pearl_c_device.py` | the vLLM device path as it stood at d9820913f, with native BLAKE3 keys and the -h1, -h2 and -h3 schemes |

The BLAKE3 in core (`verity.ml.crypto.blake3`), Pearl-C4's BLAKE3 seeds at -h0 and the frozen backends' CUDA
hashing (`verity/ml/kernels/hash_gpu/`) stayed where they were.
