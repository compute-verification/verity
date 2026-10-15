# PoUW benchmarks

The measuring side of `verity/core/protocols/pouw`: Pearl-C's kernels on the RTX PRO 6000 Blackwell (sm_120), their
harness, the served vLLM path and its audits, and the CPU accounting. How a kernel is designed, gated, timed and landed is
the kernel-engineering skill (`.agents/skills/kernel-engineering/SKILL.md`).

| Path | Runs on | What |
|---|---|---|
| `harness/` | RTX PRO 6000 | One harness for every PoUW prover arm and plain GEMM baseline: each gated at the exact shape and flags it is timed at, then timed the same way (its own README) |
| `pearl_c_sm120/` | RTX PRO 6000 | `pearl-c-sm120-v1` (G = 4) and the unpromoted record on sm_120a: `build.sh` builds the cubin and refuses it unless `sass_gate.py` passes; `fixture.py` writes the check; `run.py` checks every buffer bit for bit on the expected UUID, then times diagnostic phases |
| `pearl_c/` | CPU | Pearl-C's hashing format -h3 on the device (`hash_h3.cuh`, which `pearl_c_sm120.cu` includes), its host build (`hash_host.cpp`, `simt_twin.py`, for byte tests without a GPU) and `fake_cuda.c`, the stand-in for libcuda that `run.py`'s dry run uses. BLAKE3 and the H100 -h1 bench are in `archive/pouw-blake3/` |
| `pearl_c_vllm/` | RTX PRO 6000 | Pearl-C on every linear of a served vLLM model: `e2e.py` (gates, timing against stock FP8, retained passes, smoke), `window.sh` (the timed window), `verify.sh` (the reference replay), and the `verity-bench` arm (`bench_arm.py`) |
| `served_zk/` | CPU | A served audit window committed in the ZK statement's format from a retained pass, and judged through the proof service (`served_commit.py`, `job.sh`) |
| `hidden_zk/` | CPU | Pearl-C4's hidden-check units under C-Flock `--zk`, proved in Rust and verified by the Lean verifier (`job.sh`) |
| `pearl_c4/` | RTX PRO 6000 | Pearl-C4 (NVFP4 and MXFP4) and its harness arms |
| `nvfp4_sm120/` | RTX PRO 6000 | The plain NVFP4 GEMM as harness baseline entries (`nvf4_plain.cu` on `mainloop_nvf4.cuh`), and its standalone benches |
| `pouw_hash/` | RTX PRO 6000 | The PoUW hash kernels' bench and check (`run.sh`, `check.py`) |
| `rs_attacks/` | CPU, RTX PRO 6000 | Attacks on the committed prefix targets of (A + UVᵀ)(B + WZᵀ) with rank-r masks: `attacks.py` checks each route bit for bit and prices it (ledger and time), with the threshold r* (`check`, `table`); `prefix.cu` (`prefix.sh`, under `gpu-lease`) times the honest int8 step against the CUDA-core routes on one GPU |
| `kernels/` | RTX 4090 | NCP's native u8xs8 route-U GEMM and Pearl's e4m3 GEMM on the `ada` device (`pouw_gemm.cu`), JIT-built as a torch extension |
| `vllm_bench.py` | RTX 4090 | Overhead against plain vLLM for BF16, stock FP8, W7A7, NCP and Pearl by wrapping every `LinearBase` layer's apply; `pearl_c_vllm/` reads its prompts and model lookup |
| `gamma.py` | CPU | γ accounting: each scheme's honest reference priced term by term under W1, split into checked and unchecked work |
| `pearlc_census.py` | CPU | TT_OUT's falsification census for Pearl-C's ticket chain, bit-exact, on the H100's atom or sm_120's (`--atom`) |
| `workloads.json` | | The model and its linear shapes, the decode batches and prefill lengths, and the reference shapes (`gamma.py`) |
| `run_result.py`, `tool.py` | | The runs' result record (`research/result/v0.1`) and the store's tool declaration |

~~~bash
uv run python benchmarks/pouw/gamma.py      # CPU
~~~

Served claims (`pearl_c_vllm/`) are measured against graphed stock FP8: stock vLLM `quantization="fp8"` compiled with its
CUDA graphs, `e2e.py`'s ``fp8-graphs`` mode (``--fp8-graphs``).
