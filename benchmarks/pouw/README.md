# PoUW benchmarks

The measuring side of `verity/core/protocols/pouw`: Pearl-C's kernels on the RTX PRO 6000 Blackwell (sm_120), their
harness, the served vLLM path and its audits, and the CPU accounting. How a kernel is designed, gated, timed and landed is
the kernel-engineering skill (`.agents/skills/kernel-engineering/SKILL.md`).

| Path | Runs on | What |
|---|---|---|
| `harness/` | RTX PRO 6000 | One harness for every PoUW prover arm and plain GEMM baseline: each gated at the exact shape and flags it is timed at, then timed the same way (its own README) |
| `gpu_kit/` | RTX PRO 6000 | The timing protocol for a standalone `.cu` bench, as code: `bench.cuh` (0xA5 poisoning and its no-write control, interleaved items with NVML clocks, the dump format), `sass_gate.py` (the timed loop's opcode counts, no spills, a live sink, bank conflicts) and `compare.py` (the twin compare) |
| `opcode_probe/` | RTX PRO 6000 | One instruction class's rate a probe, gated by `gpu_kit/sass_gate.py` before it is timed, written as a census line in the machine model's schema (`run.sh`; how a probe is written is its README) |
| `fp_exact/` | RTX PRO 6000 | Whether FP8 E4M3 and FP4 (MXFP4, NVFP4, 2:4-sparse) MMA chains on restricted grids are exact integer arithmetic: every output against an exact int64 sum, over three accumulation orders, replayed through the silicon models, beside the MMAs' rates against IMMA (`run.sh`) |
| `pearl_c_sm120/` | RTX PRO 6000 | `pearl-c-sm120-v1` (G = 4) and the unpromoted record on sm_120a: `build.sh` builds the cubin and refuses it unless `sass_gate.py` passes; `fixture.py` writes the check; `hash_fast.cuh` computes `hash_h3.cuh`'s row digests and tile leaves in two launches, a thread per 1 KiB segment across every row or tile and then a thread per row or tile for the fold, which `run.py` uses unless `PEARLC_SHA=ref`; `run.py` checks every buffer bit for bit on the expected UUID, then times diagnostic phases |
| `pearl_c/` | CPU | Pearl-C's hashing format -h3 on the device (`hash_h3.cuh`, which `pearl_c_sm120.cu` includes), its host build (`hash_host.cpp`, which also builds `pearl_c_sm120/hash_fast.cuh`, and `simt_twin.py`, for byte tests without a GPU) and `fake_cuda.c`, the stand-in for libcuda that `run.py`'s dry run uses. BLAKE3 and the H100 -h1 bench are in `archive/pouw-blake3/` |
| `pearl_c_vllm/` | RTX PRO 6000 | Pearl-C on every linear of a served vLLM model: `e2e.py` (gates, timing against stock FP8, retained passes, smoke), `window.sh` (the timed window), `verify.sh` (the reference replay), and the `verity-bench` arm (`bench_arm.py`) |
| `served_zk/` | CPU | A served audit window committed in the ZK statement's format from a retained pass, and judged through the proof service (`served_commit.py`, `job.sh`) |
| `hidden_zk/` | CPU | Pearl-C4's hidden-check units under C-Flock `--zk`, proved in Rust and verified by the Lean verifier (`job.sh`) |
| `pearl_c4/` | RTX PRO 6000 | Pearl-C4 (NVFP4 and MXFP4) and its harness arms |
| `fp4_nobits/` | RTX PRO 6000 | The measurement of premise NB (NOBITS): a `kind::mxf4nvf4` MMA's FP32 accumulator holds the exact integer but, for a nonzero word, never the committed s32 word's bits, whether it starts at 0 or at the binade constant; and the cheapest exact extraction is one F2I or one IADD3 a word (SASS-counted), against the int8 IMMA whose accumulator is the word; `run.sh` under `gpu-lease 1`, clocks read |
| `nvfp4_sm120/` | RTX PRO 6000 | The plain NVFP4 GEMM as harness baseline entries (`nvf4_plain.cu` on `mainloop_nvf4.cuh`), and its standalone benches |
| `pouw_hash/` | RTX PRO 6000 | The PoUW hash kernels' bench and check (`run.sh`, `check.py`) |
| `kernels/` | RTX 4090 | NCP's native u8xs8 route-U GEMM and Pearl's e4m3 GEMM on the `ada` device (`pouw_gemm.cu`), JIT-built as a torch extension |
| `filler/` | RTX PRO 6000 | The DRAM-free filler beside the decode step, its arms and its locked scorer, one suite (below) |
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

## Where the filler's tools live

Each has one copy, in `filler/`; change it there. A new arm is a class in an arm module, and a new tile or fold is a flag
of `pfill.cuh`, not a copy of the kernel. Its tests are one suite: `uv run tools/verity/check/suites.py benchmarks/pouw/filler`.

| Tool | Files |
|---|---|
| The decode step's driver and the filler ABI (`dfill_k`, `DFill`): pfill's filler, or FILL v1's chained one with `-DDFILL_V1` | `dcap.cu`, `dcap.sh`, `dfill.sh`, `fill.sh`, `dfill_ref.py`, `share.py` |
| pfill, the proved filler: tiles `-DPFILL_BITS`, `-DPFILL_BU`, `-DPFILL_E2M1`; folds ram (NC 4), duo (NC 2), `-DPFILL_LI`, `-DPFILL_XIN13` | `pfill.cuh`, its host build `pfill_host.cpp` and Python twin `pfill_ref.py`, `pfill_probe.sh`, `tile_probe.sh` |
| The receipt folds' steps, key rules and pfill's lane starts, which `pfill.cuh` and `pfill_ref.py` call: change a fold there, never in the filler | `../receipts/` (`receipt_folds.cuh`, `receipt_folds.py`, `vectors.json`; its own README) |
| The locked scorer (filler-scorer/v5, SCORE-TP) and the arm contract with its baseline arms (`idle`, `fill-v1`) | `score.py`, `scoring.py`, `arms.py`, `dscore.cu`, `rounds.cuh` |
| The phase filler's runner, HC-SCHED's arms | `psched_score.py` on `pscore.cu`, `psched_arm.py`, `psched.sh`, `psched_probe.sh`, `psched_probe.py`, `psum.py` |
| The fused linears with filler warps, FILL's arms | `flin.cuh`, `flin.sh`, `fused_score.py` on `fscore.cu`, `fused_arm.py` |
| HC-PROVED's arms (the leader, `PFillBUDuoFPhase3`) and their runner over both drivers | `pfill_arm.py`, `pfill_fp_score.py`, `pfill_proved.py`, `pfill_phase_screen.py` |
| The filler in tensor-parallel decode's collectives | `comm/tpcomm.cu`, `comm/tpcomm.sh`, `comm/report.py` |
