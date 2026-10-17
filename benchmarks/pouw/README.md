# PoUW benchmarks

The measuring side of `verity/core/protocols/pouw`: Pearl-C's kernels on the RTX PRO 6000 Blackwell (sm_120), their
harness, the served vLLM path and its audits, and the CPU accounting. How a kernel is designed, gated, timed and landed is
the kernel-engineering skill (`.agents/skills/kernel-engineering/SKILL.md`).

| Path | Runs on | What |
|---|---|---|
| `harness/` | RTX PRO 6000 | One harness for every PoUW prover arm and plain GEMM baseline: each gated at the exact shape and flags it is timed at, then timed the same way (its own README) |
| `gpu_kit/` | RTX PRO 6000 | The timing protocol for a standalone `.cu` bench, as code: `bench.cuh` (0xA5 poisoning and its no-write control, interleaved items with NVML clocks, the dump format), `sass_gate.py` (the timed loop's opcode counts, no spills, a live sink, bank conflicts) and `compare.py` (the twin compare) |
| `opcode_probe/` | RTX PRO 6000 | One instruction class's rate a probe, gated by `gpu_kit/sass_gate.py` before it is timed, written as a census line in the machine model's schema (`run.sh`; how a probe is written is its README) |
| `tensor_kinds/` | RTX PRO 6000 | Every native MMA form's joint issue coefficient (fitted beside an IADD3+FFMA lane, which issues at 1 a SMSP-cycle alone, and beside IADD3 alone, in the warp-specialized and interleaved schedules), its issue price an output slot against IMMA s8 m16n8k32's (whether the issue form of the tensor theorem is tight), and each form's worst load on VF3e's joint rows JI3e_0/1/2 (`run.sh` under `gpu-lease 1`, `tensor_kinds.py` the fit); `tensor_kinds.py refit OUT...` prints the machine model's JI3n rows (VF3n-16) from the runs; `search.sh` (M1c) times the forms that set them over schedules (lane op, register layout, warps a SMSP, placement), and `refit OUT... --search S...` prints JI3o (VF3o-16) with the schedule margin measured |
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
| `h_falsify/` | RTX PRO 6000 | The Dither transcript's zero instance (R 0/1, B̃ periodic in U): the tiled reference and its duo receipt (`ref.py`), and honest, zero-IMMA and Four Russians arms on `gpu_kit` (`hfalsify.cu`) |
| `qstar/` | RTX PRO 6000 | q*, the fewest scalar instructions a correct fresh Dither word costs: dp4a, bit planes, Four Russians in shared memory and in L2, each against IMMA, alone and beside it in one kernel or two streams (`qstar.cu`); `count.py` counts each timed loop's instructions a word and gates its pinned histogram (`expect.json`); `run.sh` is the `research run` job |
| `kernels/` | RTX 4090 | NCP's native u8xs8 route-U GEMM and Pearl's e4m3 GEMM on the `ada` device (`pouw_gemm.cu`), JIT-built as a torch extension |
| `filler/` | RTX PRO 6000 | The DRAM-free filler beside the decode step, its arms and its locked scorer, one suite (below) |
| `floored_gemm/`, `masked_gemm/`, `edited_gemm/`, `hashed_gemm/` | RTX PRO 6000 | The noised GEMMs, from the floored int8 grid to the adopted edited GEMM with its commitments hashed in its own CTAs (below) |
| `commit_gpu/` | RTX PRO 6000 | The GPU hm96-sha512 committer, its small path beside a GEMM, and its byte gate against `verity.ml.commitments` (`reference.py`) |
| `stage_cost/`, `stage_ovl/` | RTX PRO 6000 | What the stage coins and the deferred edit cost at Llama-3.1-8B's shapes, and their overlap in prefill with the committer |
| `vllm_bench.py` | RTX 4090 | Overhead against plain vLLM for BF16, stock FP8, W7A7, NCP and Pearl by wrapping every `LinearBase` layer's apply; `pearl_c_vllm/` reads its prompts and model lookup |
| `gamma.py` | CPU | γ accounting: each scheme's honest reference priced term by term under W1, split into checked and unchecked work |
| `pearlc_census.py` | CPU | TT_OUT's falsification census for Pearl-C's ticket chain, bit-exact, on the H100's atom or sm_120's (`--atom`) |
| `machine_model/` | CPU | The sm_120 machine model's price vectors (VF … VF3l-16): the LP, the rows, census and attack families as data, and `machine-model solve NAME` (its own README, and its own suite) |
| `exact_kit/` | CPU | `verity-exact-kit`, the theory and referee scripts' primitives (F_2 and F_p linear algebra, Möbius coefficients, kernel searches, SAT) and their one check registry with a JSON verdict; no theory, no folds (its own README) |
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

## Where the GEMM kernels and the committer live

Each has one copy; a later kernel includes an earlier one rather than copying it, so change the file a row names. Each
directory's `*.sh` builds and runs it through `research run`, and each `*ref.py` judges its dumps on the CPU.

| Tool | Files | Includes |
|---|---|---|
| The served GEMM on the floored int8 grid, its receipt kernels (duo, DUOS, tb) and their reference | `floored_gemm/fgemm.cu`, `ref.py`, `run.sh`, `tb_run.sh` | `masked_gemm/sha_noise.cuh` |
| SHA-512 counter-mode noise, as one generator and as separate passes | `masked_gemm/sha_noise.cuh`, `sha_pass.cuh` | `pearl_c/sha512.cuh` |
| The masked GEMM (full and low rank), the first edited GEMM (noise from {±5, ±7}), its noise expanded from σ, P·V's noise, the one-side peel | `masked_gemm/mgemm.cu`, `egemm.cu`, `sgemm.cu`, `pvgemm.cu`, `lpgemm.cu`, each with its `*ref.py`, `*run.sh` and `*table.py` | `floored_gemm/fgemm.cu` |
| The adopted design's edited GEMM (decision 29), its receipt folds and its profile | `edited_gemm/egemm.cu`, `egemm_ref.py`, `prof.cu`, `run.sh`, `profile.sh` | `masked_gemm/lpgemm.cu` |
| That GEMM with its windowed receipts and X's tile leaves hashed in its CTAs | `hashed_gemm/hgemm.cu`, `hgref.py`, `segment_e2e.py`, `run.sh` | `edited_gemm/egemm.cu`, `commit_gpu/commit_gpu.cuh` |
| The committer: hm96-sha512 row leaves in frame-v3-sha512 trees | `commit_gpu/commit_gpu.cuh`, `sha512.cuh`, `commit_gpu.cu`, `reference.py`, `gpu.sh` | |
| Stage coins and the deferred edit | `stage_cost/stage.cu`, `prep.py`, `report.py`, `run.sh` | `filler/dcap.cu`, `served_zk/hm96_rows.cu` |
| Their overlap in prefill, with the committer's small path, and the edited arms (EGEMM's GEMM with windowed receipts) | `stage_ovl/ovl.cu`, `eg.cu`, `eg.h`, `eref.py`, `report.py`, `run.sh` | `stage_cost/stage.cu`, `commit_gpu/commit_gpu.cuh`, `edited_gemm/egemm.cu` |

`edited_gemm/egemm_ref.py` imports the E2E slice's `derive.py` from `e2e_slice/` beside it, which lands in its own PR;
`edited_gemm/receipt_folds_r9.py` is P2-FF r9's reference as recorded (art:91c156a1…), held by its sha256, so leave it
unedited.
