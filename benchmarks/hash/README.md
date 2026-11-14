# benchmarks/hash

Measurements of GPU code for PoUW's served linear, each kernel held bit for bit to its CPU reference or Definition before
anything is timed. Its commitments' hash is SHA-512 with its whole output (hm96 leaves or keyed leaves in frame-v3-sha512
trees, "One hash" in AGENTS.md); nothing here changes a committed format, and nothing a verifier runs reads this code.

| Directory | What it holds |
|---|---|
| `dither/` | PoUW's H1 receipts served in compiled vLLM: the one-stream linear (`h1o.cu`) whose noised MMAs serve y, with the peel z = A Bᵀ from the receipted words (H1V3); the A tables (`h1c.cu`); J10-R's per-call-noise units (`h1u.cu`, `j10r*.py`); each kernel gated against the credit's H1V3 reference (`h1ogate.py`), and the served overhead against stock vLLM int8, paired rep by rep (`sbench.py`, run on a leased GPU by `sbench.sh`) |
| `exactlin/` | the int8-exact linear as two Triton kernels, the vLLM integration's (`verity_vllm.engine.int8_fused`: the exact quantizer, and the GEMM with the exact epilogue and per-card configs; no copy of a kernel or a table here, only the three Triton helpers `dither/dither_a.py` inlines, in `exactlin.py`), gated bit-exact against `int8_exact` (`kgate.py`) and the int8 Definitions' replay evaluators (`replay.py`), timed against stock vLLM int8 at kernel level (`kbench.py`, `ksweep.py`) and served (`served.sh`, through `dither/sbench.py`'s exact and exact-fused arms, one engine, after the kernel gate), with the served y checked bit for bit by the vLLM integration's `int8_exact_gpu` (`values.py`), the per-step kernel table (`steps.py`) and the share of a served step PoUW doesn't credit (`share.py`) |
| `maxtput/` | a Build of record for a model that has none: the served match alone, on one RTX PRO 6000 (`build_job.sh`) |

Each script's header says how to run it, from the source tree's root, and on what hardware. They share one layout: a
run writes its results to `$RESEARCH_RUN_DIR/out` (`OUT` where a header names it) and reads the files it is given, a
build's libraries or a source archive, from `$RESEARCH_RUN_DIR/inputs` or the directory a header's variable names. GPU
work runs under a GPU lease (`gpu-lease`, owned by `GPU_LEASE_WHO`).

## Interface

What code outside `benchmarks/hash` loads. Nothing else here is an interface: any other file may change or go without a
word to another directory. The vLLM integration loads nothing from here.

| File | What is loaded | Loaded by |
|---|---|---|
| `dither/h1u.cu` | the source, compiled by path with the caller's flags, and the library's entry points `kh_digits`, `kh_words`, `kh_seeds`, `kh_seeds_at`, `kh_seeds_j10r`, `kh_unit`, `kh_units_multi`, `kh_units_multi_j10r`, `kh_units_multi_j10r_sm`, `kh_ucall_bytes`, `kh_ucallr_bytes`, `kh_smc_words`, `kh_j10r_row_fault_word`, `kh_lanes` | `benchmarks/pouw/served_h1` (`producer.py`, `partials.py`, and the builds in `commit_legs.sh`, `j10r_gpu.sh` and `producer_gpu.sh`); `benchmarks/torch_gpu/j10r_sm90` (`job.sh`, `units.py`) |
| `dither/j10r_ws.py` | `flags`, `workspace_bytes`, `JW`: the workspace a build of `h1u.cu` takes | `benchmarks/pouw/served_h1/producer.py`; `benchmarks/torch_gpu/j10r_sm90/units.py` |
| `dither/pin.sh` | run as a script: the int8 pin in a node's model cache | `benchmarks/harness/setup/int8_pin.sh` |
| `dither/numabind.py` | run as a script, from a copy sent with a job | `benchmarks/pouw/pearl_c_sm120/card.sh`; `benchmarks/pouw/pearl_c_vllm/served_step.sh` |
| `exactlin/kbench.py` | `timed`, `copies_for`, `nvml`, `ITEM_MS` | `benchmarks/torch_gpu/s8_linear` |
| `exactlin/kgate.py` | `MODELS` | `benchmarks/torch_gpu/s8_linear` |

Tests of `benchmarks/pouw` import `dither/dither_one.py`, `dither_a.py`, `j10r_units.py` and `forms.py` by name, and its
suite declares them and what they import as inputs. A test's import fixes a file's path, nothing more.
