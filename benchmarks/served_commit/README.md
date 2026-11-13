# What Commit adds to served vLLM

This directory runs the measurements: plain served vLLM (`stock`) against the same engine committing every value of every
step, interleaved rep by rep through the shared harness (`benchmarks/harness`): throughput, time to first token, and
per-token latency (p50 and p99), per model and batch, under continuous batching with staggered arrivals, greedy and seeded
stochastic. It renders no tables.

- `arm.py`: the arms' base, `ServedCommitArm`, which commits under `rows-sha512`. `prof_arm.py`'s rows arms and the
  harness's `commit` arms measure the Commit. Its docstring has what a run does, and which runs its gate accepts without a
  root (declared warm-ups, named in the verdict).
- `served_arm.py` and `served_arm.sh`: the served arm, the Commit at 4 KiB rows against compiled and eager stock in one run
  (the specs `*-rtx-served-arm`). With `PIN_BUILD` (a Build of record's `build.json`) every arm compiles the Build's
  kernels and `out/pin_check.txt` says so, which a counted compiled ratio needs. `tests/test_served_arm.py` tests both over
  stand-ins for the arms they build on.
- `run.sh`: `verity bench served_commit`, one spec's sweep in one process; `SPEC` names the spec.
- `tp_decode.py` and `tp_decode.sh`: one Commit arm against stock in one TP engine, eager (GLM-5.3 TP8).
- Measurement: `prof.sh` runs a sweep with the arms of `prof_arm.py` (roots and plans per run, native drain, rows-sha512,
  serving on a high-priority stream over the worker's hashing, NVTX ranges; `-base` and `-batch`
  arms turn the decode host-cost changes off, or all but the batched captured copies; `-ring8` and `-w64` arms size the
  staging ring at 32 windows of 256 or 64 MiB; `-cap` arms hash in launches of at most `PROF_ROWS_CAP` CTAs while the
  ring has room; `-4k` arms hash 4 KiB rows, and `-layer` arms acquire only each layer's residual stream, K and V; no
  scheme's roots), optionally under nsys,
  summarized by `nsys_summary.py`, and with `PIN_BUILD` under a Build of record's compile artifact (`served-pin` seeds one
  fresh Inductor dir for every arm and checks it after); `rows_floor.py` (`rows_floor.sh`) times the rows-sha512 kernels
  alone and beside a busy stream; `gpu_tests.sh` runs the committer's GPU byte-identity tests on a node;
  `granularity.py` lists, from a Build's request Programs, the Call boundaries a per-block or per-layer proof block would
  keep, with their bytes and hashing time (estimates; no scheme).
- `hash_prof.py` and `hash_prof.sh`: one step's rows-sha512 hashing as the windowed worker calls it, at T tokens a step, in
  the worker's form (held leaves folded many windows at once, levels copied on a copy stream) and the per-window form before
  it, with the kernels' knobs, under nsys and ncu.
- The specs: the served arm's, `benchmarks/harness/specs/*-rtx-served-arm*.toml` (one RTX PRO 6000 each, on the width sweep).
  `run.sh` and `prof.sh` take any served spec as `SPEC`, with its arms, sweep and reps unless their arguments name others.

The engines are vLLM's serving defaults (`serving_defaults`: torch.compile, full and piecewise CUDA graphs, no batch
invariance), and each arm is gated token for token against `stock` before anything is timed. The acquisition under CUDA
graphs is `adapters/vllm/verity_vllm/acquire/sources/served_compiled_source.py`.

## Interface

- **Provides:** the arms the shared harness runs (`arm.py`, `served_arm.py`, `prof_arm.py`), the tensor-parallel decode
  step (`tp_decode.py`) and the scripts that launch and profile them.
- **Loaded by:** the harness, through the specs that name `served_commit.served_arm`'s arms, and `benchmarks/moe_align`,
  which imports `tp_decode`.
- **Loads:** the vLLM integration's `commit`, `pipeline`, `engine`, `acquire`, `correspondence`, `query` and `protocols`
  packages; `verity.circuits`, `verity.definitions`, `verity.commitments` and `verity.definitions.kernels`;
  `benchmarks/harness` (`verity_bench`); torch and vLLM.
