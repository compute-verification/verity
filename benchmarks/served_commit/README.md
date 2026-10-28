# What Commit adds to served vLLM

This directory runs the measurements: plain served vLLM (`stock`) against the same engine committing every value of every
step, interleaved rep by rep through the shared harness (`benchmarks/harness`): throughput, time to first token, and
per-token latency (p50 and p99), per model and batch, under continuous batching with staggered arrivals, greedy and seeded
stochastic. It renders no tables: `benchmarks/g2/` does, from the evidence store (`curve.py` the overhead curve over every
labeled run, `served.py` the served arm's runs).

- `arm.py`: the arms' base, `ServedCommitArm`, whose `install` refuses its default scheme, `vllm-v1` (SHA-256), so an
  arm that runs names `rows-sha512`. `prof_arm.py`'s rows arms and the harness's `commit` arms measure the Commit. Its
  docstring has what a run does, and which runs its gate accepts without a root (declared warm-ups, named in the verdict).
- `served_arm.py` and `served_arm.sh`: the served arm, the Commit at 4 KiB rows against compiled and eager stock in one run
  (the specs `*-rtx-served-arm`). With `PIN_BUILD` (a Build of record's `build.json`) every arm compiles the Build's
  kernels and `out/pin_check.txt` says so, which a counted compiled ratio needs.
- `run.sh`: `verity bench served_commit`, one spec's sweep in one process; `SPEC` names the spec.
- `tp_decode.py` and `tp_decode.sh`: one Commit arm against stock in one TP engine, eager (GLM-5.3 TP8).
- Measurement: `prof.sh` runs a sweep with the arms of `prof_arm.py` (roots and plans per run, native drain, rows-sha512,
  serving on a high-priority stream over the worker's hashing, NVTX ranges; `-base` and `-batch`
  arms turn the decode host-cost changes off, or all but the batched captured copies; `-ring8` and `-w64` arms size the
  staging ring at 32 windows of 256 or 64 MiB; `-cap` arms hash in launches of at most `PROF_ROWS_CAP` CTAs while the
  ring has room; `-4k` arms hash 4 KiB rows, and `-layer` arms acquire only each layer's residual stream, K and V; no
  scheme's roots), optionally under nsys,
  summarized by `nsys_summary.py`, and with `PIN_BUILD` under a Build of record's compile artifact (`served-pin` seeds one
  fresh Inductor dir for every arm and checks it after); `gather_bench.py` times the
  gather kernel alone; `rows_floor.py` (`rows_floor.sh`) the rows-sha512 kernels alone and beside a busy stream;
  `hash_prof.py` (`hash_prof.sh`) one step's rows-sha512 hashing as the windowed worker calls it, at T tokens a step, in the
  worker's form (held leaves folded many windows at once, levels copied on a copy stream) and the per-window form before it, with
  the kernels' knobs, under nsys and ncu; `gpu_tests.sh` runs the committer's GPU byte-identity tests on a node;
  `granularity.py` lists, from a Build's request Programs, the Call boundaries a per-block or per-layer proof block would
  keep, with their bytes and hashing time, and `layout_bytes.py` a run's committed bytes by buffer category (estimates; no
  scheme).
- The specs: the served arm's, `benchmarks/specs/*-rtx-served-arm*.toml` (one RTX PRO 6000 each, on the width sweep).
  `run.sh` and `prof.sh` take any served spec as `SPEC`, with its arms, sweep and reps unless their arguments name others.

The engines are vLLM's serving defaults (`serving_defaults`: torch.compile, full and piecewise CUDA graphs, no batch
invariance), and each arm is gated token for token against `stock` before anything is timed. The acquisition under CUDA
graphs is `integrations/vllm/verity_vllm/acquire/sources/served_compiled_source.py`.
