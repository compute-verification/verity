# What Commit adds to served vLLM

Plain served vLLM (`stock`) against the same engine committing every value of every step, interleaved rep by rep through
the shared harness (`benchmarks/harness`): throughput, time to first token, and per-token latency (p50 and p99), per model
and batch, under continuous batching with staggered arrivals, greedy and seeded stochastic.

- `arm.py`: the arms' base, and its two Commit arms, `served-commit` (values retained in host memory, each run's linkage
  checked) and `served-commit-noretain` (authentication state only; openings by replay). Those two, and every `prof_arm.py`
  arm that doesn't name `rows-sha512`, are `vllm-v1` (SHA-256), the record before the Commit's default became `rows-sha512`:
  their runs of record stay readable, and their `install` refuses a new one by name. `prof_arm.py`'s rows arms and the
  harness's `commit` arms measure the Commit. Its docstring has what a run does, and which runs its gate accepts without a
  root (declared warm-ups, named in the verdict).
- `served_arm.py` and `served_arm.sh`: G2's served arm, the Commit at 4 KiB rows against compiled and eager stock in one run
  (the specs `*-rtx-served-arm`), which `benchmarks/g2/served.py` renders. With `PIN_BUILD` (a Build of record's
  `build.json`) every arm compiles the Build's kernels and `out/pin_check.txt` says so, which a counted compiled ratio needs.
- `run.sh`: `verity bench served_commit`, one spec's sweep in one process. Its header has the recorded form.
- Measurement: `prof.sh` runs a sweep with the arms of `prof_arm.py` (roots and plans per run, native drain, rows-sha512,
  serving on a high-priority stream over the worker's hashing, the finalize profile, NVTX ranges; `-base` and `-batch`
  arms turn the decode host-cost changes off, or all but the batched captured copies; `-ring8` and `-w64` arms size the
  staging ring at 32 windows of 256 or 64 MiB; `-cap` arms hash in launches of at most `PROF_ROWS_CAP` blocks while the
  ring has room; `-4k` arms hash 4 KiB rows, and `-layer` arms acquire only each layer's residual stream, K and V; no
  scheme's roots), optionally under nsys,
  summarized by `nsys_summary.py`, and with `PIN_BUILD` under a Build of record's compile artifact (`served-pin` seeds one
  fresh Inductor dir for every arm and checks it after); `gather_bench.py` times the
  gather kernel alone; `gpu_tests.sh` runs the committer's GPU byte-identity tests on a node; `granularity.py` lists, from
  a Build's request Programs, the Call boundaries a per-block or per-layer proof block would keep, with their bytes and
  hashing time (an estimate; no scheme).
- The specs: `benchmarks/specs/qwen3-8b-bf16-h200-served.toml` (the default) and
  `benchmarks/specs/llama32-1b-bf16-h200-served.toml`, each on the sweep `benchmarks/specs/served.toml` (B1, B8, B32 and
  B64, 512-token prompts, mixed traffic, greedy and stochastic), stock against `served-commit-rows`.

The engines are vLLM's serving defaults (`serving_defaults`: torch.compile, full and piecewise CUDA graphs, no batch
invariance), and each arm is gated token for token against `stock` before anything is timed. The acquisition under CUDA
graphs is `integrations/vllm/verity_vllm/acquire/sources/served_compiled_source.py`.
