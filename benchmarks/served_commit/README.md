# What Commit adds to served vLLM

While vLLM serves, the vLLM adapter commits to the values it computes, so that an audit can later sample them. This
benchmark measures what that costs serving: plain served vLLM (`stock`) against the same engine committing every value of every step,
interleaved rep by rep through the shared harness ([`benchmarks/harness`](../harness/README.md)). It measures
throughput, time to first token, and per-token latency (p50 and p99), per model and batch, under continuous batching
with staggered arrivals, greedy and seeded stochastic.

The engines are vLLM's serving defaults (`serving_defaults`: torch.compile, full and piecewise CUDA graphs, no batch
invariance), and each arm is gated token for token against `stock` before anything is timed.

## Try it

On one RTX PRO 6000, with `VERITY_BENCH_VENV` naming the node's pinned-vLLM environment:

~~~sh
SPEC=qwen3-8b-bf16-rtx-served-arm verity bench served_commit --sweep quick --reps 1
~~~

This runs one served spec's sweep in one process, its arms interleaved, and writes the result and the run's log to
`$OUT` (else `$RESEARCH_RUN_DIR/out`). `SPEC` is any of the served arm's specs,
`benchmarks/harness/specs/*-rtx-served-arm*.toml`; arguments override the spec's arms, sweep and reps.

## Profiling

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

[`DESIGN.md`](DESIGN.md) says what each arm and script is, and what loads this directory.
