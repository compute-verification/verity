# What Commit adds to served vLLM

Plain served vLLM (`stock`) against the same engine committing every value of every step, interleaved rep by rep through
the shared harness (`benchmarks/harness`): throughput, time to first token, and per-token latency (p50 and p99), per model
and batch, under continuous batching with staggered arrivals, greedy and seeded stochastic.

- `arm.py`: the two Commit arms, `served-commit` (values retained in host memory, each run's linkage checked) and
  `served-commit-noretain` (authentication state only; openings by replay). Its docstring has what a run does.
- `run.sh`: `verity bench served_commit`, one spec's sweep in one process. Its header has the recorded form.
- The specs: `benchmarks/specs/qwen3-8b-bf16-h200-served.toml` (the default) and
  `benchmarks/specs/llama32-1b-bf16-h200-served.toml`, each on the sweep `benchmarks/specs/served.toml` (B1, B8, B32 and
  B64, 512-token prompts, mixed traffic, greedy and stochastic).

The engines are vLLM's serving defaults (`serving_defaults`: torch.compile, full and piecewise CUDA graphs, no batch
invariance), and each arm is gated token for token against `stock` before anything is timed. The acquisition under CUDA
graphs is `integrations/vllm/verity_vllm/acquire/sources/served_compiled_source.py`.
