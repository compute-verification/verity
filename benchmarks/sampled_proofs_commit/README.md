# Sampled proofs' Commit in vLLM serving

`bench_arm.py` is an arm of the shared vLLM benchmark (`benchmarks/harness`, `verity-bench vllm`): serving with sampled
proofs' Commit collector installed (`sp-commit`), gated against the same engine with nothing installed (`stock-fp8`).
Its docstring says what the collector commits and what it leaves out against the Commit of record (no Build, no
FlashAttention hidden-value tap, roots-only retention).

~~~sh
verity-bench vllm --preset llama31-8b-fp8-1gpu --model-path MODEL --eager \
  --arms stock-fp8,sp-commit --arm-module benchmarks.sampled_proofs_commit.bench_arm:COMMIT --sweep quick --out OUT
~~~

The collector's hooks wrap module forwards, which CUDA graphs would skip, so the arm runs eager. `run.sh` is the recorded
run on a node (its header has the `research run` line): it builds the collector's JIT extensions outside the lease and
runs the sweep on one leased GPU.

Host memory: the collector stages each step's tree levels and window leaves in a pinned pool, about 0.4× the step's bytes,
and a step is at most `max_num_batched_tokens` (16384) tokens, about 87 GB with Llama-3.1-8B. The arm reuses the pool's
blocks by best fit in eighth-octave size classes, so the pool peaks near 47 GB over the full sweep. Over
`VERITY_SP_ARM_HOST_GB` (default 80) it refuses further steps by name and the gate fails, so the run's `--mem-gb` must
cover the cap and the engine (100 in `run.sh`'s line).
