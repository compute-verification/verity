# Benchmark specs

The specs and sweeps `verity-bench` (`benchmarks/harness`) runs, one TOML file each. The harness's README is the contract:
every key a spec may hold, how a point is timed, and what a result says. This file says what each one is for.

- A **spec** (`verity-bench/vllm-spec/v0`) is one measurement: a preset (the model at a pinned revision or a node-local
  checkpoint with its digest, its precision, the GPU, the vLLM pin and the engine settings), the arms timed against each
  other, the sweep it runs over, its reps, and its admission. Every ratio is over its baseline: `stock-fp8` when it runs,
  else `stock`, else its first arm.
- A **sweep** (`verity-bench/vllm-sweep/v0`) is the points a spec runs: the product of its batches, prompt lengths,
  traffic shapes and samplings. A spec names one by its id.

~~~text
verity-bench plan SPEC     # each point's identity, and its passing run if it has one
verity-bench run SPEC      # one recorded run for each point without a passing run (--direct NODE: on that GPU node)
verity-bench check SPEC    # exit 0 only when every point has a passing run
~~~

SPEC is a spec's id or its path. `tests/` checks that every pinned preset resolves as the vLLM integration's engine spec to
its checkpoint pin, and that every spec and sweep here has its line below.

## Stock serving

- `llama31-8b-fp8-1gpu`: Llama-3.1-8B in BF16 against FP8 (`stock`, `stock-fp8`), on one RTX PRO 6000.
- `llama31-8b-fp8-1gpu-compiled-full`: stock FP8 as vLLM serves it (compiled) against the batch-invariant eager stock
  every eager ratio is over, on the full sweep: the factor that restates an eager ratio as a served one.

## The Commit (sampled proofs)

- `llama31-8b-fp8-1gpu-commit-served`: the Commit (`rows-sha512`) against stock FP8, eager and compiled, each against its
  own stock.
- `llama31-8b-fp8-1gpu-commit-served-width`: the compiled Commit over compiled stock FP8, from batch 1 to 32.
- `llama31-8b-fp8-1gpu-commit-served-width-h200`: the same on one H200.
- `llama31-8b-fp8-1gpu-commit-served-width-b200`: the same on one B200.
- `llama31-8b-fp8-rtx-served-arm`: `served_commit`'s served arm on Llama-3.1-8B FP8, compiled and eager, against stock
  in both.
- `llama32-1b-bf16-rtx-served-arm`: the same on Llama-3.2-1B in BF16.
- `qwen3-8b-bf16-rtx-served-arm`: the compiled served arm on Qwen3-8B in BF16.
- `qwen3-8b-bf16-rtx-served-arm-eager`: the eager served arm on Qwen3-8B, in a run of its own: four Qwen3-8B engines
  don't fit one GPU beside the Commit's buffers.

## PoUW

- `llama31-8b-1gpu-pouw-h1`: H1 with fx1 on the eager exact-int8 engine, against that engine without PoUW.
- `llama31-8b-1gpu-pouw-h1-served`: the same, with stock FP8 as vLLM serves it as the baseline.
- `llama31-8b-1gpu-pouw-protocol-served`: Pearl-C against batch-invariant eager and compiled stock FP8, in one run.
- `llama31-8b-1gpu-pouw-protocol-served-b32`: the same at batch 32 alone, with 6 reps.
- `llama31-8b-1gpu-pearl-c-whole-served`: Pearl-C's whole-step graph arm beside the protocol arm and both stocks.
- `llama31-8b-1gpu-pearl-c-whole-served-h200`: the whole-step arm on one H200, against compiled stock FP8.

## PoUS's bandwidth audit

- `llama31-8b-fp8-1gpu-bandwidth`: the bandwidth audit's encoded, receipted stack against its plain stack and stock FP8.
- `llama31-8b-fp8-1gpu-bandwidth-served-r6`: the same with compiled stock FP8 too, at 6 reps.
- `llama31-8b-fp8-1gpu-bandwidth-ablation`: where the audit's time goes, by part and with the top-up off
  (`verity-bench breakdown`).

## Sweeps

- `quick`: batches 8 and 32, 512-token prompts, mixed traffic.
- `b32`: batch 32 alone, 512-token prompts, mixed traffic.
- `width`: batches 1, 8 and 32, 512-token prompts, mixed traffic.
- `served`: batches 1, 8, 32 and 64, 512-token prompts, mixed traffic, greedy and stochastic sampling.
- `full`: batches 1, 8, 32 and 128, 512- and 8,192-token prompts, decode-heavy, prefill-heavy and mixed traffic.
