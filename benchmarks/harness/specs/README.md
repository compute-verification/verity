# Benchmark specs

A spec is one measurement written down so it runs the same way every time. Each file here is one of three kinds, named
by its `schema`, and its `id` is its file's name:

- `verity-bench/vllm-spec/v0`: a preset (model and checkpoint pin, precision, GPUs, vLLM pin), the `cards` it runs on,
  the `arms` timed against each other, its sweep, its reps and its admission. `[card_settings.SHORT]` is what one card
  needs in its place; `[refused]` says why a card is not listed. Every ratio is over the baseline arm.
- `verity-bench/vllm-sweep/v0`: the points a spec runs, the product of batches, prompt lengths and traffic shapes
  (`quick`, `b32`, `width`, `full`, `prefill`).
- `verity-bench/steady-spec/v0`: the proof server's rate at steady state, one card, a row per block type, each at its
  prover counts.

`verity-bench` ([`../README.md`](../README.md)) plans and runs them. SPEC is an id or a path, and `SPEC@CARD` names the card.

## Try it

~~~sh
ls benchmarks/harness/specs/*.toml
uv run verity-bench plan llama32-1b-bf16-served-arm@h200 --local-only
~~~

The listing is every spec and sweep. `plan` prints each point's identity and its passing run, with no GPU.
