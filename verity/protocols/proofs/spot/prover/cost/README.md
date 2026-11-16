# prover/cost

SPOT proves only a random sample of a served workload's proof blocks. How the workload is cut into proof blocks, and how
many are sampled, decides both what an audit establishes and what it costs to commit, replay and prove. This directory is SPOT's cost model for one served workload under Flock. Each script plans one workload's sample
with the planner beside it (`../optimize.py`, `../candidates.py`, `../costfit.py`), from measured inputs under
`benchmarks/sampling_plan/`, each with its source and status, measured or assumed.

A plan gives the cost model fitted from proving measurements, the cuts the candidate generator finds, the frontier of ε
(at δ = 2^-40, against the workload's MACs) over the verification overhead B, the break-even granularity, and how the
plan moves with each input. It is a model, not a measurement.

## Try it

~~~sh
uv run python verity/protocols/proofs/spot/prover/cost/glm53.py
~~~

This needs nothing but the tree. It prints GLM-5.3 as served, from its proof-block census: the proof blocks, output
bits, ANDs, MACs and opened row bytes per prefill token, per decode token and per request.

## The workloads

| Script | What it plans |
| --- | --- |
| `int8_llama.py` | int8 Llama-3.1-8B, or Qwen3-8B (`--model qwen3-8b`), under each PoUW scheme |
| `served_window.py` | the served window (Llama-3.2-1B int8 on the H200), cut at each rung of a ladder, for the free-compute and exfiltration targets |
| `transformer.py` | the end-to-end demo's transformer, small or at the Phase 2 model's size |
| `h200.py` | the demo's plan on one H200 |
| `glm53.py` | GLM-5.3 as served, from its proof-block census |
| `cpu_budget.py` | a block's sample sized to a CPU proving budget |
| `recursion.py` | one recursive session's cost per monitoring period |

Each script's docstring says what it plans, what it writes and its flags.
