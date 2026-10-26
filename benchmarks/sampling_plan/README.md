# sampling_plan

The sampling plan for one served row under C-Flock, from the planner in `verity.core.protocols.sampled_proofs`
(`optimize`, `candidates`, `costfit`): the cost model fitted from proving measurements, the cuts the candidate generator
finds, the frontier of ε (at δ = 2^-40, against the row's MACs, W_ref's proxy) over the verification overhead B, the
break-even granularity, and how the plan moves with each input. A model, not a measurement: every input carries its
source and status, measured or assumed (`inputs.py`).

B is BUDGET's unit, GPU-seconds of overhead per plain GPU-second over a window (note:compute-accounting/
20261018T0349Z-draft-budget), and it is the serve-time total (ASK-25: everything is committed at serve time): every
scale's serve-time commitments under the PoUW scheme (`--scheme`: `floor`, the measured s8 A-operand floor alone;
`h1`, `h1-bf16`, `pearl-c`, each scheme's rows a token a layer) and the remote audit's store-block receipts, which every
plan pays; then proving and the verifier's CPU (at its price ratio) for the drawn units, the commitment of every unit's
outputs, replay of the layer-steps the draw meets (to build their witnesses), and the zero-work templates at their floor.

- `inputs.py`: the models (Llama-3.1-8B, Qwen3-8B); each scheme's serve-time rows a token a layer, with their leaf
  chains; the row's inputs (price ratio, rounds, throughput, replay and commitment speeds, the remote audit's receipts
  and passes, decode's step and its compression latency), each with a low and a
  high; the pieces' AND counts (#2391's pins, hm96's rows); and the observations the cost model is fitted from, C-Flock's
  zero-knowledge sessions on sm_120 (`benchmarks/one_stage/zk_cost.py`'s of s8 coordinates, and A8's of the remote
  bandwidth audit's opening unit, `BlockOpeningCheck_v1`, at twice their ANDs, its proving only), with the builds and
  hardware they key.
- `int8_llama.py`: the int8 Llama-3.1-8B row (vLLM's `cursor/vllm-int8-build-8350`). Its units are compute accounting's
  decision 106: every s8 dot checks its seed, derives its own noise and opens its rows in circuit. `--model qwen3-8b`
  runs the Qwen3-8B row the same way. A linear's coordinate may be cut into K-chunks that commit their running sums, and
  r × c tiles are offered so the width rule's price shows as a hint. It writes the fitted model (`cost-model.json`, a
  cost-model/v1 record), the serve-time commitments, the frontier, the plan at each budget, the tensors committed at
  more than one scale and what committing each once saves (`overlaps`), the block kinds' weights at each ε
  (`--weights`), the break-even table (the linears forced to each grain, per window, ε and commitment regime), the
  sensitivity (each cost group scaled, each input at its low and high) and the hints (`plan.json`), and prints the
  tables.

- `transformer.py`: the end-to-end demo's transformer (PRF-166), CMP-92's decoder block over one server-time block of
  serving, small (width 128, 2 heads, MLP 512, 2 layers, a byte vocabulary, context 32) or large (`--model large`: 100
  layers at width 8,192, 80.6 GB of INT8 linears and 13.4 GB of KV on one 96 GiB card; each dimension a flag). Every op
  is a proof unit of its Definition: the linears `ScaledMmS8Coordinate_v1`, attention `GemmCoordinate_v2`, the residual
  `Bf16Add_v1`, and CIR-100's ops (RMSNorm, the s8 quantizer, softmax, ReLU, the embedding, argmax) as stubs, each id
  `stub:`, until they land. The harm is faulty units, IntegrityPeriod's W. It writes the frontier of expected C-Flock
  cost (every serve-time commitment in it) against certified faulty units at δ, the chosen plan for `--target`, its
  certificate recomputed through `Stratified.harm_bound` and `count_bound` at δ = 1% and 1e-6, the pooled bound over a
  week, a month and a year on one server and on an assumed 10,000 (an assumed 10-minute block: neither is in the
  census), and the plan as the circuit record's templates with the `work:1` law and work table that draw its k
  (`handoff`). Its tests are this directory's suite.

```text
uv run python benchmarks/sampling_plan/int8_llama.py --out DIR [--model qwen3-8b] [--scheme pearl-c] [--weights]
uv run python benchmarks/sampling_plan/transformer.py --out DIR [--model large] [--target W] [--delta D]
```

- `cpu_budget.py`: the block's draw sized to a CPU proving budget (PRF-171). It reads `h200.py`'s `plan.json` (its
  templates, their n, S and proved ANDs), prices each template's statements (up to 8 drawn units each) from measured
  firewalled C-Flock `--zk` sessions on the Genoa proofs slots, and finds the least share whose statements fit
  `--slots` x `--hours`. It writes the per-template k and cost and the certified units per window and per period, pooled
  and each block, at δ = 1% and 1e-6 (`cpu-budget.json`, `cpu-budget.md`).

```text
uv run python benchmarks/sampling_plan/cpu_budget.py --plan PLAN.json --slots 17 --hours 2.5 --out DIR
```

- `cpu_budget.py`: the block's draw sized to a CPU proving budget (PRF-171). It reads `h200.py`'s `plan.json` (its
  templates, their n, S and proved ANDs), prices each template's statements (up to 8 drawn units each) from measured
  firewalled C-Flock `--zk` sessions on the Genoa proofs slots, and finds the least share whose statements fit
  `--slots` x `--hours`. It writes the per-template k and cost and the certified units per window and per period, pooled
  and each block, at δ = 1% and 1e-6 (`cpu-budget.json`, `cpu-budget.md`).

```text
uv run python benchmarks/sampling_plan/cpu_budget.py --plan PLAN.json --slots 17 --hours 2.5 --out DIR
```
