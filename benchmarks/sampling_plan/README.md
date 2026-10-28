# sampling_plan

The sampling plan for one served workload under C-Flock, from the planner in `verity.core.protocols.sampled_proofs`
(`optimize`, `candidates`, `costfit`): the cost model fitted from proving measurements, the cuts the candidate generator
finds, the frontier of ε (at δ = 2^-40, against the workload's MACs, W_ref's proxy) over the verification overhead B, the
break-even granularity, and how the plan moves with each input. A model, not a measurement: every input carries its
source and status, measured or assumed (`inputs.py`).

B is the budget model's unit, GPU-seconds of overhead per plain GPU-second over a window, and it is the serve-time
total (everything is committed at serve time): every
scale's serve-time commitments under the PoUW scheme (`--scheme`: `floor`, the measured s8 A-operand floor alone;
`h1`, `h1-bf16`, `pearl-c`, each scheme's rows a token a layer) and the store-block receipts of proof of useful
bandwidth (PoUB), which every plan pays; then proving and the verifier's CPU (at its price ratio) for the drawn units,
the commitment of every unit's outputs, replay of the layer-steps the draw meets (to build their witnesses), and the
zero-work subcircuits at their floor.

- `inputs.py`: the models (Llama-3.1-8B, Qwen3-8B); each scheme's serve-time rows a token a layer, with their leaf
  chains; the row's inputs (price ratio, rounds, throughput, replay and commitment speeds, PoUB's receipts
  and passes, decode's step and its compression latency), each with a low and a
  high; the pieces' AND counts (#2391's pins, hm96's rows); and the observations the cost model is fitted from, C-Flock's
  zero-knowledge sessions on sm_120 (`benchmarks/one_stage/zk_cost.py`'s of s8 coordinates, and A8's of PoUB's
  opening unit, `BlockOpeningCheck_v1`, at twice their ANDs, its proving only), with the builds and
  hardware they key.
- `int8_llama.py`: the int8 Llama-3.1-8B row (vLLM's `cursor/vllm-int8-build-8350`). Its units: every s8 dot checks its
  seed, derives its own noise and opens its rows in circuit. `--model qwen3-8b`
  runs the Qwen3-8B row the same way. A linear's coordinate may be cut into K-chunks that commit their running sums, and
  r × c tiles are offered, each a unit when its outputs fit a size class. It writes the fitted model (`cost-model.json`, a
  cost-model/v1 record), the serve-time commitments, the frontier, the plan at each budget, the tensors committed at
  more than one scale and what committing each once saves (`overlaps`), the weight classes' weights at each ε
  (`--weights`), the break-even table (the linears forced to each grain, per window, ε and commitment regime), the
  sensitivity (each cost group scaled, each input at its low and high) and the hints (`plan.json`), and prints the
  tables.

- `transformer.py`: the end-to-end demo's transformer (PRF-166), CMP-92's decoder block over one server block of
  serving, small (width 128, 2 heads, MLP 512, 2 layers, a byte vocabulary, context 32) or large (`--model large`:
  the Phase 2 model's 70 layers at width 8,192, 128 heads of 64 and an MLP of 32,768, 52.5 GiB of INT8 linears and
  35.0 GiB of KV for 1,023 sequences on one 96 GiB card, its requests a block from an estimated 195 ms step; each
  dimension a flag). Every op is a proof block of its Definition: the linears `ScaledMmS8Coordinate_v1` (or tiles of
  them), attention `GemmCoordinate_v2` on the BF16 step, the residual `Bf16Add_v1`, and the row ops (RMSNorm, the s8
  quantizer, softmax, ReLU, the embedding, argmax) as stubs, each id `stub:`, until they land, row ops whole, shared,
  chunked or recomputed. The harm is faulty ANDs, the same whatever the cut, with the faulty proof blocks (IntegrityPeriod's W)
  beside it. It writes the frontier of expected C-Flock cost (every serve-time commitment and the replay in it) against
  certified faulty ANDs at δ, the share of its ANDs the block's own GPU time buys, the chosen plan for `--share` (1e-3,
  the small model's knee, held fixed), its certificate recomputed through `Stratified.harm_bound` and `count_bound` at
  δ = 1% and 1e-6, the pooled bound over a week, a month and a year on one server and on an assumed 10,000 (an assumed
  20-minute block: neither is in the census), and the plan as the GPU loop registers it (`handoff`): one
  `IntegrityPeriod` sampling block a subcircuit (`subset:k`, its cap and K0), or the block as one sampling block under the `work:1` law.
  Its tests are this directory's suite.
- `h200.py`: the demo's plan on one H200: the frozen MQA shape (52 layers, width 16,384, 256 query heads of 64 and
  one KV head, MLP 65,536, context 32; batch and step are named stand-ins) on the demo's partition,
  one lockstep block of 31 decode steps. Each subcircuit is one `IntegrityPeriod` sampling block, one `--zk` session drawing
  `subset:k` over the tables `_cap` allows, and leads with what Lean proves: each window's K0 (IntegrityPeriodLog) and
  the pooled bound over any period (IntegrityHorizon) at δ = 1% and 1e-6. It prices each window's statements and drawn
  units, the replay and, beside them, the commitment of every unit's outputs; compares the static matmuls as 128-channel
  tiles (`--tiles`) and softmax and PV padded to the context (`--pad`); and gives the frontier over the share, the
  sampling block length in blocks that proves within its own GPU time, and the faulty ANDs (`Stratified.harm_bound`, not a Lean
  guarantee).
- `demo_a.py`: Demo A's frontier on one H200: plain INT8 weights, no P2 decode, a block of 16,384 requests at a
  6.35 s step, 32 steps a request, on the Definition map (`demos/h200/e2e_h200/plan.py`: every subcircuit a
  Definition C-Flock stages, at its measured ANDs and staged k_log) with PoUW's tiles as one more subcircuit at each K
  (J10-R's per-call lane units, `PouwH1LaneZy_v2` over v5, at their traced ANDs), and statements shared per subcircuit
  per sampling block. The harm is half skipped work (each unit's own ANDs), half leaked output bits, each half
  weighted at each budget by the least bound that budget certifies for it alone;
  each point reports `certify`'s `free_compute` and `leak` from its one draw at δ = 1% and 1e-6, the tile stratum's ε
  (`eps_at`), and the knee. The plan of record is proving within 1% of the sampling block's GPU time, for sampling
  blocks of one server block, the least that affords a plan, a week, a month and a year, with its draw a subcircuit
  and the commits beside it; then M3's uniform 8-unit draw against the block's GPU time, and each Definition's share
  of the ANDs against the step's GPU time.

```text
uv run python benchmarks/sampling_plan/int8_llama.py --out DIR [--model qwen3-8b] [--scheme pearl-c] [--weights]
uv run python benchmarks/sampling_plan/transformer.py --out DIR [--model large] [--share S] [--delta D]
uv run python benchmarks/sampling_plan/h200.py --out DIR [--share S] [--batch B] [--step-s T] [--tiles] [--pad]
uv run python benchmarks/sampling_plan/demo_a.py --out DIR [--spans NAME ...] [--no-frontier]
```

- `cpu_budget.py`: the block's draw sized to a CPU proving budget. It reads `h200.py`'s `plan.json` (its
  templates, their n, S and proved ANDs), prices each template's statements (up to 8 drawn units each) from measured
  firewalled C-Flock `--zk` sessions on the Genoa proofs slots, and finds the least share whose statements fit
  `--slots` x `--hours`. It writes the per-template k and cost and the certified units per window and per period, pooled
  and each block, at δ = 1% and 1e-6 (`cpu-budget.json`, `cpu-budget.md`).

```text
uv run python benchmarks/sampling_plan/cpu_budget.py --plan PLAN.json --slots 17 --hours 2.5 --out DIR
```
