# sampling_plan

The sampling plan for one served workload under Flock, from the planner in `verity.protocols.spot.verifier`
(`optimize`, `candidates`, `costfit`): the cost model fitted from proving measurements, the cuts the candidate generator
finds, the frontier of ε (at δ = 2^-40, against the workload's MACs, W_ref's proxy) over the verification overhead B, the
break-even granularity, and how the plan moves with each input. A model, not a measurement: every input carries its
source and status, measured or assumed (`inputs.py`).

B is the budget model's unit, GPU-seconds of overhead per plain GPU-second over a window, and it is the serve-time
total (everything is committed at serve time): every
scale's serve-time commitments under the PoUW scheme (`--scheme`: `floor`, the measured s8 A-operand floor alone;
`h1`, `h1-bf16`, `pearl-c`, each scheme's rows a token a layer) and the store-block receipts of
PoUB, which every plan pays; then proving and the verifier's CPU (at its price ratio) for the sampled proof blocks,
the commitment of every unit's outputs, replay of the requests the sample meets (serving keeps only the requests' tokens and
the weights, so each is re-run whole from its prompt to build its sampled proof blocks' witnesses), and the zero-work subcircuits at
their floor.

- `inputs.py`: the models (Llama-3.1-8B, Qwen3-8B); each scheme's serve-time rows a token a layer, with their leaf
  chains; the row's inputs (price ratio, rounds, throughput, a request's context, commitment speeds, PoUB's receipts
  and passes, decode's step and its compression latency), each with a low and a
  high; the pieces' AND counts (circuit-check's pins, hm96's rows); and the observations the cost model is fitted from,
  Flock's zero-knowledge sessions on sm_120 (`benchmarks/one_stage/zk_cost.py`'s of s8 coordinates, and A8's of PoUB's
  opening unit, `BlockOpeningCheck_v1`, at twice their ANDs, its proving only), with the builds and
  hardware they key; and the `h200` card (`CARDS`): the deployed prover's sessions on GLM-5.3's proof blocks on one
  H200 (`zk_cost.py`'s served types, arm dev), in H200 GPU-seconds at that card's price ratio.
- `int8_llama.py`: the int8 Llama-3.1-8B row (vLLM's `cursor/vllm-int8-build-8350`). Its units: every s8 dot checks its
  seed, derives its own noise and opens its rows in circuit. `--model qwen3-8b`
  runs the Qwen3-8B row the same way. A linear's coordinate may be cut into K-chunks that commit their running sums, and
  r × c tiles are offered, each a unit when its outputs fit a size class. Its replay grain is the window's requests, each
  one prompt of `context` tokens re-run through every layer at `tokens_per_s` (the engine is batch-invariant, so no
  co-scheduled request is re-run with it). It writes the fitted model (`cost-model.json`, a
  cost-model/v1 record), the serve-time commitments, the frontier, the plan at each budget, the tensors committed at
  more than one scale and what committing each once saves (`overlaps`), the weight classes' weights at each ε
  (`--weights`), the break-even table (the linears forced to each grain, per window, ε and commitment regime), the
  sensitivity (each cost group scaled, each input at its low and high) and the hints (`plan.json`), and prints the
  tables.

- `transformer.py`: the end-to-end demo's transformer, its decoder block over one server block of
  serving, small (width 128, 2 heads, MLP 512, 2 layers, a byte vocabulary, context 32) or large (`--model large`:
  the Phase 2 model's 70 layers at width 8,192, 128 heads of 64 and an MLP of 32,768, 52.5 GiB of INT8 linears and
  35.0 GiB of KV for 1,023 sequences on one 96 GiB card, its requests a block from an estimated 195 ms step; each
  dimension a flag). Every op is a proof block of its Definition: the linears `ScaledMmS8Coordinate_v1` (or tiles of
  them), attention `GemmCoordinate_v2` on the BF16 step, the residual `Bf16Add_v1`, and the row ops (RMSNorm, the s8
  quantizer, softmax, ReLU, the embedding, argmax) as stubs, each id `stub:`, until they land, row ops whole, shared,
  chunked or recomputed. The harm is incorrect ANDs, the same whatever the cut, with the incorrect proof blocks (IntegrityPeriod's W)
  beside it. It writes the frontier of expected Flock cost (every serve-time commitment and the replay in it) against
  certified incorrect ANDs at δ, the share of its ANDs the block's own GPU time buys, the chosen plan for `--share` (1e-3,
  the small model's knee, held fixed), its certificate recomputed through `Stratified.harm_bound` and `count_bound` at
  δ = 1% and 1e-6, the pooled bound over a week, a month and a year on one server and on an assumed 10,000 (an assumed
  20-minute block: neither is in the census), and the plan as the GPU loop registers it (`handoff`): one execution
  phase (`IntegrityPeriod`) a subcircuit (`subset:k`, its cap and K0), or the block as one execution phase under the
  `work:1` sampling design. Its tests are this directory's suite.
- `h200.py`: the demo's plan on one H200: the frozen MQA shape (52 layers, width 16,384, 256 query heads of 64 and
  one KV head, MLP 65,536, context 32; batch and step are named stand-ins) on the demo's partition,
  one lockstep block of 31 decode steps. Each subcircuit is one `IntegrityPeriod` execution phase, one `--zk` session sampling
  `subset:k` over the tables `_cap` allows, and leads with what Lean proves: each window's K0 (IntegrityPeriodLog) and
  the pooled bound over any period (IntegrityHorizon) at δ = 1% and 1e-6. It prices each window's statements and sampled
  proof blocks, the replay and, beside them, the commitment of every unit's outputs; compares the static matmuls as 128-channel
  tiles (`--tiles`) and softmax and PV padded to the context (`--pad`); and gives the frontier over the share, the
  execution phase length in blocks that proves within its own GPU time, and the incorrect ANDs (`Stratified.harm_bound`, not a Lean
  guarantee). `--card h200` prices it with the `h200` card's fit instead of the RTX PRO 6000's.
- `demo_a.py`: Demo A's frontier on one H200: plain INT8 weights, no P2 decode, a block of 16,384 requests at a
  6.35 s step, 32 steps a request, on the Definition map (the H200 demo's `plan.py`, in history at `5c095bb4ab70`: every subcircuit a
  Definition Flock stages, at its measured ANDs and staged k_log) with PoUW's tiles as one more subcircuit at each K
  (J10-R's per-call lane units, `PouwH1LaneZy_v2` over v5, at their traced ANDs), and statements shared per subcircuit
  per execution phase. The harm is half skipped work (each unit's own ANDs), half leaked output bits, each half
  weighted at each budget by the least bound that budget certifies for it alone;
  each point reports `certify`'s `free_compute` and `leak` from its one sample at δ = 1% and 1e-6, the tile stratum's ε
  (`eps_at`), and the knee. The plan of record is proving within 1% of the execution phase's GPU time, for execution
  phases of one server block, the least that affords a plan, a week, a month and a year, with its sample a subcircuit
  and the commits beside it; then M3's uniform 8-unit sample against the block's GPU time, and each Definition's share
  of the ANDs against the step's GPU time.

- `served_window.py`: the served window (Llama-3.2-1B int8, T 8,192 on the H200, one step a window) cut at each rung of
  a ladder (K-chunks of the int8 linears, output coordinates, tiles, and fused attention/MLP, layer and 4-layer proof
  blocks that commit only the residual and K, V), for the free-compute target (a window, and pooled over a week, a month
  and a year) and the exfiltration target, in the planner's two modes (proving only, and proving plus w times the serve-time
  commitment) at two readings of a hashed byte's cost, through `grain_costs` and `for_target`, and the second mode again
  under a code-based commitment at each rate ρ (codeword columns hashed, 1/ρ the bytes, and no row hashed in circuit),
  each total with the replay of the requests the sample meets, each re-run from its prompt at the stock step's share;
  then the chunk family's optimum against √(wβhC/(oΛ)). A rung fits when Flock's block holds each of its proof blocks (`inputs.cap`). Its
  inputs, each with its status and source, are `served_window_inputs.json`; it writes `served_window.json` and
  `served_window.md`.

```text
uv run python benchmarks/sampling_plan/int8_llama.py --out DIR [--model qwen3-8b] [--scheme pearl-c] [--weights]
uv run python benchmarks/sampling_plan/served_window.py --out DIR [--weight W] [--rates R ...] [--no-planner]
uv run python benchmarks/sampling_plan/transformer.py --out DIR [--model large] [--share S] [--delta D]
uv run python benchmarks/sampling_plan/h200.py --out DIR [--share S] [--batch B] [--step-s T] [--tiles] [--pad] [--card h200]
uv run python benchmarks/sampling_plan/demo_a.py --out DIR [--spans NAME ...] [--no-frontier]
```

- `glm53.py`: GLM-5.3 as served (vLLM, H200, TP8, FP8) as the planner's candidates, from its proof-block census
  (`glm53_blocks.json`, `benchmarks/one_stage/block_census.py`'s card over the Commit Build it names, the census in the
  store): each proof-block Definition one candidate, its units one request's count over every rank, its work its
  estimated ANDs, its outputs committed and its opened rows' bytes; the gaps, rows no cut fits, are listed by row family
  and are no candidate. It prints the proof blocks, output bits, ANDs, served MACs and opened bytes per prefill token,
  per decode token and per request, and the gaps.

```text
uv run python benchmarks/sampling_plan/glm53.py [--requests N]
```

- `cpu_budget.py`: the block's sample sized to a CPU proving budget. It reads `h200.py`'s `plan.json` (its
  templates, their n, S and proved ANDs), prices each template's statements (up to 8 sampled proof blocks each) from measured
  ZK-served Flock `--zk` sessions on the Genoa proofs slots, and finds the least share whose statements fit
  `--slots` x `--hours`. It writes the per-template k and cost and the certified units per window and per period, pooled
  and each block, at δ = 1% and 1e-6 (`cpu-budget.json`, `cpu-budget.md`).

```text
uv run python benchmarks/sampling_plan/cpu_budget.py --plan PLAN.json --slots 17 --hours 2.5 --out DIR
```

- `recursion.py`: one recursive session's cost per monitoring period. It reads a `benchmarks/recursion_e2e` result's
  measured parts (the inner run, V*'s staging, V*'s ZK-served releases and the Lean step), with the same units'
  non-recursive `--zk` session beside them, and scales them to every machine-window of the census's datacenter
  (`dc-12500-machines-600s`) and of the world (`verity/protocols/inference_only/rho`'s `world-h200` fleet, in the datacenter's windows,
  assumed) over a week, a month and a year: sessions, node-hours and nodes kept busy, prover's and auditor's
  (`recursion-periods.json`, `recursion-periods.md`).

```text
uv run python benchmarks/sampling_plan/recursion.py --result recursion_e2e.json [--sessions-per-window 1] --out DIR
```
