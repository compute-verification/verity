# prover/cost

SPOT's cost model for one served workload under Flock: each file plans a workload's sample with the planner beside
it (`../optimize.py`, `../candidates.py`, `../costfit.py`), from the measured inputs under `benchmarks/sampling_plan/`
(`inputs.json`, `glm53_blocks.json`, `served_window_inputs.json`, which `inputs.py` loads, `glm53.py` and
`served_window.py` read), each with its source and status, measured or
assumed. `benchmarks/prove_path/`'s `window_cost.py` and `class_rows.py` price the class-of-record window's commit.

Each workload's plan gives the cost model fitted from proving measurements, the cuts the candidate generator finds, the frontier of
ε (at δ = 2^-40, against the workload's MACs, W_ref's proxy) over the verification overhead B, the break-even
granularity, and how the plan moves with each input. A model, not a measurement.

B is the budget model's unit, GPU-seconds of overhead per plain GPU-second over a window, and it is the serve-time
total (everything is committed at serve time): every
scale's serve-time commitments under the PoUW scheme (`--scheme`: `floor`, the measured s8 A-operand floor alone;
`h1`, `h1-bf16`, `pearl-c`, each scheme's rows a token a layer) and the store-block receipts of
PoUB, which every plan pays; then proving and the verifier's CPU (at its price ratio) for the sampled proof blocks,
the commitment of every proof block's outputs, replay of the requests the sample meets (serving keeps only the requests' tokens and
the weights, so each is re-run whole from its prompt to build its sampled proof blocks' witnesses), and the zero-work subcircuits at
their floor.

- `int8_llama.py`: the int8 Llama-3.1-8B workload (vLLM's `cursor/vllm-int8-build-8350`). Its proof blocks: every s8 dot checks its
  seed, derives its own noise and opens its rows in circuit. `--model qwen3-8b`
  runs the Qwen3-8B workload the same way. A linear's coordinate may be cut into K-chunks that commit their running sums, and
  r × c tiles are offered, each a proof block when its outputs fit a size class. Its replay grain is the window's requests, each
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
  chunked or recomputed. The harm is incorrect ANDs, the same whatever the cut, with the incorrect proof blocks (IntegrityMonitoringPeriod's W)
  beside it. It writes the frontier of expected Flock cost (every serve-time commitment and the replay in it) against
  certified incorrect ANDs at δ, the share of its ANDs the block's own GPU time buys, the chosen plan for `--share` (1e-3,
  the small model's knee, held fixed), its certificate recomputed through `Stratified.harm_bound` and `count_bound` at
  δ = 1% and 1e-6, the pooled bound over a week, a month and a year on one server and on an assumed 10,000 (an assumed
  20-minute block: neither is in the census), and the plan as the GPU loop registers it (`handoff`): one execution
  phase a subcircuit (`subset:k`, its cap and K0), or the block as one execution phase under the
  `work:1` sampling design. Its tests are in SPOT's prover suite.
- `h200.py`: the demo's plan on one H200: the frozen MQA shape (52 layers, width 16,384, 256 query heads of 64 and
  one KV head, MLP 65,536, context 32; batch and step are named stand-ins) on the demo's partition,
  one lockstep block of 31 decode steps. Each subcircuit is one execution phase of `IntegrityMonitoringPeriod`, one `--zk` session sampling
  `subset:k` over the tables its `Template.cap` allows, and leads with what Lean proves: each window's K0 (IntegrityMonitoringPeriodLog) and
  the pooled bound over any period (IntegrityPooled) at δ = 1% and 1e-6. It prices each window's statements and sampled
  proof blocks, the replay and, beside them, the commitment of every proof block's outputs; compares the static matmuls as 128-channel
  tiles (`--tiles`) and softmax and PV padded to the context (`--pad`); and gives the frontier over the share, the
  execution phase length in blocks that proves within its own GPU time, and the incorrect ANDs (`Stratified.harm_bound`, not a Lean
  guarantee). `--card h200` prices it with the `h200` card's fit instead of the RTX PRO 6000's.
- `served_window.py`: the served window (Llama-3.2-1B int8, T 8,192 on the H200, one step a window) cut at each rung of
  a ladder (K-chunks of the int8 linears, output coordinates, tiles, and fused attention/MLP, layer and 4-layer proof
  blocks that commit only the residual and K, V), for the free-compute target (a window, and pooled over a week, a month
  and a year) and the exfiltration target, in the planner's two modes (proving only, and proving plus w times the serve-time
  commitment) at two readings of a hashed byte's cost, through `grain_costs` and `for_target`, and the second mode again
  under a code-based commitment at each rate ρ (codeword columns hashed, 1/ρ the bytes, and no row hashed in circuit),
  each total with the replay of the requests the sample meets, each re-run from its prompt at the stock step's share;
  then the chunk family's optimum against √(wβhC/(oΛ)). A rung fits when Flock's block holds each of its proof blocks (`inputs.cap`). Its
  inputs, each with its status and source, are `benchmarks/sampling_plan/served_window_inputs.json`; it writes `served_window.json` and
  `served_window.md`.

```text
uv run python verity/protocols/proofs/spot/prover/cost/int8_llama.py --out DIR [--model qwen3-8b] [--scheme pearl-c] [--weights]
uv run python verity/protocols/proofs/spot/prover/cost/served_window.py --out DIR [--weight W] [--rates R ...] [--no-planner]
uv run python verity/protocols/proofs/spot/prover/cost/transformer.py --out DIR [--model large] [--share S] [--delta D]
uv run python verity/protocols/proofs/spot/prover/cost/h200.py --out DIR [--share S] [--batch B] [--step-s T] [--tiles] [--pad] [--card h200]
```

- `glm53.py`: GLM-5.3 as served (vLLM, H200, TP8, FP8) as the planner's candidates, from its proof-block census
  (`benchmarks/sampling_plan/glm53_blocks.json`, `benchmarks/one_stage/block_census.py`'s card over the Commit Build it names, the census in the
  store): each proof-block Definition one candidate, its proof blocks one request's count over every rank, its work its
  estimated ANDs, its outputs committed and its opened rows' bytes; the gaps, rows no cut fits, are listed by row family
  and are no candidate. It prints the proof blocks, output bits, ANDs, served MACs and opened bytes per prefill token,
  per decode token and per request, and the gaps. With `--stop T` the requests are a stop class's, of at most T ids a
  response, and it adds the class's two populations, one proof block a response each: the stop blocks (the stop rule's
  Boolean Definition taken as a block of its own) and the response ties, each a candidate whose ANDs, rows and output
  bits are its Definition's. Both are registered at `rate:1/1`, and it prints what that leaves at `--delta`: the most
  blocks an accepted sample leaves incorrect, that count over the population, and those blocks' output bits, each 0
  where every block is proved (`charged`, which gives a smaller sample its count bound's). These lines read no verdict.

```text
uv run python verity/protocols/proofs/spot/prover/cost/glm53.py [--requests N] [--stop T [--delta D]]
```

- `cpu_budget.py`: the block's sample sized to a CPU proving budget. It reads `h200.py`'s `plan.json` (its
  templates, their n, S, rows and ANDs), prices each template's statements (up to 8 sampled proof blocks each) from measured
  ZK-served Flock `--zk` sessions on the Genoa proofs slots, and finds the least share whose statements fit
  `--slots` x `--hours`. It writes the per-template k and cost and the certified proof blocks per window and per period, pooled
  and each block, at δ = 1% and 1e-6 (`cpu-budget.json`, `cpu-budget.md`).

```text
uv run python verity/protocols/proofs/spot/prover/cost/cpu_budget.py --plan PLAN.json --slots 17 --hours 2.5 --out DIR
```

- `recursion.py`: one recursive session's cost per monitoring period. It reads a `benchmarks/recursion_e2e` result's
  measured parts (the inner run, V*'s staging, V*'s ZK-served releases and the Lean step), with the same proof blocks'
  non-recursive `--zk` session beside them, and scales them to every machine-window of the census's datacenter
  (`dc-12500-machines-600s`) and of the world (`benchmarks/slowdown`'s `world-h200` fleet, in the datacenter's windows,
  assumed) over a week, a month and a year: sessions, node-hours and nodes kept busy, prover's and auditor's
  (`recursion-periods.json`, `recursion-periods.md`).

```text
uv run python verity/protocols/proofs/spot/prover/cost/recursion.py --result recursion_e2e.json --census benchmarks/census/deployments.json --fleet benchmarks/slowdown/data/reference.toml [--sessions-per-window 1] --out DIR
```

## The class circuit's size and its window's commit

`../audit/class_size.py` sizes a class circuit in ANDs and Flock blocks, and `benchmarks/prove_path/class_rows.py` counts its rows by what fills them and
their SHA-512 compressions (`--probe S,NP,O` lowers one Program at a small class). `window_cost.py` prices a
window's commit from that census, or from a probe projected to a class (`--project 8,1024,64`, which reproduces the exact
census proof block for proof block), against its target of 0.20 to 0.25 of the served window. It reports in ANDs (58,120 a
compression) and in seconds (the window's rows on the GPU's hashing rates, the docket on the CPU's, which
`--measure-cpu` measures). It covers each lever alone and the plan stacked, the pad pass at three traffics, the
docket's bytes and files, and totals over a week, a month and a year. With `--stream SIZING` it also prices serving's
own stream commit (rows-sha512, made while it serves, before the window's commit) and its seal, from serving's stream of
record (`native_host.schedule_stream_bound`), whole and without the attention interiors (`fa*.m1`, lever F), which nothing
reads after the seal:

~~~text
python benchmarks/prove_path/window_cost.py PROBE.json --target h200 --project 8,1024,64 \
    [--measure-cpu SCRATCH] [--stream adapters/vllm/tests/pipeline/fixtures/admission/b16.json] [--out FILE]
~~~
