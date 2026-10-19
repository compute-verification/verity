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
  zero-knowledge sessions on sm_120 (`benchmarks/one_stage/zk_cost.py`), with the builds and hardware they key.
- `int8_llama.py`: the int8 Llama-3.1-8B row (vLLM's `cursor/vllm-int8-build-8350`). Its units are compute accounting's
  decision 106: every s8 dot checks its seed, derives its own noise and opens its rows in circuit. `--model qwen3-8b`
  runs the Qwen3-8B row the same way. A linear's coordinate may be cut into K-chunks that commit their running sums, and
  r × c tiles are offered so the width rule's price shows as a hint. It writes the fitted model (`cost-model.json`, a
  cost-model/v1 record), the serve-time commitments, the frontier, the plan at each budget, the tensors committed at
  more than one scale and what committing each once saves (`overlaps`), the block kinds' weights at each ε
  (`--weights`), the break-even table (the linears forced to each grain, per window, ε and commitment regime), the
  sensitivity (each cost group scaled, each input at its low and high) and the hints (`plan.json`), and prints the
  tables.

- `opt5.py`: the least serve-time and proving total that keeps each policy (skippable work, control, integrity; the IOP
  note's tier 2, note:proofs/20261018T2305Z-draft-iop-core-objects) within a share ε of its total, the draw weighted by
  the envelope of their public costs (`optimize.Policy`). Its variables are the cut, the replay block's tokens, the
  always-proved set, clustered draws (k units a drawn replay block, in closed form) and the serve-time options (Pearl-C
  reading down's and o_proj's A from the Commit's BF16 rows, hash's tap format), each priced. Per model it writes each
  plan with the policy that binds, every policy's level at it, the optimum per PoUW scheme, and TS1's two-stage table
  with every value committed (`opt5-MODEL.json`, `.md`).
- `opt5_eps.py`: the work policy's ε at a budget, and the total B for ε = 1e-5 and 1e-8, over a horizon (one GPU-week,
  and 1 hour, 1 day, 30 days and a year of G GPUs, one certificate pooling the horizon's draws), with each scheme's
  serve-time commitments, what one draw's statement costs, and the lifetime pool ε G T with q, a unit's draw
  probability, at δ = 2^-40 and at δ = 1% with A = 1 (`opt5-eps.json`, `.md`).

```text
uv run python benchmarks/sampling_plan/int8_llama.py --out DIR [--model qwen3-8b] [--scheme pearl-c] [--weights]
uv run python benchmarks/sampling_plan/opt5.py --out DIR --label main
uv run python benchmarks/sampling_plan/opt5_eps.py --out DIR [--gpus 100000] [--budget 0.01] [--luck 0]
```
