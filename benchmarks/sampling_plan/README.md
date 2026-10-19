# sampling_plan

The sampling plan for one served row under C-Flock, from the planner in `verity.core.protocols.sampled_proofs`
(`optimize`, `candidates`, `costfit`): the cost model fitted from proving measurements, the cuts the candidate generator
finds, the frontier of ε (at δ = 2^-40, against the row's MACs, W_ref's proxy) over the verification overhead B, the
break-even granularity, and how the plan moves with each input. A model, not a measurement: every input carries its
source and status, measured or assumed (`inputs.py`).

B is BUDGET's unit, GPU-seconds of overhead per plain GPU-second over a window (note:compute-accounting/
20261018T0349Z-draft-budget): proving and the verifier's CPU (at its price ratio) for the drawn units, serving's
commitment of every unit's outputs, replay of the layer-steps the draw meets, the zero-work templates at their floor, and
the served Program's own commitment floor, which every plan pays.

- `inputs.py`: the row's inputs (price ratio, rounds, throughput, replay and commitment speeds), each with a low and a
  high; the pieces' AND counts (#2391's pins, hm96's rows); and the observations the cost model is fitted from, C-Flock's
  zero-knowledge sessions on sm_120 (`benchmarks/one_stage/zk_cost.py`), with the builds and hardware they key.
- `int8_llama.py`: the int8 Llama-3.1-8B row (vLLM's `cursor/vllm-int8-build-8350`). Its units are compute accounting's
  decision 106: every s8 dot checks its seed, derives its own noise and opens its rows in circuit. A linear's coordinate
  may be cut into K-chunks that commit their running sums, and r × c tiles are offered so the width rule's price shows
  as a hint. It writes the fitted model (`cost-model.json`, a cost-model/v1 record), the frontier, the plan at each
  budget, the break-even table (the linears forced to each grain, per window, ε and commitment regime), the sensitivity
  (each cost group scaled, each input at its low and high) and the hints (`plan.json`), and prints the tables.

```text
uv run python benchmarks/sampling_plan/int8_llama.py --out DIR      # about a minute on one core
```
