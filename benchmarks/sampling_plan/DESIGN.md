# sampling_plan: design

What `inputs.json` holds, in full. `README.md` says what the directory is for and plans Demo A.

- `inputs.json`: the models' shapes (Llama-3.1-8B, Qwen3-8B); the rows' inputs (price ratio, rounds, throughput, a request's context, commitment speeds, PoUB's receipts
  and passes, decode's step and its compression latency), each with a low and a
  high; and the observations the cost model is fitted from,
  Flock's zero-knowledge sessions on sm_120 (`benchmarks/one_stage/zk_cost.py`'s of s8 coordinates, and A8's of PoUB's
  opening proof block, `BlockOpeningCheck_v1`, at twice their ANDs, its proving only), with the builds and
  hardware they key; and the `h200` card: the deployed prover's sessions on GLM-5.3's proof blocks on one
  H200 (`zk_cost.py`'s served types, arm dev), in H200 GPU-seconds at that card's price ratio.
