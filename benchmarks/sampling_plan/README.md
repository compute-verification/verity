# sampling_plan

The measured inputs of SPOT's cost model (`verity/protocols/proofs/spot/prover/cost/`, its README), each with its
source and status, measured or assumed, and the tests that hold the model's rows to them.

- `inputs.py`: the models (Llama-3.1-8B, Qwen3-8B); each scheme's serve-time rows a token a layer, with their leaf
  chains; the row's inputs (price ratio, rounds, throughput, a request's context, commitment speeds, PoUB's receipts
  and passes, decode's step and its compression latency), each with a low and a
  high; the pieces' AND counts (circuit-check's pins, hm96's rows); and the observations the cost model is fitted from,
  Flock's zero-knowledge sessions on sm_120 (`benchmarks/one_stage/zk_cost.py`'s of s8 coordinates, and A8's of PoUB's
  opening unit, `BlockOpeningCheck_v1`, at twice their ANDs, its proving only), with the builds and
  hardware they key; and the `h200` card (`CARDS`): the deployed prover's sessions on GLM-5.3's proof blocks on one
  H200 (`zk_cost.py`'s served types, arm dev), in H200 GPU-seconds at that card's price ratio.

This directory's suite tests the cost model's rows against these inputs (`tests/`).
