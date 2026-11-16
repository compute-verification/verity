# sampling_plan

The measured inputs of SPOT's cost model (`verity/protocols/proofs/spot/prover/cost/`, its README), each with its
source and status, measured or assumed, and the tests that hold the model's rows to them.

- `inputs.json`: the models' shapes (Llama-3.1-8B, Qwen3-8B); the rows' inputs (price ratio, rounds, throughput, a request's context, commitment speeds, PoUB's receipts
  and passes, decode's step and its compression latency), each with a low and a
  high; and the observations the cost model is fitted from,
  Flock's zero-knowledge sessions on sm_120 (`benchmarks/one_stage/zk_cost.py`'s of s8 coordinates, and A8's of PoUB's
  opening proof block, `BlockOpeningCheck_v1`, at twice their ANDs, its proving only), with the builds and
  hardware they key; and the `h200` card: the deployed prover's sessions on GLM-5.3's proof blocks on one
  H200 (`zk_cost.py`'s served types, arm dev), in H200 GPU-seconds at that card's price ratio.

This directory's suite tests the cost model's rows against these inputs (`tests/`).

`demo_a.py` (its test `tests/test_demo_a.py`) plans Demo A's frontier on one H200 with the cost model's planner: plain INT8 weights, no P2 decode, a block of 16,384 requests at a
6.35 s step, 32 steps a request, on the Definition map (the H200 demo's `plan.py`, in history at `5c095bb4ab70`: every subcircuit a
Definition Flock stages, at its measured ANDs and staged k_log) with PoUW's tiles as one more subcircuit at each K
(J10-R's per-call lane proof blocks, `PouwH1LaneZy_v2` over v5, at their traced ANDs), and statements shared per subcircuit
per execution phase. The harm is half skipped work (each proof block's own ANDs), half leaked output bits, each half
weighted at each budget by the least bound that budget certifies for it alone;
each point reports `certify`'s `free_compute` and `leak` from its one sample at δ = 1% and 1e-6, the tile stratum's ε
(`eps_at`), and the knee. The plan of record is proving within 1% of the execution phase's GPU time, for execution
phases of one server block, the least that affords a plan, a week, a month and a year, with its sample a subcircuit
and the commits beside it; then M3's uniform 8-unit sample against the block's GPU time, and each Definition's share
of the ANDs against the step's GPU time.

```text
uv run python benchmarks/sampling_plan/demo_a.py --out DIR [--spans NAME ...] [--no-frontier]
```
