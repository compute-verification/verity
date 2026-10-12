# overhead_frontier

How fast the remote bandwidth audit can get at γ ≤ 1% on node 2 (8 RTX PRO 6000), with which parameters, and which
input moves the answer most. A model, not a measurement: every input is a measured value with its run id, or an
estimate marked as one (`inputs.py`).

- `inputs.py`: each input's value, a realistic improvement and the work that would bring it, its source and status;
  `RISKS`, values that would make the frontier worse.
- `model.py`: the plain and audited decode step per GPU for a Llama-class shape in FP8 at a batch and context, and γ
  term by term under two definitions (against the node's read capacity, with top-up; per pass, the red team's).
- `frontier.py`: the search over rounds k, block size, k_s, binding granularity and the host tier's share, per
  workload, at today's inputs and with every lever improved, and the sensitivity (one lever at a time from today,
  one reverted from projected, and the risks).

```text
uv run python benchmarks/overhead_frontier/frontier.py --out DIR    # DIR/frontier.json, and the tables on stdout
uv run tools/verity/check/suites.py benchmarks/overhead_frontier
```

Results are published to the evidence store, not kept here.
