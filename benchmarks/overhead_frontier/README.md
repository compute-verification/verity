# overhead_frontier

How fast the remote bandwidth audit can get at γ ≤ 1% on node 2 (8 RTX PRO 6000), with which parameters, and which
input moves the answer most. A model, not a measurement: every input is a measured value with its run id, or an
estimate, assumption or policy marked as one (`inputs.py`).

- `inputs.py`: each input's value, a realistic improvement and the work that would bring it, its source and status;
  `FLOOR`, each lever at the best any measurement or bound supports; `RISKS`, values that would make the frontier
  worse.
- `model.py`: the plain and audited decode step per GPU for a Llama-class shape in FP8 at a batch and context, and γ
  term by term under two definitions (against the node's read capacity, with top-up and padding; per pass, the red
  team's), with memory-only helpers over the NIC priced (`link_n`). The word width scales the codec's squaring, and
  ra-codec-mix's σ_K is a codec option.
- `frontier.py`: the search over rounds k, block size, k_s, binding granularity, the host tier's share and the number
  of keyed copies, per workload, at today's inputs, with every lever improved and at the floor; the word width swept
  with the rounds (`widths`); and the sensitivity (one lever at a time from today, the greedy path of levers from
  today, one reverted from projected, and the risks).

```text
uv run python benchmarks/overhead_frontier/frontier.py --out DIR    # DIR/frontier.json, and the tables on stdout
uv run tools/verity/check/suites.py benchmarks/overhead_frontier
```

Results are published to the evidence store, not kept here.
