# overhead_frontier

How fast the remote bandwidth audit can get at γ ≤ 1% on node 2 (8 RTX PRO 6000), with which parameters, and which
input moves the answer most. A model, not a measurement: every input is a measured value with its run id, or an
estimate, assumption, proved guarantee or policy marked as one (`inputs.py`).

The claim is Conservation over HBM's R·T (Daniel, 21:00Z): no host tier, no μ_link, no identity term, and binding
receipts to z is an opt-in. The codec is the stated one, masked p1024 at k = 3, with k = 2 its fallback (20:55Z).

- `inputs.py`: each input's value, a realistic improvement and the work that would bring it, its source and status;
  `FLOOR`, each lever at the best any measurement or bound supports; `RISKS`, values that would make the frontier
  worse. Width-dependent inputs (x₁, the adversary's s and h, the decode's ratio, the squaring speed) are tables by
  bits.
- `model.py`: the plain and audited decode step per GPU for a Llama-class shape in FP8 at a batch and context, and γ
  term by term under two definitions (against HBM's capacity over time, with top-up and padding; per pass, the red
  team's). Openings are the hybrid's (every window's sampling at most 1%, averaged over a 10-minute horizon), with
  their serving cost and named λ. Every overhead is against the stock step, whose ephemeral traffic (`eph_plain`) no
  lever moves; the encoded kernels' own (`eph_read`, `eph_write`) is μ_unreg. η, the share of R·T the honest engine
  leaves unread, is counted by phase on the GPU's clock (decode, prefill, gaps, idle); `eta_allowance` is what the
  rest of γ leaves for it.
- `frontier.py`: the search over block size, k_s, binding granularity (when bound) and the number of keyed copies, per
  workload, at today's inputs (η as measured and at budgets of 0, 0.25% and 0.5%), with every lever improved and at
  the floor; steps past 4× the plain one are flagged as over the budget; the word width swept with the rounds, plain
  and masked, at 512, 768 and 1,024 bits (`widths`); per pass, the largest d that keeps γ ≤ 1% (`d_cap`); and the
  sensitivity from η = 0 (one lever at a time, the greedy path, one reverted from projected, the risks) and again
  from today's measured η.

```text
uv run python benchmarks/overhead_frontier/frontier.py --out DIR    # DIR/frontier.json, and the tables on stdout
uv run tools/verity/check/suites.py benchmarks/overhead_frontier
```

Results are published to the evidence store, not kept here.
