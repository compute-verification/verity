# spot_figures

The SPOT post's Figure 1, derived from the sampled-proofs planner (`verity.core.protocols.sampled_proofs.optimize`):
what an adversary serving GLM-5.3 on all of a scope's compute over a monitoring period can exfiltrate, infiltrate or
skip while evading detection with probability above δ, against proof overhead, with one curve per share of compute
spent on replays, interior commitments and proofs. A model, not a measurement: every input is a card marked measured or
modeled with its source, and the output carries every card beside every plotted series.

~~~sh
uv run python benchmarks/spot_figures/figure1.py --out DIR            # DIR/figure1.json and every scope's PNG and SVG
uv run python benchmarks/spot_figures/render.py DIR/figure1.json      # the renders again, from the JSON alone
research data put --kind spot-figure/v1 --tree DIR --preserve          # for the console
~~~

- `figure1.py`: the pipeline. Scopes are the world (rho's `world-h200` fleet, which computes the census's
  `world-inference-2027`) and a datacenter (the census's `dc-12500-machines-600s`), each over a week, a month and a
  year (`wipe_monitoring.PERIODS`). Costs are the scope's GPU-seconds: a replay re-runs its request at serving's MFU,
  proving a proof block takes the proof overhead times its FLOP at the H200's dense FP8 peak, and hashing takes the
  H200's SHA-512 rate. A curve's point is the least target the planner's cheapest plan certifies within the budget.
- Panels, each with its headline design solid and its comparisons dashed or dotted:
  - exfiltration, under two-stage sampling (serving commits only token ids), one-stage sampling of dot-product proof
    blocks, and request-level sampling, in `consumers.exfiltration_bound`'s form: the incorrect proof blocks' output
    bits and which ones they are;
  - infiltration, pending its consumer and the design that certifies it;
  - skipped work, under one-stage sampling and request-level sampling. One-stage needs every proof block's outputs
    hashed at serve time, outside the budget; the JSON gives that hashing's share of the scope's compute
    (`serve_hash_share`).
- `inputs.json`: the figure's own cards: δ, the budgets, the x-axis, the measured proof overhead (the vertical line,
  provisional until the proof-overhead card is defined), the hashing rate, a token's bytes and GLM-5.3's checkpoint.
  The tokens a request, the GPUs, the MFU, the peak, the output cap and the reference lines are read from the census
  and rho's reference, and a MAC's AND gates, which state each scope's compute and the skipped work in AND gates, from
  the FP8 MoE coordinate's circuit-check pin.
- `glm53_partition.py` writes `glm53_partition.json`, GLM-5.3's proof blocks a token, from the census of its request
  Program (`benchmarks/one_stage/a11_kinds.py`, the TP8 B1 row's rank 0, scaled to its 8 ranks): each matmul is its
  output coordinates, every other Call its output words. Attention's growth with context past the census's row is not
  in it.
- `render.py`: the PNG and SVG of one scope, from `figure1.json` alone, so the console and the post render the same
  figure; `figure1.png` is the world over a year.

Limits of the model: a replay is priced as its request alone, while a replay that must reproduce the served batch's
layout costs that batch; the one-stage curves leave serve-time hashing out of the budget; and the two-stage
certificate counts location bits over every proof block of the scope.
