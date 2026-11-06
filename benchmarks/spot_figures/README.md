# spot_figures

The SPOT post's Figures 1-3 for one model, derived from the sampled-proofs planner
(`verity.core.protocols.sampled_proofs.optimize`) and the model's cards. A model, not a measurement: every input is a
card marked measured or modeled with its source, and each figure's JSON carries every card it read beside every
plotted series.

~~~sh
uv run python benchmarks/spot_figures/figures.py --model GLM-5.3 --out DIR   # figure{1,2,3}.json, PNGs and SVGs
uv run python benchmarks/spot_figures/render.py DIR/figure1.json            # Figure 1 again, from the JSON alone
research data put --kind spot-figure/v1 --tree DIR --preserve                # for the console
~~~

- **Cards** (`cards.py`, schema `spot-card/v1`): every number the figures read arrives as one JSON card in a slot
  (`workload`, `proof_overhead`, `commitment`, `replay`, `design`, `references`, `infiltration`,
  `commit_hillclimb`), for a model or for every model. `cards.py` holds each slot's fields and checks a card against
  them. A card is published to the evidence store and kept in `cards/SLOT.MODEL.json` with its art id:

  ~~~sh
  uv run python benchmarks/spot_figures/cards.py check CARD.json
  research data put --kind spot-card/v1 --file CARD.json --preserve --json
  uv run python benchmarks/spot_figures/cards.py fetch art:...
  uv run python benchmarks/spot_figures/cards.py list
  ~~~

  A model without a `workload` card is refused; no census is guessed from a model's config.
- **Figure 1** (`figure1.py`, `render.py`): what an adversary serving the model on all of a scope's compute over a
  monitoring period can exfiltrate, infiltrate or skip while evading detection with probability above δ, against
  proof overhead, a curve per share of compute spent on replays, interior commitments and proofs.
  - Scopes are the world (rho's `world-h200` fleet, which computes the census's `world-inference-2027`) and a
    datacenter (the census's `dc-12500-machines-600s`), each over a week, a month and a year
    (`wipe_monitoring.PERIODS`).
  - Costs are the scope's GPU-seconds: a replay costs the `replay` card's factor times its request at serving's MFU,
    proving a proof block takes the proof overhead times its FLOP at the H200's dense FP8 peak, and hashing takes the
    `commitment` card's rate. A curve's point is the least target the planner's cheapest plan certifies within the
    budget.
  - Each panel shows its headline design solid and its comparisons dashed or dotted:
    - exfiltration: two-stage sampling (serving commits only token ids), one-stage sampling of dot-product proof
      blocks, and request-level sampling, in `consumers.exfiltration_bound`'s form (the incorrect proof blocks'
      output bits and which ones they are);
    - infiltration: the `infiltration` card's design and series, pending without one;
    - skipped work: one-stage and request-level sampling. One-stage needs every proof block's outputs hashed at
      serve time, outside the budget, and the JSON gives that hashing's share of the scope's compute
      (`serve_hash_share`).
  - The vertical line is the `proof_overhead` card's `model_overhead`. The reference lines come from rho's reference,
    the model's checkpoint and any `references` card. A MAC's AND gates, read from the FP8 MoE coordinate's
    circuit-check pin, state each scope's compute and the skipped work in AND gates.
- **Figure 2** (`figure2.py`): proof overhead per proof-block type, the `proof_overhead` card's blocks joined to the
  `workload` card's strata, beside each type's share of a token's proof blocks and MACs. A type without a measurement
  is pending, and a stand-in block is marked as one.
- **Figure 3** (`figure3.py`): the serve-time commitment's cost over fast serving. On the left, the `commit_hillclimb`
  card's series against hill-climbing effort, with each series' best so far. On the right, the model's serve-time
  cuts, priced from their bytes a token and the hash rate, beside the card's measured points for the model.
  `figure3.py card` writes the hash loop's card from its `hash-loop.json`.
- `glm53_partition.py` writes GLM-5.3's `workload` card from the census of its request Program
  (`benchmarks/one_stage/a11_kinds.py`, the TP8 B1 row's rank 0, scaled to its 8 ranks): each matmul is its output
  coordinates, every other Call its output words. Attention's growth with context past the census's row is not in it.

Limits of the model:
- A replay is priced at the `replay` card's factor; one that must reproduce the served batch's layout costs that
  batch.
- The one-stage curves leave serve-time hashing out of the budget.
- The two-stage certificate counts location bits over every proof block of the scope.
