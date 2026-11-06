# spot_figures

The SPOT post's Figures 1-3 for one model, derived from the sampled-proofs planner
(`verity.core.protocols.sampled_proofs.optimize`) and the model's cards. A model, not a measurement: every input is a
card marked measured or modeled with its source, and each figure's JSON carries every card it read beside every
plotted series.

~~~sh
uv run python benchmarks/spot_figures/figures.py --model GLM-5.3 --out DIR   # figure{1,1_overhead,2,3}.json, PNGs, SVGs
                                                       # [--request workload] [--workers N] [--points N]
uv run python benchmarks/spot_figures/total_cost.py render DIR/figure1.json  # Figure 1, from the JSON
uv run python benchmarks/spot_figures/render.py DIR/figure1_overhead.json   # the bounds against proof overhead
research data put --kind spot-figure/v1 --tree DIR --preserve                # for the console
~~~

- **Cards** (`cards.py`, schema `spot-card/v1`): every number the figures read arrives as one JSON card in a slot
  (`workload`, `proof_overhead`, `commitment`, `replay`, `design`, `references`, `infiltration`,
  `commit_hillclimb`, `serve_tax`, `partition`), for a model or for every model. `cards.py` holds each slot's fields and checks a card against
  them. A card is published to the evidence store and kept in `cards/SLOT.MODEL.json` with its art id; a model's
  `references` cards add up, each kept as `cards/references.MODEL.ART.json`:

  ~~~sh
  uv run python benchmarks/spot_figures/cards.py check CARD.json
  research data put --kind spot-card/v1 --file CARD.json --preserve --json
  uv run python benchmarks/spot_figures/cards.py fetch art:... [--replaces art:OLD]
  uv run python benchmarks/spot_figures/cards.py list
  ~~~

  A `references` card's optional `caption` is appended to the caption of each panel it has lines on. The `design`
  card's `main_lines` names, per panel and by label prefix (`{model}` for the model's name), the reference lines the
  main figure plots and each one's style (`solid` or `dashed`); a panel it names plots only those, and the appendix
  the rest. Its `notes` are sentences each caption ends with, by panel (`exfiltration_appendix` for the appendix's).

  A model without a `workload` card is refused; no census is guessed from a model's config. A `workload` card marks a
  stratum whose Definition has no cut on main `modeled` and gives the whole circuit's MACs a token
  (`served_macs_per_token`) where its proof blocks leave some out: requests are sized from the served MACs, and the
  figures show the share in no proof block and the share modeled. Its `params`, the model's total parameters, scale
  the references that go as the model's size.

  A `serve_tax` card gives each server configuration's serve tax, what its change to the production server costs on
  every token against its `baseline`, as one number with its range, `priced` or `measured`, a lower bound or not, and
  its parts. `total_cost.py priced-card` prices one from the model's cards; a measured value replaces a priced one
  configuration by configuration, and the figure plots a priced tax hollow.

  A `partition` card is another cut of the model's circuit: its strata take the place of the `workload` card's strata
  of the kinds it `replaces`, at its own `model_overhead`, its blocks proved at their MACs or at their staged witness
  bits' share of a token's proving (`price`), and its `label` says how far it has got.
- **Figure 1** (`total_cost.py`): what an adversary serving the model on all of a scope's compute over a monitoring
  period can exfiltrate or infiltrate while evading detection with probability above δ, against total cost as a share
  of serving compute, the serve tax plus β (the share spent on replays, interior commitments and proofs of the sampled
  proof blocks), at the `proof_overhead` card's marginal overhead; skipped work is a figure of its own.
  - One line per server configuration (`CONFIGURATIONS`), each starting at its own tax, hollow while the tax is
    priced, with its range as a bar: (A) tokens only (the token ids, the schedule record and the KV cache at window
    edges committed; a replay block is a request), (B) each layer's state (a replay block is one layer of one token),
    and (C) every proof-block boundary (one-stage, dashed, its tax a lower bound from hashing alone, the only line
    with skipped work). (A′) is (A)'s serving and tax on the `partition` card `per-expert`'s cut (the MoE cut per
    expert), dash-dot, labelled with the card's `label` (what of the cut is built, checked, proved and served), a
    line only for a model with that card. A new configuration, such as another cut, is a new entry with its own line.
  - Exfiltration: (A) and (B) in the capped sets' form (`card_influenceTwo_le_served_min`) and never above the
    scope's served bits (`card_influenceTwo_le_min`), the JSON saying which term binds at each β; (C) in
    `consumers.exfiltration_bound`'s form, at most the served bits by the same count. The caption names the lines
    that at β = 1% are bounded only by everything the scope serves, the trivial bound, so that value isn't read as a
    result.
  - Infiltration: the write form, the bits the incorrect proof blocks write, at the design each exfiltration line
    plans (`write_form`): each replay block charged at most its footprint, a request's served bits under (A) and a
    layer's boundary bits under (B); (C)'s is the K incorrect blocks its plan detects, each the widest block's bits.
- **The bounds against proof overhead** (`figure1.py`, `render.py`): the same model against marginal proof overhead,
  a curve per share of compute spent on replays, interior commitments and proofs.
  - Scopes are the world (rho's `world-h200` fleet, which computes the census's `world-inference-2027`) and a
    datacenter (the census's `dc-12500-machines-600s`), each over a week, a month and a year
    (`wipe_monitoring.PERIODS`).
  - Costs are the scope's GPU-seconds: a replay costs the `replay` card's factor times its request at serving's MFU,
    proving a proof block takes the marginal proof overhead (the x-axis) times its FLOP at the H200's dense FP8 peak
    plus its share of its statement's fixed cost, and hashing takes the `commitment` card's rate. A statement holds
    one replay block's sampled proof blocks of one Definition, up to the `proof_overhead` card's `statement_blocks`,
    so each design pays at its own fill: one block under one-stage, the k it samples in a request under two-stage, a
    request's blocks under request-level sampling. A one-stage or request-level curve's point is the least target the
    planner's cheapest plan certifies within the budget, and a two-stage one is the planner's design with the least
    bound at the budget (`optimize.two_stage_design`).
  - Each design's serve-time hashing is in its caption and in the table at the vertical line: every proof block's
    outputs under one-stage, outside the budget; the token ids under two-stage and request-level, inside it.
  - Each panel names the deployment it assumes (`DEPLOYMENT`: what serving commits, and the replay block) and shows
    its headline design solid and its comparisons thin, dash-dotted, dashed or dotted:
    - exfiltration: two-stage sampling (serving commits only token ids), width-stratified (the headline: the public
      size classes up to a threshold sampled together and each wider one apart), and request-level sampling. A
      two-stage bound is `CompiledTwo.Exfil`'s count (`optimize.two_stage_exfil`): every admissible set of incorrect
      proof blocks, each charged its own blocks' committed bits, so a wholly incorrect request is charged every bit it
      commits. Beside it (dash-dot) is the same design with each request charged at most the bits it serves, every one
      of its tokens as the token select's 32-bit output word, the `design` card's `token_bytes` (`served_bits`), in
      `two_stage_exfil`'s sets' form (`card_influenceTwo_le_served_min`), which still counts which of a request's proof
      blocks are incorrect, and never above the scope's served bits (`card_influenceTwo_le_min`). `--request
      workload` sizes a request as the `workload` card's instead of the census's mean. The appendix figure adds the
      uniform designs (one sample over a request's proof blocks), the patterns' form, which no lemma states yet (a
      request's incorrect sets of more than its served bits count once; at most the scope's served bits by
      `card_influenceTwo_le_two_pow_outs`), and one-stage
      sampling of dot-product proof blocks, in `consumers.exfiltration_bound`'s form (the incorrect proof blocks'
      output bits and which ones they are), where the adversary makes the widest ones incorrect first and each is
      charged the widest block's output bits;
    - infiltration: the `infiltration` card's design and series, pending without one;
    - skipped work: one-stage and request-level sampling. One-stage needs every proof block's outputs hashed at
      serve time, outside the budget, and the JSON gives that hashing's share of the scope's compute
      (`serve_hash_share`).
  - The vertical line is the `proof_overhead` card's `model_overhead`, the marginal: a token's per-block proving
    seconds, statements' fixed costs excluded, over its MACs at the peak. The reference lines come from rho's reference
    (its reference model's, SparseLoCo's syncs scaled to the served model's parameters), the model's checkpoint and
    every `references` card. A line at 0 is named in the caption, lines that coincide are plotted once, and one far
    above the headline curves is named at the axis's top. A MAC's AND gates, read from the FP8 MoE coordinate's
    circuit-check pin, state each scope's compute and the skipped work in AND gates.
- **Figure 2** (`figure2.py`): proof overhead per proof-block type, the `proof_overhead` card's blocks joined to the
  `workload` card's strata, beside each type's share of a token's proof blocks and MACs. A type without a measurement
  is pending, and a stand-in block is marked as one.
- **Figure 3** (`figure3.py`): the serve-time commitment's cost over fast serving. On the left, the `commit_hillclimb`
  card's series against hill-climbing effort, with each series' best so far. On the right, the model's serve-time
  cuts, priced from their bytes a token and the hash rate, beside the card's measured points for the model.
  `figure3.py card` writes the hash loop's card from its `hash-loop.json`.
- A model's `workload` card comes from the block census of its served circuit, every Program row of its Commit's Build
  on every rank and step of one request, one stratum per proof-block Definition; its `source` names the census.

Limits of the model:
- A replay is priced at the `replay` card's factor; one that must reproduce the served batch's layout costs that
  batch.
- The one-stage curves leave serve-time hashing out of the budget.
- The one-stage certificate counts location bits over every proof block of the scope.
- The uncapped two-stage bound charges a wholly incorrect request every bit it commits, its interior included, not
  only what it serves; the capped ones charge at most what it serves, the sets' form on
  `card_influenceTwo_le_served_min` and the patterns' form on none yet.
- The served bits count every token of a request as served, as the token select's output word alone: a circuit that
  returns more (logits, state) serves more, and the cap is then too low.
- The two-stage write forms need a harm bound at the footprint, which no lemma states yet, and (B) replays a layer of
  one token at its share of its request's replay, from state it retains.
- The registered cut's lines prove each block at its MACs and (A′)'s at its staged witness bits, so where the two
  disagree on a block's share of a token's proving, the lines price it differently; (A′)'s overhead is computed
  from staged k_log, not measured.
