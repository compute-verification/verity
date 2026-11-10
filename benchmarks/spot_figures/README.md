# spot_figures

The SPOT post's Figures 1-4 for one model, derived from the sampled-proofs planner
(`verity.core.protocols.spot.optimize`) and the model's cards. A model, not a measurement: every input is a
card marked measured or modeled with its source, and each figure's JSON carries every card it read beside every
plotted series. Lean checks the design behind every exfiltration and skipped-work point of both forms of Figure 1
(`plancheck`, which builds `sampled-proofs-plancheck` with `lake` on first use; `--no-lean` plans without it): a point
plots what Lean certifies, a two-stage one the bits at the θ Lean finds, with the planner's bits beside them
(`figure1.lean_two`, which both forms call), and a design Lean refuses is an error. Which design a budget buys is the
planner's pricing of the cards, and the infiltration panels' write forms are the planner's.

~~~sh
uv run python benchmarks/spot_figures/figures.py --model GLM-5.3 --out DIR   # figure{1,1_overhead,2,3}.json, PNGs, SVGs
                                                       # [--request workload] [--workers N] [--points N] [--no-lean]
uv run python benchmarks/spot_figures/total_cost.py render DIR/figure1.json  # Figure 1, from the JSON
uv run python benchmarks/spot_figures/render.py DIR/figure1_overhead.json   # the bounds against proof overhead
uv run python benchmarks/spot_figures/figure4.py DIR/figure1.json            # Figure 4 (figures.py writes it too)
uv run python benchmarks/spot_figures/figure5.py SURFACE.json... [--path P]  # Figure 5, from break-even surfaces
uv run python benchmarks/spot_figures/figure6.py --advice A.json [--points P...] [--old-rule POINT...] --out DIR
research data put --kind spot-figure/v1 --tree DIR --preserve                # for the console
~~~

- **Cards** (`cards.py`, schema `spot-card/v1`): every number the figures read arrives as one JSON card in a slot
  (`workload`, `proof_overhead`, `commitment`, `replay`, `design`, `references`, `infiltration`,
  `commit_hillclimb`, `serve_tax`, `partition`, `verifier_cost`), for a model or for every model. `cards.py` holds each slot's fields and checks a card against
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

  A `serve_tax` card gives each server configuration's serve tax, what its change to the serving engine costs on
  every token against its `baseline`, as one number with its range, `priced` or `measured`, a lower bound or not, and
  its parts. `total_cost.py priced-card` prices one from the model's cards; a measured value replaces a priced one
  configuration by configuration, and the figure plots a priced tax hollow.

  A `partition` card is another cut of the model's circuit: its strata take the place of the `workload` card's strata
  of the kinds it `replaces`, each with the k_log it is staged at, and its `label` says how far it has got.

  A `verifier_cost` card gives the verifier's core-seconds a statement by proof-block type (and its peak memory where
  measured), how a family's other types scale, and the core-seconds that cost one GPU-second.
- **Figure 1** (`total_cost.py`): what an adversary serving the model on all of a scope's compute over a monitoring
  period can exfiltrate or infiltrate while evading detection with probability above δ, against total cost as a share
  of serving compute, the serve tax plus β (the share spent on replays, interior commitments, proofs of the sampled
  proof blocks and their verification); skipped work is a figure of its own.
  - Each statement's verify (`verified`) is its type's core-seconds on the `verifier_cost` card at its core-seconds a
    GPU-second, added to the statement's fixed seconds; without the card, β counts proving only and the caption says
    so.
  - Every line prices a proof block one way (`priced`): at its type's seconds on the `proof_overhead` card, a block's
    and its statement's fixed ones, with its statement's cap, the type being the card's block of the same name, else
    of the same Definition and K; a type the card doesn't measure is computed from its staged witness bits (the
    measured types' median seconds a witness bit, a fixed cost on their least-squares line in k_log, a statement's
    2^35 bits). A line's marginal overhead is what its blocks come to over its MACs at the card's peak. The caption
    gives each line's computed share twice: of its marginal overhead, and at β = 1% of its sampled replay block's
    proving (`computed_proving`), where a design that samples a few blocks of a type pays its statement's whole
    fixed cost, so a type small in the marginal can be much of the proving.
  - One line per server configuration (`CONFIGURATIONS`), each starting at its own tax, hollow while the tax is
    priced, with its range as a bar: (A) tokens only (the token ids, the schedule record and the KV cache at
    replay-block edges committed; a sampled request replays its replay block, one machine's window, whole from its
    edge), (B) each layer's state (a sampled layer of one token replays its engine step's whole batch at that layer),
    and (C) every proof-block boundary (one-stage, dashed, its tax a lower bound from hashing alone, priced and not
    measured, the only line with skipped work). (A) replayed alone, which needs batch-invariant kernels, is a dotted
    comparison at one machine's registration that pays their measured tax (`REPLAYS`, `BATCH_INVARIANT`). (A′) is (A)'s serving and tax on the `partition` card `per-expert`'s cut (the MoE cut per
    expert), dash-dot, labelled with the card's `label` (what of the cut is built, checked, proved and served), a
    line only for a model with that card. A new configuration, such as another cut, is a new entry with its own line.
  - A two-stage statement holds one Definition's sampled proof blocks of a sampled replay block, or of a
    registration's (`optimize.two_stage_design`'s `registration`): a registration spans the census window on one
    machine, on a 64-machine pod (`POD`) or on all of the census datacenter's machines (`REGISTRATIONS`). A statement's
    shape is fixed before any coin, so a registration's sessions are padded to a public cap, K0 replay blocks' blocks,
    the least with P(K > K0) ≤ 2^−40 at the Q it samples (`optimize.registration_cap`), a sample past it refused;
    every registration proves its padding whatever it samples, and each Definition takes the registration's statements
    only where they cost less than its own (`optimize.two_stage_statements`). (A) and (A′) are plotted at each of the
    three, dark to light, and (B) at one machine's; each series gives Q, K0 and the shares of its proving in the
    registration's statements and on padding.
  - Exfiltration: (A) and (A′) in the patterns' form (`card_influenceTwo_le_patterns`), under its two modeling
    conditions (each request serves at most S bits, and no proof block is shared across requests), with the capped
    sets' form (`card_influenceTwo_le_served_min`) at one machine's registration beside each, thin and dashed; (B) in
    the sets' form. Each is never above the scope's served bits (`card_influenceTwo_le_two_pow_outs`,
    `card_influenceTwo_le_min`), the JSON saying which term binds at each β; (C) is in
    `consumers.exfiltration_bound`'s form, its count the bits Lean certifies, at most the served bits
    (`audit_exfiltration_delivered`). The caption names the lines
    that at β = 1% are bounded only by everything the scope serves, the trivial bound, so that value isn't read as a
    result.
  - The break-even (`breakeven`, `breakeven.txt` and `figure1.json`'s `breakeven`): for each patterns'-form line and
    scope, the marginal overhead at which its bound at β = 1% reaches the FP8 and the FP4 weights, the β at which it
    does at its own overhead, and its floor, the bound as proving's cost goes to nothing. Each is certified by Lean
    unless `--no-lean` (`certify`): at the β it gives, the bits Lean certifies for the planner's design there, or the
    served bits if fewer, are at most the weights, so it is an upper bound on the β the weights need (a lower bound on
    the overhead). Only that side is Lean's: Lean bounds a design's bits from above, so no certificate says a smaller
    β can't reach them, and which design a β buys is the planner's pricing of the cards.
  - Infiltration: the write form, the bits the incorrect proof blocks write, at the design each exfiltration line
    plans (`write_form`): each replay block charged at most its footprint, a request's served bits under (A) and a
    layer's boundary bits under (B); (C)'s is the K incorrect blocks its plan detects, each the widest block's bits.
- **The bounds against proof overhead** (`figure1.py`, `render.py`): the same model against marginal proof overhead,
  a curve per share of compute spent on replays, interior commitments and proofs.
  - Scopes are the world (rho's `world-h200` fleet, which computes the census's `world-inference-2027`) and a
    datacenter (the census's `dc-12500-machines-600s`), each over a week, a month and a year
    (`reporting.PERIODS`).
  - Costs are the scope's GPU-seconds: a replay costs the `replay` card's factor times what it re-runs at serving's
    MFU (`total_cost.batch`: a machine's replay block, an engine step's batch at one layer, or one request),
    proving a proof block takes the marginal proof overhead (the x-axis) times its FLOP at the H200's dense FP8 peak
    plus its share of its statement's fixed cost, and hashing takes the `commitment` card's rate. A statement holds
    one replay block's sampled proof blocks of one Definition, up to the `proof_overhead` card's `statement_blocks`,
    so each design pays at its own fill: one block under one-stage, the k it samples in a request under two-stage, a
    request's blocks under request-level sampling. A one-stage or request-level curve's point is the least target the
    planner's cheapest plan certifies within the budget, and a two-stage one is the bits Lean certifies for the
    planner's design with the least bound at the budget (`optimize.two_stage_design`, `figure1.lean_point`), each
    series giving the planner's bits and Lean's θ beside them.
  - Each design's serve-time hashing is in its caption and in the table at the vertical line: every proof block's
    outputs under one-stage, outside the budget; the token ids under two-stage and request-level, inside it.
  - Every caption says who the deployment's parties are and what its bounds rest on (`ROLES`, `TRUST`). Its parties
    are SPOT[Flock]'s three roles: the inference server serves and commits its values; the proof server replays each
    sampled replay block and proves it with `flock-circuit prove --zk`, with no ZK server and no recursion (the
    `proof_overhead` card's `dev` arm); and the verifier draws on the challenger's live coins and decides with the Lean
    verifier. The network certifier, which stamps each request's release order and bucket at ingress, is a layer of
    its own. The bounds are soundness alone, resting on SHA-512's collision resistance, uniform verifier randomness and
    an honest verifier, and, where a bound reads the certifier's stamp, on the certifier's own assumptions, stated in its
    layer (`verity/core/protocols/spot/README.md`).
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
      uniform designs (one sample over a request's proof blocks), the patterns' form
      (`card_influenceTwo_le_patterns`: a request's incorrect sets of more than its served bits count once; at most
      the scope's served bits by `card_influenceTwo_le_two_pow_outs`), and one-stage
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
    every `references` card. A line at 0 runs along the axis's floor and is named in the caption, lines that coincide
    are plotted once, and one far above the headline curves is named at the axis's top. Each line's label wraps to stay inside its axis, beside its
    line where it overlaps the fewest others. A MAC's AND gates, read from the FP8 MoE coordinate's
    circuit-check pin, state each scope's compute and the skipped work in AND gates.
- **Figure 2** (`figure2.py`): proof overhead per proof-block type, the `proof_overhead` card's blocks joined to the
  `workload` card's strata, beside each type's share of a token's proof blocks and MACs. A type without a measurement
  is pending, and a stand-in block is marked as one.
- **Figure 3** (`figure3.py`): the serve-time commitment's cost over fast serving. On the left, the `commit_hillclimb`
  card's series against hill-climbing effort, with each series' best so far. On the right, the model's serve-time
  cuts, priced from their bytes a token and the hash rate, beside the card's measured points for the model.
  `figure3.py card` writes the hash loop's card from its `hash-loop.json`.
- **Figure 4** (`figure4.py`), for 'Randomness is powerful': the sampling rate that keeps the model's FP4 weights in at
  δ = 2^-40, against a proof block's output bits. It is sampled in the simple form q = b ln(1/δ)/T and in the planner's
  exact one-stage form, which adds location bits, over the headline and stretch scopes. Beside it is the rate a 10%
  total cost buys at each line's marginal proof overhead, counting proving alone.
- **Figure 5** (`figure5.py`), where 10% comes from: the break-even total cost at the FP4 weights (serve tax plus
  the β at which the bound reaches them) over the headline scope, along a path of levers from today's measured system.
  Each step adds one lever to the ones before it. Beside it is the bound at a 10% total cost. It reads break-even
  surfaces (`breakeven_surface.json`) and a path (`figure5.PATH` or `--path`) giving each step's status (measured,
  modeled or assumed) and the surface case that prices it. A step no surface prices is shown as pending. A lever off
  the path (`off`) is plotted in grey with its owner. A term no step's total includes (`unpriced`) is shown as a named
  line or an arrow, and a `charged` one is stated in the caption as the condition every total holds under.
- **Figure 6** (`figure6.py`), the certified tick's trade-off:
  - (a) a timeline of continuous admission against tick admission, with the network certifier stamping release
    buckets as its own layer;
  - (b) latency and throughput against the tick period, beside each load's no-tick point, from the sweep's
    `point.json` files, pending without them; a tick point named by `--old-rule` took τ_sat from the no-tick step at
    its own load and is plotted hollow, off its load's line, with its queue growth in the caption;
  - (c) the advice's term, a machine window's bits times the scope's windows, with and without the tick, against the
    FP4 weights. It reads an advice file (`machine_windows`, and `arms` of `label`, `tick` and `bits`, with a
    `pending` reason where the arm's figure isn't a result yet, hatched).
- The headline scope (`figure1.PRIMARY`, a world year) gets the bare file names. Its exfiltration panel also plots
  (A) over the secondary scope (`figure1.SECONDARY`, a datacenter year).
- A model's `workload` card comes from the block census of its served circuit, every Program row of its Commit's Build
  on every rank and step of one request, one stratum per proof-block Definition; its `source` names the census.

Limits of the model:
- A replay re-runs its whole replay block from its edge at the `replay` card's factor. Each sampled request pays its
  own block: right while a block holds far fewer than one sample, as at β = 1%; past that, samples sharing a block
  share its replay, so the planner overprices them. Stopping at the last sampled step or starting from a nearer edge
  is left out.
- The one-stage curves leave serve-time hashing out of the budget.
- The one-stage certificate counts location bits over every proof block of the scope.
- The uncapped two-stage bound charges a wholly incorrect request every bit it commits, its interior included, not
  only what it serves; the capped ones charge at most what it serves, the sets' form on
  `card_influenceTwo_le_served_min` and the patterns' form on `card_influenceTwo_le_patterns`.
- The patterns' form takes each request to serve at most S bits and no proof block to be shared across requests: a
  batched or routed block that several requests read needs the planner changed.
- A registration's cap pads only for the first stage's count: a Definition's blocks in a sampled replay block are
  taken at their mean share of its stratum's k, and its sessions' padding to one size within a statement, not to one
  size across Definitions. A pod-wide or datacenter-wide registration needs one committer over its machines' boundary
  rows.
- The served bits count every token of a request as served, as the token select's output word alone: a circuit that
  returns more (logits, state) serves more, and the cap is then too low.
- The two-stage write forms need a harm bound at the footprint, which no lemma states yet, and (B) replays a layer of
  one token with its step's batch, the requests in flight a request's lifetime over the window (`LIVE`), from state it
  retains.
- A type the `proof_overhead` card doesn't measure is priced from its staged witness bits, not measured, its
  statement's fixed cost included; the caption gives each line's share of it.
