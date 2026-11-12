# spot_figures

The SPOT post's Figures 1-4 for one model, derived from the sampled-proofs planner
(`verity.protocols.spot.optimize`) and the model's cards. A model, not a measurement: every input is a
card marked measured or modeled with its source, and each figure's JSON carries every card it read beside every
plotted series. Lean checks the design behind every exfiltration and skipped-work point of both forms of Figure 1
(`plancheck`, which builds `sampled-proofs-plancheck` with `lake` on first use; `--no-lean` plans without it): a point
plots what Lean certifies, a two-stage one the bits at the θ Lean finds, with the planner's bits beside them
(`figure1.lean_two`, which both forms call), and a design Lean refuses is an error. `--workers` processes (default
every CPU) plan and check the points, each point one job (`figure1.processes`), and the JSON is the same at any
number of them. Which design a budget buys is the planner's pricing of the cards, and the infiltration panels' write
forms are the planner's.

~~~sh
uv run python benchmarks/spot_figures/figures.py --model GLM-5.3 --out DIR   # figure{1,1_overhead,2,3}.json, PNGs, SVGs
                                                       # [--request workload] [--workers N] [--points N] [--no-lean]
                                                       # [--serving measured]
uv run python benchmarks/spot_figures/total_cost.py render DIR/figure1.json  # Figure 1, from the JSON
uv run python benchmarks/spot_figures/render.py DIR/figure1_overhead.json   # the bounds against proof overhead
uv run python benchmarks/spot_figures/figure4.py DIR/figure1.json            # Figure 4 (figures.py writes it too)
uv run python benchmarks/spot_figures/figure5.py SURFACE.json... [--path P] [--headline CARD]  # Figure 5
uv run python benchmarks/spot_figures/figure6.py --advice A.json [--points P...] [--old-rule POINT...] --out DIR
uv run python benchmarks/spot_figures/binding.py --K 2^41.81 --classes 2^26.24 --sessions 15694 --symbols 2^23  # the first-hit bound
research data put --kind spot-figure/v1 --tree DIR --preserve                # for the console
uv run python benchmarks/spot_figures/record.py stamp META.json [--out DIR]  # a serve's stamp: the switches it has off
~~~

- **Cards** (`cards.py`, schema `spot-card/v1`): every number the figures read arrives as one JSON card in a slot
  (`workload`, `proof_overhead`, `commitment`, `replay`, `design`, `references`, `infiltration`,
  `commit_hillclimb`, `serve_tax`, `partition`, `verifier_cost`, `headline`, `serving_rate`), for a model or for every model. `cards.py` holds each slot's fields and checks a card against
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

  A `headline` card holds the headline's rows, each with its cells (by S, verifier step and scope) and its `basis`
  (`cards.BASES`): `served-measured`, the served cut with every type measured on served values and every lever
  measured; `served-card-priced`, the served cut at the proof-overhead card's prices; or `design`, a cut no serving
  path commits, or an estimate. A row short of `served-measured` names what sets its basis (`basis_from`), and a card
  is `measured` only when every row is `served-measured`.

  A model without a `workload` card is refused; no census is guessed from a model's config. A `workload` card marks a
  stratum whose Definition has no cut on main `modeled` and gives the whole circuit's MACs a token
  (`served_macs_per_token`) where its proof blocks leave some out: requests are sized from the served MACs, and the
  figures show the share in no proof block and the share modeled. Its `params`, the model's total parameters, scale
  the references that go as the model's size.

  A `serve_tax` card gives each server configuration's serve tax, what its change to the serving engine costs on
  every token against its `baseline`, as one number with its range, `priced` or `measured`, a lower bound or not, and
  its parts. A part has a name, its share with its range, its source, and how it got its share (`status`: `modeled`
  from the code and a measured rate, `simulated` on served traces, or `measured` on the served engine); a priced tax
  is the sum of its parts, and a card where it isn't is refused. (A)'s parts are the certified-tick rule, the step
  record, the two-stage boundary commitment, the history tables, the token commits with the replay-block edge
  snapshot, and the deterministic top-k, and the break-even planner's `Case.tax` reads each by name.
  `total_cost.py priced-card` prices a card from the model's cards; a measured value replaces a priced one
  configuration by configuration, and the figure plots a priced tax hollow.

  A `serving_rate` card is the engine's measured computed tokens, prompt and output, a GPU-second on the mix its
  `plan` names. It is `saturated` only where a rate above it was served and computed no more; a lower bound is
  `saturated` false on a provisional card, and `bound` says what held the rate (compute; the engine's step, where
  requests waited with room in the KV pool and the record does not tell the GPUs' arithmetic from a step's token
  budget; the KV pool; or unknown). No figure reads it by its presence. `--serving measured` plans every scope at it:
  a scope's requests are its GPU-seconds times the rate over a request's tokens, a replay takes its request's tokens
  over the rate, the scope's GPUs stay rho's, and each exfiltration row names the rate and the rate at which its bound
  would reach 300 GB by proportion (`tokens_per_gpu_second_at_goal_by_proportion`: the bound is taken to go as the
  requests, which a line planned again at that rate need not do). Under a rate that is a lower bound each caption says
  its bounds are lower bounds too. Every row priced at a rate, assumed or measured, holds at a registration of 1/q
  requests, q its sampling rate: the plan counts q·n sampled of a registration's n replay blocks
  (`optimize.registration_cap`), and the first stage draws at least ⌈q·n⌉ of them, never fewer than one (`Spot.Rate`'s
  `count`). None is what serving costs at a smaller registration, and a caption at the measured reading says so. Every
  JSON that read the rate names it under `serving` (Figure 1's two forms, Figure 3, Figure 4 and the break-even, whose
  table starts with the same sentence), and `figures.py` and `total_cost.py breakeven` write a measured reading into
  `measured/` under `--out`, never over the default's files. Without it every figure is planned at rho's fleet
  utilization, as before the card. The `serve_tax` card stays as priced; `total_cost.py priced-card --serving
  measured` prices one at the card's rate.

  A `partition` card is another cut of the model's circuit: its strata take the place of the `workload` card's strata
  of the kinds it `replaces`, each with the k_log it is staged at, and its `label` says how far it has got.

  A `verifier_cost` card gives the verifier's core-seconds a statement by proof-block type (and its peak memory where
  measured), how a family's other types scale, and the core-seconds that cost one GPU-second.
- **The post's configuration** (`record.json`, `record.py`): the 22 switches the post's claim fixes, each with the
  value the post claims and the run that sets it (a serve, the proving run or the planner). `record.py stamp` reads a
  serve's `meta.json` and writes its stamp, `{digest, observed, off}`: the SHA-512 of `record.json`, the serve's value
  of each switch its `meta.json` shows, and the switches a serve sets that it does not show at the post's value (one at
  another value, one shown only in part, one nothing in `meta.json` gives). It holds configuration only, so it is
  published where the serve's own record is private.
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
    layer (`verity/protocols/spot/verifier/README.md`).
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
  the β at which the bound reaches them) over the headline scope, along a path of levers. Each step adds one lever to
  the ones before it. Beside it is the bound at a 10% total cost. It reads break-even surfaces
  (`breakeven_surface.json`) and a path (`figure5.PATH` or `--path`) giving each step's status (measured, modeled or
  assumed), its basis (a headline row's) and the surface case that prices it; a path with a `design` step says what
  makes it one (`design`). The caption calls the first step today's measured system only when it is served and
  measured, and names the step from which every total is a design. A step no surface prices is shown as pending. A
  lever off the path (`off`) is plotted in grey with its owner. A term no step's total includes (`unpriced`) is shown as
  a named line or an arrow, and a `charged` one is stated in the caption as the condition every total holds under. The
  caption ends with the `headline` card's rows (`--headline`, else the model's card), each with its basis.
- **The break-even planner** (`breakeven/`) writes those surfaces. A case is one of Figure 1's request-grain lines,
  priced on `total_cost._point`'s path, with levers on:
  - its priced strata (`levers.py`): table reads fused into their GEMMs, proving scaled, blocks cut into columns;
  - its per-type prices (`reprice.py`): the statement cache, several provers on one GPU, the proof server's prices,
    the Lean verifier's CPU in GPU-seconds, and the 128k workload, each table saying whether it is measured, modeled
    or assumed;
  - its budget's form (`zkstack.py`).

  `sweep.py` gives each case's bound at β of 1%, 5% and 10%, its floor, and the β at which it reaches the FP8 and
  the FP4 weights:

  ~~~sh
  uv run python benchmarks/spot_figures/breakeven/sweep.py --out DIR [--workers N] [--scopes all] [--only NAME...]
  ~~~

  `--steady ROWS.json` (a case's `steady`) prices each block type the file holds a steady-state row of at what the
  row measured (`reprice.steady`: a statement's GPU-seconds, its blocks, its provers and the run or art that holds
  it), in place of the tables' price for it. Each priced stratum then names its basis, that record or the table that
  priced it, and each proof-server case's row gives every basis's share of its proving (`basis`).
- **Figure 6** (`figure6.py`), the certified tick's trade-off:
  - (a) a timeline of continuous admission against tick admission, with the network certifier stamping release
    buckets as its own layer;
  - (b) latency and throughput against the tick period, beside each load's no-tick point, from the sweep's
    `point.json` files, pending without them; a tick point named by `--old-rule` took τ_sat from the no-tick step at
    its own load and is plotted hollow, off its load's line, with its queue growth in the caption;
  - (c) the advice's term, a machine window's bits times the scope's windows, with and without the tick, against the
    FP4 weights. It reads an advice file (`machine_windows`, and `arms` of `label`, `tick` and `bits`, with a
    `pending` reason where the arm's figure isn't a result yet, hatched).
- **The first-hit bound** (`binding.py`): the two-stage first-hit finder's bound on the probability that an accepted
  run opens a value off the committed trace, evaluated in exact rational arithmetic at a design given as
  parameters. A design is stated per sampled replay block; a run at a padded first stage of K0 replay blocks
  (`--first-stage`) has K0 times its sessions and gates. Every figure is per sampled replay block or per run, and
  every line says which. The design: the gates a sampled replay block's sample needs (`--K`, or `--card`: one proof
  block a stratum of the model's `workload` card), k, a session's accepted reruns a wait (`--symbols`, `--tables`),
  and T, the calls of SHA-512 in every run, the prover's and the verifier's.
  - With `--sessions`, the sample's sessions in the mean, it is the by-position form, four terms each printed with
    the premise it holds under: the sessions' knowledge error (`--kappa`), the rewinding term under `ExpectedCR` at
    B and under `StrictBirthday` at B (`--rounds`, `--gamma`), the conflict term, and the first-hit term, which
    reads the sample's mean gates counted once and session by session (`--K`, `--Km`) and the run's own opening
    checks (`--units`, `--open-calls`).
  - Without, it is the by-gate form, which reads the most gates and d, the sessions holding one gate.
  - Each figure is printed in calls, as the premises count a finder's cost, and in compressions, converted at the
    longest call (`--longest-call`, 64 KiB unless stated). With `--classes`, the classes of gates one reveal opens
    whole, a design by position is printed in four columns: by gates and by classes, each in calls and in
    compressions.
  - K_s and K_m as given are means over the verifier's coins. With `--largest`, the most gates one block needs, it
    prints beside them the gates as a count of the sample gives them: the blocks it checks (`--checked`, the
    sessions unless stated) times that. `--largest-classes` does the same by classes.
  - It prints the sums per sampled replay block, per run and over each monitoring period, marks each above 2^-100
    (`--target`), gives the most gates that keep the collision term there, and beside them the term of the limit on
    the extractor's waits at each limit, per run (`--waits-k`).
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
