# spot_figures

The SPOT post's figures, for one model: what an adversary serving the model on all of a scope's compute could
exfiltrate, infiltrate or skip while evading detection, against what the audit costs. They are derived from the SPOT
planner (`verity.protocols.proofs.spot.prover.optimize`) and the model's cards. A model, not a measurement: every input
is a card marked measured or modeled with its source, and each figure's JSON carries every card it read beside every
plotted series.

## Try it

The first-hit bound, at a design given as parameters, evaluates in exact arithmetic in under a second:

~~~sh
uv run python benchmarks/spot_figures/binding.py --K 2^41.81 --classes 2^26.24 --sessions 15694 --symbols 2^23
~~~

It prints the bound's terms in four columns (by gates or by classes, in calls or in compressions), then for each column
the largest K at which the first-hit term stays at most 2^-100 per sampled replay block, per run, and over a week, a
month and a year of runs. The figures themselves (`figures.py`, below) plan every
point on every CPU and take minutes.

## Commands

~~~sh
uv run python benchmarks/spot_figures/figures.py --model GLM-5.3 --out DIR   # figure{1,1_overhead,2,3}.json, PNGs, SVGs
                                                       # [--request workload] [--served request] [--workers N]
                                                       # [--points N] [--no-lean] [--serving measured]
uv run python benchmarks/spot_figures/total_cost.py render DIR/figure1.json  # Figure 1, from the JSON
uv run python benchmarks/spot_figures/total_cost.py counts --line tokens_datacenter --beta 0.1 --out COUNTS.json
                                                       # a line's design as the counts the verifier's rule takes
uv run python benchmarks/spot_figures/render.py DIR/figure1_overhead.json   # the bounds against proof overhead
uv run python benchmarks/spot_figures/figure4.py DIR/figure1.json            # Figure 4 (figures.py writes it too)
uv run python benchmarks/spot_figures/figure5.py SURFACE.json... [--path P] [--headline CARD]  # Figure 5
uv run python benchmarks/spot_figures/figure6.py --advice A.json [--points P...] [--old-rule POINT...] --out DIR
uv run python benchmarks/spot_figures/binding.py --K 2^41.81 --classes 2^26.24 --sessions 15694 --symbols 2^23  # the first-hit bound
research data put --kind spot-figure/v1 --tree DIR --preserve                # for the console
uv run python benchmarks/spot_figures/record.py stamp META.json [--out DIR]  # a serve's stamp: the switches it has off
~~~

## Figure 1

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
    comparison at one machine's verification phase that pays their measured tax (`REPLAYS`, `BATCH_INVARIANT`). (A′) is (A)'s serving and tax on the `partition` card `per-expert`'s cut (the MoE cut per
    expert), dash-dot, labelled with the card's `label` (what of the cut is built, checked, proved and served), a
    line only for a model with that card; (A″) is the same on the card `head-scores-values`'s cut (a sparse-attention
    head's key block as its scores proof block and its values proof blocks on column blocks, which read the block's key rows from
    their writers' committed rows). A new configuration, such as another cut, is a new
    entry with its own line.
  - A two-stage line's first stage is the one the verifier runs: a verification phase, one committed trace and the
    sample over it, samples a whole number K of its replay blocks, never fewer than one
    (`optimize.two_stage_design`'s `registration`, `optimize.registration_sample`). A verification phase spans the
    census window on one machine, on a 64-machine pod (`POD`) or on all of the census datacenter's machines
    (`REGISTRATIONS`), and each point is priced and bounded at K over its replay blocks. Where β does not pay for
    one sampled replay block a verification phase, no design is within the budget: the point is at the scope's
    served bits, with no plan, and says what one sampled replay block costs against the phase's budget
    (`sample_budgets`); a point with a design gives the same for its design, so the least verification phase the
    design fits. Its second stage's strata are the ones the verifier's two-stage rule registers, one a block type
    of the cut (`total_cost.STRATA`, `optimize.two_stage_design`'s `stratify` "template"): each block type samples
    its own whole count of a sampled replay block's proof blocks, at least one, where the planner's width-stratified
    design samples a count among several types, which that rule cannot state. `total_cost.py counts` writes a line's
    design at a β as the rule takes it, the first stage's count and each block type's by the cut's name
    (`optimize.two_stage_counts`). A statement holds one Definition's sampled proof blocks of a sampled replay block,
    or of the verification phase's K, its shape fixed before any coin because K is, its sessions padded to one size
    and none larger than its type's statement on the `proof_overhead` card (`statement_blocks`), and each Definition
    takes the phase's statements only where they cost less than its own (`optimize.two_stage_statements`). The
    verifier's second stage reads one session of one table for each template of a verification phase
    (`Spot.SecondStage`: `second-template`, `second-tables`), and a table holds at most 2^35 witness bits
    (`total_cost.table_blocks`: 32,768 proof blocks of 2^20 bits, 256 of 2^27; a type's blocks are sized as the
    `proof_overhead` card's block that prices it is staged, not by the strata's own estimate), so a point with more
    sessions than block types (`sessions`, in each series and caption) prices sessions the verifier refuses. Beside
    each such
    line's bound at β = 1%, and in `total_cost.py counts` at any β, are the designs it could be instead
    (`total_cost.session_designs`), all modeled at the seconds a proof block the line's own statements are priced
    at. Two have one session a block type (`optimize.SESSIONS`): one whose session holds whatever the design
    samples, which the verifier refuses where it is more than one table and which is printed with its bound only
    where every session is within the tables a session of the proof system holds (`SESSION_TABLES`, 200), and
    otherwise with its largest session in proof blocks and in tables; and one whose sessions are each at most one
    table, the form the verifier's second stage reads, which samples far fewer proof blocks and bounds far more. The
    others have sessions of at most 1, 16, 64 and 200 tables of the type (`TABLES`), as many as hold its sampled
    proof blocks, each with its sessions a verification phase and the seconds between them if they run one at a
    time; the verifier's second stage refuses more than one session a template. A design, the line's own or one
    beside it, is printed as out of reach where it meets a limit of the code that runs a second stage today, with
    the first it meets (`total_cost.limits`): a session of more tables than the GPU prover (1), the CPU prover and
    the Lean verifier (64), the challenger's coin commitment (127) or the proved scope (200) takes; more lines of
    session coins than one challenger session's release log holds; or a second sample of more bytes of live coins
    than one layout holds. Its bound is then the cost model's alone, of a deployment no code runs. (A), (A′) and
    (A″) are plotted at each of the three sizes, dark to light, and (B) at one machine's; each series gives K, its
    sessions and the shares of its proving in the phase's statements and on padding. No code builds a verification
    phase of a datacenter's size, one committer over every machine's boundary rows: that line is the cost model's.
  - A two-stage point's bound is for a deployment whose second stage has the coins its cells read first: on fewer it
    refuses every honest run. Each two-stage exfiltration series carries them at every β that has a design
    (`second_stage.at_design`): a sampled replay block's cells, the proof blocks sampled of them and the bytes they
    read first; the rounds of 2^20 bytes and the challenger's 64-byte coins that hold the first reads of the K replay
    blocks a verification phase samples, and the replay blocks whose first reads one commitment of the challenger's
    holds (`challenger.MAX_COINS` coins); Lean's bound on the chance the second stage then refuses an honest run at a
    first-stage coin the first stage accepts, K·Σ_t k_t·2^−64 (`SpotFlock.secondDraw_refused_le_first`); and that
    bound added up over the replay blocks the scope samples. `total_cost.limits`' second sample is the same read with a
    block type of under one proof block a replay block counted at the chance a replay block has it; `at_design`
    counts every cell, the theorem's condition.
  - Exfiltration: (A) and (A′) in the patterns' form (`card_influenceTwo_le_patterns`), under its two modeling
    conditions (each request serves at most S bits, and no proof block is shared across requests), with the capped
    sets' form (`card_influenceTwo_le_served_min`) at one machine's verification phase beside each, thin and dashed;
    (B) in
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
  - Scopes are the world (the slowdown calculator's `world-h200` fleet, which computes the census's `world-inference-2027`) and a
    datacenter (the census's `dc-12500-machines-600s`), each over a week, a month and a year
    (`verity.protocol.MONITORING_PERIODS`).
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
    `proof_overhead` card's `dev` arm); and the verifier samples on the challenger's live coins and decides with the Lean
    verifier. The network attester, which stamps each request's release order and bucket at ingress, is a layer of
    its own. The bounds are soundness alone, resting on SHA-512's collision resistance, uniform verifier randomness and
    an honest verifier, and, where a bound reads the network attester's stamp, on network attestation's own assumptions, stated in its
    layer (`verity/protocols/proofs/spot/verifier/README.md`).
  - Each panel names the deployment it assumes (`DEPLOYMENT`: what serving commits, and the replay block) and shows
    its headline design solid and its comparisons thin, dash-dotted, dashed or dotted:
    - exfiltration: two-stage sampling (serving commits only token ids), width-stratified (the headline: the public
      size classes up to a threshold sampled together and each wider one apart), and request-level sampling. A
      two-stage bound is `CompiledTwo.Exfil`'s count (`optimize.two_stage_exfil`): every admissible set of incorrect
      proof blocks, each charged its own blocks' committed bits, so a wholly incorrect request is charged every bit it
      commits. Beside it (dash-dot) is the same design with each request charged at most the bits it serves, each of
      its tokens as the token select's 32-bit output word, the `design` card's `token_bytes` (`served_bits`): at most
      the response cap of the model's `certifier` card, the network attester's schedule (`NetworkAttesterParams`), whose cap
      ingress refuses a request past and `NetTiming.Guarantees.ResponseCapped` holds each response to, else every one
      of its tokens (`--served request` takes every token under a cap too), in
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
    seconds, statements' fixed costs excluded, over its MACs at the peak. The reference lines come from the slowdown calculator's reference
    (its reference model's, SparseLoCo's syncs scaled to the served model's parameters), the model's checkpoint and
    every `references` card. A line at 0 runs along the axis's floor and is named in the caption, lines that coincide
    are plotted once, and one far above the headline curves is named at the axis's top. Each line's label wraps to stay inside its axis, beside its
    line where it overlaps the fewest others. A MAC's AND gates, read from the FP8 MoE coordinate's
    circuit-check pin, state each scope's compute and the skipped work in AND gates.

## Limits

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
- The patterns' form takes no proof block to be shared across requests and each request to serve exactly S bits.
  Replay blocks that read one another's values (a batched or routed block that several requests read, a value
  another replay block committed) are one request there, priced at the replay grain's fan-out F as serving F S bits
  (`optimize.two_stage_exfil`'s `fanout`). Lean certifies that point only for a window whose groups each hold F replay
  blocks, and `plancheck.two` writes it only for a caller that says they do (`whole_groups`). Every figure here is at
  F = 1 (`figure1.FANOUT`).
- A verification phase's statements are sized for the first stage's count alone: a Definition's blocks in a sampled
  replay block are its own count, and its sessions' padding to one size within a statement, not to one size across
  Definitions. A pod-wide or datacenter-wide verification phase needs one committer over its machines' boundary rows.
- A total-cost line's design has the counts the verifier's two-stage rule registers and more sessions than its
  second stage reads. One session a template of a verification phase would hold every sampled proof block of the
  template, many tables of it, where a table is the proof system's limit and the verifier reads one; held to one
  table a session, the design samples far fewer proof blocks. Both are printed beside the line, modeled, and
  neither is the line. Figure 1's own panels (`figure1.py`) still plot the planner's width-stratified and uniform
  designs over one verification phase holding every replay block, which the verifier's rule registers neither.
- A two-stage bound spends δ once over the scope's replay blocks and counts them as one population, each
  verification phase's sample taken to miss its incorrect replay blocks independently of the others'. Lean checks
  that count for one design over the population (`TwoStageCheckPatternsSound`); no Lean statement carries it across
  the scope's verification phases.
- The served bits count every token of a request as served, as the token select's output word alone: a circuit that
  returns more (logits, state) serves more, and the cap is then too low. Under the network attester's response cap a
  request serves at most S, which the patterns' form reads as exactly S, and the tie of the statement's served outputs
  to the tokens the network attester releases is pending, so the network attester card is provisional.
- The two-stage write forms need a harm bound at the footprint, which no lemma states yet, and (B) replays a layer of
  one token with its step's batch, the requests in flight a request's lifetime over the window (`LIVE`), from state it
  retains.
- A type the `proof_overhead` card doesn't measure is priced from its staged witness bits, not measured, its
  statement's fixed cost included; the caption gives each line's share of it.

