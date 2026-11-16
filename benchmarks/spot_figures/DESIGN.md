# spot_figures: design

How the figures are planned and checked, the cards they read, and every figure but Figure 1. `README.md` says what
the figures show, lists the commands and holds Figure 1 and the model's limits.

## Planning and checking

The SPOT post's Figures 1-4 for one model, derived from the SPOT planner
(`verity.protocols.proofs.spot.prover.optimize`) and the model's cards. A model, not a measurement: every input is a
card marked measured or modeled with its source, and each figure's JSON carries every card it read beside every
plotted series. Lean checks the design behind every exfiltration and skipped-work point of both forms of Figure 1
(`plancheck`, which builds `spot-plancheck` with `lake` on first use; `--no-lean` plans without it): a point
plots what Lean certifies, a two-stage one the bits at the θ Lean finds, with the planner's bits beside them
(`figure1.lean_two`, which both forms call), and a design Lean refuses is an error. `--workers` processes (default
every CPU) plan and check the points, each point one job (`figure1.processes`), and the JSON is the same at any
number of them. Which design a budget buys is the planner's pricing of the cards, and the infiltration panels' write
forms are the planner's.

## Cards and figures

- **Cards** (`cards.py`, schema `spot-card/v1`): every number the figures read arrives as one JSON card in a slot
  (`workload`, `proof_overhead`, `commitment`, `replay`, `design`, `references`, `infiltration`,
  `commit_hillclimb`, `serve_tax`, `partition`, `verifier_cost`, `headline`, `serving_rate`, `certifier`), for a model or for every model. `cards.py` holds each slot's fields and checks a card against
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
  over the rate, the scope's GPUs stay the slowdown calculator's, and each exfiltration row names the rate and the rate at which its bound
  would reach 300 GB by proportion (`tokens_per_gpu_second_at_goal_by_proportion`: the bound is taken to go as the
  requests, which a line planned again at that rate need not do). Under a rate that is a lower bound each caption says
  its bounds are lower bounds too. Every two-stage row, at a rate assumed or measured, holds at its verification phase's size: the first
  stage samples a whole number of the phase's replay blocks, never fewer than one (`Spot.Rate`'s `count`), and the
  plan is priced and bounded at that count (`optimize.registration_sample`). None is what serving costs at a
  smaller verification phase, and a caption at the measured reading says so. Every JSON that read the rate names it
  under
  `serving` (Figure 1's two forms, Figure 3, Figure 4 and the break-even, whose table starts with the same sentence),
  and `figures.py` and `total_cost.py breakeven` write a measured reading into `measured/` under `--out`, never over
  the default's files. Without it every figure is planned at the slowdown calculator's fleet utilization, as before the card. The
  `serve_tax` card stays as priced; `total_cost.py priced-card --serving measured` prices one at the card's rate.

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
  - (a) a timeline of continuous admission against tick admission, with the network attester stamping release
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
  `workload_card.py` writes it from the census file and nothing else, so no stratum is edited by hand. The 128k
  workload's sparse-MLA blocks are that card's, counted over the serve plan's requests as serving splits a row
  (`breakeven/mla128.py`). The figure of record's driver (`breakeven/headline.py`) cuts the head as it is served, by
  the `partition` card `head-scores-values`: each head's key block its scores proof block and its values proof blocks on 64
  columns, which read the block's key rows from their writers' committed rows (one writer a column block for all the
  heads); its rows carry that card's label. `--whole` prices the head as the `workload` card's whole key blocks and end
  block instead, for comparison.

