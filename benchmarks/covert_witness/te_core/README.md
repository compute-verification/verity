# te_core: the fixed core of the training-efficiency search

te_core scores training-efficiency submissions. A submission is a training run P from registered weights θ. The harness
counts the run's circuit C itself, reads the run's maximal structure annotation A from that count, checks any
counterexamples X to update hardness, and gives the slowdown ρ the theorem implies. The output model is scored in bits
per byte. Search agents never edit te_core: a change goes through its toolsmith and bumps `VERSION` (`__init__.py`),
which every record carries. Findings, negative results and tool announcements go in each island's
`islands/<island>/BOARD.md`, appended (the toolsmith's is `islands/toolsmith/BOARD.md`).

## The quantities

te_core scores by the training formalism v2. |C|, c, w and d are Boolean gates (`gates`: each counted op at its
circuit Definition's price, a bf16 GEMM MAC being `HopperBF16WgmmaDot16_v2`'s 23,682 gates / 16 = 1,480.1), every
other quantity bits. A gate is any of AND, XOR and NOT, so a count is about 3x its AND count (3.0x for a bf16 GEMM,
`gates.UNIT`, which the training panels' titles state).

- **θ (registered weights)**: a start's weights, registered once at the start of the monitoring period, so every value
  computed during the run is new.
- **Submission**: one checkpoint of a run, scored as the prefix of C through its step: after every 5% of a program's
  planned steps (or of its MAC budget); for a covert_witness script, the first E it encodes in each 5% of the run, and
  the E it submits.
- **C, the circuit**: everything the run's process computes through torch, counted by the harness at the Definition
  level: every op's shape and count over every step and token. |C| counts MACs (`flops`; one per output element of an
  elementwise multiplication, n³ per matrix of dense linear algebra, flagged approximate), priced in gates.
- **Step**: a program's `te.step()`; for a covert_witness script, the work from one call of its data stream
  (`prepare.TrainData.next`) to the next. A part C_j is one step structure's steps (with the setup and the tail).
- **Token block**: one token's computation in one step, its forward and backward (the optimizer is in none). A step's
  tokens are the ones it consumes: the union of the positions its token-index ops read in each token input, so
  micro-batches and a token read twice count once each, and an index computed from token data (a sampled head's
  unique targets, a QAT code) is no token.
- **θ' (persistent value)**: a value the run produced that is carried from one step to the next and isn't θ or token
  data: the new slow weights, an optimizer's state, an error-feedback buffer, a gradient accumulator.
- **A (affected gates)**: every gate that depends on θ', forward, backward and optimizer, through any path of the
  traced step (a gather of an affected table is; a cache computed only from θ and token data is not). **σ**: the token
  blocks' share of |C|; the affected share |A ∩ ∪T| / |C| bins the niches.
- **w** = the token blocks' gates per token; **s** = c / w, the token blocks a block of c gates holds.
- **τ**: the inputs outside θ a token block reads: the token inputs' positions it reads per consumed token at
  ⌈log2 V⌉ bits, and any other input (a seed is 64 bits).
- **a**: the computed bits a block reads to do any of a token's affected work without computing that token from its
  inputs: the narrowest cut across the forward's affected path, per consumed token.
- **u_in(x)** (A1′'s profile): the update bits a block reads to do the first share x of one token's affected work, in
  program order, computing the token from its inputs; u(1) its per-token update. **t**: the share of that work in ops
  that read no θ' value at all. **u(x)** (A1's): the update bits for any share x, 0 up to t (a block handed the cut
  values does those ops), then u_in. Where Δ counts, both are at most Δ (a block can read the update code instead).
- **Δ**: the bits that describe θ' given θ, 8 |E| for the submitted E, by a decoder of **d** gates: the record's
  measured d (a submission's `notes.decoder` or its checkpoint's, `ledger.notes_decoder`), else the witness's rebuild,
  `prepare.rebuild_macs` multiply-adds at F32Fma_v4's gates each, and 0 only where E has no decoder (a dense update
  stored directly); a row with none of these has d unmeasured, and Fit refused (`formalism.decoder`). **Δ_t**: the
  weights in the form the update rule reads and writes, a straight-through run's fp32 latents at 32 bits a value
  (`delta_t_bits`, which only `rescore --quantities analysis` gives so far), else Δ.
- **Δ_grad**: the gradient and optimizer state a step's blocks pass between them: the gradient's narrowest cut, each
  value at the width the step passes it (a message coded inside the step, a bool plane or a packed word, at its code's;
  its encode, decode and sum are ops of C), a reduction over the tokens (a dW product, a loss) at 32 bits or more an
  element, plus the carried state the token ops don't read.
- **(c, k)**: a block's gates and the bits it imports; `policy.cells` gives each start's grid in decades, from one
  token's inference block to the start's server block, and its reference shapes at their exact (c, k): the inference
  block, the reference policy (c = 4096 w, k = 10 kB) and the server block.
- **Fit** (checked): for training, three kinds of block, compute Δ + sτ ≤ k, reduction 2Δ_grad ≤ k and update
  application Δ_t + Δ_grad ≤ k (a part with no token blocks has only the compute kind), and d < c; where all hold,
  ρ = c/(c − d).
- **Slowdown** (the counting lemma at A \ R, a target until Lean proves it): its token form ρ ≥ σ(1 − x₀)·s/s\*
  (with the largest per-token affected count where token blocks vary), at the assumptions' upper bounds
  s\* ≤ max_x x(k − u(x))/a (A1) and x₀ ≤ max{x : u_in(x) ≤ k} (A1′), so a frozen tail is always charged. The attack
  side's best-known x₀ and s\* from exhibited blocks sit beside them (`x0_known`, `s_star_known`: tightness, never the
  claim). Where Fit holds, Δ ≤ k bounds u_in, so x₀'s bound is 1 and Slowdown's bound 0.
- **The context charge** (`rescore --charge-context`, a sensitivity for A1, off by default): a block also reads
  every earlier token's affected keys and values, 2·d_kv·16 bits per earlier token per layer, at each layer it works
  on for a token it doesn't compute from its inputs. Its best span then starts at the sequence start, so a block
  serving position p holds p + 1 cuts; s\* becomes the harmonic mean over positions of the most such a block can do
  (`formalism.context_s_star`, the formula and the choices favourable to the block in `formalism`'s docstring).
- **ρ (slowdown)**: c/(c − d) where Fit holds for the training, else max(1, Slowdown's).
- **The one bound** (`formalism.bound`, and `formalism.at` with reading A; opt-in, `at`'s default is Slowdown's
  token form): the counting lemma over the kinds of block work that are lower bounds on their own, ρ ≥ max over the
  kinds (fresh products, token, decoding) of each kind's share of |C| over its cap a block. It returns each kind's
  term (None, with the reason, where unmeasured), the max, the kind it binds on and what each kind rests on: fresh
  products Loomis-Whitney's cap (a theorem under `io_bound`'s premises P-width and P-chain, each true of the
  reference circuit as `gates` prices it, every output entry summed along one chain of whole dot cells
  (`formalism.ORDER`): `io_bound`'s bound on the products `formalism.products_of` takes from captured steps or a
  model, or `formalism.products_of_trace` from the product rows a trace keeps from te_core 4.8 on (`capture.products`),
  measured whether or not the trace gives the order, and unmeasured only for a circuit that declares another; the
  term is the figure without P-cell, a premise on the partition, which the attacker chooses (`formalism.P_CELL`, one
  constant), with the whole-cell figure beside it as the sensitivity `fresh_cell`; where a record's traced steps
  differ in shape, the dominant step's products stand in for all of them, scaled by tokens, labelled so
  (`formalism.STAND_IN`); a record with only an op log, a ledger
  row traced before 4.8, has its log's product rows, `formalism.products_of_ops`, but no operand's freshness, so the
  term is unmeasured there with the share of MACs it leaves unknown), token A1 under reading A with its witness
  family, decoding assumption
  (D). Where Fit fails ρ is max(1, that max). Gradient reduction, update application, reading B's token form and the
  per-gate token term come out beside it as labelled sensitivities, never in the max: (R) and (U) hold only for
  partitions split by tokens, and per_gate only for blocks that are no union of witnesses. `rho_vectors.json` holds
  its test vectors, which the console's rho.ts runs (null unmeasured, "inf" ∞): the bound on the fixtures' cases,
  the fresh-products term on `io_bound`'s product sets and the 45M start's dense step (`dense_45m.products.json`),
  and `formalism.published` at a cell for each case of its rule.
- **Reading A** (A1's approved reading): a block reads at least the update bits its work reads, once, plus
  each token's computed state where its work on the token begins, the context it reads and doesn't compute included.
  A1's values are the least reads of the default packer's witness family: the runs of one traced token's path in
  segments (the trace's `entries`, which `capture.circuit.check` counts the same way as any set of the trace's ops)
  and the matrix-product tiles of the record's op log (`tiles`, from the row's `macs_log` or the trace's GEMMs),
  weight-gradient tiles among them. `formalism.packer` gives the token and decoding kinds' terms from both, closed
  under unions: a block holds at most its gates and at most the best run block plus every product's tile at k plus
  every gate no tile places (`tiles.held`), so the term is the affected gates' share of |C| over that cap a block.
  Each gate at its own best witness (`formalism.per_gate`) is a sensitivity: a block holding one gate of a product for
  its few operand bits also holds the best run or another product's tile. Without entries (a row traced before 4.4)
  or an op log the token kind is unmeasured, never ∞.
- **Where the quantities come from**: each is measured on the record's traced step (`capture.circuit`, read through
  `ledger ingest --structures`; `frontier.measured` says whether a row's were), a, u_in, u, t and Δ_grad as upper
  bounds from that witness. A row without a trace gets labelled stubs (`formalism.STUBS`, and `formalism.V23` for a row
  ingested before 4.0), and `rescore --quantities analysis` the per-record quantities of covert_witness's
  `analysis/rescore_v1.py` instead, which has no formula of its own; every row's `labels` name each quantity's source.
- **Refused rows**: a trace the capture can't vouch for is refused, never scored (`formalism.refusal`,
  `formalism.Refused`; `frontier.formalism_rows` and `rescore` list each with its reason): one with a data-dependent
  read-back (Python control flow on a value read from a tensor, which the trace's stand-in decides), one with a Python
  number carried across steps (its Δ_grad grows with the step; a program keeps such state as a device tensor), one
  traced before 4.2 (it has no read-back lists: trace it again), and one whose E is nonempty while its traced step
  shows no update (an affected share of 0 or u(1) = 0) or no step through its checkpoint reads θ'.
- **Counterexample family X**: a decoder from `codecs/` (a `codecs.Plan`) plus per-step encodings E_t, each rebuilding
  its step's updated weights from θ bit for bit within c (`prepare.declare_step` for a covert_witness script, decoded by
  `prepare.decode`). Families are checked and recorded with each run; the formalism doesn't read them.
- **bpb**: the held-out bits per byte of the output model from the integrator's fixed evaluator: `val` during the search
  (for covert_witness, a paired score on a fresh slice of its search pool, sampled after the output's digest, by its
  evaluation job after the training job ended), `test` when finally scored. An evaluator returns None for a sealed
  split, and the record then holds no test bpb.
- **l\***: a target bits per byte, the start's bpb less 0.5%, 1%, 2% or 5%.
- **H (honest)**: the least |C| over every checkpoint of every valid submission, of any class or direction, that
  reaches val bpb ≤ l\* (the coordinator's 19:50Z ruling, `frontier.target`). The reference directions' alone
  (ref-optim, ref-data) is kept as context.
- **W\***: the least |C|·ρ over the same checkpoints, per cell; **S** = W\* / H, Security's slowdown.
- **Niche**: a mechanism tag × affected-share range × start cell (`frontier.AFFECTED_EDGES`). **Elite**: a niche's
  submission with the least val bpb, ties going to the least |C|·ρ.
- **Frontier**: the submissions that no other beats on both |C|·ρ and val bpb, at a reference shape, among those
  measured by the pinned measurement commit (covert_witness's `pin.py`, a submission's `measurement_commit`); invalid,
  unscored, `other-version` and `duplicate-e` submissions (an E digest an earlier submission had) are left out, and so
  is one whose read-outs `cw.readout_leaks` lists (`read-out-leak`) (`frontier.status`; a row ingested before 3.13 gets
  its read-outs from `frontier --facts`). The scorer's tied-lookup rule (13:32Z) is retired: a tied embedding's update
  is affected from its lookup on.
- **Tiers**: tier 0 is a screen of about 3 GPU-minutes on the small start, which anyone may launch. Tier 1 is about 10
  GPU-minutes on the medium start, at most `TIER1_PER_HOUR` per agent per hour. Tier 2 is about 30 GPU-minutes, 3
  seeds, the small and medium starts. Tiers 1 and 2 run only an entry the scorer promoted to that tier.

`frontier` gives, per start, target and cell, W\*, H and S with W\*'s route and its x₀ and s\* bounds and best-known
values; `plots` writes five panels (P1-P4 and the A2 curve, the least |C| to each target against Δ, per start), each
`<name>.json` and its PNG, which are off the console; `console` cuts each to the panels route's limits and holds one
panel's PUT; `rescore` scores v2.3's records in one command, on the measured quantities beside the analysis's (module
docstring). The console's training panels, `training/algorithm-policy` and `training/frontier-fit`, are both PUT by
`python -m te_core.algorithm_policy --out DIR --publish`. They read the ledger and the quantities through `rescore`'s
loading (`rescore.arguments`, `rescore.load`), so they score the measured quantities unless told otherwise
(`--charge-context` charges the context, as `rescore`'s, and the dial can't reproduce it), the pin's records and the
pilot's unless `--pin` names the commits. `training/frontier-fit` is `frontier_fit`'s: per data (`v3` or
`v2.3-pilot`), L(C) = E + A·C^(−γ) fitted to its unconstrained frontier, the least held-out bits per byte at each total
training compute in gates (from scratch: the pilot's pretraining runs, or the v3 ladder's from `--ladder-points` and
`--ladder-fit`, else `frontier_fit.LADDER`'s, each start of the ladder's manifest at its counted C0 and held-out bpb;
each start at C0; every scored checkpoint from a start at C0 + |C|), with the points it read and their residuals; a
data with neither a start that has a scored checkpoint nor a ladder point has no fit, since nothing then converts its
from-scratch losses or prices C0. A fit takes three points (`FIT_MIN`), and one on fewer than five from scratch
(`FIT_TARGET`) is provisional: its status and note say so, the note with its points' residuals and the ladder's runs
off its frontier. `training/algorithm-policy` is `algorithm_policy`'s: per run, start and gain target g (1.1, 2 and
11, a checkpoint's gain C(L1)/C(L0) on that fit, from the start's loss L0 to its own L1, so every target is below
L0), "yes" at the run's first checkpoint reaching g, else "no" at its last, with the algorithm's inefficiency p = C/H
(H the least |C| of any checkpoint from the start
on the same data reaching g, every run's pooled; a lower bound on a "no" row's p) against the policy's slowdown ρ,
`formalism.published` at the console's default policy (`policy.CONSOLE`, the one default: `rho_default`, a number,
and `rho_console`; `rho_served` at the registered serving block, `panels.SERVED`), so S = pρ, and the inputs the
console's dial recomputes ρ from at any (c, k), in one JSON cell (`rho_inputs`; te_core 4.x's per-part Fit and
ᾱ/α_max among them, u_in exact at the dial's k and the console's); `build` refuses a row whose cells miss
`formalism.at` at a dial cell or the console's default. Beside them: its `reading` (the one bound's ρ under reading A
where the row's trace has entries and an op log: Fit's c/(c − d) where it holds, else the measured max where it
exceeds 1, partial where a kind is unmeasured, else null where a kind is unmeasured, else 1, binding none; else
reading B's, a target), the `te_core` that priced it, `binding_kind`, `rests_on` and `kinds` (the terms its max takes
and the sensitivities it leaves out). Its coverage
(gain target by data by decade of ρ, with the reached cells' ρ at the registered serving block beside the console's
default) names the cells nothing reaches, and counts the valid checkpoints `formalism.refusal`
refuses, which no cell holds (`tables/refused.json` gives each one's reason). A row `published` marks partial (a valid
kind unmeasured, so ρ is only a lower bound) has basis `partial`: the figure draws it with a caret up and the cut
picks the partial rows among themselves, but none is in a cell, on the others' front or the least S. `--publish` GETs what the route serves
and PUTs nothing without it: on a data whose served rows are at a newer te_core, or at an older one with a run this
build has no row for, both panels keep the served rows (`kept_data`, `guarded`), so a version switch never
empties a populated panel; a build with every served run replaces an older te_core's rows, however its coverage
moves. `training/frontier-branches` (`panels`'s: the frontier and each start's honest and throttled branches over
the dial) is retired from the console and no longer PUT; `panels` still holds the dial both use (`dial`, `rho_columns`,
`page_inputs`, `mismatches`). Every scoring command (`frontier`, `plots`, `rescore`, `panels`, `frontier_fit`,
`algorithm_policy`) applies one cap, `policy.eligible`: a checkpoint is scored only when its model's forward gates per
token are at most the greater of twice its start's (`policy.CAP_FALLBACK`) and the compute-optimal model's at its loss
under the ladder's fit (their `--ladder-fit FILE`, `policy.ladder_fit`), and twice its start's where there is no fit or
the fit doesn't reach its loss, so a grown model is still scored; the rest are left out as `over-cap`.

## What the integrator calls

covert_witness's islands run unmodified (`cw.py`): the harness wraps the fixed `prepare.load_start`,
`TrainData.next`, `encode_*` and `submit`, and writes each checkpoint's E to `OUT/E/`, uncounted. It scores nothing:
covert_witness's evaluation job (`cw.score_run`, run by its evaluate.py after the training job, outside that job's
sandbox) decodes each E with `prepare.decode` and scores it with `prepare.Scorer`: the output's digest, then a fresh
slice of the search pool, and the promotion set's score sealed under the digest (covert_witness's README; the final
set only in its final mode). It records each scored model's forward MACs and gates per token, counted on that model's
own forward (`cw.forward_cost`: what runs, so an MoE's active experts), and the start's; nothing in the harness caps
them (scoring does, `policy.eligible`). A batch from `TrainData.next` is token data through any op that reads it, not only an
embedding lookup (`flops.Counter.batch`). The encoders and `submit` are the program's work and are counted. The
harness's own work runs with its own instance of prepare.py, on plain copies of what the program handed it, and with the
program's torch function modes (a default device among them), global module hooks and profilers off, its own
`torch.device(...)` contexts on, and with the `sys.path` and the modules
prepare.py names as they were before the program ran. A program that rebinds what that work calls (`cw.watched`) makes
its run invalid. So does one that routes work past the counter: an op outside torch's own namespaces (a custom op), a
product op with no MAC formula, token data read out of torch (`numpy`, `tolist`, DLPack) at a value per token or more, a
thread or a child process it starts, or a module it imports that names a route the screen refuses or that it wrote during
the run (`run.Imports`). The screen reads the program and every module it imports outside the fixed side, and refuses
what would run code it never reads: `exec`, `eval` and `compile` (called or not), `runpy`, `importlib`, `__import__`,
`marshal`, pickle, `FunctionType`, `CodeType`, `__code__` and `sys.meta_path`; while the program runs no bytecode cache is
read (`run.no_bytecode`). From `benchmarks/covert_witness/`:

~~~text
python -m te_core.run --script islands/NAME/train.py --out OUT [--families DIR] [--budget-fraction F] [--tier T] \
    -- --start NAME --out SUB [--budget-fraction F] [TRAIN_ARGS...]
python -m te_core.launch --script islands/NAME/train.py --tier 0 --agent A --direction D --mechanism M \
    [--nodes FILE] [-- TRAIN_ARGS...]        # FILE: `vy-nebius-2 pool=gpu`
python -m te_core.ledger --agent A ingest OUT --direction D --mechanism M [--reference]
python -m te_core.frontier [--json OUT]              # W*, honest and S per target and (c, k) cell
python -m te_core.rescore --out DIR [--quantities measured|analysis|stubs]  # v2.3's records, one command
python -m te_core.plots --out DIR [--policy reference|inference|server|c1e14/k1e4] [--drop 0.01] [--from-scratch F.json]
python -m te_core.algorithm_policy --out DIR [--publish] [--ladder-points F.json] [--ladder-fit FIT.json]  # both training panels
python -m te_core.frontier_fit --out DIR [--publish] [--ladder-points F.json] [--ladder-fit FIT.json]      # training/frontier-fit alone
python -m te_core.panels frontier-branches --out DIR [--ladder-fit FIT.json]   # retired from the console, not PUT
python -m te_core.policy [START ...]                 # each start's reference shapes and (c, k) grid
python -m te_core.gates [--check]                     # gates per MAC by op; --check recompiles the Definitions
~~~

A script may call `te_core.hooks` (`plan`, `output`, `step`, `tokens`, `family`; each a no-op outside the harness) for
5% checkpoints of its own model or counterexample families. The rest of this section is the `train(te)` program form.

1. **The starts registry** (`starts.json`): one entry per start, `small` and `medium`, each written by
   `python -m te_core.starts --name small --theta S.pt --factory MODULE:FUNCTION --val-bpb X`, which gives `{path, sha256,
   val_bpb, ctx, vocab, forward_macs_per_token}`.
2. **The evaluator**: a function `bpb(state_dict, split, device) -> float`, with `split` either `"val"` or `"test"`,
   named as `--evaluator MODULE:FUNCTION`. The harness calls it at every checkpoint and does not count its work.
3. **A program** (`train.py`, one file, importing only the fixed modules, te_core and torch) defines `train(te)`:

   ~~~python
   def train(te):
       m = te.load(LM(SIZES["s"]))       # loads θ, moves it to te.device, registers it
       te.plan(n_steps)                   # checkpoints after every 5% of these
       for step in range(n_steps):
           with te.step():               # one training step: everything inside is counted as step `step`
               loss_of(m, next_batch()).backward(); opt.step(); opt.zero_grad()
           if te.time_left() < 0: break   # the tier's budget; past 1.2x it the harness stops the run
   ~~~

   `te.output(fn)` names the output model (e.g. merged LoRA). `te.family(name, plan)` returns a writer whose
   `put(step, E_t)` declares a counterexample. `te.tokens(batch)` marks token data that doesn't go through an embedding.
   Tokens computed outside a step make the run invalid. A program naming the harness's internals or a route around the
   dispatcher (`run.FORBIDDEN`) is refused.
4. **A run**: `python -m te_core.run --program train.py --starts starts.json --start small --tier 0 --seed 0 --evaluator
   MODULE:FUNCTION --out OUT`, or `python -m te_core.launch ...` to put it on a node. Either one writes `OUT/run.json`.
5. **Ingest**: `python -m te_core.ledger --agent A ingest OUT --direction D --mechanism M --parent ID --run-id R`
   writes one submission record per checkpoint, with the program's arguments; `--structures TRACED.json` (structure
   id -> traced structure, `capture.script`) gives its steps their U, D and the formalism's measured quantities
   (`frontier.measured`); `--readouts` judges its read-outs against the start's tensors; `--meta META.json` (else the
   run's own or its parent directory's `meta.json`, checked to be this run's submission) gives each checkpoint its
   measured decoder. Reference (honest) runs add `--reference`.
6. **The frontier**: `python -m te_core.frontier`. Its promotion candidates (`frontier.candidates`) are promoted only
   on the sealed promotion set, one bit each: covert_witness's `promote.py --tier T`, run as the scorer.

Run every command from `benchmarks/covert_witness/`, where `te_core` and `prepare` are importable.

## Files

| file | what it does |
|---|---|
| `flops.py` | `Counter`: the dispatch-mode harness counting MACs and provenance; `guard_exits`: token data read out of torch; `flags`: uncounted-work flags; `forward_macs_per_token` |
| `annotate.py` | `annotate(record, through)`: the maximal annotation, per step U, D_t, the depth profile and the cut; `with_structures`, a record's traced structures and their measured quantities |
| `counterexample.py` | `Family` (the program's writer), `check` (the sampled bit-for-bit rebuild), `coverage` |
| `score.py` | `checkpoints` (a run's prefixes), `block_c` and `C_PER_FORWARD_TOKEN` |
| `ledger.py` | `Ledger` (append-only, per agent, evidence store), `ingest`, the index |
| `launch.py` | tiers, the promotion gate and hourly cap, the `research run` command (`--pool` with gpu-lease inside, or `--gpus 1`) |
| `cw.py` | `Adapter`: a covert_witness train.py under the harness, its held-out guard and its E checkpoints; `score_run`, the evaluation job's scoring of them; `CWPlan`, the family `prepare.declare_step` writes |
| `hooks.py` | what a standalone script may call, no-ops outside the harness |
| `frontier.py` | the valid rows (`status`, `measured`), W\*, honest and S per target and cell with W\*'s x₀ and s\* bounds and best-known values, promotion candidates |
| `opnames.py` | the op-name sets the counter, the capture and the gate prices share (torch-free) |
| `gates.py` | each counted op's price in Boolean gates from the circuits' Definitions (`DEFINITIONS`, `PRICES`, `kind_of`, `of_log`) |
| `policy.py` | a start's reference shapes (inference block, reference policy, server block), the (c, k) grid in decades, and `cells`, both by key; the console's policy (`CONSOLE`), one full session of the 45M reference in gates (`session_gates`, `session`: the matrices, the LM head and attention's score and value products over the session's context, at the shape `SHAPES` states for each size), which every producer reads |
| `formalism.py` | the formalism's per-record quantities, Fit and Slowdown's ρ at (c, k) with the attack side beside it, and the one bound over the kinds that are lower bounds on their own under reading A, its sensitivities beside it (`bound`, `packer`, `per_gate`, `fresh_products`, `KINDS`, `RESTS_ON`, `SENSITIVITIES`), the fresh-products kind's inputs from captured steps (`products_of`, `ORDERS`, `ORDER`) and from a trace's product rows (`products_of_trace`, `product_kind`, `PRODUCTS_FROM`, `TRACED_PRODUCTS`, `STAND_IN`), its figure with or without P-cell (`P_CELL`), and an op log's product rows, which can't tell a fresh operand (`products_of_ops`, `LOG_KIND`, `LOG_UNKNOWN`), and the console's ρ on it (`published`, at `policy.CONSOLE` by default, `RESTS_ON_CELL`, `CELL_SENSITIVITIES`, `READING_A`); `of_record` (ingest's), `of_trace` (a traced step's), `measure` (with labelled stubs, assumptions and upper bounds), the profiles `even`, `tail` and `with_tail` (u, u_in) and their cap at Δ, `capped`; `fit_k` |
| `plots.py` | the five panels (P1-P4 of the experiments draft and the A2 curve), from the ledger, each `<name>.json` and its PNG |
| `console.py` | each panel cut to the panels route's limits (64 KiB, a 500-character note, a 120-character title, 4,096-character cells, 24 columns; `refused`), one panel's PUT and the panels the route serves (`served`) |
| `rescore.py` | v2.3's records scored on their measured quantities (or the analysis's, or the stubs): the report, the five panels and their cuts, and the measured-against-analysis comparison, one command; `arguments` and `load`, its ledger and quantities loading |
| `panels.py` | `training/frontier-branches`, retired from the console: the frontier, each branch point's honest and throttled curves and ρ's candidates, its cut and figure; the dial the training panels use (`dial`, `rho_columns`, `page_inputs`, `mismatches`) |
| `frontier_fit.py` | a console training panel, `training/frontier-fit`: per data, L(C) = E + A·C^(−γ) fitted to the unconstrained frontier (`fit`, `used`; provisional under `FIT_TARGET` points from scratch), its effective compute (`loss_at`, `c_eff`, `gain`) and the v3 ladder's points (`ladder_points`, `LADDER`'s by default) |
| `algorithm_policy.py` | the other, `training/algorithm-policy`: per run, start and gain target, p = C/H against ρ with ρ's inputs (`page_rho`, the dial's formula, which must reproduce `formalism.at`) and the dashboard's cells (`published`'s), the coverage, its console cut, figure, and the PUT of both panels over what the route serves (`guarded`) |
| `io_bound.py` | the Loomis-Whitney bound on a training step's slowdown under P-width, the fresh-products kind's calculator: one product's cap at the chain constant (`alpha`, `beta`, `mu`), Λ's factors (`SHARE`, `feed`, `whole_chain`), and the blocks a step's products need and its ρ, the larger of the root and linear forms, proved and searched, with and without P-cell (`bound`); each GEMM of a step captured by `capture_step` or `capture_fn` classified by kind (`rows_of`, `Row`), the step's \|C\| (`step_gates`), its vectors (`tests/io_bound_vectors.json`, `python -m te_core.io_bound vectors`) and the s-shape steps (`steps`) |
| `tiles.py` | reading A's matrix-product tiles: a record's ops as products (`of_log` from a `macs_log`, `of_graph` from a traced step, `products`), the most MACs a tile reading k bits holds (`volume`) and each op's tile cap (`cap`, `caps`), and the most a block's tiles hold beside each other (`held`) |
| `workloads.py` | the covert workloads of the console's *Standard training under the policy* plot, each a model: one function per workload from its dials (`Dial`) to the one bound's inputs (`formalism.Measured`), its products of two fresh operands (`Fresh`) and the most update bits one of its ops reads; at the console's policy, the only one rows are reported at (`POLICIES`: `policy.CONSOLE`), ρ over the bound's kinds (`rho`: the fresh products by `formalism.fresh_products` on the model's products, `products`, calibrated on io_bound's captured steps, summed in a chain as the standard algorithm's are, `ORDER`, its term io_bound's figure without P-cell under P-width and P-chain, `FRESH_FIGURE`; the token and decoding kinds unmeasured, reading A needing a trace, `reading_a`; the measured terms' max where it exceeds 1, partial where a kind is unmeasured, else null, never 1, while a kind is unmeasured, and 1 only where every kind is measured and none exceeds 1), its binding kind and what that rests on (`RESTS_ON`), with reading B's token form beside it as the target (`TARGET`) and the sensitivities (the bound's, and the fresh products' under P-cell, `fresh_cell`); measured records beside the model with every term on each (`Check`, `recorded`: the 45M pilot's and the standard-workloads island's rows, an island row with `formalism.published` on it as the island published it, `Check.published`), and each input's source, measured with its record or assumed with why (`input_status`); `python -m te_core.workloads --json te_core/workloads.json`, and `--records TABLES ART` to rewrite `workloads_records.json` from the island's panel |
| `workloads.json` | `workloads.py`'s reference points and checks, which the console's tests pin (`tests/test_workloads.py` fails while it's stale) |
| `workloads_records.json` | the standard-workloads island's measured rows that `workloads.py` checks the model on: each row's panel cells (dials, run, `rho_inputs`, `input_status`), `formalism.published` on it at the console's policy and reading B's ρ there, its parts and its traced steps' facts, or a generation row's rounds |
| `rho_vectors.py` | the one bound's test vectors (`rho_vectors.json`, `--write`): wildcard's grown adapter, a dense and a LoRA step on the tiny start at the dial's cells, under reading A, each case's op log beside its circuit; `fresh`, the fresh-products term without P-cell and `fresh_cell` beside it on io_bound's product sets and the 45M dense step (`DENSE_45M`); `published`, `formalism.published` at a cell for each case of its rule (`PUBLISHED`); and `sessions`, `policy.session`'s inputs and outputs at every stated size and `SESSIONS`, with the console's c, for the console's port |
| `codecs/` | identity, low rank, sparse mask, quantized delta, seed replay, `Compose`, `Plan` |
| `run.py` | `Session` (the program's API), the run CLI and its script mode |
| `starts.py` | a start's registry entry |
| `capture/` | the step circuit: one training step traced on fakes (`make_fx`, or `capture.script`'s recorder for a covert_witness script from its launch record), its \|C\|, U by update dependence, D and the cut without running, in `flops`'s record form, and the formalism's quantities on it (`capture.circuit`, reading A's checker `check` among them), with the step's matrix products as `io_bound` reads them (`capture.products`: dims, each operand's θ′-dependence, passed widths and rates keyed by the capture's values, the order unknown to the trace, the reference circuit's chain, `formalism.ORDER`, standing); its own suite. Its op sets are `flops`'s (`ELEMENTWISE_NAMES`, `SCALED_NAMES`, `FUSED_OPTIMIZER_MACS`, `LINALG_NAMES`, `GEMM_LIKE`, `NOT_GEMM`, `TOKEN_INDEX`, `GATHERS`), and so is its rule for an updated node with no multiplication |
| `tests/` | a suite (`uv run tools/verity/check/suites.py benchmarks/covert_witness/te_core`); `cw_fake/` is a fake `prepare` and island |

## Limits of the count

- **Work outside torch's dispatcher** (custom CUDA or Triton kernels, host libraries) isn't counted. The screen refuses
  the usual routes by name, and the routes it can see run make the run invalid (above). What stays open is in-process:
  a program that reaches the harness's own objects by attribute lookups (`getattr` on `hooks.ACTIVE`, frames) can pause
  the counter or rewrite its record; torch's C methods (`torch._C.TensorBase.numpy`, storages, `torch.save` into a
  buffer) read values out unseen; a grandchild process that outlives its parent isn't seen; and native code through
  ctypes is only the `native` flag. The flags `cpu_heavy`, `cpu_parallel`, `children`, `gpu_unexplained` (with
  `--peak-macs-per-s`) and `cpu_macs` show work the count can't vouch for; a check outside the training process (the
  GPU's own busy time against the counted MACs) is what bounds the in-process routes. Torch compilation is disabled and
  CUDA graph replay refused.
- **The runtime counter's provenance prices |C|'s other forward work**: by value for θ's tensors and by origin for
  everything else, so a tensor that isn't exactly one of θ's counts as differing even when it computes nothing new (a
  zero-initialized adapter factor at step 0, say), and a forward op combining it with an activation is counted. A
  start's buffers (RoPE tables) are taken as θ's.
- **The measured quantities are one traced step's** (`capture/README.md`'s limits): a record's updated steps all take
  its representative step's structure, scaled to their tokens; a period of several steps has one cut, which each of
  them carries. A cut counts a value at its dtype's width, so a sign kept as a float costs 32 bits a value; a circuit
  that wants the credit stores it narrow. Values are whole tensors: a share of a token's work is a share of its ops'
  gates, and a lookup reads its whole table.
- **Harness work is neither counted nor timed**: loading, encoding, decoding and evaluation run outside the dispatch
  mode and outside the steps' clocks; `totals` sums the steps' seconds and `process_*` the whole process.
- **The held-out guard is the integrator's sandbox, with an in-process hook inside it**: covert_witness's training job
  runs in a sandbox (its `sandbox.py`) where the held-out sets, the sealed side and other runs aren't mounted at all.
  Inside it an audit hook also refuses `prepare.UNREADABLE` (the search pool, the sealed side and learning_frontier's
  `val.bin` and hub cache) except while the harness loads a start, and other runs' outputs (the run directories beside
  `$RESEARCH_RUN_DIR`, other candidates' scratch under `prepare.ROOT/candidates`). The loaders that open files in C++ or
  Rust (`cw.c_readers`: `torch.from_file`, storages' `from_file`, safetensors' `safe_open`) raise the hook's event first;
  the screen refuses their unwrapped forms (`torch._C`, a class's bases, a function's closure) and the Arrow readers
  (pyarrow, polars, duckdb, `datasets`). Subprocesses are logged to `guard.log`; a thread or a process the program starts
  makes its run invalid, and the sandbox's process-tree record bounds any work outside the count.
- **Script steps follow the data stream**: a script that fetches all its data at once is one step, which keeps every
  count but coarsens D.
- **Weight shape (the runtime counter)**: a forward activation with a weight's exact 2-D shape, in a matmul of two
  token-indexed operands, counts as a weight.
- **Bit-for-bit rebuilds need the run's kernels**: families are checked on the run's device, right after it trains.
