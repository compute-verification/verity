# te_core: the fixed core of the training-efficiency search

te_core scores training-efficiency submissions. A submission is a training run P from registered weights θ. The harness
counts the run's circuit C itself, reads the run's maximal structure annotation A from that count, checks any
counterexamples X to update hardness, and gives the slowdown ρ the theorem implies. The output model is scored in bits
per byte. Search agents never edit te_core: a change goes through its toolsmith and bumps `VERSION` (`__init__.py`),
which every record carries. Findings, negative results and tool announcements go in each island's
`islands/<island>/BOARD.md`, appended (the toolsmith's is `islands/toolsmith/BOARD.md`).

## The quantities

te_core scores by the training formalism v2 (note:training-20261102T2215Z-report-training-formalism-v2). |C|, c, w and
d are Boolean gates (`gates`: each counted op at its circuit Definition's price, a bf16 GEMM MAC being
`HopperBF16WgmmaDot16_v2`'s 23,682 gates / 16 = 1,480.1), every other quantity bits.

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
- **Δ**: the bits that describe θ' given θ, 8 |E| for the submitted E, by a decoder of **d** gates (a stub at 0 until
  measured).
- **Δ_grad**: the gradient and optimizer state a step's blocks pass between them: the gradient's narrowest cut at
  32 bits or more an element, plus the carried state the token ops don't read.
- **(c, k)**: a block's gates and the bits it imports; `policy.cells` gives each start's grid in decades, from one
  token's inference block to the start's server block, and its reference shapes at their exact (c, k): the inference
  block, the reference policy (c = 4096 w, k = 10 kB) and the server block.
- **Fit** (checked): for training, three kinds of block, compute Δ + sτ ≤ k, reduction 2Δ_grad ≤ k and update
  application Δ + Δ_grad ≤ k (a part with no token blocks has only the compute kind); where every part fits,
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
- **Where the quantities come from**: each is measured on the record's traced step (`capture.circuit`, read through
  `ledger ingest --structures`; `frontier.measured` says whether a row's were), a, u_in, u, t and Δ_grad as upper
  bounds from that witness. A row without a trace gets labelled stubs (`formalism.STUBS`, and `formalism.V23` for a row
  ingested before 4.0), and `rescore --quantities analysis` the per-record quantities of covert_witness's
  `analysis/rescore_v1.py` instead, which has no formula of its own; every row's `labels` name each quantity's source.
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
docstring). The console's one training panel, `training/frontier-branches`, is `panels`'s: held-out bits per byte
against total training compute in gates, the best unconstrained learner (the frontier, from scratch or continuing a
start, at C0 + |C|) and per branch point (each start, `on_frontier` when the frontier passes through it) the best
unconstrained and constrained learners starting from it (honest at C0 + |C|; throttled at C0 + |C|ρ per (c, k) of its
dial, with S = W\*/H). It reads the ledger and the quantities through `rescore`'s loading (`rescore.arguments`,
`rescore.load`), so it scores the measured quantities unless told otherwise.

## What the integrator calls

covert_witness's islands run unmodified (`cw.py`): the harness wraps the fixed `prepare.load_start`,
`TrainData.next`, `encode_*` and `submit`, and writes each checkpoint's E to `OUT/E/`, uncounted. It scores nothing:
covert_witness's evaluation job (`cw.score_run`, run by its evaluate.py after the training job, outside that job's
sandbox) decodes each E with `prepare.decode` and scores it with `prepare.Scorer`: the output's digest, then a fresh
slice of the search pool, and the promotion set's score sealed under the digest (covert_witness's README; the final
set only in its final mode). It records each scored model's forward MACs and gates per token, counted on that model's
own forward (`cw.forward_cost`: what runs, so an MoE's active experts), and the start's; nothing caps them. A batch from `TrainData.next` is token data through any op that reads it, not only an
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
python -m te_core.panels frontier-branches --out DIR [--publish]   # the console's training panel; --publish PUTs it
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
   (`frontier.measured`); `--readouts` judges its read-outs against the start's tensors. Reference (honest) runs add
   `--reference`.
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
| `policy.py` | a start's reference shapes (inference block, reference policy, server block), the (c, k) grid in decades, and `cells`, both by key |
| `formalism.py` | the formalism's per-record quantities, Fit and Slowdown's ρ at (c, k) with the attack side beside it; `of_record` (ingest's), `of_trace` (a traced step's), `measure` (with labelled stubs, assumptions and upper bounds), the profiles `even`, `tail` and `with_tail` (u, u_in) and their cap at Δ, `capped`; `fit_k` |
| `plots.py` | the five panels (P1-P4 of the experiments draft and the A2 curve), from the ledger, each `<name>.json` and its PNG |
| `console.py` | each panel cut to the panels route's limits (64 KiB, a 500-character note), and one panel's PUT |
| `rescore.py` | v2.3's records scored on their measured quantities (or the analysis's, or the stubs): the report, the five panels and their cuts, and the measured-against-analysis comparison, one command; `arguments` and `load`, its ledger and quantities loading |
| `panels.py` | the console's training panel, `training/frontier-branches`: the frontier, each branch point's honest and throttled curves, its console cut, figure and PUT |
| `codecs/` | identity, low rank, sparse mask, quantized delta, seed replay, `Compose`, `Plan` |
| `run.py` | `Session` (the program's API), the run CLI and its script mode |
| `starts.py` | a start's registry entry |
| `capture/` | the step circuit: one training step traced on fakes (`make_fx`, or `capture.script`'s recorder for a covert_witness script from its launch record), its \|C\|, U by update dependence, D and the cut without running, in `flops`'s record form, and the formalism's quantities on it (`capture.circuit`); its own suite. Its op sets are `flops`'s (`ELEMENTWISE_NAMES`, `SCALED_NAMES`, `FUSED_OPTIMIZER_MACS`, `LINALG_NAMES`, `GEMM_LIKE`, `NOT_GEMM`, `TOKEN_INDEX`, `GATHERS`), and so is its rule for an updated node with no multiplication |
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
