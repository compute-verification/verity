# te_core: the fixed core of the training-efficiency search

te_core scores training-efficiency submissions. A submission is a training run P from registered weights θ. The harness
counts the run's circuit C itself, reads the run's maximal structure annotation A from that count, checks any
counterexamples X to update hardness, and gives the slowdown ρ the theorem implies. The output model is scored in bits
per byte. Search agents never edit te_core: a change goes through its toolsmith and bumps `VERSION` (`__init__.py`),
which every record carries. Findings, negative results and tool announcements go in each island's
`islands/<island>/BOARD.md`, appended (the toolsmith's is `islands/toolsmith/BOARD.md`).

## The quantities

The theorem, in MACs (a multiplication is one MAC, so the page's m is 1): for inference blocks of c MACs importing k
bits, an annotation meeting update hardness at β and deep use at D ≥ k/β gives ρ ≥ φcβ / (k(k + β)).

- **θ (registered weights)**: a start's weights, registered once at the start of the monitoring period, so every value
  computed during the run is new.
- **Submission**: one checkpoint of a run, scored as the prefix of C through its step: after every 5% of a program's
  planned steps (or of its MAC budget); for a covert_witness script, the first E it encodes in each 5% of the run, and
  the E it submits.
- **C, the circuit**: everything the run's process computes through torch, counted by the harness at the Definition
  level: every multiplication's shape and count over every step and token.
- **|C|**: C's size in multiply-accumulates (MACs). It sums the MACs of matmuls, attention and convolutions, one MAC per
  output element of each elementwise multiplication, and n³ per matrix of dense linear algebra (flagged approximate).
- **MAC (multiplication block)**: one z′ ← x·y + z, the initial accumulator z counted as an operand.
- **Token block**: one token's computation in one step.
- **Token-indexed value**: a value computed, within the current step, from the step's token data (an integer tensor
  indexed into an embedding, or a batch passed to `te.tokens`).
- **Weight and its roots**: a value that is not token-indexed. Its roots are the persistent tensors it is computed from
  that differ from θ: a registered weight whose value isn't θ's, bit for bit; any other tensor made before the step that
  isn't exactly one of θ's tensors (a recast copy that is, such as `p.data = p.data.float()`, is θ's); any value
  autograd's backward produced.
- **Step**: a program's `te.step()`; for a covert_witness script, the work from one call of its data stream
  (`prepare.TrainData.next`) to the next.
- **Updated blocks U**: the forward-pass MACs with a token-indexed operand and another operand that has roots, so their
  weight operand differs from θ. Backward products are left out of U: they read the same weights, so adding them halves
  β per MAC, doubles φ and leaves φβ unchanged. A read of an updated weight by token data (embedding, index, gather) is
  in U at its position, priced as the circuit's gather Definition prices it (`flops.gather_selects`: one multiplexer per
  word selected per level of its tree, about one per row of a table): a tied embedding's update is read at the lookup,
  so every later updated node has the lookup upstream. A lookup of θ's table is 0. `ledger ingest --lookups` adds the
  lookups of the start's own table to a record counted before 3.6 (`annotate.with_lookups`).
- **Inference block S**: a set of at most c MACs that imports at most k bits beyond θ. It holds only computed gates, so
  its inputs are counted in in(S) and are never its members.
- **in(S)**: the sender form, the bits of the gates outside S that feed a gate of S plus S's own source gates: the
  description S is computed from.
- **c**: the block size, `C_PER_FORWARD_TOKEN` = 4096 times the start model's forward MACs per token, counted by the
  harness at the start's context.
- **k**: the bits a block imports, one of 10 kB, 100 kB and 1 MB (8·10⁴, 8·10⁵ and 8·10⁶ bits, `score.K_BITS`).
- **p**: bits per updated value: by default the narrowest width an updated weight operand is multiplied at (16 under
  bf16 autocast).
- **r**: the assumed compressibility of an updated value, one of 1, 10 and 100.
- **β (update hardness, T1)**: the computed bits a block of at most c MACs imports for each updated MAC of one token it
  holds. The default is p/r.
- **Updated MACs per token block**: the smaller of a step's per-row sum over its updated nodes and its updated MACs over
  its tokens.
- **D (depth)**: the updated MACs per token upstream of a token's deepest updated multiplication, within its forward
  pass.
- **D_min**: k/β, the depth from which an updated MAC counts as deep.
- **φ (deep use, T2)**: the share of |C| in updated MACs that have at least D_min updated MACs upstream.
- **Counterexample family X**: a decoder from `codecs/` (a `codecs.Plan`) plus per-step encodings E_t. Each E_t
  rebuilds its step's updated weights from θ bit for bit, for at most c MACs; declared imported computed values travel
  in E_t. A covert_witness script declares a step with `prepare.declare_step(base, suffixes)`: its family `E` is
  decoded by the fixed `prepare.decode` (`cw.CWPlan`), so any E whose decoder rebuilds a step's weights within c (a
  seed log, low-rank, sparse or quantized updates in a fixed format) is credited, not only adapter factors, each step at
  no less than its E's `prepare.rebuild_macs`. The
  decoder may produce more weights than the step read; every weight the step read is compared against its value digest
  (`flops.KeyedDigest`, keyed from the OS when the run starts, the key in `run.json`).
- **β_t**: a step's counterexample β, bits(E_t) over the step's updated MACs per token block.
- **Coverage**: a family covers the sampled steps that rebuild bit for bit (5 sampled steps at search time, 20 at
  final, sampled with a live coin). A uniform family also covers every step it declares, if every sampled step verified.
- **U_τ (threshold annotation)**: for τ in {p/r} and the covered steps' β_t below p/r, the uncovered steps' U and the
  covered steps' with β_t ≥ τ, with β = min(p/r, τ) and φ recomputed on it over the same |C|. A cheap covered step
  leaves U_τ for every τ above its β_t, so it costs only its own share of φ.
- **ρ (slowdown)**: max(1, φcβ / (k(k + β))) if D ≥ k/β, else 1, for each U_τ, then the largest, at each (k, r).
- **bpb**: the held-out bits per byte of the output model from the integrator's fixed evaluator: `val` during the search
  (for covert_witness, a paired score on a fresh slice of its search pool, sampled after the output's digest, by its
  evaluation job after the training job ended), `test` when finally scored. An evaluator returns None for a sealed
  split, and the record then holds no test bpb.
- **l\***: a target bits per byte, the start's bpb less 0.5%, 1%, 2% or 5%.
- **honest(l\*)**: the least |C| over every checkpoint of every valid submission, of any class or direction, that
  reaches val bpb ≤ l\* (the coordinator's 19:50Z ruling, `frontier.target`). The reference directions' alone
  (ref-optim, ref-data) is printed as context.
- **W\*(l\*)**: the least |C|·ρ over the same checkpoints as honest(l\*).
- **Effective slowdown**: W\*(l\*) / honest(l\*), ≥ 1 since ρ ≥ 1.
- **Niche**: a mechanism tag × β range × φ range × start cell (`frontier.BETA_EDGES`, `PHI_EDGES`).
- **Elite**: a niche's submission with the least val bpb, ties going to the least |C|·ρ.
- **Frontier**: the submissions that no other submission beats on both |C|·ρ and val bpb, at one (k, r), among those
  measured by the pinned measurement commit (covert_witness's `pin.py`, a submission's `measurement_commit`); invalid,
  unscored, `other-version` and `duplicate-e` submissions (an E digest an earlier submission had) are left out, and so
  is one with covered steps whose ρ wasn't priced by thresholds (`re-ingest`: ingested before 3.0), one whose read-outs
  `cw.readout_leaks` lists (`read-out-leak`), and one that updates the tied embedding with no block below the top
  (`tied-lookup`) (`frontier.status`; a row ingested before 3.13 gets those facts from `frontier --facts`).
- **Tiers**: tier 0 is a screen of about 3 GPU-minutes on the small start, which anyone may launch. Tier 1 is about 10
  GPU-minutes on the medium start, at most `TIER1_PER_HOUR` per agent per hour. Tier 2 is about 30 GPU-minutes, 3
  seeds, the small and medium starts. Tiers 1 and 2 run only an entry the scorer promoted to that tier.

## The formalism's score (covert_witness v3)

covert_witness v3 scores by the training formalism v2 (note:training-20261102T2215Z-report-training-formalism-v2),
beside the theorem's ρ above, which every record still carries. |C|, c, w and d are Boolean gates: `gates` prices each
counted op by the circuits' own Definitions (a bf16 GEMM MAC is `HopperBF16WgmmaDot16_v2`'s 23,682 gates / 16 =
1,480.1). Ingest writes each submission's quantities (`formalism.of_record`: per step structure its gates, token blocks,
the gates in them (forward and backward; the optimizer is in none), its affected gates A and Δ_grad, read from 4.0's
`cut_bits` where the record has them). `frontier --formalism` gives, per start, target and cell of the start's
`policy.cells` (the (c, k) grid in decades from one token's inference block to the start's server block, and the
reference shapes at their exact (c, k): the inference block, the reference policy c = 4096 w, k = 10 kB, and the server
block), W\* = the least |C|ρ and honest = the least |C| over every checkpoint of every valid record, and Security's
S = W\*/honest. Fit is checked: for training its three kinds of block, compute Δ + sτ ≤ k, reduction 2Δ_grad ≤ k and
update application Δ + Δ_grad ≤ k, and where they hold ρ = c/(c − d), d the decoder's gates. Elsewhere ρ is Slowdown's,
the counting lemma at A \ R in its token form, ρ ≥ σ(1 − x₀)·s/s\* (σ the token blocks' share of |C|; with the largest
per-token affected count where token blocks vary), a target until Lean proves it. The claim takes the assumptions'
upper bounds, s\* ≤ max_x x(k − u(x))/a (A1) and x₀ ≤ max{x : u_in(x) ≤ k} (A1'), u and u_in counting update bits
only; a frozen tail is always charged (u is 0 up to its share t, u_in jumps to the changed update at once). The attack
side's best-known x₀ and s\* from exhibited blocks are reported beside them (`x0_known`, `s_star_known`: tightness,
never the claim). u(1) is a field of its own (it equals Δ only when every token's path reads the whole update); u, u_in
are labelled assumptions, a and Δ upper bounds from witnesses, a quantity not measured yet a stub; the row's `labels`
name each (`formalism.STUBS`, and `formalism.V23` for a row ingested before). `plots` writes five panels (P1-P4 and
the A2 curve, the least |C| to each target against Δ, per start), each `<name>.json` and its PNG, which are off the
console; `console` cuts each to the panels route's limits and holds one panel's PUT; and `rescore` re-scores v2.3's
records with the per-record quantities of covert_witness's `analysis/rescore_v1.py`, which has no formula of its own
(module docstring). The console's one training panel, `training/frontier-branches`, is `panels`'s: held-out bits per
byte against total training compute in gates, the best unconstrained learner (the frontier, from scratch or continuing
a start, at C0 + |C|) and per branch point (each start, `on_frontier` when the frontier passes through it) the best
unconstrained and constrained learners starting from it (honest at C0 + |C|; throttled at C0 + |C|ρ per (c, k) of its
dial, with S = W\*/H). It reads the ledger through `rescore`'s loading (`rescore.arguments`, `rescore.load`).

## What the integrator calls

covert_witness's islands run unmodified (`cw.py`): the harness wraps the fixed `prepare.load_start`,
`TrainData.next`, `encode_*` and `submit`, and writes each checkpoint's E to `OUT/E/`, uncounted. It scores nothing:
covert_witness's evaluation job (`cw.score_run`, run by its evaluate.py after the training job, outside that job's
sandbox) decodes each E with `prepare.decode` and scores it with `prepare.Scorer`: the output's digest, then a fresh
slice of the search pool, and the promotion set's score sealed under the digest (covert_witness's README; the final
set only in its final mode). A batch from `TrainData.next` is token data through any op that reads it, not only an
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
python -m te_core.frontier [--formalism [--json]]     # --formalism: W*, honest and S per target and (c, k) cell
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
   writes one submission record per checkpoint, with the program's arguments and, beside a covert_witness record's
   `witness.json`, the tensors it changed; `--readouts` judges its read-outs against the start's tensors. Reference
   (honest) runs add `--reference`.
6. **The frontier**: `python -m te_core.frontier`. Its promotion candidates (`frontier.candidates`) are promoted only
   on the sealed promotion set, one bit each: covert_witness's `promote.py --tier T`, run as the scorer.

Run every command from `benchmarks/covert_witness/`, where `te_core` and `prepare` are importable.

## Files

| file | what it does |
|---|---|
| `flops.py` | `Counter`: the dispatch-mode harness counting MACs and provenance; `guard_exits`: token data read out of torch; `flags`: uncounted-work flags; `forward_macs_per_token` |
| `annotate.py` | `annotate(record, through)`: the maximal annotation, per step U, D_t, the depth profile; `Annotation.phi(d_min)` |
| `counterexample.py` | `Family` (the program's writer), `check` (the sampled bit-for-bit rebuild), `coverage` |
| `score.py` | `score` (the largest ρ over the thresholds τ), `score_grid`, `checkpoints`, `w_star`, `honest`, `effective_slowdown` |
| `ledger.py` | `Ledger` (append-only, per agent, evidence store), `ingest`, the index |
| `launch.py` | tiers, the promotion gate and hourly cap, the `research run` command (`--pool` with gpu-lease inside, or `--gpus 1`) |
| `cw.py` | `Adapter`: a covert_witness train.py under the harness, its held-out guard and its E checkpoints; `score_run`, the evaluation job's scoring of them; `CWPlan`, the family `prepare.declare_step` writes |
| `hooks.py` | what a standalone script may call, no-ops outside the harness |
| `frontier.py` | frontier, elites, coverage map, W\* at the targets, promotion candidates; `--formalism`: W\*, honest and S per target and cell, with W\*'s x₀ and s\* bounds and best-known values |
| `opnames.py` | the op-name sets the counter, the capture and the gate prices share (torch-free) |
| `gates.py` | each counted op's price in Boolean gates from the circuits' Definitions (`DEFINITIONS`, `PRICES`, `kind_of`, `of_log`) |
| `policy.py` | a start's reference shapes (inference block, reference policy, server block), the (c, k) grid in decades, and `cells`, both by key |
| `formalism.py` | the formalism's per-record quantities, Fit and Slowdown's ρ at (c, k) with the attack side beside it; `of_record` (ingest's), `measure` (with labelled stubs, assumptions and upper bounds), the profiles `even` and `tail` (u, u_in), `fit_k` |
| `plots.py` | the five panels (P1-P4 of the experiments draft and the A2 curve), from the ledger, each `<name>.json` and its PNG |
| `console.py` | each panel cut to the panels route's limits (64 KiB, a 500-character note), and one panel's PUT |
| `rescore.py` | v2.3's records scored with the analysis's formalism blocks (or the stubs): the report, the five panels and their cuts, one command; `arguments` and `load`, its ledger and analysis loading |
| `panels.py` | the console's training panel, `training/frontier-branches`: the frontier, each branch point's honest and throttled curves, its console cut, figure and PUT |
| `codecs/` | identity, low rank, sparse mask, quantized delta, seed replay, `Compose`, `Plan` |
| `run.py` | `Session` (the program's API), the run CLI and its script mode |
| `starts.py` | a start's registry entry |
| `capture/` | the static count: one training step traced on fakes (`make_fx`), its \|C\|, GEMM roles and φ without running, in `flops`'s record form; its own suite. Its op sets are `flops`'s, from `opnames` (`ELEMENTWISE_NAMES`, `SCALED_NAMES`, `FUSED_OPTIMIZER_MACS`, `LINALG_NAMES`, `GEMM_LIKE`, `NOT_GEMM`, `TOKEN_INDEX`, `GATHERS`), and so is its rule for an updated node with no multiplication |
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
- **Provenance is by value for θ's tensors and by origin for everything else**: a tensor that isn't exactly one of θ's
  counts as a differing weight even when it computes nothing new (a zero-initialized adapter factor at step 0, say).
  Any annotation holding a subset of these is still valid, so this error can only raise ρ, and a counterexample family
  can lower it again. A start's buffers (RoPE tables) are taken as θ's.
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
  count but coarsens D and φ.
- **Weight shape**: a forward activation with a weight's exact 2-D shape, in a matmul of two token-indexed operands,
  counts as a weight.
- **Bit-for-bit rebuilds need the run's kernels**: families are checked on the run's device, right after it trains.
