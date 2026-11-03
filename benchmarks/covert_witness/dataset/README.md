# The training-efficiency dataset

The training-efficiency search's result is this dataset: every checkpoint the search measured at its pin, with what each
number rests on, and the search's qualitative record (each run's hypothesis and outcome, the board's lines, the methods,
each direction's ideas and the lessons). The `training/frontier-branches` panel is one view of it, and the
`branch_points` and `curves` tables hold that panel's numbers.

- `export.py` builds the dataset from te_core's ledger, the scorer's directory, the board and `curation.json`, and
  publishes it.
- `schema.json` is the schema as JSON Schema. The exporter checks every row against it before it writes anything, and
  this file is the same schema in prose.
- `curation.json` is the hand-written part: what each method is, what it taught and what is open, each direction's ideas
  with their verdicts, and the lessons, each citing its evidence by id.
- `tests/` holds the exporter's tests: `uv run tools/verity/check/suites.py benchmarks/covert_witness/dataset`.

## Building and publishing

Run the exporter from `benchmarks/covert_witness` with the workspace on the path, as te_core's commands run:

```sh
PYTHONPATH=../.. python -m dataset.export --out DIR --root LEDGER --scorer SCORER --board BOARD.md \
    --pin COMMIT [--pin COMMIT ...] [--methods METHODS.json] [--gates GATES.json] [--pretraining DIR] [--publish]
```

- `--root`, `--pin`, `--scorer`, `--facts`, `--quantities`, `--tail`, `--charge-context`, `--analysis`, `--stubs`,
  `--source` and `--manifest` are te_core.rescore's options, read as it reads them. The checkpoints are those of the
  valid records at the pins, by te_core.frontier's rules, and every other ledger row goes to `left_out` with its reason.
- `--quantities` defaults to `analysis` here (te_core.rescore's default is `measured`): the per-record quantities of
  `analysis/rescore_v1.py`. `measured` keeps only the records ingested with a traced step, which no v2.3 record is. A
  checkpoint that neither covers takes te_core.formalism's stubs on its own fields and says so in `quantities` and
  `caveats`.
- `--charge-context` adds te_core.formalism's context charge, a sensitivity for A1 that is off by default. It reads each
  run's changed tensors from `--facts`.
- `--board` is the dated board `tools/merge_board.py --out FILE` writes, with a `### YYYY-MM-DD` heading over each day's
  lines. A board without them is refused, since an entry's time needs its day.
- `--methods` is the console's `training/methods` panel body. It labels and describes the methods `curation.json`
  doesn't.
- `--gates` is the gate factors the analysis prices counted MACs at, copied into DIR as `analysis-gates.json` first.
- `--pretraining` is te_core.panels' pretraining runs, the ones the starts were taken from (by default fetched from the
  store).
- `--publish` puts DIR in the evidence store (`research data put --kind evidence/v1 --tree DIR --preserve`) and prints
  its art id. `--research` names the command it runs.

DIR gets one JSON Lines file per table (`checkpoints.jsonl` and so on, one row per line), `manifest.json`, `schema.json`,
the `curation.json` it read and `analysis-gates.json`. The tables are JSON Lines rather than Parquet because pyarrow is
not in the workspace.

## Conventions

- **Ids.** A checkpoint is its ledger id (`hybrid-000146`), a run its run id (`r` + launch date + time + four hex
  digits), a board line `<UTC minute>-<agent>-<hash>` (the hash six hex digits of the SHA-256 of its tag and text), an
  artifact its `art:` id. A
  method's key is `direction/mechanism`, each part lowercased to `[a-z0-9_-]` and cut at 64 characters, as the launcher
  tags runs.
- **Nights.** A run belongs to the night of its launch time; `manifest.nights` gives the boundary.
- **Compute.** Gates are Boolean gates (te_core.gates), MACs the harness's count and FLOPs twice the MACs.
- **Losses.** Held-out bits per byte: `search_bpb` on the search slice, and `final.mean_bpb` on the sealed final shard
  where a final rerun covers the checkpoint. A drop is `1 - bpb / l0`, with l0 the start's loss on the same slice.
- **The dial.** c is in tokens of the start's forward per block (1, 10, 100, 1000, 4096) and k in bytes (10 kB to
  100 MB); each cell also gives c in gates and k in bits. A cell's `basis` is `fit` where every part's blocks fit k
  (ρ = c / (c − d)) and `target` where ρ is Slowdown's target, max(1, the bound).
- **Quantities.** Every ρ input is `{value, basis, source}`. The basis is `measured` (counted by the pinned harness, or
  derived from counted values only), `upper-bound` (a bound the witness proves, safe for the claim), `estimated` (a stub,
  an assumption or a value derived from one) or `unmeasured` (no value). A value derived from several takes the worst of
  their bases. The source is te_core's label for the value.
- **Nulls.** A field with no value is null. A non-finite number (an infinite ρ) is written null, and
  `manifest.non_finite` counts them.

## The tables

`schema.json`'s top level has `tables`, one entry per table below, and `manifest`.

- **`checkpoints`**: one row per checkpoint of every valid record at the pin, keyed by `id`. It holds the run, the
  checkpoint's step and schedule `fraction`, `schedule_end` (whether this checkpoint ends a complete schedule), night,
  launch time, agent, direction, mechanism and `family` (how the update is carried); `start` (its id, label,
  parameters, tokens, tokens per parameter, l0 and C0 with C0's basis); `method` (key and label); `config` (optimizer,
  optimizer state bits, the update code's format and bytes, the trainable set, whether θ is trained in full, the lowest
  updated layer); `seed` (the run's and the data's); `harness` (version, measurement commit, te_core versions, witness,
  commit, program and its SHA-256, island, node, hardware); `compute` (gates, MACs, FLOPs and their basis); `loss`
  (`search_bpb`, `start_bpb`, `drop` and `final`); `quantities` (`analysis` or `stubs`); `rho_inputs`, the formalism's
  inputs as quantities (|C|, tokens, token gates, affected gates, τ, a, Δ, u(1) and the profiles u and u_in, the tail
  share, the decoder's gates d, Δ_grad, the forward per token, σ, w, the per-token affected counts, the parts and the
  context charge's inputs); `dial`, te_core.formalism at every cell; `fit_k`, the least k at which Fit holds at the
  default c, for running θ′ and for the training; `validity`; `in_panel` (whether `training/frontier-branches` plots
  it); `arts` (the record, the launch record and the run record); and `caveats`, each a sentence.
- **`left_out`**: every ledger row that is not a checkpoint, with its `reason` (`invalid`, `other-version`,
  `duplicate-e`, `re-ingested` and the rest of te_core.frontier's statuses).
- **`earlier`**: every scorer submission of a regime before the pin's (harnesses 2.0 to 2.2), with its regime, harness,
  compute, loss and update code bytes, kept apart from the checkpoints since an earlier harness measured them.
- **`runs`**: one row per run any source names, keyed by `run`: kind (`search`, `final-rerun` or `unrecorded`), night,
  agent, direction, mechanism, start, node, state, harness, `hypothesis`, `config`, `seed`, `outcome` (its checkpoints,
  what was left out, its best checkpoint and the first to reach each target drop), `rerun_of`, and the board lines that
  name it, split into `findings` and `negatives`.
- **`board`**: every line of the board, keyed by `id`: time, night, agent, tag (`FINDING`, `NEGATIVE`, `LOOPHOLE`,
  `DIGEST`, `TOOL`, `QUESTION` and the rest), text, and the `runs`, `checkpoints` and `arts` its text names, resolved
  against the other tables. What reads as an id but resolves to nothing is in `unresolved`; `copies` counts identical
  lines that several boards carried.
- **`methods`**: one row per method key any record has: label, description and `description_source` (`curation`, the
  methods panel or a launch note), `learned` and `open` (curated), family, counts of runs, checkpoints and left-out
  rows, starts, nights, the `best` checkpoint per start, and the findings and negatives of its runs.
- **`directions`**: one row per direction: `what` it tries, its `ideas` (each with a `verdict`: `gain`, `no-gain`,
  `mixed`, `untested` or `loophole`, an outcome and evidence), `learned`, `open`, evidence, and counts of runs,
  checkpoints, mechanisms, starts, findings and negatives.
- **`lessons`**: the search's lessons, each with a `kind` (`loophole`, `measurement`, `method` or `process`), the lesson,
  its `fix`, a `status` (`fixed`, `rule`, `standing` or `open`), a `rank` (1 to 5 for the five most important, else
  null) and evidence.
- **`starts`**: one row per start: label, shape, the forward per token in gates, l0 on the search slice and on the final
  shard, C0 with its basis and source, whether it is on the pretraining frontier, and its counts.
- **`branch_points`**: per start and target drop (0.5%, 1%, 2% and 5%): l*, whether any checkpoint reaches it, the blue
  envelope H (the least compute of any checkpoint that reaches l*) and, per dial cell, the red envelope: W* (the least
  compute times ρ), S = W*/H, ρ and its basis, the checkpoint that sets it, x₀ and s* with the attack side's best-known
  values.
- **`curves`**: `training/frontier-branches`' rows, uncut, with the start's and the checkpoint's ids: curve, start,
  compute before and after the start, loss, S, run, method, Δ and basis.

`manifest.json` says what went in and what the numbers rest on: the dataset's name, te_core's version, the checkout and
the pins, the newest ledger row's time, each source it read, the nights, every table's count and the
breakdowns by night, start, direction, family, quantities and harness, each ρ input's and each dial cell's bases,
te_core's labels with their counts, C0's rate, the directions `curation.json` doesn't describe (empty when it is
complete), the five ranked lessons, and the caveats.

## The curated record

`curation.json` has three keys. `methods` maps a method key to its `label`, `description` (at most 200 characters),
`learned` and `open`, each a list of sentences. `directions` maps a direction to `what`, `ideas`, `learned`, `open` and
`evidence`, an idea being `{idea, verdict, outcome, evidence}`. `lessons` is a list of `{id, kind, lesson, fix, status,
rank, evidence}`. Evidence is a list of ids, each a board line's, a run's, a checkpoint's or an `art:` prefix of at least
12 hex digits. The exporter refuses a curation that cites an id no table holds, describes a method no record has or
repeats a lesson id.

## What the numbers rest on

`manifest.caveats` lists them, and a checkpoint's own `caveats` repeat the ones that apply to it. In short:

- The ρ inputs are the analysis's (or stubs), not measured on each record's traced step. On full-training routes the
  analysis's ρ is far above the traced one.
- Fit's update clause is checked as Δ + Δ_grad ≤ k with Δ the update code, as te_core prices it. The clause reads Δ_t,
  what the update block reads, which te_core does not price yet, so for a run that trains full-precision latents (family
  `update-code`) Fit is optimistic.
- The search's losses are single runs. The sealed final reruns (three seeds) check the checkpoints they cover.
- Every start's C0 is estimated: its training tokens at dense AdamW's counted gates per token.
