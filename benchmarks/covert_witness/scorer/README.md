# scorer: the training-efficiency search's ledger, scores, promotions and digests

The scorer reads only the evidence store and writes the ledger and scores under `$TE_SCORER_STATE` (default
`~/.research/te-scorer`), all of it rebuilt from the store by `collect`. It writes its board lines to its own file,
`islands/scorer/BOARD.md` on its branch (`cursor/te-scorer-516a`), which the coordinator merges into `BOARD.md` on
`cursor/te-digest-4207`, and edits nothing of the fixed side or te_core.

~~~text
python scorer.py collect                               # new manifests -> index.jsonl, index.md, runs.json, starts.json
python scorer.py score                                 # score.json, score.md, submissions.json
python scorer.py board --promote --digest [--append]   # this hour's PROMOTED and DIGEST lines (--append pushes them)
python scorer.py post TAG TEXT                         # one line onto islands/scorer/BOARD.md, pushed
python scorer.py final entries|plan [--begin HH:MM] [--deadline HH:MM]  # the FINAL lines; the reruns that fit
python scorer.py final launch ID... [--seeds 101,102,103] [--node N] [--smoke] [--dry-run]
python scorer.py final dispatch [--nodes N1,N2,...] [--deadline HH:MM] [--dry-run]  # one pass: the plan's next reruns
python scorer.py final collect|table                   # final-set reruns (below)
~~~

The nodes launch.py and promote.py take are the coordinator's shared list `/tmp/te-nodes.txt` (11:55Z) when it exists
(`TE_NODES`), else the harness's `nodes.txt`. The hourly lines are two runs: `board --digest --append`, then
`board --promote --append`, which waits on the sealed set (a `cpu=gpu` node decides in its gpu line). Appends take a
lock, so the two never collide.

- **Records**: evidence/v1 artifacts with `meta.covert_witness.class` (launch.py's, one per candidate) and te_core
  ledger records (`meta.schema` te-core/submission/v1), written since `TE_SCORER_SINCE`. The start registry's bpbs and
  tensors come from launch.py's start-scores record. A record's agent is the `cursor/te-<direction>-4207` branch holding
  its island's commit (witness v4's `island.commit`; before it, `candidate.json`'s commit), else its run's commit, else
  the direction its meta names, else its class. The index names each record's witness schema and version and its
  harness's te_core.
- **Who is scored** (the coordinator's 14:57Z ruling, priced by `te_core.frontier.target` of te_core 3.5.3, a worktree
  at `TE_PIN` on `cursor/te-core-313-on-pin-3a8f`; `target_row`): honest(l*) is the least |C| over every valid record,
  of any class or direction, that reaches l* by a complete schedule, and W*(l*) in each (k, r) the least |C|ρ over the
  same records, so W*/honest >= 1 is the effective slowdown. A complete schedule is a run's last checkpoint
  (`schedule_ends`; the coordinator, 12:05Z: a checkpoint inside a longer schedule is not what a run of that length
  reaches) or any checkpoint of a fork ladder (`fork_ladder`: the reference island's train.py with `--forks` and an
  empty `--snap-at` encodes only at its cooldown branches' ends, each decayed to 0; its FLOPs include the earlier
  branches' passes, which overstates honest there by up to about a quarter). Where only a mid-schedule checkpoint
  reaches l*, both read it and te_core labels the target 'upper bound (checkpoint)'. The reference-only honest, the
  rule before (full-θ runs by `ref-optim` or `ref-data` and the reference smokes, `is_reference`), is kept beside it as
  context. A W* entry whose counted φ is under a tenth of its position φ (`soft_phi`, the soft spot) shows both. The
  measurement auditor's records (direction `audit`, or a note starting `audit:`) are in the ledger and in nothing scored.
- **What a record must show to count**: launch.py uploads a record from an agent's VM, so its `witness.json` and
  `run/run.json` must be the node's (`custody_problems`: the hashes of `out/witness.json` and `out/run/run.json` in the
  run-record the node's runner publishes, the attempt's `outputs.run_record`), and its fixed side must be the
  harness's: a witness v4 record's `measurement_commit` one that `pinned.json` on `TE_BRANCH` has named (launch.py ships
  the pin and only the island), an earlier record's every fixed-side file at its commit (`benchmarks/covert_witness`
  outside `islands/`, `scorer/` and docs) one the harness's branches (`cursor/covert-witness-4207`,
  `cursor/te-core-4207`) held since 30 Oct (`fixed_side`). A record whose E (sha256) is
  another direction's earlier record's (`replays`; the coordinator, 14:22Z: an agent's own relaunch counts), or
  whose harness counted under half the FLOPs of the run's own steps (`over_train_flops` < `COUNTED_FLOOR`), is a
  counted-compute escape and unscored (the coordinator, 10:35Z), unless the scorer has read its op table and found the
  gap accounted for op by op (`FLOOR_ACCOUNTED`, by art id; the coordinator, 13:42Z). A record whose updated set is the tied embedding with
  no block below the top (embed only, or embed and the top block) is priced wrong (tied lookup) and out of the frontier:
  v2.3's counter prices the lookup at 0 (the coordinator, 13:32Z). A pin record whose `run.json` te_core 3.5.3's
  `cw.readout_leaks` lists a step of (`judge_readouts`, run under `TE_TORCH_PYTHON` since cw imports torch; cached in
  `$TE_SCORER_STATE/readouts.json`) took its read-out allowance from index reads no counted work consumed and is
  invalid (the coordinator, 15:00Z); one without its `run.json` is unjudged. Otherwise the record is invalid, with the
  reason in the ledger.
- **Regime** (`regime_of`): a record measured by a commit pinned under the pin's version (`pinned.json` at
  `TE_BRANCH`'s head and its history, `pins`: a re-pin that keeps the version fixes the harness, not the measurement, as
  v2.2's second pin did) is `current`, and only those make the frontier, honest, the promotions, the cells and the finals (the coordinator, 12:55Z: the v2.3
  pin is the last tonight), each against the start bpb its records carry (the median). The rest are labelled context,
  scored apart in `score.md` and in the digest's context lines: `pin V` (an earlier pin's), `paired` (te_core 2.2's
  paired slices before the pins: `score.start_bpb` and every checkpoint's `val_slice`), and `earlier` (the fixed search
  set).
- **ρ**: te_core's (`TE_CORE_PATH`, its `score.score_grid` and `frontier.TARGETS`).
  A record made inside te_core's harness (witness v2/v3, covert_witness v2.1) is priced by the harness's count: `collect`
  ingests the node's `run.json` with the te_core of the official branch with the highest VERSION that holds `PAIRED`
  (`te_current`, a worktree at `TE_CURRENT`), so a pricing fix re-prices every record, the pin's included (the
  coordinator, 12:55Z: re-ingest, not a re-pin, is how a later ρ reaches them). One submission per checkpoint
  carries |C|, U, D, φ and ρ, and the record is refused if te_core's |C| isn't the witness's or a ρ is below 1. Its
  entry takes |C|, φ and D from the last checkpoint. The te_core records agents publish themselves are listed and never
  scored. A witness v1 record has no te_core count, so its annotation is analytic (`scorer.analytic`):
  - U is every forward multiplication whose weight operand is trained: a fully trained tensor's matmul or norm gain, and
    both factors of an adapter. A frozen W·x is θ's, except in a run that merges its pieces (`notes.mech` relora with
    `notes.pieces` > 1), whose targets are full for the whole run.
  - A node's upstream is the updated MACs per token of the nodes before it in (layer, stage) order.
  - |C| is the training FLOPs over 2, p = 16, and c = 4096 times the start's forward MACs per token (weight matmuls, norm
    gains and 2·d·ctx attention MACs a layer).
  - The declared trainable set is checked by the witness to cover every changed tensor, so U, φ and D are never
    understated, but the step log's tokens are self-reported.
  - Each curve point is a submission (its prefix of the run). Scores and digests mark each W* entry (c) counted or (a)
    analytic.
  - The harness's count prices an unmerged adapter at its own matmuls, though the rebuilt model reads W + sBA at every
    adapted matrix. The coordinator (13:42Z) ruled that ρ a real result, and the digest names the route a soft spot
    (`soft_spots`): every record whose counted φ is under a tenth of its witness's (`prepare.deep_share`), and the
    newest one at ρ 1 in every cell with φ ≥ 0.3 by deep_share.
- **Counterexample credit** (`scorer.credit`, witness v1 only; v2's families are the harness's, verified by
  `prepare.declare_step`): an E whose decoder rebuilds a step's updated weights from θ within c is a T1 family for that
  step. A seed log (`E.format` seedlog) with one coefficient a step (MeZO) is one: its prefix E_t rebuilds step t in
  (t + 1)·n MACs for n trained values, so it covers the steps with (t + 1)·n ≤ c, at β_t = bits(E_t) / updated MACs per
  token (bounded by the smallest snapshot's E), and te_core's option (a) takes that β. A seed log of fewer coefficients
  than steps (a trained subspace, read from E's header), adapter and full E's, and a piece run's snapshots show no step
  rebuilt bit for bit and get none; an adapter's ρ is low because its φ is.
- **Cells**: mechanism (the note's first word) x |E| range (decades from 10 kB) x φ range (te_core's `PHI_EDGES`, on
  te_core's φ for a v2 record, else the witness's φ over the tensors that actually changed) x start. A cell is thin at two
  valid records or fewer.
- **Promotions** (hourly): the candidates are each direction's top third by relative bpb drop from its start, and the
  best record of each thin cell, among valid current-regime records launched from 08:30Z. Each candidate's submitted
  output goes to the harness's `promote.py` (`sealed_bits`), which decides one bit per output on its node from the
  sealed promotion value; only a true bit is PROMOTED, and the DIGEST names the candidates the sealed set didn't hold.
  A decision run that promote.py stops waiting on (research run's 20-minute cap, behind a deep gpu line) is kept in
  `$TE_SCORER_STATE/sealed/pending.json` and read when it ends, never relaunched; its candidates are listed as not yet
  decided until then.
  `board` without `--append` is a dry run that lists the candidates and decides nothing. Re-screens (a mechanism's note,
  less rates and seeds, on a class and start another direction screened first, or its own direction swept three times
  already) are flagged.
- **Final scoring** (`scorer.py final`): an entry (a valid current-regime search record) is rerun at fresh seeds (101,
  102, 103 on the `--seed` and `--data-seed` flags its train.py takes; once if it takes neither, since the harness seeds
  nothing for a `--script` run) by the harness's `launch.py CLASS START --direction scorer --final --no-wait ... --
  TRAIN_ARGS` with `--island` and `--program` the entry's (TRAIN_ARGS by `entry_argv`: the record's `args` joined
  by spaces, with each string-valued flag's value taken whole from the program's parsed `train_args`, so an empty
  `--snap-at` or a spaced `--arms` survives; a launch whose args differ is `superseded` and relaunched at its seed),
  run from `overlay`: a commit whose tree is the harness branch
  head's (`TE_BRANCH`) with the entry's `islands/`, its parents both. launch.py ships the pin and takes only that
  island (checked to be the entry's tree), so every rerun is measured by the pin with the entry's own island code and
  train args, and `final table` refuses one measured by any other commit. In final mode (`CW_FINAL=1`) every score is the final set's, and launch.py preserves the
  record with meta `final: true` and ingests it into no ledger. `final collect` runs `launch.py --collect` once per ended
  run and reads its witness and te_core's submissions from its `run.json`. `final table` refuses a rerun that isn't
  done, valid, scored on the final split, priced at the witness's |C|, ρ ≥ 1, over the counter floor and with no
  read-out leak. It averages the seeds per checkpoint and writes the final table of W* and W*/honest per start, target,
  k and r, against the final set's start bpb: every current-regime record, an entry's reruns replacing its search
  values where they exist (the coordinator, 15:27Z) and a search value carried onto the final start at the same drop
  from its search start. Beside it go the reruns alone and the search set alone, and the frontier plots, into
  `$TE_SCORER_STATE/final/`. `final entries` reads each direction's last `[FINAL]` line (its own board on its
  `cursor/te-*` branch or the merged board; art ids or run ids). `final plan` orders the reruns (the coordinator,
  12:55Z): the entries that set honest or W* at some target in the pin's table, shortest first, then the first
  `FINAL_PER_DIRECTION` (3) records each direction's FINAL line names (an art id and its run's id are one record): every
  direction's first, shortest first, then every second, then every third. It keeps the seeds that end by `--deadline` (17:00Z)
  starting at `--begin` (15:00Z) on the listed nodes' GPUs (8 a node; a rerun takes its record's GPU seconds plus 10
  minutes). `final dispatch` launches the plan in that order, recomputed each pass: a node (`--nodes`, in order: n1,
  then n3 and n2 as their search runs free them) takes the next rerun while fewer than four of the scorer's reruns wait
  there for a GPU slot (slot.py's `slot.json` not `held`), and nothing is launched that would end after `--deadline`
  (its record's GPU seconds plus 5 minutes). A `--smoke` launch checks
  the pipeline and never enters the table.
