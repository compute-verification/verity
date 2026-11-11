# covert_witness: the least compute that lowers a start's loss to a target, at every policy (c, k)

The training formalism v2 prices a training
computation C at a policy (c, k) by its slowdown ρ, so that |C|ρ is what the computation costs under that policy. This
speedrun measures, per start (θ, ℓ0) and target ℓ*, the least |C|ρ of a valid candidate's checkpoint that reaches ℓ*
(W*), against the least |C| of any valid candidate's checkpoint that reaches it (honest H), and S = W*/H ≥ 1. The inputs are
the start, the target and the policy, with c (gates per block) and k (bits a block reads beyond θ) as two independent
axes and n the per-period cap; the red team chooses everything else: no class caps the update, its size, its format or
its depth. Every candidate carries witnesses, which the evaluator checks, and every record carries the formalism's
quantities, which te_core measures.

It is autoresearch-shaped: a fixed side (`prepare.py`, `evaluate.py`, `candidate.sh`, `sandbox.py`, `launch.py`,
`pin.py` and the harness `te_core/`, with `data.py`, `starts.py`, `starts_store.py` and `model.py` behind them) and
islands (`islands/<name>/`, one per direction or seed program), each with one editable `train.py` and its program,
`README.md`, for a researcher agent.

**Version**: v3.0 (`prepare.VERSION`, in every witness record, schema `covert-witness/witness/v5`; data version
`sealed.DATA` v3, `data_v3.py`: OLMo 2's stage-1 mix and its 32,000-token BPE; v2.2's FineWeb-Edu, `data.py`, is the
v2.3 harness's, labelled `v2.3-pilot` where its records are scored). It measures every candidate as v2.3 did (te_core's harness counts its MACs and times its steps in a
sandboxed training job; a separate evaluation job rebuilds and scores the model on a fresh slice of the search pool;
promotions are decided on a sealed set and the final set is sealed until the scorer's final scoring; one pinned commit,
`pinned.json`, measures every candidate, taking only its island from the candidate's commit). It drops the classes and
scores by the formalism. `pinned.json` names v3's commit, and v2.3's two as its `pilot`.

## Policy and score

- **Gates** (`te_core.gates`): |C|, c and w are Boolean gates. Each counted MAC is priced by the circuits' own
  Definitions, their gate counts (AND, XOR, NOT) as `verity.circuits.evaluation.bits.compile` gives them, the AND
  counts checked against circuit-check's pins (`verity/definitions/ops/boolean/*.circuit-check.json`;
  `python -m te_core.gates --check` recompiles them):

  | op | gates per MAC | from |
  |---|---|---|
  | bf16 or f16 GEMM | 1,480.1 | `HopperBF16WgmmaDot16_v2`, 23,682 gates / 16 |
  | e4m3 / e5m2 GEMM | 861.1 / 856.7 | `BlackwellE4m3QmmaDot32_v2` 27,555 / 32, `BlackwellE5m2QmmaDot32_v2` 27,413 / 32 |
  | fp4 GEMM | 334.8 | `BlackwellNvf4OmmaDot64_v2`, 21,429 / 64 |
  | f32 GEMM, optimizer, linear algebra | 12,637 | `F32Fma_v4` |
  | bf16 elementwise | 3,947 | `Bf16MulBf16_v2` 2,053 + `Bf16Add_v2` 1,894 |
  | gather (one 16-bit word) | 48 | a word multiplexer, 3 gates per bit |

  An op with no dtype in the log is priced as bf16, optimizer and linear algebra as f32, and a dtype with no
  Definition (int8) as bf16, each noted in the record (`assumed`). There is no sm_120 bf16 Definition yet, so bf16 is
  the Hopper dot's.
- **Policy grid** (`te_core.policy`, from each start's `shape` in `starts_store.json`): every decade of c and of k
  from one token's inference block (c = its forward in gates, k = its id and its context's keys and values) to the
  start's server block (c = C0, k = its training state beyond θ, 80 bits a parameter), with the reference policy of the
  experiments draft (c = 4096 w, one 4,096-token session per block, and k = 10 kB) between them. m100: c 1e11 to 1e21
  gates, k 1e4 to 1e10 bits, 77 cells; s100: c 1e10 to 1e20. `python -m te_core.policy` prints each start's.
- **Per record** (`te_core.formalism`, written at ingest): per step structure, its gates, the gates in token blocks
  (σ, their share of |C|), the affected gates A and Δ_grad; τ, a, Δ (the whole update, Fit's), d (the decoder's gates),
  u(1) (the per-token update; equal to Δ only when every token's path reads the whole update) and the update profiles
  u(x) and u_in(x) (a block computing the token from its inputs, in dependency order), both counting update bits only.
  At (c, k): w, s = c/w, then Fit, checked per kind of block (compute Δ + sτ ≤ k, reduction 2Δ_grad ≤ k, update
  application Δ + Δ_grad ≤ k), where ρ = c/(c − d), and elsewhere Slowdown's token form ρ ≥ σ(1 − x₀)·s/s\* (with the
  largest per-token affected count where token blocks vary), a target until Lean proves it. The claim takes the
  assumptions' upper bounds, s\* ≤ max_x x(k − u(x))/a (A1) and x₀ ≤ max{x : u_in(x) ≤ k} (A1'); a frozen tail of
  share t is always charged (u is 0 up to t, u_in jumps to the changed update at once). The attack side's best-known x₀
  and s\* are reported beside them (`x0_known`, `s_star_known`: tightness, never the claim). u and u_in are labelled
  assumptions, a and Δ upper bounds from witnesses, a quantity te_core doesn't measure yet a stub; the record's
  `labels` name each.
- **Score** (`python -m te_core.frontier --formalism`): per start, target (`TARGETS`, drops below the start's bpb) and
  cell, W* = the least |C|ρ over every checkpoint of every valid record that reaches ℓ*, H = the least |C| over the same
  checkpoints (the coordinator's 19:50Z rule), and Security's S = W*/H. **Plots** (`python -m te_core.plots --out
  DIR`): P1-P4 of the experiments draft and the A2 curve (the least compute to each target against Δ, per start; Δ's
  bins, never caps), as five panels, each `<name>.json` and a PNG made from it, off the console; `python -m
  te_core.console DIR` cuts each to the panels route's limits. The console's training panels are
  `training/algorithm-policy` (per run, start and gain target, a gain being effective compute over the start's own, its
  algorithm's inefficiency p = C/H against its policy's slowdown ρ, reached or not) and `training/frontier-fit` (per
  data, the scaling law fitted to its unconstrained frontier, which turns a loss into effective compute), both `python
  -m te_core.algorithm_policy --out DIR --publish`; `training/frontier-branches` is retired.
  **Re-scoring v2.3's records** (`python -m te_core.rescore --out DIR`): the same score and panels over the pin's
  valid records, on the quantities measured on each record's traced step (te_core 4.0's default; `--charge-context`
  charges earlier tokens' affected keys and values), or `analysis/rescore_v1.py`'s per-record quantities with
  `--quantities analysis` (from the scorer's directory), or the stubs with `--stubs`.

**Budget**: 2x the MACs the harness counts ≤ 9e16 x 1.1 (`TRAIN_FLOP_BUDGET`, about 10 GPU-minutes of m's dense AdamW at
its measured 1.49e14 FLOP/s, times `HARNESS_FLOP_SLACK` for the elementwise and optimizer work the analytic meter leaves
out) and train.py's steps' wall seconds under the harness ≤ 3000 (`GPU_SECONDS_CAP`; the harness runs eager and counted,
about 2.4x a compiled run). The budget bounds a run's node time; it is not a class. The long tier (`prepare.budget(start,
"long")`, `launch.py --long 'HH:MMZ who'`, valid only for a run the board assigns it with that line): up to `LONG_C0` = 100
times the start's C0 in FLOPs and `LONG_GPU_SECONDS`, 16 GPU-hours, of steps.

The seed islands (`islands/<name>/train.py`), each a program to start from:

| island | seed |
|---|---|
| `reference` | Muon on the blocks' matrices, AdamW on the embedding and norms; E every tensor |
| `e1mb` | LoRA on every block matrix and the norm gains, sized to E = 1,000,000 bytes (`--e-bytes`) |
| `e100kb` | LoRA on each block's o and down projections and the norm gains, sized to 100,000 bytes |
| `shallow` | AdamW on the top 2 blocks and the final norm; the tied head stays frozen (it is layer 0 too) |
| `seed` | MeZO: two forwards a step, the step's coefficient stored, z replayed from seeds; 100,000 bytes of log |

## The fixed side

- **Data** (`data_v3.py`, data version v3; its docstring has the rules): a proportional sample of OLMo 2's stage-1 mix
  (`allenai/olmo-mix-1124` at `data_v3.REVISION`, `FRACTION` of each source's bytes; the 278 chosen files with their
  sha256 in `data_v3.json`), split into units each a proportional sample of the mix. Unit 0 is held out: its documents
  in hash order, less any whose text matches a training document's (6,567 of 104,971), each taken whole until a set
  holds its bpe32k tokens: the search pool (2^23, `CW_ROOT/heldout/v3/search-pool`), the promotion and final sets (2^21
  each, sealed under `heldout/v3/`) and the ladder's validation stream (2^22, `CW_ROOT/train/v3/val.bin`, in no
  training job's view). Each set is in bpe32k's and GPT-2's tokens (final also in Pythia's), with every document's tokens
  and bytes, pinned (`prepare.SET_SHA256`), so every node scores the same text; `CW_ROOT/heldout/v3/data.json` records
  them. Bits per byte divides by the documents' UTF-8 bytes, so scores compare across tokenizers. The tokenizer,
  `bpe32k`, is a byte-level BPE of 32,000 tokens (end-of-text 0) trained on the training stream's first 8 parts
  (`CW_ROOT/train/v3/tokenizer.json`); models tie their embeddings. The training stream is units 1-277,
  `CW_ROOT/train/v3/parts/NNN.bin` (uint16 bpe32k tokens, each document after end-of-text, about 40B tokens, about 4.23
  bytes a token) with its records, `manifest.json` last, read through `prepare.TrainData("bpe32k")` (`parts=N`: the
  first N); Pythia's shard 000 (`train/pythia.bin`) is the sealed start's.
- **Search scores** (`prepare.Scorer`): every search evaluation, the evaluation job's, hashes the weights and then
  samples a slice of the pool (`prepare.draw_slice`: a coin of 16 bytes from the OS, and 1,024 of the pool's 8,192
  evaluation windows, 2^20 tokens, sampled by a generator seeded with sha256(coin ‖ digest)). The score is the start's bpb
  on the whole pool plus the candidate's minus the start's on the same windows, so a slice's difficulty cancels; the
  start's windows are scored once per process. The record names the coin and the digest, which reproduce the slice.
- **The sealed side** (`sealed.py`; `CW_SEALED`, `/workspace/research/te/sealed` on the node): the promotion and final
  sets, the final start, the sealed values and the decisions. At each checkpoint the evaluation job also scores the
  promotion set's first 2^20 tokens (`prepare.PROMOTION_TOKENS`) and writes the value (the search and promotion bpb,
  and the start's) once under the weights' digest
  (the checkpoint's `output_sha256`); nothing returns it. Promotions are the scorer's `promote.py`: it sends candidates
  (`{id, output_sha256}`) to `sealed.py decide` on the node that holds their values and gets one bit each, promote iff
  the search gain over the start is positive and the promotion gain is at least half of it (`sealed.PROMOTION_SHARE`),
  with the value record's salted sha256; each decision is appended to `decisions.jsonl`. `promote.py --tier T` promotes
  te_core's frontier members and elites at tier T - 1 this way (`python -m te_core.frontier --promote` is refused).
- **Final mode** (`CW_FINAL=1`, the scorer's, after the search; `launch.py --final` with direction `scorer`): the
  evaluation job scores the final set (`evaluate.py --split final` refuses outside final mode), and the sealed start
  loads (its directory is mounted into the training job only then): `pythia160m`, EleutherAI's Pythia-160M at its recorded revision cast to fp32 (`prepare.register_pythia`). It
  trained on the Pile, so the same final text is a domain shift for it, and its architecture (GPT-NeoX, family
  `hf-neox`, whose parallel residual makes its φ an upper bound) is not the search's. A final run is preserved but not
  ingested into the search's ledger.
- **Starts** (`starts.py`): `s50`, `s100`, `m50`, `m100`, learning_frontier's dense s and m (seed 0, data order 0, its
  best rates) at 50% and 100% of their 8-tokens-per-parameter runs, retrained here since that branch kept no final
  checkpoint. Each is `<name>/theta.safetensors` with `start.json` (the digest θ, `prepare.digest`; config, tokenizer and
  provenance), and all four are one preserved artifact in the evidence store, named with every file's sha256 in
  `starts_store.json`. Each node installs them into one fixed directory, `/workspace/research/te/starts` (`CW_STARTS`,
  `prepare.STARTS`), with `python3 starts_store.py install --keep`, which checks every file against the manifest; no run
  reads another run's directory. Each start's manifest entry has its `shape` (`starts_store.shape`), which the policy
  grid reads. These four are GPT-2 starts on v2.2's data, the pilot's. v3's starts are the v3 starts ladder's
  (`islands/ladder-v3/ladder.py`, dense lf models at vocabulary 32,000 on bpe32k's stream, at several tokens per
  parameter with C0 counted), which join the store with tokenizer `bpe32k` in their `shape` when it publishes, each
  batch in a commit that changes only `starts_store.json` on the pin, pinned with it as `same_version`. At v3 a branch
  starts only from v3's starts: launch.py refuses a pilot start, and scoring leaves out a v3 record from one
  (`pilot-start`).
- **Evaluator** (`evaluate.py`, `prepare.evaluate_bpb`): bits per byte at context 1024 of decode(θ, E). The model has
  θ's tensors, shapes and dtypes, or with E in the model format (`prepare.encode_model`, CWMODL1) its own lf
  architecture with θ's vocabulary, context and tying (`prepare.model_config`), grown, pruned or distilled, every tensor
  θ doesn't give it in E. Nothing in the harness caps its cost: the evaluation job records each model's parameters and
  forward MACs and gates per token (`te_core.cw.forward_cost`), and scoring admits a checkpoint only within the cap
  (`te_core.policy.eligible`: at most the greater of twice the start's forward and the compute-optimal model's at its
  loss under the v3 ladder's fit, and twice the start's forward until that fit exists).
- **Training job** (`candidate.sh`, `sandbox.py`, the harness `te_core/` and its README): the island's program runs as
  `python -m te_core.run --script islands/<island>/train.py`, which counts every multiply-accumulate it computes through
  torch, times its steps (a step is one `TrainData.next` to the next), writes the first E in each 5% of the run to its
  output (a checkpoint, scored later) and refuses programs that route around the dispatcher. It runs in sandbox.py: new
  mount, PID, network, IPC and UTS namespaces, as the run's user with no privileges, seeing the pinned commit's
  `benchmarks/covert_witness/` with only its own island under `islands/`, the starts, the training stream
  (`train/v3/parts/`, its tokenizer and manifest; and the Pythia stream), the Python and its leased GPU, all read-only,
  and one output directory; no held-out set, sealed side, validation stream, other run or home directory is mounted, and
  there is no network. An audit hook also refuses `prepare.UNREADABLE` (the search pool, the sealed side, v2.0's
  `eval/`, learning_frontier's `val.bin` and hub cache, and v3's `val.bin`) and other runs' outputs.
  The submission holds E, the step log and the declarations; no model. sandbox.py records the job's whole process tree:
  every process's CPU seconds (the namespace's init reaps them all) and each GPU compute process `nvidia-smi` sees in
  it.
- **Evaluation job** (`evaluate.py`, after every process of the training job is gone, outside its sandbox, in the same
  leased GPU slot): rebuilds each checkpoint's model from θ and its E with the fixed decoder (each E file checked against
  the digest the harness recorded) and scores it (`te_core.cw.score_run`); the submitted E's is the score.
- **Witness checker** (`evaluate.py`): θ's digest (`prepare.START_DIGESTS`); decode(θ, E) has θ's tensors, shapes and
  dtypes (it is the model); 8·|E| bits; the declared changed set covers what differs, and gives φ and the lowest updated layer (`prepare.deep_share`); the declared
  trainable set covers the changed set; the harness's record is this E's and found nothing invalid, and its FLOPs and
  seconds are within the budget; the training job's process tree did no work the harness didn't count (`tree_counted`:
  every CPU second but the harness's own within 10 s or 1%, no other process on a GPU, no GPU it didn't lease; the
  seconds from its start to E's write are recorded); the source is the measurement commit's (`source_clean`) and the
  island is recorded (`island_recorded`). Nothing caps E's size, format or depth. A failed check makes the ledger's records invalid. A step the island declares (`prepare.declare_step`) is a
  counterexample family the harness checks by rebuilding sampled steps with `prepare.decode`, bit for bit, within c MACs
  (te_core's README). The analytic meter (`prepare.train_flops`, from the step log) is recorded beside the harness's
  count, and is that count's matmuls and attention (`te_core/tests/test_meter.py`), so an island sizes its steps by it.
  Decoders are fixed (`prepare.decode`: `full`, `lora`, `lora2`, `seedlog`, `sparse`, `quant`) and read only θ and E;
  the witness records E's rebuild cost in MACs (`prepare.rebuild_macs`), and each curve point's (the harness's
  checkpoints).
- **Pinned version** (`pin.py`, `pinned.json` on `cursor/covert-witness-4207`): the one commit whose fixed side measures
  every candidate. launch.py ships it (a clean worktree of it under `~/.research/cw-pinned/`) whatever the candidate's
  checkout holds, and the witness and ledger records name it (`measurement_commit`); te_core's frontier leaves out a
  record of any other commit than it and the earlier commits of its version (`same_version`) (`other-version`), and
  one whose E an earlier record at the pin, or another agent's at any commit, submitted (`duplicate-e`). `pilot` names
  an earlier version's commits, whose records only the console's training panels (`te_core.panels`,
  `te_core.algorithm_policy`) show beside the pin's, each labelled by its data.
- **Nodes** (`nodes.txt`, `prep.sh`): a node is listed (`NODE pool=GPU_POOL cpu=CPU_POOL`) once `prep.sh`, run there
  in its CPU pool, has installed the starts, built the held-out sets (checked against `prepare.SET_SHA256`, left
  read-only) and registered the sealed start; on a prepared node it only checks. vy-nebius-1, -2 and -3 are listed.
  Each launch takes the list's next node in turn (`launch.py --node` names one).
- **Launcher** (`launch.py` on an agent VM, `candidate.sh` on the node): one candidate a run, `research run --on NODE
  --pool GPU_POOL` from the pinned commit, the island sent as a tar (`git archive HEAD:islands/<island>`), with
  `gpu-lease 1` inside (vy-nebius-2 takes runs only through its slot pools, and `--gpus` is refused there);
  the record goes to the store as `research data put --kind evidence/v1 --tree OUT --preserve`, and the harness's record
  into te_core's ledger (`python -m te_core.ledger --agent D --publish ingest`, D the direction from the branch
  `cursor/te-<D>-NNNN`, the mechanism the note's first word), after which the local frontier is printed.

**Run**: `python benchmarks/covert_witness/launch.py START [--island NAME] --note TEXT [--program FILE] --
TRAIN_ARGS` from a checkout whose island is committed: the island is NAME, else `islands/<direction>/` when HEAD has its
FILE (default `train.py`), and nothing else of the checkout is used.
`starts.py --size s|m` built v2.3's starts once (one GPU); `data_v3.py build --push NODE ...` installs data v3 on every
node from one; `prep.sh` prepares a node (`data_v3.py verify --keep`, `prepare.register_pythia()`, the starts from the
store); `evaluate.py --start NAME --out F` scores a start on the whole
search pool. `CW_DEVICE=cpu CW_LEASE=none` runs candidate.sh on a CPU for a smoke test.
`analysis/rescore_v1.py` gives the scorer's current records' per-record quantities under the training formalism v2, analytically: gates, token blocks' gates, σ, the affected
gates, Δ, Δ_grad, u and u_in, the frozen tail's share t and P4's family. It has no formula of its own: `python -m te_core.rescore --quantities analysis` scores them. Its docstring
lists the approximations.

**Reads and writes**: the starts are in `/workspace/research/te/starts` and the sealed side in
`/workspace/research/te/sealed` (above); everything else lives under `/workspace/research/scratch/covert-witness`
(`CW_ROOT`, a retention the training lane owns): `heldout/<data version>/` (the search pool), `train/`, `ckpt/`, and `candidates/`
(each candidate's island, training job output and sandbox root, removed when it ends).
A candidate's run `out/` holds `run/run.json` (the harness's record as the evaluation job scored it, schema
`te-core/run/v1`, with `evaluation`), `witness.json` (schema `covert-witness/witness/v5`), `tree.json` (the process
tree), `island.json` and `island.tar`, `timing.json`, `meta.json`, `steps.jsonl`, `E.bin` when at most 1 MB, `train.py`
and its logs. **Backs**: no Lean guarantee or claim; its numbers are measurements of training, scored by the
formalism's Fit (checked), Slowdown (a target until Lean proves it) and Security (S = W*/H).
