# covert_witness: the best compute efficiency at lowering loss, per class of a training computation's structure

The theorem prices a training computation's slowdown from its structural parameters: the update's size (β, K′), its depth
D, its deep share φ and its block size m. This speedrun measures the other factor: for each class of those parameters, the
lowest held-out loss a candidate reaches from a registered start at a fixed training-FLOP budget, and its FLOPs-to-target
curve. The attacker's effective cost in a class is the theorem's slowdown for the class times that rate. Every candidate
carries witnesses to its parameters, which the evaluator checks; nothing here meters a chain of blocks.

It is autoresearch-shaped: a fixed side (`prepare.py`, `evaluate.py`, `candidate.sh`, `sandbox.py`, `launch.py`,
`pin.py` and the harness `te_core/`, with `data.py`, `starts.py`, `starts_store.py` and `model.py` behind them) and one
island per class (or per direction, `islands/<direction>/`), each with one editable `train.py` and its program,
`README.md`, for a researcher agent.

**Version**: v2.3 (`prepare.VERSION`, in every witness record, schema `covert-witness/witness/v4`; the harness is te_core
3.6; data version `sealed.DATA` v2.2). Since v2, every candidate runs inside te_core's harness, which counts its MACs and
times its steps, and every run is ingested into te_core's ledger, which prices ρ. Since v2.1, every search score is on a
fresh slice of the search pool drawn after the weights are hashed, promotions are decided on a sealed set one bit each,
and the final set and a final start are sealed until the scorer's final scoring. Since v2.2, one pinned commit
(`pinned.json`) measures every candidate, taking only its island from the candidate's commit; a candidate is a training
job, sandboxed so that no held-out text is on any path it can open, that writes only its E files and step log, and a
separate evaluation job that rebuilds and scores the model; the held-out sets leave out documents that are training
documents. v2.3, the night's last pin, adds te_core 2.7-3.5 to it: the counter's routes and product forms, the guard over
other runs' outputs and the C++ and Rust loaders, the screen over code it never reads (runpy, exec/compile, marshal,
pickle, modules written during the run), ρ by thresholds, a declared step's rebuild-cost floor, the `sparse`, `lora2`
and `quant` formats with every E's rebuild cost, and the meter's tied-embedding adapters.

## Classes

Caps on the witnessed parameters (`prepare.CLASSES`); every class also has the budget below.

| class | cap | seed (`islands/<class>/train.py`) |
|---|---|---|
| `reference` | none | Muon on the blocks' matrices, AdamW on the embedding and norms; E every tensor |
| `e1mb` | E ≤ 1,000,000 bytes | LoRA on every block matrix and the norm gains, the largest rank that fits (bf16) |
| `e100kb` | E ≤ 100,000 bytes | LoRA on each block's o and down projections and the norm gains |
| `shallow` | only the top 2 blocks and the final norm differ | AdamW on those; the tied head stays frozen (it is layer 0 too) |
| `seed` | E a seed log, ≤ 100,000 bytes | MeZO: two forwards a step, the step's coefficient stored, z replayed from seeds |

**Budget**: 2x the MACs the harness counts ≤ 9e16 x 1.1 (`TRAIN_FLOP_BUDGET`, about 10 GPU-minutes of m's dense AdamW at
its measured 1.49e14 FLOP/s, times `HARNESS_FLOP_SLACK` for the elementwise and optimizer work the analytic meter leaves
out) and train.py's steps' wall seconds under the harness ≤ 3000 (`GPU_SECONDS_CAP`; the harness runs eager and counted,
about 2.4x a compiled run). A class's score at a start is its lowest valid search bpb.

## The fixed side

- **Held-out sets** (`data.py`, data version v2.2, each with every document's tokens and bytes): the search pool,
  fineweb-edu shard 013's first whole documents up to 2^23 GPT-2 tokens (8,282 documents, 38,780,980 bytes), in
  `CW_ROOT/heldout/v2.2/search-pool`; the promotion set, shard 011's by the same rule up to 2^21 tokens (2,155
  documents), and the final set, shard 010's up to 2^21 (2,213 documents) in GPT-2's and Pythia's tokens, both sealed.
  A document whose text is one of the training shards' documents (sha256 of the text, 2,163,660 documents in shards
  000-002) is left out: 157, 41 and 39 of them, each named by row and hash in its set's manifest (`dedup`). Each set's
  tokens are pinned (`prepare.SET_SHA256`), so every node scores the same text. Bits per byte divides by the documents'
  UTF-8 bytes, so scores compare across tokenizers. Training streams: learning_frontier's GPT-2 shards 000-002
  (`train.bin`; its `val.bin`, the first 2M tokens of v2.0's search set, is not in the training job's view), and shard
  000 in Pythia's tokens.
- **Search scores** (`prepare.Scorer`): every search evaluation, the evaluation job's, hashes the weights and then
  draws a slice of the pool (`prepare.draw_slice`: a coin of 16 bytes from the OS, and 1,024 of the pool's 8,192
  evaluation windows, 2^20 tokens, drawn by a generator seeded with sha256(coin ‖ digest)). The score is the start's bpb
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
  reads another run's directory.
- **Evaluator** (`evaluate.py`, `prepare.evaluate_bpb`): bits per byte at context 1024 of decode(θ, E). The model must
  have θ's tensors, shapes and dtypes, so its parameter count, evaluation FLOPs and weight bytes read per byte (every
  parameter's bytes once per window, a lookup table in full) are the start's: no lookup table rides free.
- **Training job** (`candidate.sh`, `sandbox.py`, the harness `te_core/` and its README): the island's program runs as
  `python -m te_core.run --script islands/<island>/train.py`, which counts every multiply-accumulate it computes through
  torch, times its steps (a step is one `TrainData.next` to the next), writes the first E in each 5% of the run to its
  output (a checkpoint, scored later) and refuses programs that route around the dispatcher. It runs in sandbox.py: new
  mount, PID, network, IPC and UTS namespaces, as the run's user with no privileges, seeing the pinned commit's
  `benchmarks/covert_witness/` with only its own island under `islands/`, the starts, `train.bin` (and the Pythia
  stream), the Python and its leased GPU, all read-only, and one output directory; no held-out set, sealed side,
  other run or home directory is mounted, and there is no network. An audit hook also refuses `prepare.UNREADABLE` (the search pool,
  the sealed side, v2.0's `eval/`, and learning_frontier's `val.bin` and hub cache) and other runs' outputs.
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
  island is recorded (`island_recorded`); the class's caps. A failed check makes the ledger's records invalid. A step the island declares (`prepare.declare_step`) is a
  counterexample family the harness checks by rebuilding sampled steps with `prepare.decode`, bit for bit, within c MACs
  (te_core's README). The analytic meter (`prepare.train_flops`, from the step log) is recorded beside the harness's
  count. Decoders are fixed (`prepare.decode`: `full`, `lora`, `lora2`, `seedlog`, `sparse`, `quant`) and read only θ and
  E; the witness records E's rebuild cost in MACs (`prepare.rebuild_macs`), and each curve point's (the harness's
  checkpoints).
- **Pinned version** (`pin.py`, `pinned.json` on `cursor/covert-witness-4207`): the one commit whose fixed side measures
  every candidate. launch.py ships it (a clean worktree of it under `~/.research/cw-pinned/`) whatever the candidate's
  checkout holds, and the witness and ledger records name it (`measurement_commit`); te_core's frontier leaves out a
  record of any other commit (`other-version`) and one whose E an earlier record at the pin, or another agent's at any
  commit, submitted (`duplicate-e`).
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

**Run**: `python benchmarks/covert_witness/launch.py CLASS START --note TEXT [--island NAME] [--program FILE] --
TRAIN_ARGS` from a checkout whose island is committed: the island is `islands/<direction>/` when HEAD has its FILE
(default `train.py`), else `islands/CLASS/`, and nothing else of the checkout is used.
`starts.py --size s|m` built the starts once (one GPU); `prep.sh` prepares a node (`data.py`,
`prepare.register_pythia()`, the starts from the store); `evaluate.py --start NAME --out F` scores a start on the whole
search pool. `CW_DEVICE=cpu CW_LEASE=none` runs candidate.sh on a CPU for a smoke test.

**Reads and writes**: the starts are in `/workspace/research/te/starts` and the sealed side in
`/workspace/research/te/sealed` (above); everything else lives under `/workspace/research/scratch/covert-witness`
(`CW_ROOT`, kept under @training): `heldout/<data version>/` (the search pool), `train/`, `ckpt/`, and `candidates/`
(each candidate's island, training job output and sandbox root, removed when it ends).
A candidate's run `out/` holds `run/run.json` (the harness's record as the evaluation job scored it, schema
`te-core/run/v1`, with `evaluation`), `witness.json` (schema `covert-witness/witness/v4`), `tree.json` (the process
tree), `island.json` and `island.tar`, `timing.json`, `meta.json`, `steps.jsonl`, `E.bin` when at most 1 MB, `train.py`
and its logs. **Backs**: no Lean guarantee or claim; its numbers are
measurements of training under structural caps.
