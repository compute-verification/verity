# Island `e1mb`: lower search bpb within the class's caps

This is the island's program for a researcher agent (autoresearch's program.md; the tree names it README.md).

**Class caps**: 8·|E| ≤ 8,000,000 bits (E.bin at most 1,000,000 bytes), any of the fixed formats. The fixed training-FLOP budget and GPU-second cap hold in every class.

**Seed** (`train.py` as committed): LoRA on every block matrix plus the norm gains, the largest rank that fits in bf16 (r = 4 on m), E the factors (`encode_lora`).

Anything that fits the bytes is fair: which tensors, rank per tensor, bf16 or f16 factors, norms or not, a sparse set of full tensors through `encode_full`, a seed log.

## Your loop

1. Pick a start (`s50`, `s100`, `m50`, `m100`; m100 is the main one) and a hypothesis.
2. Edit `islands/e1mb/train.py` only. Commit it (the run ships HEAD and its record names the commit).
3. Run the candidate: `python benchmarks/covert_witness/launch.py e1mb START --note "HYPOTHESIS" -- TRAIN_ARGS`. It runs on
   vy-nebius-2 (one GPU, the `gpu` slot pool), waits for it, puts the run's `out/` in the evidence store
   (`research data put --kind evidence/v1 --tree ... --preserve`) and prints one JSON row (also appended to
   `~/.research/covert-witness.jsonl`): `valid`, `bpb`, `E_bytes`, `phi`, `lowest_updated_layer`, `train_flops`
   (the harness's), `gpu_seconds`, the checks, the run, the art id, and the te_core ledger's ids and ρ at the submission. `--no-wait` returns at once; `--collect RUN` collects later.
4. Keep or discard: keep the commit when the row is `valid` and its `bpb` is below the best kept for that start; otherwise
   revert it. Every candidate is logged either way: its record is in the store, and the ledger keeps the row.
5. Report each kept result as the run id, its art id and the witness record's `score`, `E`, `structure` and `train`.

Keep at most two candidates of this island in flight, so the islands share the node.

## The score and the budget

- Score: `score.bpb`, bits per byte of the search set (shard 013's first 2,092 documents, 9.8 MB of text) at context 1024,
  computed by `evaluate.py` on decode(θ, E). Lower is better. Compare only within a start.
- Budget: train.py runs inside the harness (`te_core.run --script`, `te_core/README.md`), which counts every
  multiply-accumulate it computes through torch and times its steps. 2x its MACs ≤ `prepare.TRAIN_FLOP_BUDGET` x
  `prepare.HARNESS_FLOP_SLACK` (9e16 x 1.1: the count includes the elementwise and optimizer work `prepare.train_flops`
  leaves out, 1.06-1.09x of it on the reference seed), and its steps' wall seconds ≤ `prepare.GPU_SECONDS_CAP` (3000: the
  harness runs eager and counted, about 2.4x a compiled run, and stops a run at 1.2x the cap). A step is the work from one
  `prepare.TrainData.next` call to the next. Your step log is still priced (the witness record's `train`) but no longer
  decides the budget.
- Curve: the harness scores the first E you encode in each 5% of the run, and the one you submit, as it runs; the witness
  record's `curve` is those checkpoints at the harness's FLOPs.
- ρ: the launcher ingests the run into the te_core ledger, one submission per checkpoint with ρ from the theorem at each
  (k, r); the scorer's W\*(l\*) is the least |C|·ρ at or below a target. A step whose weights you declare as encodings
  from θ (`prepare.declare_step(base, suffixes)`, its docstring) is credited as a counterexample when `prepare.decode`
  rebuilds them bit for bit within c = 4096 x the start's forward MACs per token, which lowers ρ to about 1 for a seed
  log or any small fixed-format update; the seed island's train.py declares every step.

## The witnesses, which evaluate.py checks

θ's digest; E and its format, decode(θ, E) bit-equal to the model you submit, with θ's tensors, shapes and dtypes (so the
model's size and its evaluation FLOPs and weight bytes per byte are the start's); 8·|E| bits; the declared `changed` set,
which must cover every tensor that differs, from which φ (the share of forward weight-matmul FLOPs whose weight differs and
has an updated node upstream) and the lowest updated layer are computed; `trainable`, which must cover `changed`; the
harness's record, which must be this E's and find nothing invalid (tokens computed outside a step, weights written where
the counter can't see, a failed script), and its FLOPs and seconds; the class's caps. `valid` is every check passing.

## Rules

- Never edit `prepare.py`, `evaluate.py`, `candidate.sh`, `launch.py`, `te_core/`, `data.py`, `starts.py`, `model.py`,
  `sealed.py` or `promote.py`, and never run `evaluate.py` yourself. Never read `prepare.UNREADABLE` (the search pool and
  the sealed side), from train.py or anything it starts; the harness refuses it in train.py's process. Each score you see
  is on a fresh slice of the search pool drawn after your weights are hashed; promotions are decided on a sealed set, and
  the final set is never scored for the search.
- Train only on `prepare.TrainData` (the start's tokenizer's training stream) or data you generate from θ.
- train.py may not name the harness's internals or a route around torch's dispatcher (`te_core/run.py`'s `FORBIDDEN`:
  Triton, CUDA graphs, custom extensions, subprocesses and the like); the harness refuses such a file.
- Report any loophole in the evaluator, the checker or the harness instead of exploiting it: work the harness doesn't
  count, a way to make E smaller than the update it carries, a model that differs from decode(θ, E) and passes, a way to
  the held-out sets. A decoder change is an evaluator change (ask, don't add one).
