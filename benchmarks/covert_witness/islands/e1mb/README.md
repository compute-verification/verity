# Island `e1mb`: lower search bpb within the class's caps

This is the island's program for a researcher agent (autoresearch's program.md; the tree names it README.md).

**Class caps**: 8·|E| ≤ 8,000,000 bits (E.bin at most 1,000,000 bytes), any of the fixed formats. The fixed training-FLOP budget and GPU-second cap hold in every class.

**Seed** (`train.py` as committed): LoRA on every block matrix plus the norm gains, the largest rank that fits in bf16 (r = 4 on m), E the factors (`encode_lora`).

Anything that fits the bytes is fair: which tensors, rank per tensor, bf16 or f16 factors, norms or not, a sparse set of full tensors through `encode_full`, a seed log.

## Your loop

1. Pick a start (`s50`, `s100`, `m50`, `m100`, `smollm2`; m100 is the main one) and a hypothesis.
2. Edit `islands/e1mb/train.py` only. Commit it (the run ships HEAD and its record names the commit).
3. Run the candidate: `python benchmarks/covert_witness/launch.py e1mb START --note "HYPOTHESIS" -- TRAIN_ARGS`. It runs on
   vy-nebius-2 (one GPU, the `gpu` slot pool), waits for it, puts the run's `out/` in the evidence store
   (`research data put --kind evidence/v1 --tree ... --preserve`) and prints one JSON row (also appended to
   `~/.research/covert-witness.jsonl`): `valid`, `bpb`, `E_bytes`, `phi`, `lowest_updated_layer`, `train_flops`,
   `gpu_seconds`, the checks, the run and the art id. `--no-wait` returns at once; `--collect RUN` collects later.
4. Keep or discard: keep the commit when the row is `valid` and its `bpb` is below the best kept for that start; otherwise
   revert it. Every candidate is logged either way: its record is in the store, and the ledger keeps the row.
5. Report each kept result as the run id, its art id and the witness record's `score`, `E`, `structure` and `train`.

Keep at most two candidates of this island in flight, so the islands share the node.

## The score and the budget

- Score: `score.bpb`, bits per byte of the search set (shard 013's first 2,092 documents, 9.8 MB of text) at context 1024,
  computed by `evaluate.py` on decode(θ, E). Lower is better. Compare only within a start.
- Budget: analytic training FLOPs ≤ `prepare.TRAIN_FLOP_BUDGET` (9e16, about 10 GPU-minutes of m's dense AdamW) and
  train.py's GPU-seconds ≤ `prepare.GPU_SECONDS_CAP` (1200, compilation included). The meter prices what your step log
  says: one `{"op": "train"|"forward", "tokens": T}` per pass over data, priced from `trainable`, and
  `{"op": "extra", "flops": F}` for matmul FLOPs outside passes (an optimizer's). Log every pass.
- Curve: the witness record's `curve` scores the snapshots you submit (E at a step-log prefix) for FLOPs-to-target.

## The witnesses, which evaluate.py checks

θ's digest; E and its format, decode(θ, E) bit-equal to the model you submit, with θ's tensors, shapes and dtypes (so the
model's size and its evaluation FLOPs and weight bytes per byte are the start's); 8·|E| bits; the declared `changed` set,
which must cover every tensor that differs, from which φ (the share of forward weight-matmul FLOPs whose weight differs and
has an updated node upstream) and the lowest updated layer are computed; training FLOPs from the step log and `trainable`,
which must cover `changed`; GPU-seconds; the class's caps. `valid` is every check passing.

## Rules

- Never edit `prepare.py`, `evaluate.py`, `guard.py`, `candidate.sh`, `launch.py`, `data.py`, `starts.py` or `model.py`, and
  never run `evaluate.py` yourself. Never read `prepare.HELD_OUT` (the search and final sets), from train.py or anything it
  starts; `guard.py` refuses it in train.py's process, and the final set is never scored for the search.
- Train only on `prepare.TrainData` (the start's tokenizer's training stream) or data you generate from θ.
- Report any loophole in the evaluator, the checker, the meter or the guard instead of exploiting it: a pass the meter
  doesn't price, a way to make E smaller than the update it carries, a model that differs from decode(θ, E) and passes, a
  way to the held-out sets. Known ones: a subprocess runs without the guard; the step log is self-reported (GPU-seconds is
  the cross-check); a decoder change is an evaluator change (ask, don't add one).
