# Island mb10: class 10MB (k = 10^7 bytes (8e7 bits))

You are a researcher agent searching training methods in one structural class: a LoRA-style update sized to fit 10 MB per block. Lower the entry's SEARCH
bits per byte at a fixed charged budget, by editing `train.py` in this directory and nothing else.

## The setup (fixed: never edit `prepare.py`, `meter.py`, `job.sh`, `launch.py` or another island)

- **Starts** (registered theta, `python benchmarks/covert_search/prepare.py show` on the node, or the witness's `start`):
  `s50`, `s100`, `m50`, `m100` (our dense s and m, 19M and 57M non-embedding parameters, at 50% and 100% of their first-curves
  runs, retrained with the same configuration) and `smol` (SmolLM2-135M, its own tokenizer, the same text). Every start
  continues on the fineweb-edu training stream after the sequences its own run read; you can't choose the data.
- **A block** is one call of `block(ctx, carry) -> carry`; block 0 gets `carry=None`. After the budget, one more block calls
  `export(ctx, carry)`, which returns the entry: `{parameter name: tensor}` overrides on theta (the start's architecture,
  `ctx.params()`'s names and shapes).
- **c** (per start, `start.json` `c_flops`) is the FLOPs of one forward and backward over 64 sequences of 1024 tokens on the
  start, as torch's FlopCounterMode counts them (matrix products and attention). A block may do at most c; charged compute
  is blocks x c whether you use it or not.
- **k** = 10^7 bytes (8e7 bits): the most a block may import beyond theta. Counted: the carry it receives (canonical serialization: per
  tensor its name, 3 bytes, 8 per dimension, and the raw values), every sequence it takes (`ctx.sequences(n)`, 1025 tokens
  x 16 bits each), and its 64-bit seed. `ctx.bits_of(carry)` and `ctx.sequence_bits(n)` compute it; `ctx.k_bits` is k.
- **Budget**: 1000 blocks plus the export block, 1001 c, at every start (about 6-10 GPU-minutes at m). Fixed; don't
  pass `--budget-blocks` to compare candidates.
- **Score**: the evaluator (`prepare.py evaluate`, its own process) loads the exported entry onto the start, refuses it if
  its forward costs more FLOPs per 1024-token window than the start's (the eval-compute cap per byte), and scores every
  full window of the SEARCH split at context 1024 in bits per byte. The witness's trajectory has it at blocks 0, 250, 500,
  750 and the end. Lower is better. Compare candidates at the same start and seed.

## What `ctx` gives a block

`ctx.model` (theta, frozen; restored from a pristine copy before every block), `ctx.params()`, `ctx.sequences(n)` (the next
n training sequences, `[n, 1025]` int64 on the GPU), `ctx.logits(x, overrides=None)` and `ctx.loss(seqs, overrides=None)`
(the start's forward under bf16 autocast, with parameter overrides through `torch.func.functional_call`), `ctx.seed` (this
block's 64-bit seed; torch's global RNG is seeded from it too), `ctx.block`, `ctx.blocks`, `ctx.k_bits`, `ctx.c_flops`,
`ctx.device`, `ctx.hp` (train.py's `HP` with any `--hp` overrides).

## The seed

The seed trains rank-r factor pairs on the hidden matrices with AdamW and merges them at export; `plan` picks the largest rank whose carry (factors and both AdamW moments, fp32) and 64 sequences fit under k. Directions: what the carry holds and at what precision (bf16 or 8-bit moments buy rank), which matrices, more sequences per block (the frozen weights take no weight gradient, so 64 sequences use about 2/3 of c), the rate, re-drawing or merging factors (a merged factor is still carry: the update since theta crosses every boundary), other parameterizations of a small update.

## The loop

1. Edit `train.py`, commit it (the launcher runs the commit at HEAD; a dirty tree is refused).
2. Run it: `python benchmarks/covert_search/launch.py mb10 --start m50 [--seed 0] [--hp KEY=VALUE ...] --note "what changed"`.
   It runs one candidate on vy-nebius-2 (`research run --on vy-nebius-2 --pool research`, one GPU from gpu-lease), puts the run's `out/`
   (witness.json, blocks.jsonl) in the evidence store (`research data put --kind evidence/v1 --preserve`), appends a row
   to `results.tsv` here (untracked: start and final bits per byte, charged FLOPs, GPU-seconds, run id, art id) and prints
   it. A refusal comes back as status `refused: <reason>`.
3. Keep or discard: if the final SEARCH bits per byte beat this island's best at that start, keep the commit; otherwise
   `git reset --hard` to the best. Log every candidate, kept, discarded, refused or crashed: the launcher's row is the
   log; add a `--note`.
4. Repeat. One candidate at a time per island unless you were told how many GPUs are yours.

## Rules

- Everything a block uses beyond theta arrives in its carry or through `ctx.sequences`. No files, no network, no state
  outside the carry: no module-level variables that change, no attributes left on the model or on imported modules, no
  mutable default arguments, closures or caches that survive a block. The meter checks some of this (train.py's
  globals, the model's modules, theta's values, gradients) and refuses what it sees; what it can't see is still a
  violation.
- All compute goes through PyTorch operators the counter sees. No torch.compile (the meter disables it), no custom CUDA,
  Triton or other kernels, no work moved into operators the counter doesn't count (elementwise ops, `torch.cumsum`, sorting)
  to do what a matrix product would.
- Training code never reads held-out data: not the SEARCH or FINAL files, not the evaluator's outputs beyond the score
  the launcher reports. Never tune on FINAL; you can't run it.
- Never edit `prepare.py`, the meter, `job.sh` or `launch.py`, and never pass `--budget-blocks`, `--eval-windows` or a
  different class to compare candidates.
- **Report any loophole** you find in the evaluator or the meter (something it allows that the rules forbid, or that
  would let a method look better than it is) instead of exploiting it: a row with status `loophole` and a `--note`
  saying what and how, and tell the integrator. Known ones: elementwise work is uncounted (its GPU-seconds are
  recorded); the meter can't see every Python channel; SmolLM2 was pretrained on a fineweb-edu mix, so `smol`'s SEARCH
  text may not be held out from it.
