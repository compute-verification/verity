[2026-11-13T06:43Z] daf956fc63b2
GROWN, Thu 12 Nov, 10:50 PM PST.

**Of the nine runs, route `grow` takes the two depth-18 runs as they stand. The six `ref-data/grow-muon` runs need three dials it lacks (Muon's scalar norm, a decay share and a warmup share), plus an average of the weights for their averaged checkpoints. The run grown "in width" grows depth too (8 to 10 blocks) and widens d_ff from 2048 to 3072 with noised copies of rows, which no origin map can express.**

The nine side by side. All rows are READ from each run's `out/candidate.json`, `out/meta.json` and `out/run/run.json` (`script_args`, `steps`, `structures`, `checkpoints`). Every run uses batch 128 in 4 forwards of 32 rows, with 32,768 tokens a forward. The "steps" column is the record's `len(steps)`, which includes the program's non-training steps (hybrid: 1 θ measurement + n + 24 evaluations; ref-data: n + 1 final).

| run | program (sha256[:12]) | start, grown shape | arguments that decide the step | steps (modal) | tier | parameters (last checkpoint) | route `grow` as it stands |
|---|---|---|---|---|---|---|---|
| r20261103-161953-18b2 | hybrid grow.py e39bf543e822 (the held run's) | m-t20 8 to 18 blocks | `--grow depth --depth 18 --zero 1 --lr 0.008 --wd 0.05 --adam-lr 3e-3 --embed-lr 6e-3 --warmup 200 --micro 32 --budget-fraction 0.02 --tier long` | 10,217 (10,190) | long | 152,006,400 | yes: depth 18, steps 10217, the held entry's other dials |
| r20261103-063831-714f | hybrid grow.py 390165eae35c (older) | m-t20 8 to 18 | the same as 18b2, except that `--tier` is absent (the tier came from the environment, long) | 10,217 (10,190) | long | 152,006,400 | yes: the same config as 18b2 |
| r20261103-185042-b248 | ref-data train.py 8e35558c8f97 | s-t20 6 to 9 | `--grow-layers 9 --mat-lr 0.003 --embed-lr 0.003 --norm-lr 3e-3 --wd 0.05 --warmup 0.02 --sched wsd --decay-frac 0.25 --ema-frac 0.1 --windows fresh --budget-fraction 0.85` | 1,563 (1,558) | search | 45,295,104 | no |
| r20261104-001417-6111 | ref-data 8e35558c8f97 | s-t20 6 to 12 | as b248 but `--grow-layers 12 --mat-lr 0.006 --embed-lr 0.006 --budget-fraction 0.013` | 2,465 (2,460) | long | 54,931,968 | no |
| r20261104-013912-2137 | ref-data 8e35558c8f97 | s-t20 6 to 12 | as 6111 but `--budget-fraction 0.024` | 4,550 (4,545) | long | 54,931,968 | no |
| r20261103-064517-7ace | ref-data bffb4d8dda74 (older; differs only in its FLOP-plan helper) | s-t20 6 to 12 | as b248 but `--grow-layers 12 --budget-fraction 0.98`, no snapshots | 1,481 (1,478) | search | 54,931,968 | no |
| r20261103-192526-ca11 | ref-data 8e35558c8f97 | s-t20 6 to 15 | as b248 but `--grow-layers 15 --budget-fraction 0.0235` | 3,782 (3,777) | long | 64,568,832 | no |
| r20261103-095914-f13e | ref-data bffb4d8dda74 | s-t20 6 to 12 | as 7ace but `--budget-fraction 0.030` | 5,686 (5,683) | long | 54,931,968 | no |
| r20261104-060905-8d56 | hybrid grow.py e39bf543e822 | m-t20 8 to 10 blocks, d_ff 2048 to 3072 | `--arms grow=mlp,depth=10,mlp_mult=1.5 --zero 1 --lr 0.0065 --wd 0.05 --adam-lr 3e-3 --embed-lr 4.5e-3 --warmup 200 --budget-fraction 0.011 --tier long` | 7,265 (7,238) | long | 118,963,968 | no |

Where the records differ from your table:

- s-t20 has 6 blocks (`starts.toml`), not 8. The ref-data growths are 6 to 9, 6 to 12 and 6 to 15. READ.
- The "grown in width" run widens and also deepens: `grow=mlp` with `depth=10` copies block ⌊8j/10⌋ and zeroes the repeated copies' `wo` and `w_down` (blocks 1 and 6). It also widens every block's MLP to round(1.5 × 2048) = 3072. Each block draws its own 1024 random source rows for `w_gate` and `w_up`, adds 0.01 × the row's RMS × Gaussian noise from a generator seeded with `--seed` (0), and gives `w_down` 1024 zero columns. READ, `out/train.py` `build`.
- Every parameter count and step count in your table matches the record. READ.

Why each "no":

- **ref-data (six runs).** The program grows depth exactly as `grow` does: block j copies ⌊6j/N⌋, and a copy that repeats the block before it gets `wo` and `w_down` zeroed. With `--grow-d` and `--grow-ff` at 0 it widens nothing. READ, `train.py` `grow`. It then trains every tensor: Muon on the matrices (Nesterov, 0.95, five Newton-Schulz steps, decay 0.05), and fused AdamW on the embedding (at `--embed-lr`) and on the gains (at `--norm-lr`). This is `grow`'s optimizer split. Four things differ:
  1. Muon divides by the matrix's norm as a scalar. The record's modal step has `other/param div [[512,512], []]`, where `grow` fixes `stack=True` (`[1,1]`), so the count differs. READ.
  2. The schedule is WSD: flat, then falling linearly over the last 25% (`--decay-frac 0.25`). That is `subset`'s `decay = 0.25`, and `grow` fixes `decay = 1.0`. READ.
  3. The warmup is 2% of the steps (`--warmup 0.02`). `grow`'s `warmup` is an int number of steps, and 0.02 × 1563 is 31.26. READ.
  4. It keeps an average of every tensor (`--ema-frac 0.1`, one `_foreach_lerp_` a step), and at each checkpoint it hands out whichever of the weights and the average scores lower on held-back windows. READ. The checkpoints:
     - b248: average, average, weights.
     - 6111 and 2137: average ×3.
     - ca11: average, average, weights.
     - 7ace and f13e: weights (one checkpoint each).

  So these runs are entry pairs, NAME and NAME-ema, as the fullmuon runs are. Which checkpoint each frontier point is I don't have (INFERRED: ca11's one point is its last). The loss (row means, summed over the rows kept) is the `loss` rule's case, since there is no `--trunc`. `--windows fresh` maps to the config's `windows = "fresh"`. The full design for these runs, with each dial's default and why it leaves the held entry unchanged, is the part 2 message I send next. I edit nothing in `grow.py` before your yes.
- **8d56 (width).** `origin` maps a tensor only to a θ tensor of the same shape, or to zeros, and `run.start_state` refuses any other shape. The widened `w_gate`/`w_up` (θ's rows plus noised copies) and `w_down` (θ's columns plus zeros) are neither. `evaluate.py` also refuses a d_ff other than the start's. The design goes in the part 3 message.

Two things for you:

- **Names, and the twin.** 714f and 18b2 have the same dials, seed 0 and steps. Only the program revision differs: the older program reads the running loss into Python as a float each forward. So 714f's record should lack the 4 `fwd/theta mul [[]]` lines the held entry names as `record_only` (INFERRED until the comparison runs). Two entries with one config text share one digest, so they are the same job. Unless you say otherwise, I write `m-t20-2x-hybrid-grow-depth-18` (18b2, 4 points) and `m-t20-2x-hybrid-grow-depth-18-b` (714f, 1 point), identical but for the name, and I say so in the body.
- **Suffixes and targets.** For the ref-data entries I follow the README's suffix-by-steps naming, with the target set by the tier (search 1.1x, long 2x, as the held entries have). For example: `s-t20-1.1x-ref-data-grow-muon-1563` and its `-ema` twin.

Next: the two depth-18 configs and their `of_record` runs on a CPU node tonight, then the part 2 and part 3 design messages.

