[2026-11-13T07:03Z] 5fbc9b92ac8e
GROWN, Thu 12 Nov, 11:03 PM PST.

**714f is held to `m-t20-2x-hybrid-grow-depth-18`'s config: "equal, 1 expected difference", with no `record_only` line, where 18b2 needs the one it has. The `-b` copy is gone from the branch, and the held entry and `-18` are equal as before.**

RAN, r20261113-064950-da37 (vy-nebius-cpu-11, CPU, exit 0; "rc held=0 714f=0"). It ran the branch at 4da2178230; the next commit only deletes the `-b` copy. The command is `of_record` itself:
- once on the tree's own `of_record.toml`, for the held entry and `-18`;
- once with `--held`, a temporary data file whose one entry is `[held."m-t20-2x-hybrid-grow-depth-18"]`, naming r20261103-063831-714f (art `24056ffd…`, run.json sha256 `88d66c68…`), with the same `partial` and `step` as the tree's entry and no `record_only`.

No change to `of_record.py`. Outputs: `out/of_record.txt` and `out/of_record-714f.txt` of the run.

| entry, run | verdict | updated | log | named |
|---|---|---|---|---|
| `m-t20-2x-hybrid-grow-depth`, d64c (held, unchanged) | equal, 1 expected difference, 1 named line | 444 of 444 | 47 of 49 | `fwd/theta mul [[]] x4, 4 MACs` |
| `-18`, 18b2 | equal, 1 expected difference, 1 named line | 660 of 660 | 47 of 49 | `fwd/theta mul [[]] x4, 4 MACs` |
| `-18`, 714f (`--held`) | equal, 1 expected difference | 660 of 660 | 48 of 49 | none |

**The lines that differ between 714f's verdict and 18b2's:**
- `log`: 48 of the record's 49 products for 714f, 47 for 18b2.
- 18b2's `named` line: "in the record's step and not the config's: fwd/theta mul [[]] x4, 4 MACs; of_record.toml names it: the program's running total of the loss…".
- Everything else is the same: `start` m-t20 (3669435ebb3d), `steps` 10217, step 3 against the modal structure 10190 of 10217 steps have, `updated` 660 of 660, `E` "608,038,092 B after 3 steps; the record's model, 608,040,849 B after 10217", and the one expected difference, `norm` (148 multiplies by an updated gain, 148 normalisations, 1332 products of the backward's).

**Why** (READ, the two records' `structures`): 18b2's program adds each forward's loss times its share into a running tensor, so its modal step has `fwd/theta mul [[]]` x8 where the config's has x4. 714f's older program reads the loss into Python, so its record has the config's x4 and nothing to name.

The body will list 714f's point as the same job as `-18`.

[2026-11-13T07:03Z] eb79f0394cd0
GROWN, Thu 12 Nov, 11:03 PM PST.

**Part 2 design: `grow` gains four dials, all LATER, plus an `output`, and imports `subset`'s decay share and `ema`'s average. A prototype of this, run locally, holds both b248 entries, and the held entry stays equal with the `output` added. I edit `grow.py` on the branch only after your yes.**

## What `grow` imports

**From `subset`: `decay`, with its name and meaning.** It is the last share of the steps, where the rate falls linearly to 0; 1 means from the first step.
- It moves out of `FIXED` into `DIALS`, LATER at 1.0.
- The schedule is still `subset.schedule`, reached through `subset.step` as now.
- I add no `schedule` dial. All six runs are `linear` (WSD is `linear` with `decay`), so no entry would set it. If you want it anyway, it is one line, LATER at `"linear"`.

**From `ema`: its average, by import.** Today `ema` finds its tensors through `subset.trained`, the `trainable` pattern, which `grow` has no dial for. So `ema` gets two helpers, and `averages` takes an optional list of names:
- `track(model, names)` registers the buffers.
- `update(model, dials, t, names)` is today's step body: the copy or `lerp_` by `keeps`.
- `ema.build` and `ema.step` call them with `subset.trained(...)`. `ema`'s computation, VERSION and every `ema` entry's digest stay as they are.
- `grow` names its trained tensors as those with `requires_grad`.

**The dial: `ema_horizon`, LATER at 0.0.** Its nonzero values mean what they mean in `ema`. In `grow`, 0 means no average. In `ema`, 0 falls back to `ema_decay`, which `grow` doesn't take because no run uses it.

**`output`: `model.state_dict() | averages`, the averages only under a horizon.**
- Effect on the held job: run locally, the held entry with an `output` that returns the state dict alone is "1 equal, 0 differ", exit 0.
- What does change for every `grow` job (READ `run.py` 439-482, 742-755): `train` registers `te.output`, so each checkpoint records the trained model's digest beside the output's (the same model when there is no average), and a continued job checks its kept model against that digest.
- The count doesn't change: a state dict computes nothing.

## Warmup

**The program's rule** (READ `train.py` `lr_mult`): the ramp is min(1, (t+1)/(0.02·n)), where n is the program's training steps, the record's steps less its final evaluation. The ramp lengths 0.02·n per run:

| run | 0.02·n |
|---|---|
| b248 | 31.24 |
| 6111 | 49.28 |
| 2137 | 90.98 |
| 7ace | 29.6 |
| ca11 | 75.62 |
| f13e | 113.7 |

**b248's first rates against the nearest integers:**
- The program: 0.0320, 0.0640, 0.0960, …, reaching 1 at t = 31.
- `warmup = 31`: 0.0323, 0.0645, …, reaching 1 at t = 30.
- `warmup = 32`: 0.03125, ….

**No integer gives any of the six**, since each ramp length is not a whole number.

**So, a new dial: `warmup_share`**, a float, LATER at 0.0 (0 means `warmup`'s steps), in an `EITHER` group with `warmup`. The step hands `subset` `warmup = warmup_share`, which `subset` already reads as a share of the steps when it is below 1. `warmup` itself can't take 0.02: it is an int dial, and making its default a float turns the held configs' 200 into 200.0 and changes their digests.

**What remains, a finding** (INFERRED from READ `of_record.py` 488 and the programs): `of_record` makes the config's `budget.steps` equal the record's `len(steps)`, which counts the program's non-training steps too.
- In ref-data, the record's steps are n + 1. So the job's ramp is (t+1)/(0.02·(n+1)), 0.06% (b248) to 0.02% (f13e) below the program's, and the job trains one step more than the run did.
- In the held entry it is larger. The config has 7713 steps against the program's n of 7688: one θ measurement and 24 evaluations are steps in the record. So its job trains 25 steps more than the run did, on a schedule over 7713 steps.
- Neither changes the step `of_record` compares. Both change how much the job trains against the run. Whether to compare training steps instead is yours; I change nothing there.

## Muon's norm

**`stack`, LATER at `true`** (the held entry's form), passed as `Muon(..., stack=dials["stack"])`.
- `_optimizers.Muon` puts `stack` in the group only when it is true. So the held entry's optimizer state is as it is today, and a `stack = false` config's has no key.
- The program's Muon computes the same update (READ `train.py` 60-70 and 110-132): Nesterov 0.95, five Newton-Schulz steps in bf16 with the same coefficients, the scale √max(1, rows/cols), and decoupled decay. Only the divisor is a scalar norm.

## The entries

Names and targets as proposed. Every entry shares:
- route `grow`, start `s-t20`, `windows = "fresh"`, batch 128;
- `adam_lr = 3e-3` (the program's `--norm-lr`), `weight_decay = 0.05`, `warmup_share = 0.02`, `decay = 0.25`, `stack = false`, `micro_batches = 4`;
- `step = {counted = 3}`, with the held entry's why.

| entry | run | depth | steps | lr = embed_lr | form |
|---|---|---|---|---|---|
| `s-t20-1.1x-ref-data-grow-muon-1563` | r20261103-185042-b248 | 9 | 1563 | 0.003 | weights |
| `s-t20-1.1x-ref-data-grow-muon-1563-ema` | r20261103-185042-b248 | 9 | 1563 | 0.003 | average |
| `s-t20-2x-ref-data-grow-muon-2465-ema` | r20261104-001417-6111 | 12 | 2465 | 0.006 | average |
| `s-t20-2x-ref-data-grow-muon-4550-ema` | r20261104-013912-2137 | 12 | 4550 | 0.006 | average |
| `s-t20-1.1x-ref-data-grow-muon-1481` | r20261103-064517-7ace | 12 | 1481 | 0.003 | weights |
| `s-t20-2x-ref-data-grow-muon-3782` | r20261103-192526-ca11 | 15 | 3782 | 0.003 | weights |
| `s-t20-2x-ref-data-grow-muon-3782-ema` | r20261103-192526-ca11 | 15 | 3782 | 0.003 | average |
| `s-t20-2x-ref-data-grow-muon-5686` | r20261103-095914-f13e | 12 | 5686 | 0.003 | weights |

**How each form is held:**
- A weights entry carries one `record_only` line: the program's one `_foreach_lerp_` a step, its average, worded as the fullmuon weights entries word it.
- An `-ema` entry is held through the `average` rule (the program's arguments have `--ema-frac`).

**Prototype, RAN locally** (CPU VM, no run id, a scratch worktree off my branch, nothing pushed):
- `s-t20-1.1x-ref-data-grow-muon-1563`: "equal, 2 expected differences, 1 named line of the record's program alone". The two are `norm` and `loss`, and the named line is `other/param _foreach_lerp_ [[512] x38, [512,512] x72, [512,1408] x18, [1408,512] x36, [32000,512] x2] x1, 45,295,104 MACs`.
- `s-t20-1.1x-ref-data-grow-muon-1563-ema`: "equal, 3 expected differences": `norm`, `loss`, and `average` ("83 of them … the record's 45,295,104, the config's 45,295,104").
- `updated` 336 of 336 and `log` 46 of 51 on both. The program's fused AdamW lists the embedding's group before the gains', where `grow` lists the gains' first; the comparison doesn't see it.

**Your per-form rule changes two things outside `grow`:**
- **The test.** `test_the_weights_and_their_average_are_each_an_entry` requires `NAME` beside every `NAME-ema`, with route `subset`. I would have it:
  - require `NAME` only where it exists, with the `-ema` entry's own base route (`subset` under `ema`, `grow` under `grow`);
  - still check that the average differs from the weights, by running the route without the horizon;
  - so 6111 and 2137 stand alone.
- **The README.** Its lines 70-73 say such a run "is two entries"; they become one entry for each form a checkpoint of the run handed out.

`REFERENCES`: the `grown` reference gains the decay share, the warmup share, the scalar norm (a value both forms of the norm compute alike) and the average (torch's own `AveragedModel` over the horizon, as `averaged` takes it).

## Still to come

The 714f comparison (r20261113-064950-da37, on vy-nebius-cpu-11, running) and the part 3 design follow in their own messages.

