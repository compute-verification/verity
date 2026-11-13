[2026-11-13T02:03Z] 96a8383a627a
TOPCODE, Thu 6:03 PM PST

**Your guard is in (5a2f22555d) and the 81M entry is equal under it. But the record's program trained 5,387 steps, not 5,389: two of the record's 5,389 steps are forward-only cut probes. I need your ruling on the config's steps.**

The guard. Commit 5a2f22555d replaces 45c881f613's line. G is the config's excess of `bwd/grad mul [x, g]`, and each held backward key's excess, less `updated[0]` for `mul [x, x]`, must be a whole multiple of G, or 0 where G is 0. The docstring states it in your sentence.
- `norms()` now holds autograd's backward: `div [x]`, `mul [x, x]` ×2 plus one per updated gain, `mul [x, s]`, `mul [x]`, `mul [s, s]`, `mul [s]`, and `mul [x, g]` for each gain multiply whose normalisation passes a gradient. It takes `still`, the normalisations on an input that needs no gradient.
- New test: with `still` = 1 and with `still` = 5, beside plain normalisations, the rule fires; with G = 0 and a stray `div [x]`, it is refused. The detached case is still refused. RAN, `tests/test_of_record.py` on my VM: 20 pass.
- Mutations (RAN): with the old guard, the plain-normalisation test and the new test fail; with no guard, the detached test and the new test fail.
- Your lines against the counted record: they hold (RAN r20261113-014310-d364, counted step 3 and counted step 4 identical in all 17 lines). G = 108; the five single products are each 108; `mul [x, x]` is 327 − 111 = 216 = 2 × 108; the gain's gradient is a `mul [x, x]`.

The entry under it. RAN r20261113-015658-21de at 5a2f22555d: "equal, 1 expected difference, 7 named lines of the record's program alone". The expected difference is the norm: 111 multiplies by an updated gain, 111 normalisations, 975 products of the backward's, beside 3 plain normalisations on both sides. The 7 named lines are R's construction.

The other entries. RAN, a script over all 34 held entries' records on my VM, reading each record's modal step as `of_record` does (`modal`, `as_counted`, `per_row`): only `m-t20-2x-wildcard-topcode-coded` holds a backward product over the norm's shapes alone ([1, ctx, d], [1, ctx, 1], [d]). The other 33 hold none, so `held` has no backward key, the guard's `rest` is empty, and their verdicts can't move. So f3a4 (GROW's head) and 3b8e (dea3ccc1a1, before the new guard) stand for those 33. My entry's verdict at the new head is 21de's.

**The steps (needs your ruling).** READ, the record's `train.py`:
- `n_steps = int((flop_budget(a) - reserve) // per_step)`, and `for step in range(n_steps)`, with the WSD schedule's `frac` over `n_steps`.
- `cut_probes("theta")` before the loop and `cut_probes("end")` after it, each under `step_of()`.
- `meta.json` and the first line of `train.log` say `"steps": 5387`.

RAN, the record's `run.json`: steps 1 and 5,389 have no backward (15 and 20 forward log rows, 262,144 tokens each: the probe windows). The 5,387 between them are training steps. So the program trained 5,387 steps with its schedule over 5,387. The config's 5,389 trains two steps more and stretches the schedule over 5,389. The uncoded controls' records have no forward-only step (`m-t20-1.1x-wildcard-topcode` 2,385 and `s-t20-1.1x-wildcard-topcode` 1,481, equal to their configs).

The program wins, so I'd set the config's steps to 5,387. But `of_record`'s `steps` fact compares `len(rec["steps"])`, and would then print "the steps: the config's 5387, the record's 5389". Two ways out:
- (a) `of_record` counts only the record's steps that have a backward. This is a rule in code, its own commit with its test, and it moves no other entry, since none has a forward-only step.
- (b) The entry states the probe steps in `of_record.toml`, as `step` does.

I lean to (a). Until you rule, the config stays at 5,389.

Condition 9, in training steps. RAN r20261113-014310-d364: the config's steps have structures [A, B, C, C], so steps 1 and 2 differ from the later ones and step 3 is the first with the later structure. The record's structures run probe, D, E, M, M, … M, probe, with M modal (5,385 steps). Its first modal step is harness step 4, which is training step 3, because harness step 1 is the theta probe. So the same training step. The record's training step 2 has 51 fused norms with updated gains (the start's 8 blocks × 2 + the final norm, in 3 forwards); its step 3 has 111. That fits `step.why`.

Condition 11 (81M). RAN, the same run's counted record: the trace at batch 192 and context 1,024 has `a_bits` 3,088.0, beside the program's `a_fwd_bits` 3,088 (READ, `train.log`'s `cuts`), with `refused` None and `data_dependent` and `carried_scalars` both [].

Point 7 (238M), commit e52ae72e5e, on the trace RAN r20261113-014549-a81c at 36 blocks:
- 337,075,475,841,024 MACs a step, the same at 32 and at 64 rows a forward: grouped by everything but their token dims, the two splits' products are equal, so only the rows a forward change.
- `a_bits` 4,112 (= 1,024 × 4 + 16), and no `refused`, `data_dependent` or `carried_scalars`.
- Steps ⌊1.01e22 / (3.3708e14 × 1,480.7)⌋ = 20,236. The plan's `cap_gates` is 1.0084763e22 (READ, `plan.json` of r20261106-183414-1416, run 16), which gives 20,205. I used your 1.01e22; say if you want the plan's exact cap.
- Batch 192: run 16 states none (READ, the same `plan.json` and `PLAN.md`'s row 16).
- Split, INFERRED from the trace's product sizes at 32 rows a forward: activations ≈ 32.0 GiB (q, k, v and o 9.0; gate and up 12.4; down 2.25; attention outputs 2.25; the code's rotations 0.25; logits 1.95 bf16 plus 3.9 fp32). The CPU trace's 36 fp32 score products, 2 GiB each, are the math kernel's; the GPU's flash kernel keeps none. Add 5.3 GiB of fp32 weights, gradients and Muon momentum: 37.3 GiB of the card's 96 GiB (READ, `rtx-pro-6000-server.toml`'s `dram`). At 64 rows a forward it is 69.3 GiB before the elementwise intermediates the product rows don't show. So six forwards of 32.
- minutes 3060, seeds `{ search = [0] }`, and `honest = "l16-t20-2x-fullmuon-fe-wsd-40326"`.
- Dry runs in flight: r20261113-020146-1c6c (both configs, and the 238M one again with its honest line dropped in the run's clone).

Also on the branch: abedd475dc (the code rounds each token as one group, the record's `[tokens, 1, d]` view), ef6125b0b3 (the README's routes and codecs rows; "the trace reads the forward's plane"; no width for the backward), and dea3ccc1a1 (the 7 `record_only` lines).

Next: open the draft PR tonight, and fill in the every-entry verdicts when f3a4 and 3b8e finish.

