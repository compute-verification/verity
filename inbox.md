[2026-11-16T19:48Z] b88deac01b4a
From STEP to the training lead, Mon 16 Nov 11:48 AM PST, SIZING: the library's 64 entries under the Glossary's slowdown. Read on origin/main 5f724969058.

Answer: no subset is scorable now without a design decision. The tree holds no record, so nothing is scorable on CPU from the tree alone. From stored records, 24 entries have everything except a defined T_ref. The question is at the end.

1. The quantity [read]
- The Glossary's slowdown(𝒢,𝒜,η) = T_𝒢(η;𝒜)/T_𝒢(η;𝓕) belongs to a task and a set of computations, not to one algorithm.
- For one entry e, the Glossary-terms figure is W(e) = cost of an admissible run of e that passes the task, over T_ref.
  - Numerator: e's run to target, executed inside the policy's blocks.
  - Denominator: T_ref, the reference's cost to the same target with no protocol. It bounds T(𝓕) from above.
  - W(e) bounds the slowdown from above only if e's admissible execution is exhibited and T_ref = T(𝓕). The lower bound T_min/T_ref is the task's, not e's.
- The task here: held-out bits per byte at or below `grade.target` = E + (l0−E)·g^−γ, on `grade.fit` through the registered starts (`starts.toml`). Here g = 1.1 for 61 entries and 2 for 3. Success is the first scored checkpoint at or below it (`grade.seed_at`); η is over seeds.
- A record needs: cumulative cost at each checkpoint, the val_bpb there, and the start's l0. The runs of record have all of these: MACs, and gates by `grade.checkpoints`. [ran]
- Where the Glossary and the code part:
  - Unit: the Glossary costs GPU-hours by whole servers; the code counts Boolean gates.
  - The task is not static: the fit has 4 points and is provisional. On today's fit, s-t20-…-758's run no longer reaches gain 1.1: it reaches 1.0949. [ran]
  - 𝒜 is modelled only as "fits the console policy's (c,k) blocks".
  - No reference is named: 58 of 64 configs name no `honest` config.

2. Table [ran: library run r20261115-173329-4bc4, 40 artifacts fetched and sha256-checked, `grade.checkpoints`/`seed_at` locally; reach is on today's fit, search split, seed 0, one seed only]
| entries | today | class |
|---|---|---|
| 24: s-t20 ref-optim-{747,961,1027}, ref-optim-fullmuon-841, fullmuon(-ema), fullmuon-fe-wsd(-ema), wildcard-topcode; m-t20 fullmuon-wsd(-4611)(-ema), ref-optim-1372, ref-optim-fullmuon, wildcard-topcode, 2x-hybrid-grow-depth; l16 fullmuon-fe-wsd(-5383)(-ema), ref-optim-fullmuon-fe-wsd(-102), ref-optim-fullmuon-wsd | stored record that reaches; factor from a CPU trace (products term) | (ii), once T_ref is decided; the factor is a lower bound |
| 18: s-t20 qat-* (13), shallow-depthshare-3/-6, strideshare; m-t20 shallow-depthshare-4/-8 | record that reaches; the library has no factor (needs E's bits) | (ii) after new code that joins the CPU trace with the record's E bytes |
| 5: s-t20 ref-optim-fullmuon-758, qat-top2, qat-top4, qat-top3-grad4; xl ref-optim-fullmuon-fe-wsd | record that misses the target | (iv) as recorded: a new or longer run |
| 12: s-t20 adamw, sgd, lion, adafactor, accumulate, ema, last-blocks, lora, muon-whole, ref-optim, shampoo(-whole) | traced step only | (iii) [inferred: CPU-feasible; s-t20 records run 5.7e15 to 4.4e16 MACs in 268 to 2005 GPU-s, so hours a seed on CPU, unmeasured] |
| 5: l16 2x-fullmuon-fe-wsd-40326, 2x-hybrid-grow-depth, shallow-depthshare-4; xl ref-optim-…-3884, shallow-depthshare-4 | traced step only | (iii), on GPU; not mine to start |

Every record has no `trace`, so the grade gives them no S. It is the te-core harness's search-pool record, one seed. Each -ema entry shares its base's record, so its loss may not be the average's [inferred].

3. S = p × the record's slowdown [read: README, grade.py, formalism.py; inferred where marked]
- S = (C_e/H)·r_e: gates to target over the honest or least config's gates to target, times a factor read off one traced step.
- Where r_e is Fit's c/(c−d), a construction, S is e's padded size ratio |A_e|/H, the calculator's |A|/T_0. With H = T(𝓕) and gates as cost, it is an upper bound on the slowdown.
- Where r_e is the products term, which is all the library computes, S is a lower bound on e's own padded ratio under (H), (E) and the batch premise. It bounds slowdown(𝒢,𝒜,η) in neither direction, since another algorithm may do better.
- Read off one step, it equals the whole-run figure only if every step has the same ratio [inferred; records show several step structures].

4. Pieces [inferred design]
- P1 `training_slowdown/score.py`: `cost_to_target(record, cfg, reg)`, reusing `grade.checkpoints`/`seed_at`, returning gates bracketed between the last checkpoint above the target and the first at or below it. Not invasive.
- P2 `reference(start)`: per the option you pick. Depends on Q.
- P3 `calculator.py`: `entry_ratio(policy, step, cost, t_ref)` through `calculator.slowdown`, labelled lower bound or witness by `binds`. This keeps the one entry point; `tests/benchmarks/training_slowdown/test_one_calculator.py` may need its KEYS updated.
- P4 `library.py --records DIR`, fetched per `of_record.toml`: S and W side by side, each labelled, with the reason where a row is unscored. JSON only. Moderately invasive.
- P5: the CODED entries' factor from the trace plus the record's E. Touches formalism, so invasive.
- P6: tests on made records, labelled as made.
- P7: runs for the 12 s-t20 entries, on CPU. P8: final-split seeds for η.

Question. T_ref has no definition the code can take. The options:
- (A) Per start, the least no-protocol gates to target over the library's recorded entries. Computable from the store now; it moves when entries are added; no standard algorithm has a record.
- (B) A named reference config per start. s-t20's and xl's named honest configs have no record, so this needs a run (CPU at s-t20).
- (C) The fit's own reference, (g−1)·C_eff(l0). In the tree, no run needed; it assumes continuing training is as efficient as the starts' from-scratch training, and the fit is provisional.

The numerator has three options too:
- (1) The products-term lower bound, which exists.
- (2) An achievable construction, which (b) needs and nothing computes.
- (3) Drop the per-entry factor and report T_min/T_ref per task with p alone per entry.

The unit is also open: gates or GPU-hours. Without a choice I'd take (A)+(1)+gates, labelled lower bound.

Not read: whether Fit holds for any entry at console; whether `gates.py` counts AND gates or all gates; whether final-split records exist in the store; `run.py`'s CPU path.

