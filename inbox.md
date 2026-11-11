[2026-11-11T08:18Z] 9007f29aa3b3
From library (TRN-100), Wed 11 Nov 12:19 AM PST. The first PR's five points are met. Branch cursor/training-algorithm-library-6e3e, head 91e731be6dd, pushed. Draft PR #8840, body rewritten, NOT marked.

THE TABLE, run not read: run r20261111-074845-09b2, CPU node vy-nebius-cpu-3, commit 7dba4eaacc98 (the head differs by one README paragraph), `python -m training_slowdown.library --out "$OUT"`, about five minutes. Start s-t20 (35,658,240 parameters), batch 64, 65,536 tokens a step, policy console (k = 65,536 bits). Slowdown is the calculator's lower bound under (H) and (E). No entry was trained.

entry                  route   slowdown   what binds
s-t20-1.1x-adamw       subset  7.831e+04  HEShared on M, 84% of the MACs whose two operands the update reaches
s-t20-1.1x-sgd         subset  7.831e+04  same
s-t20-1.1x-lion        subset  7.831e+04  same
s-t20-1.1x-adafactor   subset  7.831e+04  same
s-t20-1.1x-ema         ema     7.831e+04  same
s-t20-1.1x-accumulate  subset  7.792e+04  same
s-t20-1.1x-ref-optim   subset  4.548e+04  HEShared on M, 71%
s-t20-1.1x-shampoo     subset  4.445e+04  HEShared on M, 72%
s-t20-1.1x-last-blocks subset  2.081e+04  HEShared on M, 69%
s-t20-1.1x-lora        lora    882        HE on the weight gradients alone, 4%, an operand reached through a rank-16 bottleneck
s-t20-1.1x-shallow-strideshare subset -   needs a record: its E is coded, so the grade reads it

The planned run (838 steps) at its slowdown, as a share of compute, scaled not proved: adamw 2.28e24 AND gates, of one datacenter (reference-h200, 12,500 locations) 1.33e-4 a week, 3.11e-5 a month, 2.56e-6 a year; of the world (world-h200, 4.687e7) 3.56e-8, 8.30e-9, 6.82e-10. lora 1.88e22: 1.10e-6, 2.57e-7, 2.11e-8; world 2.94e-10, 6.85e-11, 5.63e-12. Every entry's are in the body and the run's out/table.txt.

SPREAD, AND ONE THING YOU SHOULD KNOW. Six distinct values. The five at 7.831e4 differ only by the optimizer's gates (sixth digit). Muon and Shampoo are lower NOT because of the optimizer: `subset` has no rule for the embedding beside them, so their entries leave it untrained, and the tied head's three products (3 x 65,472 x 512 x 32,000 MACs, about half of N) are not charged. The stack's test holds that refusal, so I left it and listed the fix (the embedding under AdamW beside them) as a follow-up. Until then plot 1 should not read their rows as "Muon costs less".

ALSO: the nine new configs' rates are standard values, untuned; nothing shows any reaches gain 1.1 in 838 steps (the pause).

THE TEN: eight are configs of subset (adamw, accumulate, sgd, lion, adafactor, shampoo, last-blocks, ref-optim); ema (52 lines) and lora (81 lines) are routes. output(model, dials) is traced at the end of every step and charged; tests/test_run.py holds it by a matmul. No new config has the `honest` key.

CHECKED: VM, tiny shape, 72 passed, ruff clean. Node, suites.py --changed 7b7ae0802eab --quick at 7dba4eaacc98, run r20261111-073623-99e3: benchmarks/training_slowdown 303 passed, its grader 25, repository passed; 57 of 61 passed, none failed, four still running (verity-vllm, verity-numerical, verity-ops, verity-core-evaluate; none reads the change). I wait on it and put the verdict on the issue.

ONECALC: trial merge with a4ed47cb711 has one conflict, the benchmark's README; my tests pass on the merged tree. After #8854 the calculator changes, so the table must be rerun.

FOR YOU TO FILE (in the body, one line each): the neutral name for attacks/; the schema's `honest` key; the embedding beside Muon and Shampoo; six kinds the route API can't express (distillation to a student, RL and decode-time, unlike steps such as DiLoCo, values read into Python, data chosen by the algorithm, a changed architecture).

TWO FINDINGS OUTSIDE THE CHANGE: (1) torch's bf16 Newton-Schulz was not deterministic on 4 threads on this VM (50 of 9,000 calls differed, 6 NaN; none on 1 thread, cause not found); the stack's Muon tests in tests/test_routes.py compare two such results bit for bit and can flake; mine pin one thread. (2) `research queue open` warned that TRN-100 has no label naming a live goal.

Not started, as you said: porting the search's programs. Hand-off note: lanes/training/20261111T0812Z-handoff-trn100-library-ten-entries-table.

