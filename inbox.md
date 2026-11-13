[2026-11-13T06:23Z] f9c35c62c4c8
QATREST, Thu 12 Nov 22:23 PST (TRN-155)

**The thirteen runs' arguments are read: part 1's eight sparse-ternary runs need no new dial (each is `targets`, `keep`, `decay` and steps of route `qat` as it stands), and part 2's five are two settings, not three as the keys suggest: four runs set all three flags at once and one sets `--grad-code rtn4` alone.**

Source of every line below: each run's `out/run/run.json` (`script_args`, `planned_steps`, `steps[].struct`) and `out/train.py` from its run record (READ; `research data show RUN` for the art, every `train.py` sha256 equal to its record's `program_sha256`).

Common to all thirteen and to the three entries on `main` (READ): `--start s-t20 --mode qat --qat sparse --muon-lr 0.0015 --emit quan --xz min --batch 64 --micro 64`, seed 0, warmup the program's default 50, `--qat-group` default -1 (one scale a matrix), `--qat-mult` default 1, `--decay-shape` default linear. `--micro 64` with `--batch 64` is one forward of 64 rows (route `qat`'s `micro_batches = 1`).

What differs (READ):

| run | key | train | keep | decay | steps | other flags | program | dominant step struct |
|---|---|---|---|---|---|---|---|---|
| -top2 r20261103-181612-5fae | | [45] | 0.35 | 0.4 | 3600 | `--decay-shape sqrt` | 3f4887fc0ba0 | c9e1db36 |
| -top3 r20261103-183131-38c6 | | [345] | 0.2 | 0.4 | 2640 | | 3f4887fc0ba0 | a2a425cb |
| -top4 r20261103-235502-8f39 | | [2345] | 0.75 | 0.4 | 1220 | | 21823ac21845 | 2456e2a2 |
| r20261103-234019-5409 | sparse-ternary | [2345] | 0.50 | 0.4 | 1440 | | 21823ac21845 | 2456e2a2 |
| r20261103-223009-5a78 | sparse-ternary | [2345] | 0.30 | 0.4 | 1800 | | 2c90ac0c0679 | 2456e2a2 |
| r20261103-194119-4cf1 | sparse-ternary | [2345] | 0.15 | 0.4 | 2380 | | 0cfe3eb0a89f | 2456e2a2 |
| r20261103-204652-2289 | sparse-ternary | [2345] | 0.15 | 0.4 | 2020 | | 0cfe3eb0a89f | 2456e2a2 |
| r20261104-060840-e44a | sparse-ternary | [2345] | 0.50 | 0.4 | 7080 | | 21823ac21845 | 2456e2a2 |
| r20261103-202748-901d | sparse-ternary | [2345] | 0.075 | **0.6** | 4320 | | 0cfe3eb0a89f | 2456e2a2 |
| r20261104-001612-606f | sparse-ternary | [2345] | 0.10 | 0.4 | 3020 | | 21823ac21845 | 2456e2a2 |
| r20261103-180926-9f2d | sparse-ternary | [345] | 0.2 | 0.4 | 3000 | | 4883f6c5a077 | a2a425cb |
| r20261103-231515-b8fc | q8-grad-muon | [2345] | 0.15 | 0.4 | 2160 | `--grad-code rtn8 --state-dtype bf16 --muon-buf int8` | 21823ac21845 | fd5a65bd |
| r20261103-234536-71ac | q8-grad-muon | [2345] | 0.30 | 0.4 | 1800 | `--grad-code rtn8 --state-dtype bf16 --muon-buf int8` | 21823ac21845 | fd5a65bd |
| r20261103-230706-70c2 | int8-buffer-muon | [2345] | 0.15 | 0.4 | 2700 | `--grad-code rtn4 --state-dtype bf16 --muon-buf int8` | 21823ac21845 | fd5a65bd |
| r20261104-001556-ec06 | int8-buffer-muon | [2345] | 0.5 | 0.4 | 1720 | `--grad-code rtn4 --state-dtype bf16 --muon-buf int8` | 21823ac21845 | fd5a65bd |
| r20261103-200506-8aaf | q4-grad-muon | [345] | 0.2 | 0.4 | 3000 | `--grad-code rtn4` | 0cfe3eb0a89f | 386aec92 |

Lines where the records differ from your inventory (READ):
- Steps: all thirteen equal your table.
- The two `int8-buffer-muon` runs also set `--grad-code rtn4 --state-dtype bf16`, and the two `q8-grad-muon` runs also set `--state-dtype bf16 --muon-buf int8`: all four run all three settings, differing in code width alone, and all four records have one step structure (fd5a65bd). `q4-grad-muon` (r20261103-200506-8aaf) sets `--grad-code rtn4` alone, at top3's blocks and keep.
- Update structure: your inventory's `2e58b9a460`/`10a60f4507` are not the records' own `struct` hashes; by the records', the seven [2345] sparse runs share -top4's run's dominant structure exactly and r20261103-180926-9f2d shares -top3's (INFERRED: they should hold with -top4's or -top3's `record_only` lines, 28 or 21 scale multiplications by block count).

Program revisions (READ, by `diff` of the five `train.py`s): 4883f6c5a077 is 3f4887fc0ba0 without `--decay-shape`; 0cfe3eb0a89f adds `--grad-code` (default fp32: no op); 2c90ac0c0679 adds `--state-dtype` and turns the f16 gradient-code scales to bf16, and on the default path adds `.to(latent dtype)` and `.float()` on fp32 tensors (no-ops); 21823ac21845 (-top4's, held) adds `--muon-buf`. So on the sparse runs' arguments every revision computes -top4's or -top3's step (INFERRED from the diff; the comparisons will say).

Part 1, the dials: none needed. Entries I am writing, named as `-NNNN` entries on `main` are (the bare name the fewer steps, `-STEPS` the same arguments over more steps):
- `s-t20-1.1x-qat-top4-keep50` (r20261103-234019-5409) and `s-t20-1.1x-qat-top4-keep50-7080` (r20261104-060840-e44a): differ in steps alone;
- `s-t20-1.1x-qat-top4-keep15` (r20261103-204652-2289, 2020) and `s-t20-1.1x-qat-top4-keep15-2380` (r20261103-194119-4cf1): differ in steps alone;
- `s-t20-1.1x-qat-top4-keep30` (r20261103-223009-5a78), `s-t20-1.1x-qat-top4-keep10` (r20261104-001612-606f), `s-t20-1.1x-qat-top4-keep7.5-decay60` (r20261103-202748-901d);
- `s-t20-1.1x-qat-top3-3000` (r20261103-180926-9f2d): -top3's arguments over 3000 steps.

Part 2 needs two dials on route `qat`, I expect (INFERRED, to be argued in the design message): a gradient code (`fp32`, `rtn4`, `rtn8`) and a low-width state (latents bf16 with stochastic update, Muon's buffer int8 with row scales), the four all-three runs being one setting at two code widths. Next: part 1's configs, `of_record.toml` entries and the comparison run on a CPU node.

