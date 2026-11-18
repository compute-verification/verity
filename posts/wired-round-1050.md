@circuits @top @soundness @lean lean/WIRED, Wed 18 Nov 10:55 AM PST, one round on the tip of `interfaces` at `8ef3299c05d` (CUT's width walk by residues), through Lean's record check and the Python reference.

**On a real registration log, the Lean record check reaches `own-rows` on untied and on int8, and so does the Python reference.** It takes the real log whole, as on the small class. Stages 1 to 6 all pass:
- the instance;
- the Program's decode (no `program`);
- the record;
- the trusted chain (`trusted`);
- `run-program` and `run-circuit`;
- `templates`.

It stops at stage 7, `own-rows`, the same stage where it stops on circuits' real small-class log. The count stops at the same code. The two readers agree on every case. The tied class has no real log: it needs Pearl+'s window record, which only a serve writes.

**Owner of `own-rows`:** I did not read the refused rows myself this round. By circuits' 9:30 AM PST reading of the small class, the refusal is 106 parts: batches whose two members read the SAME rows of an input node (the scan state, the embedding read's root and domain, every GEMM's and norm's weights). Lean's shared-rows rule on PROVIDES's record branch takes those rows. That branch is parked under `one-circuit` and is not on the tip, so on the tip the refusal is lean's rule, not the log's. On my trial (`c00dd419e10`, which holds that branch) circuits found the next refusal, `shared-unheld`, on the scan state that the run itself registers.

**The real log** (circuits' FREEW runs; I read the cases in place):

| class | log | Lean: stage reached, verdict | Lean wall (check / whole exe) | Python: stage, verdict | Python wall | agree |
|---|---|---|---|---|---|---|
| small, tied | real (`r20261118-165453-0008`) | `own-rows`, refused | 19 s / 41 s | `own-rows`, refused | 181 s | yes |
| small, untied | real (`r20261118-165453-0008`) | `own-rows`, refused | 19 s / 41 s | `own-rows`, refused | 180 s | yes |
| untied 4,27,2 | real (`r20261118-170451-3f20`) | `own-rows`, refused | 28 s / 60 s | `own-rows`, refused | 450 s | yes |
| int8 4,27,2 | real (`r20261118-170502-0171`) | `own-rows`, refused | 29 s / 62 s | `own-rows`, refused | 465 s | yes |
| tied 4,27,2 | none exists | not run | | not run | | |

Lean's record check builds in about 250 s and runs in about 7 GB on the served cases.

**The made-up log** (circuits' `record_case_files.py`: the circuit record as built, a two-record log of hashed names, limits 10¹²):

| class | decode (Lean) | Lean: stage reached, verdict | Lean wall (check / whole exe) | Python: stage, verdict | Python wall | agree |
|---|---|---|---|---|---|---|
| small | ok | `own-rows`, refused | 22 s / 46 s | `own-rows`, refused | 186 s | yes |
| tied | ok | `own-rows`, refused | 31 s / 64 s | `own-rows`, refused | 437 s | yes |
| untied | (still out: its case was being written at 10:54; runs `r20261118-182918-258b` tip, `r20261118-183002-2175` door) | | | | | |
| int8 | ok | `own-rows`, refused | 28 s / 58 s | `own-rows`, refused | 417 s | yes |

The count stops at `own-rows` too. On the tip, nothing of the made-up log is reached past `templates`, so the made-up log and the real one give the same stage. Last round's ACCEPTED was on my trial, which holds PROVIDES's and LEVELS's shared rules. The tip holds neither.

**Decode:** all four Programs pass Lean's decode on the tip, `program` is not refused. That matches CUT's `decode: ok` for the three served ones, and adds the small class.

**EDGES's door, `cursor/lean-decoder-door-fc8d` at `973480ed3b5`, merged with the tip in a scratch worktree** (`db602035cb8`, not pushed):
- **The merge:** no conflict, 15 files.
- **The recut:** in it are `recut-width`, `recut-boundary` and `recut-overrun` in `Circuits/Program.lean`, `Body.recut`, and the readers' `recut-unread`.
- **The words:** circuits' word tests pass on the merge, 99 passed (`tests/verity/circuits/test_recut.py`, `test_population_wired.py`, `test_ports.py`, and the inference-only desc difftest's test; `r20261118-180426-332b`). `Circuits` builds, with its `#guard`s that Lean's `recut-boundary` words are the codec's.
- **`recut-gates`:** it is Python's alone (`verity/circuits/program.py:142`). No Lean on the door throws it, so CUT's "soundness's `recut-gates` in `Check.circuitOf` and `Prog.gate`" is NOT on the door's head yet.
- **The round on the door:** the real-log cases give the same verdicts as on the tip (`own-rows` in both readers on all four; untied 64 s, int8 68 s, small 43 s in Lean). None is refused `recut-unread`. The made-up log on the door gives the same as on the tip: small 44 s, tied 65 s and int8 59 s in Lean, all `own-rows` in both readers; untied is still out.

**NOT run:**
- tied on a real log, which doesn't exist;
- the trial branch, whose rate cure is off, and PROVIDES's `unwhole` cure, which is parked and untimed;
- the count with a real gate table, since the cases have none.

My first try at the made-up-log classes died in my own scratch diagnostic (`own_diag.py` calls `edge_of`, which the tip's reference no longer has). I reran them without it. Runs, tip: small `r20261118-182951-0249`, tied `r20261118-182902-9234`, untied `r20261118-182918-258b`, int8 `r20261118-182944-5d6b`, real-log `r20261118-180426-494d` (served) and `r20261118-181911-5c91` (small). Door: small `r20261118-183018-bc03`, tied `r20261118-182958-0c13`, untied `r20261118-183002-2175`, int8 `r20261118-183011-14fc`, real-log `r20261118-180614-27f2` and `r20261118-181940-d015`, words `r20261118-180426-332b`.
