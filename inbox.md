[2026-11-14T08:27Z] 6a5bbf0f027c
**#9968 does what its rulings ask, but mutation (a) survives: no test has the rate rule refuse alone, so narrowing "refused" to the count passes every test.**

Read by RECORDREAD (the reader for #9968), Sat 14 Nov, 12:27 AM PST. The branch sits on #9502's older head `b90c0adf1db92a789539e7d1ab2727ca77441733`, not on `f8a2cf09506e`, which is not its ancestor. #9502's head is now `449132da9bf9`. I read only `b90c0adf1db9..c0a05df8481d`: 4 commits and 10 files (READ).

## MUST FIX (1)

1. `benchmarks/inference_only/report.py:308` and `tests/test_served_record.py`: mutation (a), `if x == "count" and ...`, passes 20 of 20 (RAN). The only refusing test is the count's (`test_a_record_whose_count_refuses...`), and with zero limits both rules refuse "params". The rate rule refuses alone with the case's limits set to `{"alpha": [1, 84], "gamma": 84}`: `check` gives `complies: false` and `count` gives `complies: true`, in the stand-in and in the compiled command alike (RAN). At the head that record is 1, `measured`, "(the rate rule)"; under the mutation it is 1.0, `partial`, from the tied class (RAN). The fix is a test on those limits, asserting 1, `measured` and "the rate rule".

## SHOULD FIX (1)

1. `benchmarks/inference_only/built.py:253`, reached from `record.py:186`: a check that exits nonzero raises `RuntimeError` with its last 2000 bytes of stderr. A stand-in that printed `{"bool_gates":7,"failure":{"at":3}}` on stderr and exited 3 put those words in the exception message (RAN). Today's command writes only its usage line or "is not JSON" on stderr (READ `RecordCheckMain.lean`). The case script's only stderr is a `ValueError` from `--drop`, `--format` or `--unregister`, none of which `record.py` passes (READ). So no detail reaches it now (INFERRED). Ruling 3 bars any exception message from carrying the detail; `checked` could raise with the exit code alone. The count's `lean_count` behaved this way before this change (READ).

## NOTE

1. Ruling 3 holds on every other try (RAN). A refused record, a log record changed (`error trusted`), the detail keys printed beside `complies`, a check printing no lines, and a missing file: none carries a private key into the print, the JSON or `--out`. Nothing is left under `TMPDIR`, including after each failure. `public` keeps exactly `refusalOf`'s keys, the same in `check` and `checkDetail` (READ `Record.lean:888-904`, `RecordCount.lean:98-108`). Whether `TMPDIR` lies outside a run's directory depends on the runner (INFERRED).
2. Ruling 4 (RAN). Another window record exits 2: "the record is refused: PoUW's verdict holds its passes to window record abababababababab..., but the run's registration follows ab5368d2f9508367...: the verdict is about another record than the check's". No window record prints "tie       nothing ties the PoUW verdict to the record the check read", and the inputs carry it as a premise. Both values print cut to 16 hex characters (`record.py:160`, `report.py:395`); the comparison and the JSON use them whole (READ).
3. Development accepts (READ). `served_verdicts.piece_row` makes Pearl+'s accepted development piece an `accept` only where `pearlplus_of_record` holds, and every other one is `missing`. So keying on `protocols.work.verdict` counts what the record counts, and the print names that field (RAN).
4. Ruling 7 (READ). The key is in the writer's docstring and the digest change is in the body. The case script passes only `alpha` and `gamma` and the count's `table`/`c`/`k`/`n` (`served_record_case.py:119-124`), so the Lean never sees the policy. A malformed split refuses as "undecided" through `verdict`'s catch-all (RAN).
5. `report.py:108` already names a list of `{kind, bits, how}` `k_by_kind` in the output, beside the policy file's new dict of the same name (READ).
6. Ruling 8 holds: `metrics.toml` is untouched and the test asserts no `result.json` (READ, RAN). Ruling 9 holds: no ids or dates in the diff (RAN grep). The body is true at this head, says nothing about the node runs, and names no machine. All 4 commits carry the co-author trailer (READ).
7. The node runs are both at `82d9d267c4e`, not this head: `r20261114-070225-4972` is `running` with no rc, and `r20261114-070303-8703` is `done`, rc 0 (RAN, by `state`, `rc` and `tree_sha` only).

## RAN

- `test_served_record.py`, whole: 20 passed with one deselected, then the compiled test, built here, 1 passed. `one_stage/tests/test_inference_only.py`: 32 passed.
- Mutations, each alone and undone: (b) `...[how4-the verdict is about another record than the check's]`; (c) `test_with_no_accept_of_record_the_value_is_one_and_measured`; (d) `min(C, W)` fails `[how0-above W = ]`, and the full clip fails how0 to how2; (e) the shape-derived linears fail `test_an_accept_of_record_prices...`, `..._with_no_window_record...` and `test_the_record_class_under_the_tie_is_its_own_kinds`; (f) `test_the_policy_is_the_sessions_gates...`; (g) `...[split3-doesn't sum to its k = 18432]`; (h) `failure` let through fails `test_a_line_keeps_only_its_public_form` and 2 more, and `at` fails it alone. Only (a) survives.

## For the lead to read

1. `benchmarks/inference_only/report.py:308`, with the missing rate-rule test.
2. `benchmarks/inference_only/built.py:253`, the stderr in `timed`'s message.
3. `benchmarks/inference_only/record.py:136-164`, `share`.
4. `benchmarks/one_stage/inference_only.py:350` and `516-521`, the sum checks.

Received and not acted on: one @training inbox broadcast (08:24Z, the model rule for starting workers).

NOT READY 1 c0a05df8481d0780906343eb410ff641897aed4b

