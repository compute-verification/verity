[2026-11-14T04:30Z] 5ee8e354f245
KEY, Fri 13 Nov, 8:29 PM PST. The fix for #9474 is pushed: 5a6d7a8d04bf2467af3e3832e542bddb04c638fc on cursor/record-check-less-block-rows-10f7.

RAN: one commit, parent 2d8e9bac34c7, one file, network `tests/test_nci_record_check.py` (+9/−2). It adds a helper `_as_written(made)`: the made case less `block_rows` (where it has one), with `spot` None and `rows` []. Both asserts use it. Its docstring says it in the test's words: a made case may carry `block_rows`, the rows count's limits (`gamma`, `alpha`, `table`) for the difftest; the record check does not read them, so the script writes none.
READ: checked against dd6555ae618a. `finish` sets `"block_rows": rows_rule`, from `rows_limits`, which returns `{"table", "gamma", "alpha"}`. RecordCheckMain reads only the case's `instance`, `trusted` and `count`.
RAN: on the VM at 5a6d7a8d04bf (#9474 alone), `pytest tests/test_nci_record_check.py`: `4 passed in 32.83s`.
RAN: on the VM at my head 14ca2e3c12 with 5a6d7a8d04bf merged (local ee4dff6f3972, not pushed): `5 passed in 30.71s`. That merge has one conflict in that file: my `_more` (rebased on `_restated`) and the new `_as_written` are added at the same place. Both kept; I'll settle it the same way in the real merge.
RAN: the confirmation run, main + 2d8e9bac + dd6555ae without my change (840da2df89, r20261114-041421-5673): `1 suites: 0 passed (0 cached), 1 failed; 7.5 min of test time`, the network suite `94 passed, 2 failed`. The failures are `test_nci_record_check.py::test_the_script_reads_a_runs_files_as_the_case_they_make` and `::test_each_option_makes_the_case_it_says`, the same two as at my head.
RAN: test 5 at my head ended (r20261114-034532-d03d): `59 suites: 58 passed (0 cached), 1 failed; 119.1 min of test time`. The failed one is the network suite, with those two tests. `repository` passed (361), so `test_public_boundary.py` is green.
Pushed nothing more to #9502. Test 6 (the served pair, r20261114-034525-11dd) is still running. I wait for #9474's new head.

