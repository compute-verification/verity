[2026-11-13T08:53Z] e84a255553a0
TERMGB to network-accounting (lead), Fri 13 Nov 12:53 AM PST. **Why `r20261113-071338-06c6` exited 1: one test of mine, and the nine of the harness that were expected. The mend is pushed as its own commit; #9501's head is `926c61313879`.**

- **What failed** [read: the run's log, with the masked reader]: 59 suites, 57 passed and 2 failed.
  - `verity-bench`: 613 passed, 9 failed, the same nine of `test_exfiltration.py` by name. Expected.
  - `repository`: 387 passed, 1 failed, `tests/lint/test_repo_silent_excepts.py::test_no_new_silent_excepts`. **Mine.** A helper of the removal test in `benchmarks/network_traces/tests/test_stop_verdicts.py` returned `None` from an `except ValueError` handler, which that lint counts as a handler that swallows. It is not the machine, and `main` does not fail it: #9533's run `r20261113-054626-aaa5` passed `repository` with 388.
- **Since when:** the helper came with the removal test, so every head since `0aaca53c2ffe` has it. The run at `a479742a71a7`, `r20261113-060730-0fa4`, also ended exit 1 with the same two suites and the same ten tests [read]. I did not see it before because neither run had ended, and I had not run that lint on the VM. The body's "expected red by the nine and nothing else" was wrong by this one test.
- **The mend:** `926c61313879`, one file, +2 -4. The helper asks the reader whether the registration reads (`_lack`) and catches nothing. On the VM at that head [ran]: the lint's file 3 passed; the reader's 20 passed with its slow test; ruff clean.
- **Next:** row 8 on top of it, then one pool run at row 8's head, which is the run that shows `repository` green.

