[2026-11-13T02:58Z] 37aa820af04e
TERMGB to the lead, 6:58 PM PT, Thu 12 Nov. #9501's head, for proofs.

**#9501's branch is at `aee474a20bf752443649b5bd08076c501c3a8dd1`** [ran]: one merge of #9533's `2c431b69a93d` into `ca49f22a3f7f`, pushed as a fast-forward. It holds `main` at `49d12bdc366a` through #9454's `de9bd72fdddd` and #9533, and meets `main` at `8c11a1d5abff` with no conflict [ran: `git merge-tree`].

- The one conflict was the calculator, resolved as on my scratch: `Kind` holds `verdicts` and `mark`, each row of the document carries its `mark`, and there is one `RUN_NOT_PROVED`, beside `MARKS` and `SIMULATED`, for a run's verdicts and for the `scan` row.
- At it [ran, on this machine]: the calculator's and the reader's tests 34 passed, the runner's 15, ruff clean on the calculator. **The harness's file: 27 passed, 9 failed**, the nine of my 6:43 PM PT: the seven of NET-403 (now lines 439, 527, 599, 632, 706, 773, 868), #9467's `test_a_run_not_proved_is_printed_with_its_runs_and_verifier_and_counted_apart_from_measured` (line 1247) and the digests' pin (line 1223), which I left unmoved.
- No pool run at this head, and none launched: it would end red on those nine.

