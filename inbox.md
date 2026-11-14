[2026-11-14T01:55Z] 8f9d2c27b2ad
TERMGB to network-accounting (lead), Fri 13 Nov 5:55 PM PST. **#9329's run at its mended head ended 0: `r20261114-012103-6071`, exit 0, at 5:52 PM PST, head printed `fe9296fde681d052f4d703d9a3973144e4a116fb`, the head on origin.** [read: the run's record and its files]

- The six lint files: exit 0, 38 passed.
- `suites.py --changed 0a9a1bc70963 --quick`: exit 0. 4 suites, 4 passed: `benchmarks/one_stage` 1,121 passed, with `test_audit_two_members.py` run whole, slow tests included; `repository` 361; `verity-graph` 220; `benchmarks/prove_path` 195.
- What I did not read: the mended test's own line. It is in the pool's log of the suite, which the run's files do not hold. What I read is that its file ran whole and the suite failed nothing; at `aba14110ec6f` the same file run whole gave the one failure.
- **One thing I said at 5:23 PM PST is wrong:** that `benchmarks/one_stage` would be red in `r20261114-001345-b5be` at `0a9a1bc70963`. It passed there, 1,119 passed, because that file is marked slow and `--quick` runs a slow test only where the change reaches its file, which the merge from `aba14110ec6f` does not. So that run does not hold the test; this one does. That run goes on: 122 of 132 passed, 0 failed, 10 left.
- Body of #9329 set at `fe9296fde681` and read back: the tie's arm green at `aba14110ec6f` with each tie's line, the suites' one failure there, the mend, both runs. It says unmarked.
- #9501: 54 of 61 passed, `verity-bench` failed by the eight, 6 left. #9811: 49 passed and 1 kept, `verity-bench` the same, 10 left. [read at 5:53 PM PST]

TIMER: none.

