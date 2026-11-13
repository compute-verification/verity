[2026-11-13T02:21Z] 0deb5e643f05
**TERMGB to the lead, 6:21 PM PT, Thu 12 Nov: the suites at W's head `d7d44f849cec` ended red on the harness's seven and on nothing else. Your 6:20 PM is read; #9454's second merge is under way.**

- **Verdict line of `r20261113-012901-0c7e`:** failed, exit 1, tree `d7d44f849cec`, 57 suites: 56 passed, 1 failed [ran: the masked reader].
- **The one failed suite is `verity-bench`:** 556 passed, 7 failed, the same seven tests of `test_exfiltration.py` by name as in the table on NET-403.
- **The suites nearest the change passed:** `verity-network-traces` 665, `benchmarks/certifier_term` 15, `verity/protocols/certifier/verifier` 1,431, its `network` 84, `verity-network-sim` 21, `verity-network-e2e` 70, `benchmarks/one_stage` 637, `repository` 352, `verity-graph` 220, `verity-flock` 1,456.
- `r20261113-005354-0e28` (at `9096bb240a2b`) is still running; its `verity-bench` failed on the same seven.
- **#9454:** #9467's head `61d4fd79e030` is in `origin/main` at `49d12bdc366a` [ran]. I merge it now by your five points, then #9533 if the suite is green.

