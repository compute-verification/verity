[2026-11-14T17:16Z] da2dd552485a
GATERELAY2 to network-accounting (lead). TIMER: none. Item 1: both verdicts are in their bodies (`research queue edit`), and no head moved. Both heads, and #10147's cdd92734d4776df95cda3d8494c14e8cc2205b2b, merge cleanly with `main` 723560c29ec1c06f82fda210e974a0f9ebee218b (`git merge-tree`, rc 0). [ran]
- #10139 at e4fab5d158814f41961a3f1fceeb59108376c1e7: `r20261114-155336-af87` done, rc 0, 8 suites, 8 passed, 0 failed. [read: its stdout.log]
- #10155 at 2af58c6adcf48663d0ecd296bcdecd42bb7aa808: `r20261114-160454-ee7f` done, rc 0, 4 suites, 4 passed, 0 failed. Its body gains "Statements for the lead" with your two yeses (the class beside the default, Refs NET-435). [read: its stdout.log]
- Next: the stopped-run rc in #10147. `r20261114-155842-c7ef` is still running at its old head.

[2026-11-14T17:19Z] 73927ccbcedd
GATERELAY2 to network-accounting (lead). TIMER: none. Item 2, the stopped run's rc, is in #10147, now at e3a59c7eb9b1c1c6fc613bbbc263a94c0dcb6707 (base `main`; it merges cleanly with 723560c29ec1c06f82fda210e974a0f9ebee218b).
- The finding is true. A session and a store's write whose runner exits 0 on the unit's SIGTERM ended with rc 0 beside `stopped by its unit (SIGTERM)` at cdd92734d4776df95cda3d8494c14e8cc2205b2b, whose rc lines are #10070's unchanged. [ran]
- The mend: when `stop.signalled`, rc is the runner's when nonzero, else 2, as for a stop before the runner starts and for a session's tunnel closing (`rc or 2`). It covers `certifier_run` and `_weights_run`, and the docstring says so. Its readers [read]: `certifier_piece.py` l.452 (`close` exits 0 only on rc 0), l.572 (accepts only rc 0), `wipe_egress.py` l.102 (likewise), and `served_session.py` l.332 (reports it). The two new tests fail without the mend and pass with it. The two test files pass (157), and ruff equals the base (40). [ran]
- Runs: `r20261114-155842-c7ef` (at cdd92734) is still running, and its verdict goes to the body and to you. `r20261114-171755-b693` (at e3a59c7e, `--changed 723560c29ec1 --quick`, launched before your 9:30 AM line) is waiting for a slot. Both stay as they are.
- From your 9:30 AM line: no served run, gate session, rehearsal or capture was on my list, so I drop nothing.
- Next: the manifest, on your 8:10 AM kinds.

