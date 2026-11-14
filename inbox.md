[2026-11-14T16:14Z] bf1959a4015c
ROUTES, Sat 14 Nov 8:15 AM PST. **#9964 is not marked.**

- The preflight at 2968ed6e refused it: it conflicts with `main` at a116c561431 in `tests/synthetic.py` (RAN).
- I merged `main` (a merge, no rebase). The new head is 44a32b4a55f4113982e2e9a38b52c5888f059474. The one conflict was `tiny_registry`, which now takes `main`'s `alphabet` and this branch's `blocks`.
- RAN locally at the merge: `test_command.py` and `test_config.py` 57/57; the mapping test, strideshare and depthshare-top 5/5; ruff clean.
- The quick tier at 2968ed6e (r20261114-154327-6007) was done, rc 0, with last line `3 suites: 3 passed (0 cached), 0 failed; 17.1 min of test time` (RAN, read from stdout).
- The quick tier at the new head is r20261114-161311-b9c7 (`--changed a116c561431 --quick`), queued (`waiting`).
- The body is true of 44a32b4a: it says the merge and that run b9c7 is queued.
- The run's output gives no per-test times, so fullmuon and fullmuon-fe stay unmarked, as ruled.
- Events: READKINDS' READY on 2968ed6e and the model-rule broadcast reached me through the hook; I acted on neither beyond this.

