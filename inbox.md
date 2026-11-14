[2026-11-14T18:33Z] 72cdc23d494f
LOWRISKGUARD to network-accounting (lead), for ci. **LOWRISKGUARD is open as [#10227](https://github.com/compute-verification/verity/pull/10227), unmarked, at head `833629fe3110e0048754cbdd67394e2fa57faeaa`.** Branch `cursor/ci-lowrisk-guard-b51f`, off `main` `ddf9519e929e6ef1f48cd5389f816f4ff011a3fd`. The body starts `Lane ci` and has `Refs CI-485`. Nothing is marked, and no reader was asked.

- **What changed:**
  - `tests/test_lowrisk_excludes.py`, new. It uses the queue's own `DriveFlags.parse` and `research.queue.lowrisk` over `git ls-files`.
  - The root `pyproject.toml` lists the test under `untraced`, because it runs `git`.
  - `drive.toml` gets ci's two entries, `verity/protocols/accounting/registration/verifier/` and `verity/protocols/compose/verifier/`, with comment lines in the list's comment.
- **The test is not weakened.** It failed on `main`'s list, naming those two directories [ran, at `7d1005933734`: root suite 314 passed, 1 failed, read guard clean]. With the fix it passes, and so does `test_drive_toml.py` [ran: 4 passed].
- **The remote run:** `r20261114-183221-3934` at `833629fe3110`, the `repository` and `verity-check` suites, is waiting for a slot, about 33 minutes from starting. I withdrew `r20261114-183015-6c83` before its check started, because it was at `7d1005933734`, before ci's fix.

