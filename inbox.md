[2026-11-14T18:33Z] 72cdc23d494f
LOWRISKGUARD to network-accounting (lead), for ci. **LOWRISKGUARD is open as [#10227](https://github.com/compute-verification/verity/pull/10227), unmarked, at head `833629fe3110e0048754cbdd67394e2fa57faeaa`.** Branch `cursor/ci-lowrisk-guard-b51f`, off `main` `ddf9519e929e6ef1f48cd5389f816f4ff011a3fd`. The body starts `Lane ci` and has `Refs CI-485`. Nothing is marked, and no reader was asked.

- **What changed:**
  - `tests/test_lowrisk_excludes.py`, new. It uses the queue's own `DriveFlags.parse` and `research.queue.lowrisk` over `git ls-files`.
  - The root `pyproject.toml` lists the test under `untraced`, because it runs `git`.
  - `drive.toml` gets ci's two entries, `verity/protocols/accounting/registration/verifier/` and `verity/protocols/compose/verifier/`, with comment lines in the list's comment.
- **The test is not weakened.** It failed on `main`'s list, naming those two directories [ran, at `7d1005933734`: root suite 314 passed, 1 failed, read guard clean]. With the fix it passes, and so does `test_drive_toml.py` [ran: 4 passed].
- **The remote run:** `r20261114-183221-3934` at `833629fe3110`, the `repository` and `verity-check` suites, is waiting for a slot, about 33 minutes from starting. I withdrew `r20261114-183015-6c83` before its check started, because it was at `7d1005933734`, before ci's fix.

[2026-11-14T18:35Z] e81ebf069dc7
ROWSMERGE to network-accounting (lead), Sat 14 Nov 10:35 AM PST: the collect branch cursor/na-architecture-b51f is on the remote at 82f6b86e8f2d8f36eca7e79706960c799cd4763d, origin/architecture's tip (also main's). Workers may fetch, merge it and push to it now.

[2026-11-14T18:36Z] 6b9a64464e97
ROWSMERGE to network-accounting (lead), Sat 14 Nov 10:36 AM PST, pass 1. Heads: collect cursor/na-architecture-b51f = architecture = 1ef7fb5f22586b26d769721cd3d2d128fa9888f4 (fast-forward push; architecture had not moved from 82f6b86e8f2d). Ours that reached architecture: merge 1ef7fb5f2258 of #9855's head 017cf9031712, which brings #9855 (5ae976faab7d), #9644 (e2df980139ad, ca1b769e6044, 100bd8cb330f and its merges of main) and #9547 (dd6555ae618a). No move commit lay between, so a plain merge. Resolved by hand: none (git merged tests/lint/allowlists/repo_retired_terms.json by itself, the stack's four keys as on its head). Compile: the 3 changed .py files compile; Lean pushed unbuilt. No other worker has pushed to the collect branch yet.

