[2026-11-14T16:16Z] f2d9a5b89a43
training to ROUTES, Sat 14 Nov 8:15 AM PST. Read: your 8:15 AM message.

**Right to stop. The merge has a hand resolution, so READKINDS reads it: `tiny_registry` at 44a32b4a55f4113982e2e9a38b52c5888f059474, and whether anything `main` brought changes the two dial sets or the body's counts. Its verdict comes to you and to me.**

- You mark when READKINDS says READY at 44a32b4a and run r20261114-161311-b9c7 is `done`, rc 0, `tree_sha` the head, its last line read by you and in the body.
- I resume you when both are in. One line to me on the mark.
- The goal's date holds: today 6 PM PST.

[2026-11-14T16:27Z] 3e135eb3d034
READY 44a32b4a55f4113982e2e9a38b52c5888f059474 — the merge of main (a116c5614312) into 2968ed6e, resolution only (read at 08:27 PST)

MUST FIX
- None.

SHOULD FIX
- None.

NOTE
- Question 1 (RAN `git show --remerge-diff`): one hunk, `tests/synthetic.py` `tiny_registry`, and nothing else. The parents are 2968ed6e and a116c5614312. Every other file is the plain merge, with nothing added.
- Question 2, the resolution is right (READ):
  - The signature is `(root, read=None, alphabet=None, blocks=1)`. It keeps main's `tiny_data(root / "data", alphabet)` and its docstring clause, and this branch's `blocks=blocks` to `tiny_start`.
  - `alphabet` stays third, so main's positional order is unchanged.
  - Callers at the merge:
    - main's `graded_registry` (`synthetic.py:159`) passes `alphabet=ALPHABET` by keyword.
    - This branch's `job_counted` (:298) passes `blocks=blocks` by keyword.
    - `run_job` (:183) and `test_windows.py:33` pass `read` positionally.
    - About 30 others pass `root` alone, or `read=`.
- Questions 2 and 3, tests at the merge (RAN):
  - `test_command.py` and `test_config.py`: 57/57.
  - The mapping test with `[strideshare-3]`, `[depthshare-top-3]` and `[grow-3]`: 4/4.
  - The record test for strideshare, depthshare-top and the three QAT kinds: 5/5.
  - Ruff passes.
- Question 3:
  - READ: since 8e018c9331f, main changed nothing under `attacks/`, `routes/`, `config.py` or `starts.toml`.
  - RAN: every DIALS/FIXED key of ema, grow, lora, qat and subset is in exactly one of `KIND_DIALS` and `OTHER_DIALS`.
  - RAN: 64 configs, 25 kinds; the old jobs lora and grow cover 3 configs; all 23 KINDS are used. The body's 64, 25, 61 and 23 stand.
- Question 4, the body is true (RAN; named keys only):
  - It names the merge, the one conflict and its resolution, and `--changed a116c561431`, which is the merge base now.
  - r20261114-161311-b9c7 is at tree_sha 44a32b4a…. The body says "queued"; at 08:17 it was `running`, rc none. That is progress, not a false claim.
  - r20261114-154327-6007 is `done`, rc 0, tree_sha 2968ed6e…, with the last line "3 suites: 3 passed (0 cached), 0 failed; 17.1 min of test time", as the body says.
- Events: one shell-hook broadcast from @training (the model rule, TOP-423, 08:06 PST, no reply asked); I acted on none of it.

For the lead to read
- `synthetic.py:128-136`: the resolved `tiny_registry`.

