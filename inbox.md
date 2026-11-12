[2026-11-12T19:49Z] d38d17ddb346
SORT, TRN-146: carry of #8177. What it is for: the Lean verifier's `Slots.popOk` runs from the strata alone (`popOkRanges`, swapped in by `@[csimp] popOk_eq_popOkRanges`), so the population check builds no proof block. READ (the PR title and the diff).

- Branch `cursor/carry-8177-dc90`, head 92579572ddcb8a320540f9e6d980501112ff941e: #8177's head, `main` 30d6fcc92ff7 merged in. It crossed the three moves via `tools/move/restack.py`. RAN (`git ls-remote`).
- The diff against `main` is 1 file, `verity/protocols/spot/verifier/lean/Spot/Slots.lean`, +156. RAN (`git diff --stat`).
- Quick suites `r20261112-185823-bcfb`, rc 0, last line: `61 suites: 61 passed (0 cached), 0 failed; 140.5 min of test time`. RAN (stdout.log of the fetched run).
- `verity lean try …/Spot/Slots.lean --siblings --on auto`, run `r20261112-190057-99ec`, rc 0, which is lean's own status. Its stdout came back empty, so I have no axiom-probe line to quote; the local waiter gave up at its 20-minute cap (exit 79) before the run ended. RAN.
- Records run `r20261112-185903-4d95` (`lean_changed.py --records verity/protocols/spot/verifier/lean --update`): still waiting for a slot. I'll send its result, and any records it rewrites, when it ends. RAN.
- Question: none. It adds definitions and an equality proof that `@[csimp]` uses; no guarantee's statement is touched. READ (the diff).

