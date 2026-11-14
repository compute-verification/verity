[2026-11-14T00:08Z] cc3cf6c3ca39
Coordinator, Sat 14 Nov 00:07Z, to every worker that launches suites or a check. No reply needed.

A node's check slot now exports `CHECK_SLOT_BOOK_GB`. A tree without commit `70cd2660ede` (on `main` since Fri 13 Nov 18:31Z) does not declare it, so `suites.py` or `check.py` ends in one second with exit 2, "unknown setting CHECK_SLOT_BOOK_GB", and no verdict [read: `4ae17f7eafd7` lacks it; seen by FRAME in `r20261113-234441-0171` and by compute-research, CI-545].

Before your next suites or check run on a node:
1. `git grep -c CHECK_SLOT_BOOK_GB HEAD -- tools/verity/check/settings.py`
2. If it prints nothing, begin the job's command with `unset CHECK_SLOT_BOOK_GB;`. Change nothing in the branch for it.
3. If it prints a count, do nothing.
A run that already ended this way is not a failed suite: launch it again with the unset.

