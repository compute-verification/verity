[2026-11-12T11:21Z] 860557a9a732
BOUNDTIE, NET-349, #9181 at `ad421f210f89073ab84e1eabb980d591b5a7f309`, unmarked, 3:22 AM PST Thu 12 Nov. **55 of the 56 suites have a pass at this head. The circuit-check suite is still running, well inside its limits.** Each status below read by me at 11:20 UTC [ran: `research fetch --all`, each run's `stdout.log`].

- `r20261112-103233-2f27`: done, exit 0, "3 suites: 3 passed (0 cached), 0 failed". `verity/definitions` 950 passed, 2 skipped; `verity-numerical` 641 passed, 7 skipped; `verity-core-evaluate` 67 passed, 1 skipped.
- `r20261112-103244-4dd6`: done, exit 0, "3 suites: 3 passed (0 cached), 0 failed". `verity/circuits` 1,125 passed; `verity/commitments` 592 passed; `verity/commitments/windows` 95 passed, 1 skipped.
- `r20261112-103207-83b5`: the circuit-check suite, **running**. It waited 148 s for a slot and took one at about 10:34 UTC, so it is 46 minutes in. Its `pyproject.toml` says a pass can take up to an hour and gives it 7,200 s [read]; the slot's limit is 12,000 s from taking the slot, which is about 5:54 AM PST. So its own limit ends it by about 4:35 AM PST, 80 minutes before the slot's [argued]. Its log holds no verdict line yet.

**The 56:** 49 passed in `r20261112-070459-b6c4`, 6 passed in the two runs above, 1 is out. No test has failed at this head in any of them.

- I have not merged `main` in and will not.
- The body's Runs section says the above, with its `Goal` and `Refs` lines kept.
- I read the last run once more in 40 minutes and send its verdict. If it is near either limit then, I say so before it is cancelled, and I do not launch it again.

