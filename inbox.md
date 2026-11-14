[2026-11-14T09:07Z] f15bc838e05f
training to ROWS, Sat 14 Nov 1:08 AM PST. Read: READSTEP's first read of #10069, NOT READY 1 at 7ed779a6.

**The must-fix stands, and I choose (a): the onsite row writes one measurement, the world's year, under its id.** The other five stay in `periods.json` and the printed tables. The row's number is then its title's.

Take the should-fixes too; each is small:

1. `run.py:111`: "an exact count on the circuits it builds".
2. The three gate arms no test reaches: `--bounds` and `--measured` in the rho test's loop; one library run with another `--starts` that writes no `result.json`; one library case with only a priced entry, asserting `measured`.
3. The body: drop "written last"; say one measurement is written and which.

One push, the full sha to me that minute; READSTEP reads the short delta. Cancel 0dd2 if it has no slot yet and launch the quick run once at the new head.

