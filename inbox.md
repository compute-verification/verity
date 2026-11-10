[2026-11-10T02:43Z] ddde46eeca0e
From the coordinator, Mon 9 Nov 6:43 PM PST. Nothing changes in NET-289. One thing to know while you write it:

NET-293 (new, under NET-261) adds a period bound whose count leaves each window's checked inputs out: `PeriodLeakChecked`, which is `PeriodLeakAccepted` with `chk` and `valid` (`CheckedUnique`, `CheckedWitness`), on your operator's new shape, with `PeriodLeakAccepted` as its instance. It is planned as your next job once NET-289's PR is open. Don't start it. TIGHTBOUND sizes it on NET-293 by Tue 10 Nov 6 PM PST.

What it asks of NET-289 now: where it costs nothing, keep `Period.lean`'s counting lemmas (`encard_periodGood_le`, `mem_accSeqsSampled_of_windowGood`) stated over the set of accepted sequences, not over `reg` alone, so the checked form reuses them. If that costs more than a few lines, leave them and say so in your Wed 11 Nov comment.

