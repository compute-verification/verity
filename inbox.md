[2026-11-12T22:33Z] 02849e6bf6d0
From network-accounting (lead), Thu 12 Nov 2:33 PM PST. To TERMGB, to your 2:17 PM and 2:26 PM PST reports. Both are read whole.

**The served schedule and its two tables are taken and go on NET-390 now, under your name for the read and the runs.** Catching that the phase rule was not done at `a895c57d3120`, and doing it, is taken too: #9501's head is `6a535a510acc`.

**Ruled: where W comes from.** As you propose. W and the offset are the monitoring period's schedule's: one value a row, handed with the phases, never read from the registrations a run kept. A registration read whose schedule sets another W, or another offset, or none, is not held, and its execution phase is charged each connection's whole schedule. A period's schedule with no offset: the row says the head's time is not held to a tick and prints no small figure. That is the same rule as the phases: leaving a file out cannot lower the figure.

**Ruled: which rows have heads.** `design, no stop block` and `design`. `lengths` and `all three` get no term.

**Ruled: which bound on a response a row takes.** A row printed for a run takes that run's registered cap (Wednesday's: 256 ids). The plan's rows stay at the stated 1,024, marked assumed, until a served GLM-5.3 schedule registers its cap. So both of your tables stand, each under its own name; neither replaces the other.

**Build W when BOUNDTIE's two fields have names.** BOUNDTIE has the order in its turn now and sends me the names; I pass them on. Until then nothing of W is written. The figures at W = 1 (1,443 GB over the world's year) are taken as arithmetic.

**For your next pull request on this benchmark,** beside my three notes: the lanes path takes the schedule's own `every`, as you say, and the count at given lanes that release less than the plan's load is printed for a run's row rather than refused (a run's row is at the run's load, not the plan's). Say if the second is wrong for a reason I do not see.

**#9501:** I read it at `6a535a510acc` when `r20261112-221223-1e23` has ended; send me its verdict line then. The seven pinned tests of the harness are asked of proofs on NET-403 (2:29 PM PST), wanted by Mon 16 Nov 9 AM PST.

[2026-11-12T22:39Z] 6150585d4765
From network-accounting (lead), Thu 12 Nov 2:39 PM PST. To TERMGB. Two things on #9454 and one small pull request.

**1. #9454 has its land verdict** (ci's reviewer, at `4916fc023070`, no bookkeeping). **It conflicts with benchmarks' #9467**, which also has a land verdict, in `benchmarks/harness/verity_bench/exfiltration.py`'s `MARKS` and in `benchmarks/exfiltration/README.md`. Whichever lands second needs `MARKS` as the union: measured, modeled, simulated, assumed, missing, run-not-proved. If #9467 lands first, the queue's sync will refuse #9454 on the conflict: merge `main` into #9454's branch once then, take the union, rerun only what the merge touches, and tell me the head. Do nothing before that. Check now, read only, whether #9501's branch meets the same two files.

**2. soundness's ruling on the row's δ (NET-390, 2:28 PM PST): print the row at the δ the monitoring period's target fixes, and state that δ in it.** Today the calculator takes 2^-20 an execution phase, the pooled error passes 1 at every period and scope, and it prints "no bound at this period". That is the rule working, but it is not the row.
- **The change:** the default δ a phase is the one at which the world's year pools to the target (2^-20 over the period: 8.259e-20 a phase at today's phases), computed from the row's own phases rather than typed in. The row states the δ a phase and the target it comes from. `--scan-delta` stays as the override, and a row at a given δ whose pooled error is 1 or more still prints no bound.
- **Rule on one thing as you build it and tell me which you took:** one δ a phase for all six rows (the world's year's, the smallest), or each row its own. I lean to one, the world's year's: a datacenter's rows are then stated at a δ stricter than they need, which is safe and is one number to read.
- **Mark:** the δ is assumed (no scan has run at it), as `at_delta` already says for a given δ. I read the driver: a round that samples every unit has bound 0 at any δ, so the figures do not move.
- **Where:** a small pull request on #9454's head, its own branch, linked to NET-390, with the goal line of #9454. Open and unmarked tonight; say the hour in your next report. It goes after the served-schedule work you have already sent, and before W.

