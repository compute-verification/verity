[2026-11-11T21:02Z] d0d26b94de23
grade to training, 2026-11-11T21:02:39Z (from `date`). Resumed. I wait on the third pair; Part 1 follows from it.

RAN: #9029's branch cursor/training-trace-job-dtype-0cd4 is clean at 435bfa4688c7, equal to its remote; not marked.
READ, as they stand: both runs of the third pair are RUNNING, launched once at 20:09Z and not relaunched. They waited
for a slot behind checks on vy-nebius-cpu-7: r20261111-200933-cdc9 (corrected base) took its slot at 20:37Z and is in
its last stage, the as-if tables; r20261111-200946-a35e (published layer) took its at 20:47Z and is in its control
build. Peaks so far 50.7 and 34.6 GB of 56 declared. Each ends about 45 min after its slot.
READ from cdc9's as-if ledger: 809 runs rewritten, 25 unread, as I said at 20:15Z (the second pair: 800 and 34);
13,806 ledger rows rewritten (13,636).

