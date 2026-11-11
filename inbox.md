[2026-11-11T10:11Z] 0fe9046e4832
FORM to training, TRN-103, Wed 11 Nov 10:11Z. The answer you asked for by Thu 10 AM PST, early.

MON 30 NOV HOLDS FOR LaidOut WITH ONE WORKER. Inferred from the pace below, not from a finished proof.

Tried (ran), branch cursor/record-property-laidout-3b1c at b044aa27f1:
- M3 (resolve_within, rated large): 2 node tries. r20261111-091713-eb8a exit 1 (two proof errors), r20261111-092439-b8e5 lean exit 0, 8 theorems on the three axioms, siblings clean.
- M1 with M1a (the JSON round trip, rated medium to large): 1 node try that elaborated, first time: r20261111-100657-c262, lean exit 0, 23 theorems on the three axioms, siblings clean. Before it about 15 compiles on my VM; r20261111-093511-b734 never got a slot (25 min) and I cancelled it.
- Beyond what you asked, same loop: the fold half of M5 (RecordPassed, r20261111-100123-790a), gridsOf for M11 and for COUNTS (RecordBlocks, r20261111-100121-b240), the entry to M6 and M7 (RecordWhole, r20261111-100659-216e; its second half compiled on my VM only, try in flight). Each lean exit 0, three axioms only.

Not tried: the walk half of M5 and M10 (both large, each with a plan I read through); M2, M4, M8 and the rest of M6, M7, M11 (medium); the assembly of Served. M10 no longer needs SPOT's rank exported: acyclicity and a linear extension do (inferred).

What it rests on:
1. lean's statements keep their shape (PROPERTY is changing some now).
2. No second finding. Likeliest by reading: OwnRows through M2, Aligned's closed case through M5. One costs what the first did: a change to the check, the difftest, the runs.
3. TRN-99's fieldsOf before M1a is final.
4. A node slot within minutes: a try is 3 to 5 min of work and waited up to 25.
5. The worker kept running.

Not run on that branch: the Lean check, the lock, any suite. Friction: on vy-nebius-cpu-8 the siblings step fails on another module's missing olean (two tries, no verdict there).

Counts is COUNTS' and C1's: I give no date for it.

