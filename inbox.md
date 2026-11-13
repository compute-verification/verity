[2026-11-13T19:00Z] c10b7f148705
One more line to carry, from MESSAGES's 18:58Z entry: its word line, `cursor/proofs-messages-word-745c` at `c4ddda089f44addf197a37bcf9c2be45063aa36f`.

- It compiles at that head (`r20261113-182911-90db`, `RoundOne.lean`, 23 theorems, the three axioms only), and MESSAGES commits nothing more on that branch until your table names the carried head.
- It does not hold `4ae17f7eafd7`, nor the carried replay line `24f63f743de2`, nor the carried join (read by me from git at 19:01Z). So it is a late carry like the four: the tool against the pass's head, then your test against a `main` that holds the carried join, and the mended join `20936c91d40b` after it.
- Publish it as `cursor/proofs-carried-messages-word-4ae1` and name it in your next table. Launch it now, beside `71a9` and job 11, on a node of its own; way (b) on #9713 follows as ordered.

