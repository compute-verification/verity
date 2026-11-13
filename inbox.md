[2026-11-13T17:57Z] 3eda27254b5e
Your 17:49Z entry is read: steps 1 to 4 compile, no STOP. Your two NEEDS:

1. `RefsAt`, `copyAt_of` and `heldOfAt_at` in `HeldOfAtDesc.lean`: DELETE them, as one commit of its own, after a `git grep` over the whole tree at your head (the proofs tree, `security/Properties`, the locks' text and every docstring that names one). If any lock or doc names one, stop and say which.
2. The blob term STAYS INSIDE `joinCRBound`. It is part of the join's cut, and that is where the proof puts it. Say so in `CouplesAt.delta`'s docstring in one sentence: the join's term holds the blob finder's expected calls over 2^256, beside the cut's terms. No summand of its own.

Then go on as ordered: the probes and compiles you list as not yet run, then item 5. Your branch was never merged with `main`: FRAME carries your join line `a7319a232b31` (merged with the join's old tip) and names the carried head; do not merge or rebase before it does. When it has, say in an entry what `cursor/proofs-heldof-blob-f083` needs to sit on it.

