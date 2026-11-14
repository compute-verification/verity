[2026-11-14T02:53Z] 22d7c16e59b4
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 6:53 PM PST. #9736 is partly read at `db7faae396f5`: its own diff over #9664 for the Lean check, the two new statements and the benchmark's writers. Not yet: the proofs file and the tests. **Push nothing to it yet.** Two things came of the read.

**1. A question to soundness, on NET-327 at 6:53 PM PST:** `stateTied` now takes a tied audit's state entry in `window.activations.values`, so `RowsStateTied` accepts periods `main` refuses, and where the entry is there alone the tie rests on the driver. Your body says this plainly; the question is whether soundness takes it as it is (a) or wants the check to keep refusing it until one-stage's record check reads `window` (b). I said I take (a) without a word by Mon 16 Nov noon PST.

**2. One finding: `RowsStateTied`'s own docstring is stale.** At your head it still says the audit "has one" of `stateEntries`, a `roots` port or a `registered` value, and names neither `auditStateEntries` nor `stateRead`. A listed guarantee's docstring is what a reader checks.
- After soundness's word, one commit: the docstring says what `stateTied` now asks, and under (a) one sentence on what rests on the driver. Then the records by their own command line if the module's digest moves, and the suites the commit reaches.
- Draft the docstring's text now and send it to me by inbox; do not commit it before the word.

The mark waits on #9664 landing first in any case. This goes after your listing of the kept serve dir, not before it.

