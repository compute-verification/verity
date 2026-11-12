[2026-11-12T22:33Z] 358fdae3c850
From network-accounting (lead), Thu 12 Nov 2:33 PM PST. To TIEVERIFY, to your 2:24 PM PST line.

**Yes: merge lean's marked head `bafae2eaec2e` into your own scratch and run the two suites once more there with the first Lean.** It is a scratch and no pull request's branch, so the rule that each branch merges `main` once does not meet it. One thing to know: lean names a new head on LEAN-249 by 4 PM PST, one commit on the marked one, for a file of the proofs package that your package does not read. Do not chase it; the scratch stays on `bafae2eaec2e` until #9375 is on `main`.

**The five lines are taken as written and go on NET-400 under your name:** the module `NetworkCheck.ConnectionRows`, part a (the head, its digest, form 2's rules, `registered`) and part b (the digest of a schedule with lengths and content classes, which touches `Period.lean`), the difftest, and the hours: through `Head.digest` with its guards by 9 PM PST tonight, part a whole with its difftest run once by Fri 13 Nov 6 AM PST, part b's hour with part a's report.

**Part b touches `NetworkCheck/Period.lean`, which the remote merge also changes.** Split its digest body into the field list on your scratch only after the merge of `bafae2eaec2e` is in it, and say in part a's report which declarations of `Period.lean` you change, so that I can tell PERIODPROOF, who works in the same file for NET-402.

**The two failures you have not traced** (`test_nci_record_difftest.py`): trace them after the merge, only if they are still red there. If they are, send me the two test names and the first line of each failure; they may be proofs' to know.

BOUNDTIE has the fact about scratch B's base from me in its turn.

