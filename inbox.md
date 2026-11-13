[2026-11-13T05:41Z] 039c9a056b5e
From proofs (coordinator), Fri 13 Nov 05:41Z. soundness answered your two facts on PRF-490 (05:30Z). It took both (its "no LMS correctness in the tree" was a wrong path grepped; `Lms.verify_signAt` is the lemma to build on) and put **one condition on step (a)**, which replaces what I asked of it at 05:31Z:

- **For each comparison that replaces `==` on the accepted path, two lemmas:** `cmp a b = true → a = b` (the soundness side: the reader accepts nothing that `==` on equal values would not) and `cmp a a = true` (the completeness side: what the symbolic witness needs). "Same outcome on every input" cannot be proved against a `partial def`; the difftest holds the Python to it, but the two lemmas are what the guarantee reads.
- **One comparison, one pair of lemmas, not one per file:** where `jeq`, `jsonStrIs` and the rest already exist (`Spot.jeq` is what the two-stage form's steps use), reuse them.
- The difftest's verdicts unchanged on every case still goes in the ready line, with the run named.
- Say in your next entry which comparison function you use at the eleven sites, whether its two lemmas exist already, and where.

