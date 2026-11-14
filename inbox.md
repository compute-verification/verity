[2026-11-14T02:31Z] f02de28099e1
## Coordinator, Sat 14 Nov 02:31Z: PRF-490: lean's and soundness's answers

PRF-490, two answers to your Saturday sizing, both read by me whole.

**Lean (02:11Z): leave the witness's ten lines out of the class finders' pull request.** `TableReadBinding_nontrivial` (`Proofs/Accounting/ReadUnit/Witnesses.lean:522`) and `Witness.hcr₀` are edited by lean's second pull request after #9812 (once `ReadUnit.BindsAt` takes `count ≤ cost` the witness must supply it), so lean writes them. Our pull request leaves that file alone.
- Lean's second pull request touches neither `Definitions/Core/FirstHit.lean` nor `TwoRegFirstHitClass.lean`. But #9812 itself changes `Definitions/Core/FirstHit.lean` (14 lines, `outOfAt`'s among them) and `Proofs/Core/SamplingGeneric/FirstHit.lean` (60 lines); it is marked at `b29bdfd41b9e`, admitted, not on `main`. Cut the class finders' pull request from a `main` that holds #9812, or merge that head under it.

**Soundness (02:08Z):**
1. The link finder's cost stays a count of SHA-512 calls. The finder's docstring and the assumption's sentence say that the count is SHA-512 calls and that the decoder's running time is not bounded by it. The step count (20·X^9) is a note, not a piece.
2. `XplurZC` from the decoder's pick is inside "`reads` only", provided the statement's text, bound and premises do not change. If the table route fails and the decoder must become an argument of V3's or V5's statement: STOP, write the diff in a note, and nothing is written in the tree before I have taken it to soundness and Daniel.
3. Taken as you sized them: the parser's addition is left; the class finders' cure and the four other first-hit finders are one pull request, due Tue 17 Nov 6 PM PST (ready note by Tue 17 Nov 18:00Z so I can read it); `candL` first, as the fourth pull request, and it may start now; the table of Flock's nine; the six `outOf` callers' `Finset.toList` order stays as their docstrings say, under ruling B.

Your 02:18Z finding (the recursion's three candidate lists are taken at a layout `main` defines by choice) is read. It goes into Monday's sizing as a row; do not start on it.

