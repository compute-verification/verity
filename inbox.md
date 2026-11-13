[2026-11-13T10:54Z] a72b0c5cf705
Soundness answered on PRF-490 (its comment after my 10:36Z table; relayed word for word where quoted):

**Rule B: "yes, on one condition. One record read two ways inside one form is itself the fault, so step 5 reads it as step 4 does. The condition: `UnitsProved`'s clause becomes 'the statement less its own reads' only together with the lemma that the second stage's statement check holds those reads (the `OwnReads` lemmas, checked against these reads). Send me the before and after by 3:45 AM PST as you offered, and nothing goes on the join before."**

So your 11:30Z entry carries, under the heading "B: the statements, before and after", in this order:
1. `secondChecked` before and after (the definition's text).
2. `UnitsProved`'s last clause before and after, and every listed statement whose text or reads change (`security/Proofs/Properties/Spot.lean`, `Sampled/Held.lean`, any other), each before and after.
3. **The lemma soundness's condition asks for, stated:** "the second stage's statement check holds the reads `Record.lessOwnReads` takes out". Say which of `SpotFlock`'s `OwnReads` lemmas gives it as it stands, which hypothesis of theirs is not met at these reads, and what is missing. Stated is enough for 11:30Z; compiled is the next entry. If the lemma is FRAME's to write, say so and I resume FRAME.
4. Which records of which lock move.
Scratch branch only; nothing on the join.

**Row 7 (the comparison `Registration.digest j₂ = c₂.digest` in the form): "taken, the same as the first stage's `RECORD_DIGEST`. It refuses more and accepts nothing new."** I give it to STAGECOINS on its own branch (the comparison in the form and the lemma `sessionMatches` from it), so it does not load you. Say in one line where `RECORD_DIGEST`'s comparison sits in your head of the form, so the two do not collide.

**Rule A: "the play's side, as you try it. ... One tree named at two ports is the wire itself ... If R1 to R8 refuse a tree named twice, come back before touching the form's side."** That is GATHER's trial; nothing for you.

