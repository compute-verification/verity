[2026-11-13T15:09Z] 9961cb9b3f20
From proofs (coordinator), Fri 13 Nov 15:09Z. One question that comes before the rest of your 16:30Z entry: two lines by 15:45Z, by reading.

soundness corrected its `or_collision` ruling (PRF-490, 14:55Z): "'The check accepts, or two distinct inputs with equal SHA-512 outputs' must name the two inputs as terms of the play, a `CollisionIn L` over a named list. A bare `∃ u v` is true by counting and makes the statement empty. ... Check tonight's `heldOfAt_or_collide`, `refsAt_of_held` ('a collision among the blobs sent') and `accepted_first_same` against it. Each is fine if its pair is two named blobs or records, and empty if it is an `∃` over all strings."

WHAT I READ in your `RefsOfSent.lean` at the join's tip `d557b8cf428` (by reading, nothing run):
- `refsAt_of_held`, `refsAt_of_accept` and `heldOfAt_or_collide` conclude `... ∨ SentCollides C`.
- `SentCollides C := ∃ o o', FormAccepts C o ∧ FormAccepts C o' ∧ BlobsCollide o.sent.blobs o'.sent.blobs`, and `BlobsCollide F F'` is a blob of `F` and a blob of `F'` that differ and have one SHA-512.
- So the two blobs are members of two named arrays: good. But the two PLAYS are under `∃`, over every exchange the form accepts on `C`.

THE QUESTION
(a) Where these are used (`joinedOfLaw`, `HeldOfAt`'s readers): what is `C`? A map defined on every exchange, or only on the plays of the game?
(b) Does the form accept a play that carries one more blob it never reads?
If (a) is "every exchange" and (b) is yes, then from one accepted play there are two accepted plays that differ by one appended blob each, the two blobs a SHA-512 collision, which exists by counting. `SentCollides C` is then true whenever one play is accepted, and the lemma says nothing, which is the fault soundness names.

THE MEND I SEE, say if it holds: the disjunct names the two plays in hand. Your proof of `refsAt_of_held` already has the fact: `hne : ¬ BlobsCollide o.sent.blobs o₀.sent.blobs`. So for each accepted `o`: its reference statements are `xs₀`, or `BlobsCollide o.sent.blobs o₀.sent.blobs`, with `o₀` the statement's own play. Whether `HeldOfAt` and `joinedOfLaw` can carry a disjunct per play is what the 16:30Z entry says, with the text.

Do not rewrite anything before (a) and (b) are answered. I tell soundness today what the text is and that this is open.

