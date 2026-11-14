[2026-11-14T04:25Z] cc107aa85ebb
## Coordinator, Sat 14 Nov 04:25Z: ACCEPTS: RELAX re-cuts Registered (TOP-415)

ACCEPTS, from proofs (lead), Sat 14 Nov 04:28Z. One thing to know, nothing to stop for.

RELAX opens a pull request by Sun 15 Nov noon PST that re-cuts `Properties.Spot.Registered` (TOP-415, ASK-307 A): the caller's facts leave it for a new statement about the thin protocol's check: `DeclaredDay` out of `R2`, the stream, prover, salt and pool out of `R6`, and `PoolChained`. `R1` and `R7` stay in SPOT's statement. It re-cuts `acceptMeans_iff` in `security/Proofs/Properties/Spot.lean`.

Your conjunct for NET-436's condition 2 sits beside `R7`, so it stays in SPOT's statement under that cut. Both changes edit `Registered` and `acceptMeans_iff`. Do not wait for RELAX: whichever lands second merges the other. In your ready note say in one line where your conjunct sits in `Registered` and which lemma of `acceptMeans_iff`'s proof you touch, so the merge is quick. If your reading says the conjunct is a caller's fact (it reads something outside the record and SPOT's own log), say so in a note before you open the pull request.

