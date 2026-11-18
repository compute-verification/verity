[2026-11-18T09:57Z] 765d9556bfff
From proofs lead, 09:58Z: one more from the Lean lead: `Spot.Edges.EdgesExact` (security/Properties/Spot/Edges.lean:69) is a relation, not a statement a lock can carry. Name the open statement as a PROPOSITION in that properties file (e.g. `EdgesOfExact : Prop := forall ..., EdgesExact P (edgesOf P)`, stated against the NEW wire shape, unproved, no sorry/axiom) so the lock can list it as open: owner proofs, date 30 Nov. Say its name and line in your note; the Lean lead lists it when it exists.

