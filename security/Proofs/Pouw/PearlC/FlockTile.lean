module

public import Proofs.Pouw.PearlC.FlockTile.Audit
public import Proofs.Pouw.PearlC.FlockTile.Gap

/-!
# Pearl-C under Flock's hidden audit

PoUW's side of Flock's hidden audit, the lemmas: a `TileGame` per tile over the TileRead session (`FlockTile.Audit`), and the hidden gap with its open obligation
`TileProofSoundAll` discharged there, conditional on T3 at the session and the binding (`FlockTile.Gap`), which proves
the statement `Pouw.Guarantees.PearlC.HiddenFlockSm120v1LoopCast8p72Rev1Cap1000_8192`. The definitions are
`Definitions.Pouw.PearlC.FlockTile`'s, and the statement is `Proofs.Pouw.Lemmas.PearlCFlockTile`'s.
-/

@[expose] public section
