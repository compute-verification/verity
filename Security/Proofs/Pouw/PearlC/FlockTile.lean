module

public import Proofs.Pouw.PearlC.FlockTile.Bridge
public import Proofs.Pouw.PearlC.FlockTile.Audit
public import Proofs.Pouw.PearlC.FlockTile.Gamma

/-!
# Pearl-C under Flock's hidden audit

PoUW's side of Flock's hidden audit, the lemmas: the bridge from Flock's games to core's (`FlockTile.Bridge`), a
`TileGame` per tile over the TileRead session (`FlockTile.Audit`), and the hidden γ with its open obligation
`TileProofSoundAll` discharged there, conditional on T3 at the session and the binding (`FlockTile.Gamma`), which proves
the guarantee `Pouw.Guarantees.PearlC.HiddenFlockSm120v1LoopCast8p72Rev1Cap1000_8192`. The definitions are
`Definitions.Pouw.PearlC.FlockTile`'s, and the statement is `Definitions.Pouw.Guarantees.PearlCFlockTile`'s.
-/

@[expose] public section
