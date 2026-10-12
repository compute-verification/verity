import Proofs.Pouw.PearlC.FlockTile.Bridge
import Proofs.Pouw.PearlC.FlockTile.Audit
import Proofs.Pouw.PearlC.FlockTile.Statement
import Proofs.Pouw.PearlC.FlockTile.Gamma

/-!
# Pearl-C under C-Flock's hidden audit

PoUW's side of C-Flock's hidden audit: C-Flock's games as core's (`FlockTile.Bridge`), C-Flock's `HiddenAudit` value
over the TileRead session with a `TileGame` per tile (`FlockTile.Audit`), the hidden γ with its open obligation
`TileProofSoundAll` discharged there, conditional on T3 at the session and the binding (`FlockTile.Gamma`), and that γ
as the guarantee `Pouw.Guarantees.PearlC.HiddenFlockSm120v1LoopCast8p72Rev1Cap1000_8192` (`FlockTile.Statement`).
-/
